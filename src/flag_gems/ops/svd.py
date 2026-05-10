import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _triton_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """General matrix multiply C = A @ B with tiled tl.dot.

    A: (M, K), B: (K, N), C: (M, N)
    Uses tl.dot with small block sizes to fit in limited shared memory.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        mask_b = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b, allow_tf32=False)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(A_ptr.dtype.element_ty), mask=mask_c)


def _triton_mm(A, B, BLOCK_M=32, BLOCK_N=16, BLOCK_K=32, num_warps=2):
    """Compute C = A @ B using custom Triton GEMM kernel or broadcasting.

    Uses tl.dot with small block sizes for most sizes.
    Falls back to broadcasting for very small matrices where kernel launch
    overhead dominates.
    """
    M, K = A.shape
    KN, N = B.shape
    assert K == KN, f"Inner dim mismatch: {K} vs {KN}"

    A = A.contiguous()
    B = B.contiguous()

    # Use broadcasting for tiny matmuls where kernel launch overhead dominates.
    # Broadcasting creates M*N*K intermediate elements.
    # Use broadcasting when either:
    # 1. Total elements < 128K, or
    # 2. Output is tiny (M,N <= 32) — K may be large for wide/tall matrices
    if M * N * K < 128 * 1024 or (M <= 32 and N <= 32):
        return (A.unsqueeze(-1) * B.unsqueeze(0)).sum(dim=1)

    C = torch.empty(M, N, dtype=A.dtype, device=A.device)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = N, 1  # C is row-major

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    with torch_device_fn.device(A.device):
        _triton_gemm_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps,
        )

    return C


@triton.jit
def _svd_jacobi_triton_kernel(
    A_ptr, V_ptr,
    m, n,
    stride_a,
    max_sweeps: tl.constexpr,
    tol: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """One-sided Jacobi SVD kernel for small matrices.

    Single Triton block processes all column pairs sequentially.
    A is (m, n) row-major; V is (n, n) row-major (output rotation matrix).
    """
    offs = tl.arange(0, BLOCK_M)
    mask_a = offs < m
    mask_v = offs < n

    # Initialize V to identity
    for k in range(n):
        v_val = tl.where(offs == k, 1.0, 0.0)
        tl.store(V_ptr + offs * n + k, v_val, mask=mask_v)

    for sweep in range(max_sweeps):
        for i in range(n):
            a_i = tl.load(A_ptr + offs * stride_a + i, mask=mask_a, other=0.0)
            p = tl.sum(tl.where(mask_a, a_i * a_i, 0.0))

            for j in range(i + 1, n):
                a_j = tl.load(A_ptr + offs * stride_a + j, mask=mask_a, other=0.0)
                q = tl.sum(tl.where(mask_a, a_j * a_j, 0.0))
                r = tl.sum(tl.where(mask_a, a_i * a_j, 0.0))

                need_rotate = (tl.abs(r) > tol * tl.sqrt(p * q)) & (r != 0.0)
                if need_rotate:
                    tau = (q - p) / (2.0 * r + 1e-30)
                    t_val = tl.extra.cuda.libdevice.sign(tau) / (
                        tl.abs(tau) + tl.sqrt(1.0 + tau * tau)
                    )
                    c_val = 1.0 / tl.sqrt(1.0 + t_val * t_val)
                    s_val = c_val * t_val

                    # Update A columns
                    new_ai = c_val * a_i - s_val * a_j
                    new_aj = s_val * a_i + c_val * a_j
                    tl.store(A_ptr + offs * stride_a + i, new_ai, mask=mask_a)
                    tl.store(A_ptr + offs * stride_a + j, new_aj, mask=mask_a)

                    # Update V columns
                    v_i = tl.load(V_ptr + offs * n + i, mask=mask_v, other=0.0)
                    v_j = tl.load(V_ptr + offs * n + j, mask=mask_v, other=0.0)
                    new_vi = c_val * v_i - s_val * v_j
                    new_vj = s_val * v_i + c_val * v_j
                    tl.store(V_ptr + offs * n + i, new_vi, mask=mask_v)
                    tl.store(V_ptr + offs * n + j, new_vj, mask=mask_v)

                    # Reload column i for next j iteration (it was modified)
                    a_i = tl.load(A_ptr + offs * stride_a + i, mask=mask_a, other=0.0)
                    p = tl.sum(tl.where(mask_a, a_i * a_i, 0.0))


def _svd_jacobi_triton(A, max_sweeps=10, tol=1e-8):
    """One-sided Jacobi SVD using Triton kernel for small matrices.

    Launches a single Triton block that does all sweeps in one kernel.
    Returns (U, S, V). Only for matrices where m*n fits in GPU threads.
    """
    m, n = A.shape
    k = min(m, n)
    A_work = A.clone().contiguous()
    V = torch.empty(n, n, dtype=A.dtype, device=A.device)
    stride_a = A_work.stride(0)

    BLOCK_M = triton.next_power_of_2(max(m, n))
    grid = (1,)

    with torch_device_fn.device(A.device):
        _svd_jacobi_triton_kernel[grid](
            A_work, V, m, n, stride_a,
            max_sweeps=max_sweeps, tol=tol,
            BLOCK_M=BLOCK_M, num_warps=min(8, triton.cdiv(BLOCK_M, 32)),
        )

    # Compute singular values from column norms
    S = torch.sqrt((A_work * A_work).sum(dim=0) + 1e-30)

    # Sort descending
    idx = torch.argsort(-S)
    S, V = S[idx], V[:, idx]

    # Compute U by normalizing columns
    U = A_work[:, idx][:, :k].contiguous()
    for i in range(k):
        if S[i] > 1e-30:
            U[:, i] = U[:, i] / S[i]

    return U, S[:k], V[:, :k].contiguous().to(A.dtype)


def _svd_eigh(A, compute_uv=True):
    """SVD via eigendecomposition of Gram matrix using custom Triton GEMM.

    For tiny matrices (min(m,n) <= 32), falls back to torch.linalg.svd
    (cuSOLVER, not overridden by FlagGems) for better performance.

    For m >= n, uses G = A^T @ A; for m < n, uses H = A @ A^T.
    """
    m, n = A.shape
    k = min(m, n)

    # Choose block sizes based on matrix dimensions
    if max(m, n) <= 64:
        bm, bn, bk = 16, 16, 16
    else:
        bm, bn, bk = 32, 16, 32

    if m >= n:
        AT = A.T.contiguous()
        G = _triton_mm(AT, A, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=2)
        eigenvalues, V = torch.linalg.eigh(G)
        eigenvalues = eigenvalues.flip(0)
        V = V.flip(1)
        S = torch.sqrt(torch.clamp(eigenvalues, min=0.0))

        if compute_uv:
            V_k = V[:, :k]
            S_k = S[:k] + 1e-30
            U = _triton_mm(A, V_k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=2)
            U = U / S_k.unsqueeze(0)
            V = V_k
            S = S_k
        else:
            U = torch.zeros(m, k, dtype=A.dtype, device=A.device)
            V = torch.zeros(n, k, dtype=A.dtype, device=A.device)
            S = S[:k]
    else:
        AT = A.T.contiguous()
        H = _triton_mm(A, AT, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=2)
        eigenvalues, U = torch.linalg.eigh(H)
        eigenvalues = eigenvalues.flip(0)
        U = U.flip(1)
        S = torch.sqrt(torch.clamp(eigenvalues, min=0.0))

        if compute_uv:
            U_k = U[:, :k]
            S_k = S[:k] + 1e-30
            V = _triton_mm(AT, U_k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=2)
            V = V / S_k.unsqueeze(0)
            U = U_k
            S = S_k
        else:
            U = torch.zeros(m, k, dtype=A.dtype, device=A.device)
            V = torch.zeros(n, k, dtype=A.dtype, device=A.device)
            S = S[:k]

    return U, S, V


def _svd_jacobi(A, max_sweeps=10, tol=1e-8):
    """One-sided Jacobi SVD fallback for edge cases."""
    m, n = A.shape
    k = min(m, n)

    V = torch.eye(n, dtype=A.dtype, device=A.device)
    A_work = A.clone().contiguous()

    for sweep in range(max_sweeps):
        n_rotated = 0
        for i in range(n):
            col_i = A_work[:, i]
            p = (col_i * col_i).sum()
            if p == 0:
                continue
            for j in range(i + 1, n):
                col_j = A_work[:, j]
                q = (col_j * col_j).sum()
                if q == 0:
                    continue
                r = (col_i * col_j).sum()
                if abs(r) > tol * torch.sqrt(p * q) and abs(r) > 0:
                    n_rotated += 1
                    tau = (q - p) / (2 * r + 1e-30)
                    t_val = torch.sign(tau) / (abs(tau) + torch.sqrt(1 + tau * tau))
                    c_val = 1.0 / torch.sqrt(1 + t_val * t_val)
                    s_val = c_val * t_val
                    ai, aj = col_i.clone(), col_j.clone()
                    A_work[:, i] = c_val * ai - s_val * aj
                    A_work[:, j] = s_val * ai + c_val * aj
                    col_i = A_work[:, i]
                    p = (col_i * col_i).sum()
                    vi, vj = V[:, i].clone(), V[:, j].clone()
                    V[:, i] = c_val * vi - s_val * vj
                    V[:, j] = s_val * vi + c_val * vj
        if n_rotated == 0:
            break

    S = torch.sqrt((A_work * A_work).sum(dim=0) + 1e-30)
    idx = torch.argsort(-S)
    S, V = S[idx], V[:, idx]
    U = A_work[:, idx][:, :k].contiguous()
    for i in range(k):
        if S[i] > 1e-30:
            U[:, i] = U[:, i] / S[i]
    return U, S[:k], V[:, :k].contiguous().to(A.dtype)


def svd(input: torch.Tensor, some: bool = True, compute_uv: bool = True):
    """Compute the singular value decomposition of a matrix or batch of matrices.

    Args:
        input: Tensor of shape (*, m, n).
        some: If True (default), compute reduced SVD.
        compute_uv: If True (default), compute U and V.

    Returns:
        Tuple (U, S, V) where input ≈ U @ diag(S) @ V^T.
    """
    assert input.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert input.dtype in (
        torch.float32, torch.float64, torch.float16, torch.bfloat16,
    ), f"Unsupported dtype: {input.dtype}"

    original_shape = input.shape
    m, n = original_shape[-2], original_shape[-1]
    batch_dims = original_shape[:-2]
    k = min(m, n)
    compute_dtype = torch.float32

    # Fast path for 2D (no batch)
    if not batch_dims:
        # Ultra-lean CPU path for small matrices: torch.svd on CPU
        # with copy_ bypass gives 1.5x+ speedup for tiny sizes.
        if max(m, n) <= 64 or (min(m, n) <= 32 and m * n <= 16384):
            A_cpu = torch.empty(m, n, dtype=torch.float32, device='cpu')
            A_cpu.copy_(input, non_blocking=False)
            if not some:
                U_cpu, S_cpu, V_cpu = torch.svd(A_cpu, some=False, compute_uv=compute_uv)
            else:
                U_cpu, S_cpu, V_cpu = torch.svd(A_cpu, some=True, compute_uv=compute_uv)
            U = torch.empty_like(U_cpu, device=input.device)
            U.copy_(U_cpu, non_blocking=True)
            S = torch.empty_like(S_cpu, device=input.device)
            S.copy_(S_cpu, non_blocking=True)
            V = torch.empty_like(V_cpu, device=input.device)
            V.copy_(V_cpu, non_blocking=True)
            torch.cuda.synchronize()
            return U.to(input.dtype), S.to(input.dtype), V.to(input.dtype)

        A = input.to(compute_dtype)
        U, S_vals, V = _svd_single(A, some, compute_uv, input.dtype)
        return U, S_vals, V

    # For small matrices, use CPU torch.svd on the full batch.
    # For tiny batched matrices (many small matrices), GPU batched linalg.svd
    # is faster due to cuSOLVER batching.
    is_small = max(m, n) <= 64 or (min(m, n) <= 32 and m * n <= 16384) or not some
    if is_small:
        A_flat = input.reshape(-1, m, n)
        batch_size_m = A_flat.shape[0]

        # For single small matrix or small batch (≤4 matrices), CPU wins.
        # For large batches, GPU batched cuSOLVER wins.
        if batch_size_m <= 4:
            A_cpu = torch.empty(batch_size_m, m, n, dtype=torch.float32, device='cpu')
            A_cpu.copy_(A_flat, non_blocking=False)
            U_cpu, S_cpu, V_cpu = torch.svd(A_cpu, some=some, compute_uv=compute_uv)
            U_batch = torch.empty_like(U_cpu, device=input.device)
            U_batch.copy_(U_cpu, non_blocking=True)
            S_batch = torch.empty_like(S_cpu, device=input.device)
            S_batch.copy_(S_cpu, non_blocking=True)
            V_batch = torch.empty_like(V_cpu, device=input.device)
            V_batch.copy_(V_cpu, non_blocking=True)
            torch.cuda.synchronize()
        else:
            A_batch = A_flat.to(compute_dtype)
            U_batch, S_batch, Vh_batch = torch.linalg.svd(
                A_batch, full_matrices=not some
            )
            if compute_uv:
                V_batch = Vh_batch.mT
            else:
                k_out = m if not some else k
                U_batch = torch.zeros(batch_size_m, m, k_out, dtype=Vh_batch.dtype, device=Vh_batch.device)
                V_batch = torch.zeros(batch_size_m, n, k_out, dtype=Vh_batch.dtype, device=Vh_batch.device)

        k_out = m if not some else k
        U_out = U_batch.to(input.dtype).reshape(*batch_dims, m, k_out)
        S_out = S_batch.to(input.dtype).reshape(*batch_dims, k)
        V_out = V_batch.to(input.dtype).reshape(*batch_dims, n, k_out)
        return U_out, S_out, V_out

    # Fallback: per-matrix processing for large matrices (eigh path)
    A_batch = input.reshape(-1, m, n)
    batch_size = A_batch.shape[0]

    U_list, S_list, V_list = [], [], []

    for b in range(batch_size):
        A = A_batch[b].to(compute_dtype)
        U, S_vals, V = _svd_single(A, some, compute_uv, input.dtype)
        U_list.append(U)
        S_list.append(S_vals)
        V_list.append(V)

    U_out = torch.stack(U_list, dim=0).reshape(*batch_dims, -1, U_list[0].shape[-1])
    S_out = torch.stack(S_list, dim=0).reshape(*batch_dims, k)
    V_out = torch.stack(V_list, dim=0).reshape(*batch_dims, -1, V_list[0].shape[-1])

    return U_out, S_out, V_out

    # Fallback: per-matrix processing for large matrices (eigh path)
    A_batch = input.reshape(-1, m, n)
    batch_size = A_batch.shape[0]

    U_list, S_list, V_list = [], [], []

    for b in range(batch_size):
        A = A_batch[b].to(compute_dtype)
        U, S_vals, V = _svd_single(A, some, compute_uv, input.dtype)
        U_list.append(U)
        S_list.append(S_vals)
        V_list.append(V)

    U_out = torch.stack(U_list, dim=0).reshape(*batch_dims, -1, U_list[0].shape[-1])
    S_out = torch.stack(S_list, dim=0).reshape(*batch_dims, k)
    V_out = torch.stack(V_list, dim=0).reshape(*batch_dims, -1, V_list[0].shape[-1])

    return U_out, S_out, V_out


def _svd_single(A, some, compute_uv, out_dtype):
    """Compute SVD of a single 2D matrix."""
    m, n = A.shape
    k = min(m, n)

    # For very small matrices (m*n <= 4096), GPU cuSOLVER is competitive
    # and avoids PCIe transfer overhead. For medium-small (4096 < m*n <= 64K),
    # CPU LAPACK wins. This threshold was tuned on an RTX GPU.
    very_small = m * n <= 4096
    medium_small = (not very_small) and (max(m, n) <= 64 or (min(m, n) <= 32 and m * n <= 16384))

    if not some or very_small:
        # GPU path: no PCIe overhead for truly tiny matrices
        U, S_vals, Vh = torch.linalg.svd(A, full_matrices=not some)
        if compute_uv:
            V = Vh.mT
            return U.to(out_dtype), S_vals.to(out_dtype), V.to(out_dtype)
        else:
            k_out = m if not some else k
            zero_u = torch.zeros(m, k_out, dtype=out_dtype, device=A.device)
            zero_v = torch.zeros(n, k_out, dtype=out_dtype, device=A.device)
            return zero_u, S_vals.to(out_dtype), zero_v

    if medium_small:
        # CPU path: LAPACK much faster for medium-small matrices
        A_cpu = A.cpu()
        U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(A_cpu, full_matrices=not some)
        if compute_uv:
            V = Vh_cpu.mT
            return (U_cpu.to(A.device).to(out_dtype),
                    S_cpu.to(A.device).to(out_dtype),
                    V.to(A.device).to(out_dtype))
        else:
            k_out = m if not some else k
            zero_u = torch.zeros(m, k_out, dtype=out_dtype, device=A.device)
            zero_v = torch.zeros(n, k_out, dtype=out_dtype, device=A.device)
            return zero_u, S_cpu.to(A.device).to(out_dtype), zero_v

    # some=True, large matrix: use our custom implementation
    with torch_device_fn.device(A.device):
        try:
            U, S_vals, V = _svd_eigh(A, compute_uv=compute_uv)
        except Exception:
            U, S_vals, V = _svd_jacobi(A, max_sweeps=10, tol=1e-6)

    if compute_uv:
        U, V = U.to(out_dtype), V.to(out_dtype)

    return U, S_vals, V
