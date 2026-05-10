"""One-sided Jacobi SVD with fused GPU kernels.

Optimizations:
- Fused Jacobi step kernel (Gram + rotation + A_t update + V update)
- Pairs tensors pre-built once, reused across sweeps
- All pairs for all steps packed into one GPU tensor (2D layout)
- Adaptive BLOCK_M sizing
- Convergence check every few sweeps via GPU reduction
- V stored transposed (Vt) for contiguous column access
- full_matrices=True uses torch.linalg.qr for stable complement

Reference: Hestenes one-sided Jacobi method with Brent-Luk parallel ordering.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Brent-Luk parallel ordering
# ---------------------------------------------------------------------------

def _brent_luk_pairs(n: int):
    if n <= 1:
        return []
    all_steps = []
    n_eff = n if n % 2 == 0 else n + 1
    for step in range(n_eff - 1):
        pairs = []
        for k in range(n_eff // 2):
            i = (step + k) % (n_eff - 1)
            j = n_eff - 1 if k == 0 else (step + n_eff - 1 - k) % (n_eff - 1)
            if i < n and j < n:
                pairs.append((i, j))
        all_steps.append(pairs)
    return all_steps


def _build_pairs_tensor(all_steps, device):
    """Pack all Jacobi step pairs into a single GPU tensor.

    Layout: (total_steps, 2 * max_pairs)  — row-major per step.
    Each row stores [i0, i1, ..., iP, j0, j1, ..., jP] with zero-padding.
    Returns (pairs_all, max_pairs, step_offsets).
    """
    num_steps = len(all_steps)
    max_pairs = max((len(s) for s in all_steps), default=0)
    if max_pairs == 0:
        return None, 0, []
    flat = []
    step_offsets = []
    for step_pairs in all_steps:
        step_offsets.append(len(flat))
        for i, j in step_pairs:
            flat.append(i)
        for i, j in step_pairs:
            flat.append(j)
        pad_n = max_pairs - len(step_pairs)
        flat.extend([0] * pad_n + [0] * pad_n)
    return torch.tensor(flat, device=device, dtype=torch.int32), max_pairs, step_offsets


# ---------------------------------------------------------------------------
# Kernel: fused Jacobi step
# ---------------------------------------------------------------------------

@triton.jit
def jacobi_step_kernel(
    A_t_ptr,
    V_ptr,         # (batch, n, n) row-major
    m: tl.constexpr,
    n: tl.constexpr,
    pairs_ptr,     # row in pairs_all for current step
    num_pairs: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pair_id = pid % num_pairs
    batch_id = pid // num_pairs

    i = tl.load(pairs_ptr + pair_id).to(tl.int32)
    j = tl.load(pairs_ptr + pair_id + num_pairs).to(tl.int32)

    row_i = batch_id * n * m + i * m
    row_j = batch_id * n * m + j * m

    # --- Gram accumulation ---
    a_ii = tl.zeros([1], dtype=tl.float32)
    a_ij = tl.zeros([1], dtype=tl.float32)
    a_jj = tl.zeros([1], dtype=tl.float32)

    for m_start in range(0, m, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < m
        ci = tl.load(A_t_ptr + row_i + offs, mask=mask, other=0.0).to(tl.float32)
        cj = tl.load(A_t_ptr + row_j + offs, mask=mask, other=0.0).to(tl.float32)
        a_ii += tl.sum(ci * ci, axis=0)
        a_ij += tl.sum(ci * cj, axis=0)
        a_jj += tl.sum(cj * cj, axis=0)

    a_ii_s = tl.sum(a_ii, axis=0)
    a_ij_s = tl.sum(a_ij, axis=0)
    a_jj_s = tl.sum(a_jj, axis=0)

    # --- Jacobi rotation ---
    tau = (a_ii_s - a_jj_s) / (2.0 * a_ij_s + 1e-30)
    t_pos = 1.0 / (tau + tl.sqrt(1.0 + tau * tau))
    t_neg = -1.0 / (-tau + tl.sqrt(1.0 + tau * tau))
    t = tl.where(tau >= 0.0, t_pos, t_neg)
    off = tl.abs(a_ij_s) / (tl.sqrt(tl.abs(a_ii_s * a_jj_s)) + 1e-30)
    t = tl.where(off < 1e-10, 0.0, t)

    cos = 1.0 / tl.sqrt(1.0 + t * t)
    sin = t * cos

    # --- Apply rotation to A_t columns ---
    for m_start in range(0, m, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < m
        ci = tl.load(A_t_ptr + row_i + offs, mask=mask, other=0.0).to(tl.float32)
        cj = tl.load(A_t_ptr + row_j + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(A_t_ptr + row_i + offs, cos * ci + sin * cj, mask=mask)
        tl.store(A_t_ptr + row_j + offs, -sin * ci + cos * cj, mask=mask)

    # --- Apply rotation to V columns ---
    v_batch_off = batch_id * n * n
    for r in range(0, n, 64):
        rows = r + tl.arange(0, 64)
        mask = rows < n
        vi = tl.load(V_ptr + v_batch_off + rows * n + i, mask=mask, other=0.0).to(tl.float32)
        vj = tl.load(V_ptr + v_batch_off + rows * n + j, mask=mask, other=0.0).to(tl.float32)
        tl.store(V_ptr + v_batch_off + rows * n + i, cos * vi + sin * vj, mask=mask)
        tl.store(V_ptr + v_batch_off + rows * n + j, -sin * vi + cos * vj, mask=mask)


# ---------------------------------------------------------------------------
# Kernel: normalize
# ---------------------------------------------------------------------------

@triton.jit
def normalize_kernel(
    A_t_ptr, S_ptr, U_ptr,
    m: tl.constexpr, n: tl.constexpr, BLOCK_M: tl.constexpr,
):
    col_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    row_off = batch_id * n * m + col_id * m

    acc = tl.zeros([1], dtype=tl.float32)
    for m_start in range(0, m, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < m
        vals = tl.load(A_t_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals * vals, axis=0)

    sigma = tl.sqrt(tl.sum(acc, axis=0))
    tl.store(S_ptr + batch_id * n + col_id, sigma)

    safe_sigma = sigma + 1e-30
    for m_start in range(0, m, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < m
        vals = tl.load(A_t_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(U_ptr + batch_id * m * n + offs * n + col_id, vals / safe_sigma, mask=mask)


# ---------------------------------------------------------------------------
# Helper: full_matrices U extension via QR
# ---------------------------------------------------------------------------

def _extend_U_to_square(U: torch.Tensor) -> torch.Tensor:
    """Extend U from (batch, m, n) to (batch, m, m) using QR on a random complement."""
    batch_size, m, n = U.shape
    if m - n <= 0:
        return U
    device, dtype = U.device, U.dtype
    R = torch.randn(batch_size, m, m - n, device=device, dtype=dtype)
    # Subtract projection onto existing U columns
    Ut = U.transpose(-2, -1)
    R = R - U @ (Ut @ R)
    Q, _ = torch.linalg.qr(R)
    return torch.cat([U, Q], dim=-1).to(dtype)


# ---------------------------------------------------------------------------
# Convergence check (CPU-side, called every few sweeps)
# ---------------------------------------------------------------------------

def _check_converged(A_t, step0, tol):
    """Check if first column pair in step0 is nearly orthogonal."""
    if len(step0) == 0:
        return True
    i, j = step0[0]
    ci = A_t[:, i, :]
    cj = A_t[:, j, :]
    a_ij = (ci * cj).sum(dim=-1).abs()
    a_ii = (ci * ci).sum(dim=-1)
    a_jj = (cj * cj).sum(dim=-1)
    off = a_ij / (a_ii * a_jj).sqrt().clamp_min(1e-30)
    return off.max().item() < tol


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

def svd(A: torch.Tensor, full_matrices: bool = True):
    """One-sided Jacobi SVD with fused GPU kernels.

    Args:
        A: tensor of shape (..., m, n)
        full_matrices: controls U/Vh shape (see torch.linalg.svd).

    Returns:
        (U, S, Vh)
    """
    if A.ndim < 2:
        raise RuntimeError(f"linalg.svd: input must have >= 2 dims, got {A.ndim}")

    original_batch_shape = A.shape[:-2]
    m_orig, n_orig = A.shape[-2], A.shape[-1]
    A = A.reshape(-1, m_orig, n_orig) if len(original_batch_shape) > 0 else A.unsqueeze(0)
    batch_size = A.shape[0]

    swapped = m_orig < n_orig
    if swapped:
        A = A.transpose(-2, -1)
    m, n = A.shape[-2], A.shape[-1]
    device, dtype = A.device, A.dtype

    # Empty shortcut
    if m == 0 or n == 0:
        k = 0
        U = torch.zeros(batch_size, m_orig, m_orig if full_matrices else k, device=device, dtype=dtype)
        S = torch.zeros(batch_size, k, device=device, dtype=dtype)
        Vh = torch.zeros(batch_size, n_orig if full_matrices else k, n_orig, device=device, dtype=dtype)
        if len(original_batch_shape) == 0:
            U, S, Vh = U.squeeze(0), S.squeeze(0), Vh.squeeze(0)
        else:
            U = U.reshape(*original_batch_shape, *U.shape[-2:])
            S = S.reshape(*original_batch_shape, S.shape[-1])
            Vh = Vh.reshape(*original_batch_shape, *Vh.shape[-2:])
        return U, S, Vh

    # A_t: (batch, n, m), V: (batch, n, n) — row-major
    A_t = A.transpose(-2, -1).contiguous()
    V = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n, n).contiguous()

    # Adaptive block size
    BLOCK_M = 128 if m <= 128 else (256 if m <= 512 else (512 if m <= 2048 else 1024))

    all_steps = _brent_luk_pairs(n)
    if len(all_steps) > 0:
        # Pre-build all pairs into a single GPU tensor
        pairs_all, max_pairs, step_offsets = _build_pairs_tensor(all_steps, device)

        max_sweeps, tol = 30, 1e-6
        for sweep in range(max_sweeps):
            for si, step_pairs in enumerate(all_steps):
                num_pairs = len(step_pairs)
                if num_pairs == 0:
                    continue
                jacobi_step_kernel[(num_pairs * batch_size,)](
                    A_t, V, m, n,
                    pairs_all[step_offsets[si]:],
                    num_pairs=num_pairs, BLOCK_M=BLOCK_M,
                )

            # Convergence check every 5 sweeps
            if sweep >= 4 and (sweep + 1) % 5 == 0:
                if _check_converged(A_t, all_steps[0], tol):
                    break

    # Normalize
    S = torch.empty(batch_size, n, device=device, dtype=torch.float32)
    U = torch.empty(batch_size, m, n, device=device, dtype=torch.float32)
    normalize_kernel[(n, batch_size)](A_t, S, U, m, n, BLOCK_M=BLOCK_M)

    # Sort
    S_sorted, idx = torch.sort(S, dim=-1, descending=True)
    U_sorted = torch.gather(U, -1, idx.unsqueeze(-2).expand(-1, m, -1))
    V_sorted = torch.gather(V, -1, idx.unsqueeze(1).expand(-1, n, -1))
    Vh = V_sorted.transpose(-2, -1)

    # full_matrices
    if full_matrices and m > n:
        U_final = _extend_U_to_square(U_sorted)
        S_final, Vh_final = S_sorted, Vh
    else:
        U_final, S_final, Vh_final = U_sorted, S_sorted, Vh

    if swapped:
        U_final, Vh_final = Vh_final.transpose(-2, -1), U_final.transpose(-2, -1)

    if len(original_batch_shape) == 0:
        U_final, S_final, Vh_final = U_final.squeeze(0), S_final.squeeze(0), Vh_final.squeeze(0)
    else:
        U_final = U_final.reshape(*original_batch_shape, *U_final.shape[-2:])
        S_final = S_final.reshape(*original_batch_shape, S_final.shape[-1])
        Vh_final = Vh_final.reshape(*original_batch_shape, *Vh_final.shape[-2:])

    if dtype != torch.float32:
        U_final, S_final, Vh_final = U_final.to(dtype), S_final.to(dtype), Vh_final.to(dtype)

    return U_final, S_final, Vh_final
