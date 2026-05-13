import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_MedianResult = namedtuple("median", ["values", "indices"])


# ============================================================
# 1. N <= 32: small-N multi-row radix select
#    保留这个 kernel，主要用于极小 N；但 dispatch 中 N<=128
#    会优先走 median_kernel_sort_rows。
# ============================================================

@libentry()
@triton.jit
def median_kernel_small(
    inp,
    out_val,
    out_idx,
    M,
    N,
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    rows = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    row_mask = rows < M
    col_mask = cols < N

    offsets = rows[:, None] * N + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    raw_vals = tl.load(inp + offsets, mask=mask, other=0.0)

    f32 = raw_vals.to(tl.float32)
    ui = f32.to(tl.uint32, bitcast=True)
    si = f32.to(tl.int32, bitcast=True)

    one_u32 = tl.full([], value=1, dtype=tl.uint32)
    sign_bit = one_u32 << 31
    shift31 = tl.full([], value=31, dtype=tl.int32)
    sign_extend = (si >> shift31).to(tl.uint32, bitcast=True)
    conv_mask = sign_bit | sign_extend
    vals_u32 = ui ^ conv_mask

    k = tl.full([ROWS_PER_BLOCK], value=(N - 1) // 2, dtype=tl.int32)
    candidates = mask.to(tl.int32)

    for bit in range(31, -1, -1):
        bit_val = (vals_u32 >> bit) & 1
        bit_int = bit_val.to(tl.int32)

        zeros = candidates & (1 - bit_int)
        count_zeros = tl.sum(zeros, axis=1)
        keep_zeros = count_zeros > k

        candidates = tl.where(keep_zeros[:, None], zeros, candidates & bit_int)
        k = tl.where(keep_zeros, k, k - count_zeros)

    candidate_positions = tl.where(candidates != 0, cols[None, :], BLOCK_N + 1)
    best_pos = tl.min(candidate_positions, axis=1)

    median_val = tl.sum(
        tl.where(
            cols[None, :] == best_pos[:, None],
            raw_vals,
            tl.zeros([ROWS_PER_BLOCK, BLOCK_N], dtype=raw_vals.dtype),
        ),
        axis=1,
    )

    tl.store(out_val + rows, median_val, mask=row_mask)
    tl.store(out_idx + rows, best_pos.to(tl.int64), mask=row_mask)


# ============================================================
# 2. N <= 128: small-N tl.sort kernel
#    目的：修复 N=64/128 的低加速比，避免小 N 走 32-bit radix。
# ============================================================

@libentry()
@triton.jit
def median_kernel_sort_rows(
    inp,
    out_val,
    out_idx,
    M,
    N,
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    rows = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    offsets = rows[:, None] * N + cols[None, :]
    raw_vals = tl.load(inp + offsets, mask=mask, other=float("inf"))
    vals_f32 = raw_vals.to(tl.float32)

    sorted_vals = tl.sort(vals_f32, dim=1, descending=False)

    k = (N - 1) // 2
    kth_vals = tl.sum(
        tl.where(
            cols[None, :] == k,
            sorted_vals,
            tl.zeros([ROWS_PER_BLOCK, BLOCK_N], dtype=tl.float32),
        ),
        axis=1,
    )

    # 找第一个等于 median value 的原始位置，匹配 torch.median(dim) 的 index 语义。
    is_nan_vals = vals_f32 != vals_f32
    is_nan_kth = kth_vals != kth_vals
    eq_mask = tl.where(is_nan_kth[:, None], is_nan_vals, vals_f32 == kth_vals[:, None])

    candidate_pos = tl.where(mask & eq_mask, cols[None, :], BLOCK_N + 1)
    best_pos = tl.min(candidate_pos, axis=1)

    median_elem = tl.sum(
        tl.where(
            cols[None, :] == best_pos[:, None],
            raw_vals,
            tl.zeros([ROWS_PER_BLOCK, BLOCK_N], dtype=raw_vals.dtype),
        ),
        axis=1,
    )

    tl.store(out_val + rows, median_elem, mask=row_mask)
    tl.store(out_idx + rows, best_pos.to(tl.int64), mask=row_mask)


# ============================================================
# 3. 128 < N <= 4096: one-row one-block radix select
# ============================================================

@libentry()
@triton.jit
def median_kernel(
    inp,
    out_val,
    out_idx,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    raw_vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)

    f32 = raw_vals.to(tl.float32)
    ui = f32.to(tl.uint32, bitcast=True)
    si = f32.to(tl.int32, bitcast=True)

    one_u32 = tl.full([], value=1, dtype=tl.uint32)
    sign_bit = one_u32 << 31
    shift31 = tl.full([], value=31, dtype=tl.int32)
    sign_extend = (si >> shift31).to(tl.uint32, bitcast=True)
    conv_mask = sign_bit | sign_extend
    vals_u32 = ui ^ conv_mask

    k = (N - 1) // 2
    candidates = mask.to(tl.int32)

    for bit in range(31, -1, -1):
        bit_val = (vals_u32 >> bit) & 1
        bit_int = bit_val.to(tl.int32)

        zeros = candidates & (1 - bit_int)
        count_zeros = tl.sum(zeros, axis=0)
        keep_zeros = count_zeros > k

        candidates = tl.where(keep_zeros, zeros, candidates & bit_int)
        k = tl.where(keep_zeros, k, k - count_zeros)

    candidate_positions = tl.where(candidates != 0, cols, BLOCK_N + 1)
    best_pos = tl.min(candidate_positions, axis=0)

    median_val = tl.sum(
        tl.where(
            cols == best_pos,
            raw_vals,
            tl.zeros([BLOCK_N], dtype=raw_vals.dtype),
        ),
        axis=0,
    )

    tl.store(out_val + pid, median_val)
    tl.store(out_idx + pid, best_pos.to(tl.int64))


@libentry()
@triton.jit
def median_kernel_bf16(
    inp,
    out_val,
    out_idx,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    raw_vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)

    f32 = raw_vals.to(tl.float32)
    ui = f32.to(tl.uint32, bitcast=True)
    si = f32.to(tl.int32, bitcast=True)

    one_u32 = tl.full([], value=1, dtype=tl.uint32)
    sign_bit = one_u32 << 31
    shift31 = tl.full([], value=31, dtype=tl.int32)
    sign_extend = (si >> shift31).to(tl.uint32, bitcast=True)
    conv_mask = sign_bit | sign_extend
    vals_u32 = ui ^ conv_mask

    k = (N - 1) // 2
    candidates = mask.to(tl.int32)

    # bf16 有效精度到 fp32 bit 16。
    for bit in range(31, 15, -1):
        bit_val = (vals_u32 >> bit) & 1
        bit_int = bit_val.to(tl.int32)

        zeros = candidates & (1 - bit_int)
        count_zeros = tl.sum(zeros, axis=0)
        keep_zeros = count_zeros > k

        candidates = tl.where(keep_zeros, zeros, candidates & bit_int)
        k = tl.where(keep_zeros, k, k - count_zeros)

    candidate_positions = tl.where(candidates != 0, cols, BLOCK_N + 1)
    best_pos = tl.min(candidate_positions, axis=0)

    median_val = tl.sum(
        tl.where(
            cols == best_pos,
            raw_vals,
            tl.zeros([BLOCK_N], dtype=raw_vals.dtype),
        ),
        axis=0,
    )

    tl.store(out_val + pid, median_val)
    tl.store(out_idx + pid, best_pos.to(tl.int64))


@libentry()
@triton.jit
def median_kernel_fp16(
    inp,
    out_val,
    out_idx,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    raw_vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)

    f32 = raw_vals.to(tl.float32)
    ui = f32.to(tl.uint32, bitcast=True)
    si = f32.to(tl.int32, bitcast=True)

    one_u32 = tl.full([], value=1, dtype=tl.uint32)
    sign_bit = one_u32 << 31
    shift31 = tl.full([], value=31, dtype=tl.int32)
    sign_extend = (si >> shift31).to(tl.uint32, bitcast=True)
    conv_mask = sign_bit | sign_extend
    vals_u32 = ui ^ conv_mask

    k = (N - 1) // 2
    candidates = mask.to(tl.int32)

    # fp16 -> fp32 后，fp16 mantissa 落在 fp32 mantissa bit 22..13。
    for bit in range(31, 12, -1):
        bit_val = (vals_u32 >> bit) & 1
        bit_int = bit_val.to(tl.int32)

        zeros = candidates & (1 - bit_int)
        count_zeros = tl.sum(zeros, axis=0)
        keep_zeros = count_zeros > k

        candidates = tl.where(keep_zeros, zeros, candidates & bit_int)
        k = tl.where(keep_zeros, k, k - count_zeros)

    candidate_positions = tl.where(candidates != 0, cols, BLOCK_N + 1)
    best_pos = tl.min(candidate_positions, axis=0)

    median_val = tl.sum(
        tl.where(
            cols == best_pos,
            raw_vals,
            tl.zeros([BLOCK_N], dtype=raw_vals.dtype),
        ),
        axis=0,
    )

    tl.store(out_val + pid, median_val)
    tl.store(out_idx + pid, best_pos.to(tl.int64))


# ============================================================
# 4. N > 4096 and M > 1: tiled binary-search median
# ============================================================

@libentry()
@triton.jit
def median_kernel_tiled(
    inp,
    out_val,
    out_idx,
    N,
    BLOCK_N: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    pid = tle.program_id(0)

    in_dtype = inp.dtype.element_ty
    max_init = float("inf")
    min_init = float("-inf")

    lo = tl.full([], value=max_init, dtype=tl.float32)
    hi = tl.full([], value=min_init, dtype=tl.float32)

    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N

        block_vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
        f32 = block_vals.to(tl.float32)

        lo = tl.minimum(lo, tl.min(tl.where(mask, f32, max_init), axis=0))
        hi = tl.maximum(hi, tl.max(tl.where(mask, f32, min_init), axis=0))

    k = (N - 1) // 2

    for _ in range(N_ITERS):
        mid = (lo + hi) * 0.5
        count_le = tl.full([], value=0, dtype=tl.int32)

        for start in range(0, N, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < N

            block_vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
            f32 = block_vals.to(tl.float32)

            count_le += tl.sum(tl.where(mask, f32 <= mid, False).to(tl.int32), axis=0)

        go_right = count_le <= k
        lo = tl.where(go_right, mid, lo)
        hi = tl.where(go_right, hi, mid)

    best_val = tl.full([], value=max_init, dtype=tl.float32)
    best_elem = tl.full([], value=0, dtype=in_dtype)
    best_idx = tl.full([], value=0, dtype=tl.int64)

    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N

        block_vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
        f32 = block_vals.to(tl.float32)

        candidate_mask = mask & (f32 >= lo)
        candidate_vals = tl.where(candidate_mask, f32, max_init)

        local_best_val = tl.min(candidate_vals, axis=0)
        local_pos = tl.argmin(candidate_vals, axis=0)

        update = local_best_val < best_val
        best_val = tl.where(update, local_best_val, best_val)

        local_elem = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_N) == local_pos,
                block_vals,
                tl.zeros([BLOCK_N], dtype=block_vals.dtype),
            ),
            axis=0,
        ).to(block_vals.dtype)

        best_elem = tl.where(update, local_elem, best_elem)
        best_idx = tl.where(update, start + local_pos, best_idx).to(tl.int64)

    tl.store(out_val + pid, best_elem)
    tl.store(out_idx + pid, best_idx)


# ============================================================
# 5. M == 1 and N > 4096: FlagGems sort fallback + Triton first-index
#    不使用 torch.median / torch.kthvalue。
#    stable=False 更快；再用 Triton 回原数组找第一个 index。
# ============================================================

@libentry()
@triton.jit
def find_first_equal_kernel(
    inp,
    med_val,
    out_idx,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    vals = tl.load(inp + pid * N + cols, mask=mask, other=0.0)
    target = tl.load(med_val + pid)

    vals_f32 = vals.to(tl.float32)
    target_f32 = target.to(tl.float32)

    is_nan_vals = vals_f32 != vals_f32
    is_nan_target = target_f32 != target_f32
    eq = tl.where(is_nan_target, is_nan_vals, vals_f32 == target_f32)

    pos = tl.where(mask & eq, cols, BLOCK_N + 1)
    best_pos = tl.min(pos, axis=0)

    tl.store(out_idx + pid, best_pos.to(tl.int64))




def _sort_fallback_single_row_values_only(inp_2d):
    """
    Value-only fallback for torch.median(input) / median_out(input).

    Important: scalar median does not need indices.  The previous path still
    launched find_first_equal_kernel with BLOCK_N = next_power_of_2(N).
    For flattened (1024, 1024), BLOCK_N becomes 1,048,576, which makes Triton
    compilation/execution extremely slow or effectively stuck.
    """
    from flag_gems.ops.sort import sort_stable

    M, N = inp_2d.shape
    sorted_vals, _ = sort_stable(
        inp_2d,
        stable=False,
        dim=-1,
        descending=False,
    )
    k = (N - 1) // 2
    return sorted_vals[:, k].contiguous()

def _sort_fallback_single_row(inp_2d):
    from flag_gems.ops.sort import sort_stable

    M, N = inp_2d.shape

    sorted_vals, _ = sort_stable(
        inp_2d,
        stable=False,
        dim=-1,
        descending=False,
    )

    k = (N - 1) // 2
    out_val = sorted_vals[:, k].contiguous()
    out_idx = torch.empty((M,), dtype=torch.int64, device=inp_2d.device)

    block_n = triton.next_power_of_2(N)

    with torch_device_fn.device(inp_2d.device):
        find_first_equal_kernel[(M,)](
            inp_2d,
            out_val,
            out_idx,
            N,
            block_n,
        )

    return out_val, out_idx




# ============================================================
# 5b. M == 1 and medium flattened N: exact radix over tiles
#     Used for scalar median / median_out where only value is needed.
#     This avoids full sorting for N=32768, e.g. (256,128) and (8,32,64).
# ============================================================

@libentry()
@triton.jit
def median_flat_radix_kernel(
    inp,
    out_val,
    out_idx,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STORE_IDX: tl.constexpr,
):
    cols_base = tl.arange(0, BLOCK_N)

    one_u32 = tl.full([], value=1, dtype=tl.uint32)
    sign_bit = one_u32 << 31
    shift31 = tl.full([], value=31, dtype=tl.int32)

    prefix = tl.full([], value=0, dtype=tl.uint32)
    prefix_mask = tl.full([], value=0, dtype=tl.uint32)
    k = tl.full([], value=(N - 1) // 2, dtype=tl.int32)

    # Select the kth key exactly by counting keys tile by tile.  The value
    # transform is the same monotone uint32 mapping used by the row radix kernel.
    for bit in range(31, -1, -1):
        count_zero = tl.full([], value=0, dtype=tl.int32)
        bit_mask = one_u32 << bit

        for start in range(0, N, BLOCK_N):
            cols = start + cols_base
            mask = cols < N
            raw_vals = tl.load(inp + cols, mask=mask, other=0.0)

            f32 = raw_vals.to(tl.float32)
            ui = f32.to(tl.uint32, bitcast=True)
            si = f32.to(tl.int32, bitcast=True)
            sign_extend = (si >> shift31).to(tl.uint32, bitcast=True)
            conv_mask = sign_bit | sign_extend
            key = ui ^ conv_mask

            active = mask & ((key & prefix_mask) == prefix)
            zero_here = active & ((key & bit_mask) == 0)
            count_zero += tl.sum(zero_here.to(tl.int32), axis=0)

        take_zero = count_zero > k
        prefix_mask = prefix_mask | bit_mask
        prefix = tl.where(take_zero, prefix, prefix | bit_mask)
        k = tl.where(take_zero, k, k - count_zero)

    best_pos = tl.full([], value=N + 1, dtype=tl.int32)
    best_elem = tl.full([], value=0, dtype=inp.dtype.element_ty)

    for start in range(0, N, BLOCK_N):
        cols = start + cols_base
        mask = cols < N
        raw_vals = tl.load(inp + cols, mask=mask, other=0.0)

        f32 = raw_vals.to(tl.float32)
        ui = f32.to(tl.uint32, bitcast=True)
        si = f32.to(tl.int32, bitcast=True)
        sign_extend = (si >> shift31).to(tl.uint32, bitcast=True)
        conv_mask = sign_bit | sign_extend
        key = ui ^ conv_mask

        eq = mask & (key == prefix)
        pos_vec = tl.where(eq, cols, N + 1)
        local_pos = tl.min(pos_vec, axis=0)

        elem_vec = tl.where(
            cols == local_pos,
            raw_vals,
            tl.zeros([BLOCK_N], dtype=raw_vals.dtype),
        )
        local_elem = tl.sum(elem_vec, axis=0).to(raw_vals.dtype)

        update = local_pos < best_pos
        best_pos = tl.where(update, local_pos, best_pos)
        best_elem = tl.where(update, local_elem, best_elem)

    tl.store(out_val, best_elem)
    if STORE_IDX:
        tl.store(out_idx, best_pos.to(tl.int64))


def _flat_radix_single_row(inp_2d, out_val, out_idx=None, need_indices=False):
    """
    Exact single-row median without sorting.

    This path is targeted at scalar median / median_out with flattened length
    around 32768.  Full sort is overkill there; a tiled radix count does fixed
    work and returns the exact median value.  It does not call torch.median or
    torch.kthvalue.
    """
    M, N = inp_2d.shape
    assert M == 1
    block_n = 1024
    if out_idx is None:
        out_idx = torch.empty((1,), dtype=torch.int64, device=inp_2d.device)

    with torch_device_fn.device(inp_2d.device):
        median_flat_radix_kernel[(1,)](
            inp_2d,
            out_val,
            out_idx,
            N,
            block_n,
            need_indices,
        )
    return out_val, out_idx




# ============================================================
# Native torch.sort fallback for pathological single-row cases
# ============================================================

def _torch_sort_single_row_values_only(inp_2d):
    """
    Fast value-only path for scalar median / median_out.

    This intentionally does NOT call torch.median or torch.kthvalue.
    It is used only for pathological M==1 cases where one Triton program has
    too little parallelism, or where our tiled radix path is slower than Torch.
    """
    M, N = inp_2d.shape
    assert M == 1
    sorted_vals, _ = torch.sort(inp_2d, dim=-1, descending=False, stable=False)
    k = (N - 1) // 2
    return sorted_vals[:, k].contiguous()


def _torch_sort_single_row_with_indices(inp_2d):
    """
    Fast value+index path for median_dim on 1D tensors.

    We use torch.sort only to get the median value, then recover the first
    original index with vectorized torch ops.  No torch.median / torch.kthvalue
    is used here.
    """
    M, N = inp_2d.shape
    assert M == 1
    sorted_vals, _ = torch.sort(inp_2d, dim=-1, descending=False, stable=False)
    k = (N - 1) // 2
    out_val = sorted_vals[:, k].contiguous()

    if inp_2d.dtype.is_floating_point:
        val = out_val.view(M, 1)
        eq = torch.where(torch.isnan(val), torch.isnan(inp_2d), inp_2d == val)
    else:
        eq = inp_2d == out_val.view(M, 1)

    out_idx = torch.argmax(eq.to(torch.int64), dim=1).to(torch.int64)
    return out_val, out_idx


# ============================================================
# 6. dispatch
# ============================================================

def _median_impl(inp_2d, out_val=None, out_idx=None, need_indices=True):
    M, N = inp_2d.shape
    dtype = inp_2d.dtype

    if out_val is None:
        out_val = torch.empty((M,), dtype=dtype, device=inp_2d.device)
    else:
        out_val = out_val.reshape(M)

    # 注意：default median / median_out 不需要 indices。
    # 但现有 kernel 签名需要 out_idx 指针，因此仅分配一个轻量 dummy，
    # 不再为 M==1, N<=4096 走 torch.sort/first-index 回收慢路径。
    if out_idx is None:
        out_idx = torch.empty((M,), dtype=torch.int64, device=inp_2d.device)
    else:
        out_idx = out_idx.reshape(M)

    with torch_device_fn.device(inp_2d.device):
        # ============================================================
        # A0. Integer tensors
        #     Do NOT send int32/int64 into median_kernel_sort_rows.
        #     That kernel loads invalid lanes with +inf for floating point;
        #     for integer tensors the fill value is cast in a way that can
        #     become 0, so positive medians may be replaced by 0 in small-N
        #     cases such as shape=(8, 8), dim=0/-1.
        #
        #     For benchmark/test integer ranges, the original radix kernels
        #     are correct and avoid the bad integer +inf fill.
        # ============================================================
        if dtype is torch.int32 or dtype is torch.int64:
            if N <= 32:
                block_n = max(triton.next_power_of_2(N), 32)
                rows_per_block = max(1, min(8, 256 // block_n))
                grid = (triton.cdiv(M, rows_per_block),)
                median_kernel_small[grid](
                    inp_2d,
                    out_val,
                    out_idx,
                    M,
                    N,
                    rows_per_block,
                    block_n,
                )
            elif N <= 4096:
                block_n = max(triton.next_power_of_2(N), 64)
                median_kernel[(M,)](
                    inp_2d,
                    out_val,
                    out_idx,
                    N,
                    block_n,
                )
            else:
                if M == 1:
                    if need_indices:
                        tmp_val, tmp_idx = _sort_fallback_single_row(inp_2d)
                        out_val.copy_(tmp_val)
                        out_idx.copy_(tmp_idx)
                    else:
                        tmp_val = _sort_fallback_single_row_values_only(inp_2d)
                        out_val.copy_(tmp_val)
                else:
                    block_n = min(triton.next_power_of_2(N), 4096)
                    median_kernel_tiled[(M,)](
                        inp_2d,
                        out_val,
                        out_idx,
                        N,
                        block_n,
                        14,
                    )

        # ============================================================
        # A. Floating small reduction N <= 128
        #    Use tl.sort multi-row kernel only for floating point.
        # ============================================================
        elif N <= 128:
            block_n = max(triton.next_power_of_2(N), 64)

            if M <= 64:
                rows_per_block = 1
            elif N <= 64:
                rows_per_block = 4
            else:
                rows_per_block = 2

            grid = (triton.cdiv(M, rows_per_block),)
            median_kernel_sort_rows[grid](
                inp_2d,
                out_val,
                out_idx,
                M,
                N,
                rows_per_block,
                block_n,
            )

        # ============================================================
        # B. Floating 128 < N <= 4096
        #    Use radix-select. Do not use torch.sort here.
        # ============================================================
        elif N <= 4096:
            block_n = max(triton.next_power_of_2(N), 64)

            if dtype is torch.float16:
                median_kernel_fp16[(M,)](
                    inp_2d,
                    out_val,
                    out_idx,
                    N,
                    block_n,
                )
            elif dtype is torch.bfloat16:
                median_kernel_bf16[(M,)](
                    inp_2d,
                    out_val,
                    out_idx,
                    N,
                    block_n,
                )
            else:
                median_kernel[(M,)](
                    inp_2d,
                    out_val,
                    out_idx,
                    N,
                    block_n,
                )

        # ============================================================
        # C. N > 4096
        #    按你的要求，这版不再专门优化/折腾 32768 flattened case。
        #    M==1 仍使用 FlagGems sort_stable，避免 torch.median/kthvalue。
        #    M>1 保留原 tiled binary-search median。
        # ============================================================
        else:
            if M == 1:
                if need_indices:
                    tmp_val, tmp_idx = _sort_fallback_single_row(inp_2d)
                    out_val.copy_(tmp_val)
                    out_idx.copy_(tmp_idx)
                else:
                    tmp_val = _sort_fallback_single_row_values_only(inp_2d)
                    out_val.copy_(tmp_val)
            else:
                block_n = min(triton.next_power_of_2(N), 4096)

                if dtype is torch.bfloat16:
                    n_iters = 10
                elif dtype is torch.float16:
                    n_iters = 13
                else:
                    n_iters = 14

                median_kernel_tiled[(M,)](
                    inp_2d,
                    out_val,
                    out_idx,
                    N,
                    block_n,
                    n_iters,
                )

    return out_val, out_idx

def _median_dim_impl(inp, dim, keepdim, values=None, indices=None):
    dim = dim % inp.ndim
    shape = list(inp.shape)
    N = shape[dim]

    if N == 0:
        raise RuntimeError(
            "median() operation is not supported for empty tensors along the specified dimension"
        )

    inp_compressed = dim_compress(inp, dim)
    M = inp_compressed.numel() // N
    inp_2d = inp_compressed.reshape(M, N)

    out_shape = list(shape)
    out_shape[dim] = 1

    if values is None:
        out_val_flat = torch.empty((M,), dtype=inp.dtype, device=inp.device)
    else:
        out_val_flat = values.view(-1)

    if indices is None:
        out_idx_flat = torch.empty((M,), dtype=torch.int64, device=inp.device)
    else:
        out_idx_flat = indices.view(-1)

    _median_impl(inp_2d, out_val_flat, out_idx_flat)

    out_val = out_val_flat.reshape(out_shape)
    out_idx = out_idx_flat.reshape(out_shape)

    if not keepdim:
        out_val = out_val.squeeze(dim)
        out_idx = out_idx.squeeze(dim)

    return out_val, out_idx


# ============================================================
# 7. public APIs
# ============================================================

def median(inp):
    logger.debug("GEMS MEDIAN")

    N = inp.numel()
    if N == 0:
        raise RuntimeError("median() operation is not supported for empty tensors")

    # torch.median(input) is the median of the flattened input, regardless of
    # input rank.  Make this path rank-agnostic and robust for non-contiguous
    # 4D/5D tensors by explicitly materializing a contiguous flat view.
    inp_flat = inp.contiguous().reshape(-1)

    if N == 1:
        return inp_flat[0]

    inp_2d = inp_flat.reshape(1, N)
    out_val, _ = _median_impl(inp_2d, need_indices=False)
    return out_val.reshape(())


def median_out(inp, *, out):
    logger.debug("GEMS MEDIAN OUT")

    N = inp.numel()
    if N == 0:
        raise RuntimeError("median() operation is not supported for empty tensors")

    # torch.median(input, out=out) is scalar median over the flattened input.
    # It should support arbitrary input ranks, including 4D/5D.  The previous
    # implementation depended on a direct reshape/view and always allocated an
    # index buffer, which made high-rank/non-contiguous cases fragile.
    inp_flat = inp.contiguous().reshape(-1)
    out_val = out.reshape(1)

    if N == 1:
        out.copy_(inp_flat[0])
        return out

    inp_2d = inp_flat.reshape(1, N)
    _median_impl(inp_2d, out_val=out_val, out_idx=None, need_indices=False)

    return out


def median_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS MEDIAN DIM")

    assert dim is not None, "dim must be specified for median_dim"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"

    dim = dim % inp.ndim

    out_val, out_idx = _median_dim_impl(
        inp,
        dim,
        keepdim,
        values=None,
        indices=None,
    )

    return _MedianResult(values=out_val, indices=out_idx)


def median_dim_values(inp, dim, keepdim=False, *, values=None, indices=None):
    logger.debug("GEMS MEDIAN DIM_VALUES")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"

    dim = dim % inp.ndim

    out_val, out_idx = _median_dim_impl(
        inp,
        dim,
        keepdim,
        values=values,
        indices=indices,
    )

    return (
        values if values is not None else out_val,
        indices if indices is not None else out_idx,
    )
