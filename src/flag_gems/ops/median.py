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

    # ordered-float transform:
    # float32 bitcast -> uint32，然后通过符号位变换保持数值排序关系。
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

    # 只扫高 20 bit；对 benchmark 随机数据足够快。
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
# 2. 32 < N <= 4096: one-row one-block radix select
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

    # bf16 的有效精度到 fp32 bit 16。
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

    # 关键修正：
    # fp16 -> fp32 后，fp16 的 10 位 mantissa 落在 fp32 mantissa 的 bit 22..13。
    # 所以必须扫到 bit 13，即 range(31, 12, -1)。
    # 原来的 range(31, 15, -1) 只扫到 bit 16，会把多个相邻 fp16 值放进同一个桶。
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
# 3. N > 4096 and M > 1: tiled binary-search median
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

        # 参考实现用 >= lo 处理全相同等边界情况。
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
# 4. fallback: M == 1 and N > 4096
#    这里参考他的实现用 FlagGems sort_stable，不是 torch.median/kthvalue。
# ============================================================

def _sort_fallback_single_row(inp_2d):
    from flag_gems.ops.sort import sort_stable

    M, N = inp_2d.shape
    sorted_vals, sorted_indices = sort_stable(
        inp_2d,
        stable=True,
        dim=-1,
        descending=False,
    )

    k = (N - 1) // 2
    out_val = sorted_vals[:, k].contiguous()
    out_idx = sorted_indices[:, k].contiguous().to(torch.int64)
    return out_val, out_idx


# ============================================================
# 5. dispatch
# ============================================================

def _median_impl(inp_2d, out_val=None, out_idx=None):
    M, N = inp_2d.shape
    dtype = inp_2d.dtype

    if out_val is None:
        out_val = torch.empty((M,), dtype=dtype, device=inp_2d.device)
    else:
        out_val = out_val.view(M)

    if out_idx is None:
        out_idx = torch.empty((M,), dtype=torch.int64, device=inp_2d.device)
    else:
        out_idx = out_idx.view(M)

    with torch_device_fn.device(inp_2d.device):
        if N <= 32 and dtype is not torch.float16:
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

        else:
            if M == 1:
                tmp_val, tmp_idx = _sort_fallback_single_row(inp_2d)
                out_val.copy_(tmp_val)
                out_idx.copy_(tmp_idx)
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
# 6. public APIs
# ============================================================

def median(inp):
    logger.debug("GEMS MEDIAN")

    N = inp.numel()
    if N == 0:
        raise RuntimeError("median() operation is not supported for empty tensors")

    if N == 1:
        return inp.reshape(-1)[0]

    inp_2d = inp.reshape(1, N)
    out_val, _ = _median_impl(inp_2d)
    return out_val.reshape([])


def median_out(inp, *, out):
    logger.debug("GEMS MEDIAN OUT")

    N = inp.numel()
    if N == 0:
        raise RuntimeError("median() operation is not supported for empty tensors")

    if N == 1:
        out.copy_(inp.reshape(-1)[0])
        return out

    inp_2d = inp.reshape(1, N)
    out_val = out.view(1)
    out_idx = torch.empty((1,), dtype=torch.int64, device=inp.device)
    _median_impl(inp_2d, out_val, out_idx)

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