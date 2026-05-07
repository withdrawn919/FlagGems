import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("tril"), key=["M", "N", "B", "ELEM_BYTES"])
@triton.jit(do_not_specialize=["diag", "ELEM_BYTES"])
def tril_kernel(
    X,
    Y,
    M,
    N,
    B,
    diag,
    ELEM_BYTES,
    IS_INPLACE: tl.constexpr,
    ITER_COL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if ITER_COL:
        pid = tle.program_id(0)
        grid_m = tle.num_programs(0) // B
        pid_b = pid // grid_m
        pid_m = pid % grid_m

        row_start = pid_m * BLOCK_M
        offs_m = row_start + tl.arange(0, BLOCK_M)[:, None]
        row_mask = offs_m < M
        block_min_row = row_start
        block_max_row = tl.minimum(block_min_row + BLOCK_M - 1, M - 1)
        base = pid_b * M * N
        row_base = base + offs_m * N

        for col_start in range(0, N, BLOCK_N):
            offs_n = col_start + tl.arange(0, BLOCK_N)[None, :]
            col_mask = offs_n < N
            mask = row_mask & col_mask
            idxs = row_base + offs_n

            block_min_col = col_start
            block_max_col = tl.minimum(col_start + BLOCK_N - 1, N - 1)

            all_above = block_min_col > block_max_row + diag
            all_below = block_max_col <= block_min_row + diag

            if all_below:
                if not IS_INPLACE:
                    x = tl.load(X + idxs, mask=mask, other=0.0)
                    tl.store(Y + idxs, x, mask=mask)
            elif all_above:
                tl.store(Y + idxs, 0.0, mask=mask)
            elif IS_INPLACE:
                zero_above = offs_n > (offs_m + diag)
                tl.store(Y + idxs, 0.0, mask=mask & zero_above)
            else:
                keep = offs_n <= (offs_m + diag)
                load_mask = mask & keep
                x = tl.load(X + idxs, mask=load_mask, other=0.0)
                tl.store(Y + idxs, x, mask=mask)
    else:
        pid_m = tle.program_id(0)
        pid_n = tle.program_id(1)
        pid_b = tle.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        mask = (offs_m < M) & (offs_n < N)

        base = pid_b * M * N
        idxs = base + offs_m * N + offs_n

        block_min_row = pid_m * BLOCK_M
        block_max_row = block_min_row + BLOCK_M - 1
        block_min_col = pid_n * BLOCK_N
        block_max_col = block_min_col + BLOCK_N - 1

        all_above = block_min_col > block_max_row + diag
        all_below = block_max_col <= block_min_row + diag

        if all_below:
            if IS_INPLACE:
                return
            x = tl.load(X + idxs, mask=mask, other=0.0)
            tl.store(Y + idxs, x, mask=mask)
            return

        if all_above:
            tl.store(Y + idxs, 0.0, mask=mask)
            return

        keep = offs_n <= (offs_m + diag)
        load_mask = mask & keep
        x = tl.load(X + idxs, mask=load_mask, other=0.0)
        tl.store(Y + idxs, x, mask=mask)


def _check_batch_contiguous(tensor, allow_zero_stride=True):
    if tensor.is_contiguous():
        return True, tensor

    dims = tensor.dim()

    if dims >= 2:
        n = tensor.size(-1)
        stride_row, stride_col = tensor.stride(-2), tensor.stride(-1)

        if not (stride_col == 1 and stride_row == n):
            return False, tensor.contiguous()

    if allow_zero_stride and dims <= 3:
        return True, tensor

    expected_stride = tensor.size(-1) * tensor.size(-2)
    for i in range(dims - 3, -1, -1):
        if (
            allow_zero_stride
            and i == 0
            and (tensor.stride(i) == 0 or tensor.size(i) == 1)
        ):
            continue

        if tensor.stride(i) != expected_stride:
            return False, tensor.contiguous()

        expected_stride *= tensor.size(i)

    return True, tensor


def _launch_tril_kernel(X, Y, M, N, B, diagonal, elem_bytes, is_inplace, iter_col):
    if iter_col:
        max_relevant = max(0, N - diagonal)
        effective_rows = min(M, max_relevant)
        if B == 1 and effective_rows == M and N >= M and M >= 512:
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]),
                1,
            )
            tril_kernel[grid](X, Y, M, N, 1, diagonal, elem_bytes, is_inplace, False)
            return
        grid = lambda meta: (triton.cdiv(effective_rows, meta["BLOCK_M"]) * B,)
        tril_kernel[grid](X, Y, M, N, B, diagonal, elem_bytes, is_inplace, True)
    else:
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
            B,
        )
        tril_kernel[grid](X, Y, M, N, B, diagonal, elem_bytes, is_inplace, False)


def tril(A, diagonal=0):
    logger.debug("GEMS TRIL")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"

    can_use_directly, A_input = _check_batch_contiguous(A, allow_zero_stride=False)

    out = torch.empty(
        A.shape, dtype=A.dtype, device=A.device, memory_format=torch.contiguous_format
    )

    M, N = A_input.shape[-2:]
    B = A_input.numel() // (M * N)
    elem_bytes = A_input.dtype.itemsize

    with torch_device_fn.device(A_input.device):
        _launch_tril_kernel(A_input, out, M, N, B, diagonal, elem_bytes, False, False)

    return out


def tril_(A, diagonal=0):
    logger.debug("GEMS TRIL_ (inplace)")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    diagonal = int(diagonal)
    M, N = A.shape[-2:]
    B = A.numel() // (M * N)

    can_use_directly, A_to_use = _check_batch_contiguous(A, allow_zero_stride=True)
    elem_bytes = A.dtype.itemsize

    if not can_use_directly:
        logger.debug(
            "Input tensor does not satisfy contiguity requirements, "
            "using temporary tensor for computation"
        )

        result_temp = torch.empty_like(A_to_use, memory_format=torch.contiguous_format)

        with torch_device_fn.device(A.device):
            _launch_tril_kernel(A_to_use, result_temp, M, N, B, diagonal, elem_bytes, False, False)

        A.copy_(result_temp)
    else:
        with torch_device_fn.device(A.device):
            max_relevant = max(0, N - diagonal)
            effective_rows = min(M, max_relevant)
            if effective_rows <= 0:
                return A
            _launch_tril_kernel(A, A, M, N, B, diagonal, elem_bytes, True, True)

    return A


def tril_out(input: torch.Tensor, diagonal: int = 0, out: torch.Tensor = None):
    if out is None:
        out = torch.empty_like(input)
    assert out.shape == input.shape, "Input and output must have the same shape"
    assert out.dtype == input.dtype, "Input and output must have the same dtype"
    result = tril(input, diagonal)
    out.copy_(result)
    return out
