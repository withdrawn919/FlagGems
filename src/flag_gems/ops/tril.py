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
@triton.autotune(configs=runtime.get_tuned_config("tril"), key=["M", "N"])
@triton.jit(do_not_specialize=["diagonal"])
def tril_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    IS_INPLACE: tl.constexpr,
    M_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    row_start = pid * M_BLOCK_SIZE
    offs_m = row_start + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = offs_m < M

    block_min_row = row_start
    block_max_row = tl.minimum(block_min_row + M_BLOCK_SIZE - 1, M - 1)

    X += offs_m * N
    Y += offs_m * N

    for n_start in range(0, N, N_BLOCK_SIZE):
        offs_n = n_start + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = offs_n < N
        mask = m_mask & n_mask

        block_min_col = n_start
        block_max_col = tl.minimum(n_start + N_BLOCK_SIZE - 1, N - 1)

        all_above = block_min_col > block_max_row + diagonal
        all_below = block_max_col <= block_min_row + diagonal

        if all_below:
            if not IS_INPLACE:
                x = tl.load(X + offs_n, mask=mask, other=0.0)
                tl.store(Y + offs_n, x, mask=mask)
        elif all_above:
            tl.store(Y + offs_n, 0.0, mask=mask)
        elif IS_INPLACE:
            zero_above = offs_n > (offs_m + diagonal)
            tl.store(Y + offs_n, 0.0, mask=mask & zero_above)
        else:
            x = tl.load(X + offs_n, mask=mask, other=0.0)
            y = tl.where(offs_n <= (offs_m + diagonal), x, 0.0)
            tl.store(Y + offs_n, y, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("tril_batch"),
    key=["batch", "MN", "N", "diagonal"],
)
@triton.jit(do_not_specialize=["diagonal"])
def tril_batch_kernel(
    X,
    Y,
    batch,
    MN,
    N,
    diagonal,
    IS_INPLACE: tl.constexpr,
    BATCH_BLOCK_SIZE: tl.constexpr,
    MN_BLOCK_SIZE: tl.constexpr,
):
    batch_id = tle.program_id(0)
    mn_id = tle.program_id(1)
    row = batch_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)[:, None]
    batch_mask = row < batch
    X += row * MN
    Y += row * MN

    cols = mn_id * MN_BLOCK_SIZE + tl.arange(0, MN_BLOCK_SIZE)[None, :]
    mn_mask = cols < MN
    mask = batch_mask & mn_mask
    m = cols // N
    n = cols % N

    if IS_INPLACE:
        zero_above = n > (m + diagonal)
        tl.store(Y + cols, 0.0, mask=mask & zero_above)
    else:
        x = tl.load(X + cols, mask, other=0.0)
        y = tl.where(n <= m + diagonal, x, 0.0)
        tl.store(Y + cols, y, mask=mask)


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


def tril(A, diagonal=0):
    logger.debug("GEMS TRIL")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"

    can_use_directly, A_input = _check_batch_contiguous(A, allow_zero_stride=False)

    out = torch.empty(
        A.shape, dtype=A.dtype, device=A.device, memory_format=torch.contiguous_format
    )

    M, N = A_input.shape[-2:]

    with torch_device_fn.device(A_input.device):
        if len(A_input.shape) == 2:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            tril_kernel[grid](A_input, out, M, N, diagonal, False)
        else:
            batch = int(torch.numel(A_input) / M / N)
            B = A_input.view(batch, -1)
            grid = lambda meta: (
                triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
            )
            tril_batch_kernel[grid](B, out, batch, M * N, N, diagonal, False)
            out = out.view(A.shape)

    return out


def tril_(A, diagonal=0):
    logger.debug("GEMS TRIL_ (inplace)")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    diagonal = int(diagonal)
    M, N = A.shape[-2:]

    can_use_directly, A_to_use = _check_batch_contiguous(A, allow_zero_stride=True)

    if not can_use_directly:
        logger.debug(
            "Input tensor does not satisfy contiguity requirements, "
            "using temporary tensor for computation"
        )

        result_temp = torch.empty_like(A_to_use, memory_format=torch.contiguous_format)

        with torch_device_fn.device(A.device):
            if len(A.shape) == 2:
                grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
                tril_kernel[grid](A_to_use, result_temp, M, N, diagonal, False)
            else:
                batch = int(torch.numel(A) / M / N)
                B = A_to_use.view(batch, -1)
                result_temp_flat = result_temp.view(batch, -1)
                grid = lambda meta: (
                    triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                    triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
                )
                tril_batch_kernel[grid](B, result_temp_flat, batch, M * N, N, diagonal, False)

        A.copy_(result_temp)
    else:
        with torch_device_fn.device(A.device):
            if len(A.shape) == 2:
                grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
                tril_kernel[grid](A, A, M, N, diagonal, True)
            else:
                batch = int(torch.numel(A) / M / N)
                B = A.view(batch, -1)
                grid = lambda meta: (
                    triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),
                    triton.cdiv(M * N, meta["MN_BLOCK_SIZE"]),
                )
                tril_batch_kernel[grid](B, B, batch, M * N, N, diagonal, True)

    return A


def tril_out(input: torch.Tensor, diagonal: int = 0, out: torch.Tensor = None):
    if out is None:
        out = torch.empty_like(input)
    assert out.shape == input.shape, "Input and output must have the same shape"
    assert out.dtype == input.dtype, "Input and output must have the same dtype"
    result = tril(input, diagonal)
    out.copy_(result)
    return out
