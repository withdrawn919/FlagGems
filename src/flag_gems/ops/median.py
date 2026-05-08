import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val, argsort
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

MAX_BITONIC_M = 256
MAX_GRID = 2048


@libentry()
@triton.jit
def median_dim_kernel(
    inp,
    out_value,
    out_index,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pids = tle.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)

    if IS_FLOAT:
        mask_val = _get_finfo_val(inp.dtype.element_ty, return_max=True)
    else:
        mask_val = _get_iinfo_val(inp.dtype.element_ty, return_max=True)

    for row_idx in range(pid, M, num_pids):
        mask = cols < N
        row_ptr = inp + row_idx * N
        in_val = tl.load(row_ptr + cols, mask=mask, other=mask_val)
        in_val = tl.where(in_val.dtype.is_fp64(), in_val, in_val.to(tl.float32))

        ids = tl.arange(0, BLOCK_SIZE)
        sorted_vals, sorted_ids = argsort(in_val, ids, 0, descending=False)

        median_idx = (N - 1) // 2
        idx_range = tl.arange(0, BLOCK_SIZE)
        median_mask = idx_range == median_idx
        median_mask_f = median_mask.to(in_val.dtype)

        out_val = tl.sum(sorted_vals * median_mask_f)
        out_idx = tl.sum(tl.where(median_mask, sorted_ids, 0))

        tl.store(out_value + row_idx, out_val.to(inp.dtype.element_ty))
        tl.store(out_index + row_idx, out_idx)


def median(inp):
    logger.debug("GEMS MEDIAN")
    out = median_dim(inp.ravel(), dim=0, keepdim=False)
    return out.values


def median_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS MEDIAN DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    N = inp.shape[dim]

    if N <= MAX_BITONIC_M:
        shape = list(inp.shape)
        out_shape = list(shape)
        out_shape[dim] = 1

        out_value = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        out_index = torch.empty(out_shape, dtype=torch.int64, device=inp.device)

        inp = dim_compress(inp, dim)
        M = inp.numel() // N
        inp_2d = inp.reshape(M, N)
        BLOCK_SIZE = triton.next_power_of_2(N)
        is_float = inp.is_floating_point()
        grid_size = min(M, MAX_GRID)
        with torch_device_fn.device(inp.device):
            median_dim_kernel[(grid_size,)](
                inp_2d,
                out_value.view(-1),
                out_index.view(-1),
                N,
                M,
                BLOCK_SIZE,
                is_float,
            )

        if not keepdim:
            out_value = torch.squeeze(out_value, dim)
            out_index = torch.squeeze(out_index, dim)
    else:
        k = (N - 1) // 2 + 1
        out_value, out_index = torch.kthvalue(inp, k, dim=dim, keepdim=keepdim)

    Median_out = namedtuple("median", ["values", "indices"])
    return Median_out(values=out_value, indices=out_index)
