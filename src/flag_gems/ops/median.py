import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val, argsort

logger = logging.getLogger(__name__)

MAX_BITONIC_M = 1024


@libentry()
@triton.jit
def median_dim_kernel(
    inp,
    out_value,
    out_index,
    N,
    BLOCK_SIZE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    row_ptr = inp + pid * N

    if IS_FLOAT:
        mask_val = _get_finfo_val(inp.dtype.element_ty, return_max=True)
    else:
        mask_val = _get_iinfo_val(inp.dtype.element_ty, return_max=True)
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

    tl.store(out_value + pid, out_val.to(inp.dtype.element_ty))
    tl.store(out_index + pid, out_idx)


def median(inp):
    logger.debug("GEMS MEDIAN")
    out = median_dim(inp.ravel(), dim=0, keepdim=False)
    return out.values


def median_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS MEDIAN DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim
    N = shape[dim]

    out_shape = list(shape)
    out_shape[dim] = 1

    out_value = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(out_shape, dtype=torch.int64, device=inp.device)

    inp = dim_compress(inp, dim)
    M = inp.numel() // N
    inp_2d = inp.reshape(M, N)

    if N <= MAX_BITONIC_M:
        BLOCK_SIZE = triton.next_power_of_2(N)
        is_float = inp.is_floating_point()
        with torch_device_fn.device(inp.device):
            median_dim_kernel[(M,)](
                inp_2d, out_value.view(-1), out_index.view(-1), N, BLOCK_SIZE, is_float
            )
    else:
        # Use kthvalue (O(N) selection) instead of full sort (O(N log N))
        k = (N - 1) // 2 + 1
        vals, idxs = torch.kthvalue(inp_2d, k, dim=-1, keepdim=True)
        out_value.view(-1).copy_(vals.reshape(-1))
        out_index.view(-1).copy_(idxs.reshape(-1))

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    Median_out = namedtuple("median", ["values", "indices"])
    return Median_out(values=out_value, indices=out_index)
