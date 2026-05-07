import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]

MEDIAN_SHAPES = (
    utils.REDUCTION_SHAPES
    if cfg.QUICK_MODE
    else utils.REDUCTION_SHAPES + [(16, 512), (8, 1024), (32, 2048)]
)


@pytest.mark.median
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_median(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.median_dim
@pytest.mark.parametrize("shape", MEDIAN_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_median_dim(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out.values, ref_out.values)
    # Indices may differ when there are duplicate values around the median (tie-breaking).
    # Validate that the returned index actually points to the median value.
    res_idx = res_out.indices
    if res_idx.ndim < inp.ndim:
        res_idx = res_idx.unsqueeze(dim)
    gathered = inp.gather(dim, res_idx.to(inp.device))
    if not keepdim:
        gathered = gathered.squeeze(dim)
    utils.gems_assert_equal(gathered, res_out.values)


@pytest.mark.median_dim
@pytest.mark.parametrize("shape", [(1, 1), (8, 1), (1, 256)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_small_n(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim)

    utils.gems_assert_equal(res_out.values, ref_out.values)
    # Indices may differ when there are duplicate values around the median (tie-breaking).
    # Validate that the returned index actually points to the median value.
    res_idx = res_out.indices
    if res_idx.ndim < inp.ndim:
        res_idx = res_idx.unsqueeze(dim)
    gathered = inp.gather(dim, res_idx.to(inp.device)).squeeze(dim)
    utils.gems_assert_equal(gathered, res_out.values)
