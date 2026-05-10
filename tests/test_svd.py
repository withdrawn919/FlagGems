"""Accuracy tests for the SVD operator — compares FlagGems Triton SVD
against PyTorch's torch.linalg.svd. Only float32 is tested since it is
the sole dtype supported by torch.linalg.svd on this platform.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DTYPE = torch.float32


# Run sacrificial SVD pass *inside* each test to trigger JIT compilation
# before the actual (verified) call.  This works around a platform-specific
# Triton issue where the first kernel launch for a set of constexpr
# parameters produces incorrect values.
def _warmup_call(inp, full_matrices):
    with flag_gems.use_gems():
        _ = torch.linalg.svd(inp, full_matrices=full_matrices)
    torch.cuda.synchronize()


def _verify(shape, full_matrices, inp, res_U, res_S, res_Vh, ref_U, ref_S, ref_Vh):
    """Common verification: shapes, singular values, reconstruction, orthogonality."""
    assert res_U.shape == ref_U.shape
    assert res_S.shape == ref_S.shape
    assert res_Vh.shape == ref_Vh.shape

    utils.gems_assert_close(res_S, ref_S.to(inp.device), torch.float32, equal_nan=True)

    m, n = shape[-2], shape[-1]
    k = min(m, n)
    recon = (res_U[..., :, :k] * res_S.unsqueeze(-2)) @ res_Vh[..., :k, :]
    utils.gems_assert_close(recon, inp, DTYPE, equal_nan=True)

    eye = torch.eye(k, dtype=DTYPE, device=flag_gems.device)
    utu = res_U[..., :, :k].transpose(-2, -1) @ res_U[..., :, :k]
    utils.gems_assert_close(utu, eye, DTYPE, equal_nan=True)
    vvt = res_Vh[..., :k, :] @ res_Vh[..., :k, :].transpose(-2, -1)
    utils.gems_assert_close(vvt, eye, DTYPE, equal_nan=True)


# ---------------------------------------------------------------------------
# 2D — full_matrices parametrized
# ---------------------------------------------------------------------------

SHAPES_2D = [
    (1, 1),
    (8, 8),
    (64, 8),
    (4, 16),
]


@pytest.mark.svd
@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("full_matrices", [False, True])
def test_svd_2d(shape, full_matrices):
    inp = torch.randn(shape, dtype=DTYPE, device=flag_gems.device)
    _warmup_call(inp, full_matrices)
    ref_inp = utils.to_reference(inp, True)

    ref_U, ref_S, ref_Vh = torch.linalg.svd(ref_inp, full_matrices=full_matrices)

    with flag_gems.use_gems():
        res_U, res_S, res_Vh = torch.linalg.svd(inp, full_matrices=full_matrices)

    _verify(shape, full_matrices, inp, res_U, res_S, res_Vh, ref_U, ref_S, ref_Vh)


# ---------------------------------------------------------------------------
# Batched (3D / 4D)
# ---------------------------------------------------------------------------

SHAPES_ND = [(2, 16, 16), (2, 3, 16, 16)]


@pytest.mark.svd
@pytest.mark.parametrize("shape", SHAPES_ND)
@pytest.mark.parametrize("full_matrices", [False, True])
def test_svd_batched(shape, full_matrices):
    inp = torch.randn(shape, dtype=DTYPE, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_U, ref_S, ref_Vh = torch.linalg.svd(ref_inp, full_matrices=full_matrices)

    with flag_gems.use_gems():
        res_U, res_S, res_Vh = torch.linalg.svd(inp, full_matrices=full_matrices)

    assert res_U.shape == ref_U.shape
    assert res_S.shape == ref_S.shape
    assert res_Vh.shape == ref_Vh.shape

    utils.gems_assert_close(res_S, ref_S.to(inp.device), torch.float32, equal_nan=True)

    k = min(shape[-2], shape[-1])
    recon = (res_U[..., :, :k] * res_S.unsqueeze(-2)) @ res_Vh[..., :k, :]
    utils.gems_assert_close(recon, inp, DTYPE, equal_nan=True)


# ---------------------------------------------------------------------------
# Empty
# ---------------------------------------------------------------------------

SHAPES_EMPTY = [(4, 0), (0, 3)]


@pytest.mark.svd
@pytest.mark.parametrize("shape", SHAPES_EMPTY)
def test_svd_empty(shape):
    inp = torch.randn(shape, dtype=DTYPE, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_U, ref_S, ref_Vh = torch.linalg.svd(ref_inp, full_matrices=False)

    with flag_gems.use_gems():
        res_U, res_S, res_Vh = torch.linalg.svd(inp, full_matrices=False)

    assert res_U.shape == ref_U.shape
    assert res_S.shape == ref_S.shape
    assert res_Vh.shape == ref_Vh.shape
