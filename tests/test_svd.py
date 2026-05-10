import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SVD_SHAPES = [
    # Small square
    (1, 1),
    (3, 3),
    (8, 8),
    (16, 16),
    # Regular square
    (32, 32),
    (64, 64),
    # Rectangular
    (5, 3),
    (64, 32),
    (32, 64),
    (10, 5),
    (5, 10),
    # Batch
    (2, 16, 16),
    (2, 3, 32, 32),
]


@pytest.mark.svd
@pytest.mark.parametrize("shape", SVD_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_svd_accuracy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp)

    ref_U, ref_S, ref_V = torch.svd(ref_inp)

    k = min(shape[-2], shape[-1])

    # Check shapes
    assert res_U.shape == ref_U.shape, f"U shape mismatch: {res_U.shape} vs {ref_U.shape}"
    assert res_S.shape == ref_S.shape, f"S shape mismatch: {res_S.shape} vs {ref_S.shape}"
    assert res_V.shape == ref_V.shape, f"V shape mismatch: {res_V.shape} vs {ref_V.shape}"

    # Reconstruction: U @ diag(S) @ V^T ≈ input (use float32 for diag_embed)
    compute_dtype = torch.float32
    recon_gems = res_U.float() @ torch.diag_embed(res_S.float()) @ res_V.float().transpose(-2, -1)
    utils.gems_assert_close(recon_gems.float(), inp.float(), torch.float32, reduce_dim=k, atol=1e-3)

    # Singular values: gems values should be close to ref values
    # Small singular values near zero are sensitive to numerical errors in A^T @ A
    sv_atol = 4e-3 if dtype == torch.float16 else 2e-3
    utils.gems_assert_close(res_S.float(), ref_S.to(torch.float32), torch.float32, reduce_dim=1, atol=sv_atol)

    # Orthonormality of U columns: U^T @ U ≈ I
    if res_U.shape[-1] <= 128:  # Skip ortho check for large matrices
        u_f = res_U.float()
        utu = u_f.transpose(-2, -1) @ u_f
        eye_u = torch.eye(utu.shape[-1], dtype=torch.float32, device=flag_gems.device)
        # Repeat batch dims
        for _ in range(utu.dim() - 2):
            eye_u = eye_u.unsqueeze(0)
        eye_u = eye_u.expand_as(utu)
        u_atol = 2e-3 if dtype == torch.float16 else 1e-3
        utils.gems_assert_close(utu, eye_u, torch.float32, reduce_dim=k, atol=u_atol)

    # Orthonormality of V columns: V^T @ V ≈ I
    if res_V.shape[-1] <= 128:
        v_f = res_V.float()
        vtv = v_f.transpose(-2, -1) @ v_f
        eye_v = torch.eye(vtv.shape[-1], dtype=torch.float32, device=flag_gems.device)
        for _ in range(vtv.dim() - 2):
            eye_v = eye_v.unsqueeze(0)
        eye_v = eye_v.expand_as(vtv)
        # Use larger atol for float16 due to reduced precision
        v_atol = 2e-3 if dtype == torch.float16 else 1e-3
        utils.gems_assert_close(vtv, eye_v, torch.float32, reduce_dim=k, atol=v_atol)

    # Singular values in descending order
    if res_S.dim() > 1:
        flat_S = res_S.reshape(-1, res_S.shape[-1])
    else:
        flat_S = res_S.unsqueeze(0)
    for b_idx in range(flat_S.shape[0]):
        s = flat_S[b_idx]
        diffs = s[:-1] - s[1:]
        assert torch.all(diffs >= -1e-6), f"Singular values not descending at batch {b_idx}: {s}"


@pytest.mark.svd
@pytest.mark.parametrize("some,compute_uv", [
    (True, True),
    (False, True),
    (True, False),
    (False, False),
])
def test_svd_modes(some, compute_uv):
    shape = (8, 8)
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp, some=some, compute_uv=compute_uv)

    ref_U, ref_S, ref_V = torch.svd(ref_inp, some=some, compute_uv=compute_uv)

    # Check shapes match
    assert res_U.shape == ref_U.shape, f"U shape mismatch: {res_U.shape} vs {ref_U.shape}"
    assert res_S.shape == ref_S.shape, f"S shape mismatch"
    assert res_V.shape == ref_V.shape, f"V shape mismatch"

    if compute_uv:
        # Singular values should be close
        utils.gems_assert_close(res_S, ref_S.to(dtype), dtype, reduce_dim=1, atol=1e-3)
    else:
        # U and V should be zeros
        assert torch.all(res_U == 0), "U should be zero when compute_uv=False"
        assert torch.all(res_V == 0), "V should be zero when compute_uv=False"


@pytest.mark.svd
@pytest.mark.parametrize("shape", [
    (5, 3),
    (16, 8),
    (8, 16),
    (64, 32),
    (32, 64),
])
def test_svd_rectangular(shape):
    """Test SVD on rectangular matrices."""
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp)

    ref_U, ref_S, ref_V = torch.svd(ref_inp)
    k = min(shape[-2], shape[-1])

    # Reconstruction
    recon_gems = res_U @ torch.diag_embed(res_S) @ res_V.transpose(-2, -1)
    utils.gems_assert_close(recon_gems, inp, dtype, reduce_dim=k, atol=1e-3)

    # Singular values close to reference
    utils.gems_assert_close(res_S, ref_S.to(dtype), dtype, reduce_dim=1, atol=1e-3)


@pytest.mark.svd
def test_svd_zero_matrix():
    """Test SVD on zero matrix."""
    dtype = torch.float32
    shape = (4, 4)
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp)

    # All singular values should be zero (or very close)
    assert torch.all(res_S < 1e-6), f"Singular values of zero matrix should be zero: {res_S}"


@pytest.mark.svd
def test_svd_identity():
    """Test SVD on identity matrix."""
    dtype = torch.float32
    shape = (5, 5)
    inp = torch.eye(shape[0], dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp)

    # All singular values should be 1
    utils.gems_assert_close(res_S, torch.ones(5, dtype=dtype, device=flag_gems.device),
                             dtype, reduce_dim=1, atol=1e-3)

    # Reconstruction should be identity
    recon = res_U @ torch.diag_embed(res_S) @ res_V.transpose(-2, -1)
    utils.gems_assert_close(recon, inp, dtype, reduce_dim=5, atol=1e-3)


@pytest.mark.svd
def test_svd_diagonal():
    """Test SVD on diagonal matrix with known singular values."""
    dtype = torch.float32
    s_vals = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=dtype, device=flag_gems.device)
    inp = torch.diag(s_vals)

    with flag_gems.use_gems():
        res_U, res_S, res_V = torch.svd(inp)

    # Singular values should be sorted descending: [5, 4, 3, 2, 1]
    expected_S = s_vals  # Already sorted
    utils.gems_assert_close(res_S, expected_S, dtype, reduce_dim=1, atol=1e-3)
