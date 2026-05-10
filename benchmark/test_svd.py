import time

import pytest
import torch

import flag_gems
from flag_gems.ops.svd import svd as gems_svd


@pytest.mark.svd
def test_perf_svd():
    """Minimal SVD benchmark: one shape, float32, Gems vs PyTorch."""
    shape = (32, 32)
    dtype = torch.float32
    warmup = 5
    iters = 10

    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # FlagGems
    for _ in range(warmup):
        _ = gems_svd(A, full_matrices=False)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = gems_svd(A, full_matrices=False)
    torch.cuda.synchronize()
    gems_ms = (time.time() - t0) / iters * 1000

    # PyTorch
    for _ in range(warmup):
        _ = torch.linalg.svd(A, full_matrices=False)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = torch.linalg.svd(A, full_matrices=False)
    torch.cuda.synchronize()
    torch_ms = (time.time() - t0) / iters * 1000

    speedup = torch_ms / gems_ms
    print(
        f"\n  SVD {shape} {dtype}: "
        f"FlagGems={gems_ms:.2f}ms, PyTorch={torch_ms:.2f}ms, "
        f"speedup={speedup:.2f}x"
    )
