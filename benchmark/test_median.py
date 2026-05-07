import math

import pytest
import torch

from . import base, consts


class MedianBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]
    MAX_N = 2**20
    MAX_BITONIC_M = 1024

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        filtered = []
        for shape in self.shapes:
            if any(dim >= self.MAX_N for dim in shape):
                continue
            N = shape[-1]
            M = math.prod(shape[:-1]) if len(shape) > 1 else 1
            if N <= 1024 and M > self.MAX_BITONIC_M:
                continue
            if len(shape) >= 3 and N <= 1024 and M > 1024:
                continue
            filtered.append(shape)
        self.shapes = filtered

    def set_more_shapes(self):
        bitonic_2d = [
            (256, 64),
            (512, 128),
            (768, 256),
        ]
        sort_2d = [
            (1024, 2048),
            (512, 4096),
            (256, 8192),
            (128, 16384),
            (64, 32768),
        ]
        return bitonic_2d + sort_2d

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = sum([base.shape_utils.size_in_bytes(item) for item in [inp, inp]])
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            yield inp, -1


@pytest.mark.median
def test_median():
    bench = MedianBenchmark(
        op_name="median", torch_op=torch.median, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
