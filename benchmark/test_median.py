import math

import pytest
import torch

from . import base, consts

# Benchmark shapes organized by input dimension and size category
BENCH_SHAPES = {
    "1d": [(8,), (64,), (1024,), (4096,), (32768,)],
    "2d_small": [(1, 1), (8, 8), (16, 32), (32, 64), (7, 13), (37, 99)],
    "2d_regular": [(64, 64), (256, 128), (128, 256), (77, 233)],
    "2d_large": [(1024, 1024), (1024, 2048), (333, 1333)],
    "3d": [(8, 32, 64), (32, 128, 512), (16, 64, 1024), (11, 23, 47)],
    "4d": [(2, 4, 8, 16), (4, 8, 16, 512), (4, 8, 16, 1024), (3, 5, 7, 11)],
    "5d": [(2, 2, 4, 8, 16), (2, 4, 8, 16, 512), (2, 3, 5, 7, 13)],
}


class MedianBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]
    MAX_N = 2**20
    MAX_BITONIC_M = 256
    MAX_TOTAL_ELEMS = 4 * 1024 * 1024

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        filtered = []
        for shape in self.shapes:
            if any(dim >= self.MAX_N for dim in shape):
                continue
            N = shape[-1]
            M = math.prod(shape[:-1]) if len(shape) > 1 else 1
            if N <= 1024 and M > 16 * 1024:
                continue
            if math.prod(shape) > self.MAX_TOTAL_ELEMS:
                continue
            filtered.append(shape)
        self.shapes = filtered

    def set_more_shapes(self):
        shapes = []
        for cat in ["1d", "2d_small", "2d_regular", "2d_large", "3d", "4d", "5d"]:
            shapes.extend(BENCH_SHAPES[cat])
        return shapes

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = sum([base.shape_utils.size_in_bytes(item) for item in [inp, inp]])
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            if inp.ndim > 1:
                yield inp, -1
            else:
                yield inp,


@pytest.mark.median
def test_median():
    bench = MedianBenchmark(
        op_name="median", torch_op=torch.median, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
