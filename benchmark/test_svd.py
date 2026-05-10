import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


SVD_CORE_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (64, 128),
    (128, 64),
]


class SVDBenchmark(base.Benchmark):
    DEFAULT_SHAPES = SVD_CORE_SHAPES
    DEFAULT_SHAPE_DESC = "M, N"
    DEFAULT_METRICS = base.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def init_user_config(self):
        super().init_user_config()
        # Override shapes with SVD-appropriate 2D shapes.
        # The YAML config may provide shapes that are too large or 1D.
        self.shapes = self.DEFAULT_SHAPES[:]
        if base.Config.bench_level == attrs.BenchLevel.COMPREHENSIVE:
            self.shapes += self.set_more_shapes()

    def set_more_shapes(self):
        return [
            (32, 32),
            (512, 512),
            (256, 64),
            (32, 128),
            (2, 64, 64),
            (2, 3, 32, 32),
        ]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)

    def get_tflops(self, op, *args, **kwargs):
        inp = args[0]
        m, n = inp.shape[-2], inp.shape[-1]
        k = min(m, n)
        batch_size = inp.numel() // (m * n)
        return batch_size * (6 * m * n * k + 8 * k ** 3)


def svd_input_fn(shape, dtype, device):
    # torch.svd (reference) does not support float16/bfloat16, so always
    # generate float32 tensors. Our implementation upcasts internally.
    inp = base.generate_tensor_input(shape, torch.float32, device)
    yield inp,

    if base.Config.bench_level == attrs.BenchLevel.COMPREHENSIVE:
        yield inp, False, True
        yield inp, True, False


@pytest.mark.svd
def test_svd():
    bench = SVDBenchmark(
        op_name="svd",
        input_fn=svd_input_fn,
        torch_op=torch.svd,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
