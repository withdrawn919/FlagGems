import os

import pytest
import torch

from . import base, consts

MEDIAN_SHAPE_FILE = os.path.join(os.path.dirname(__file__), "core_shapes.yaml")


class MedianBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]
    DEFAULT_SHAPE_FILES = MEDIAN_SHAPE_FILE

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


# ===========================================================================
# median.out benchmark — out variant of default median
# ===========================================================================
class MedianOutBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]
    DEFAULT_SHAPE_FILES = MEDIAN_SHAPE_FILE

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = sum([base.shape_utils.size_in_bytes(item) for item in [inp, inp]])
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            out = torch.empty([], dtype=inp.dtype, device=inp.device)
            yield inp, {"out": out}


@pytest.mark.median_out
def test_median_out_bench():
    bench = MedianOutBenchmark(
        op_name="median_out",
        torch_op=lambda inp, *, out: torch.ops.aten.median.out(inp, out=out),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# ===========================================================================
# median.dim benchmark — dim variant returning (values, indices)
# ===========================================================================
class MedianDimBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]
    DEFAULT_SHAPE_FILES = MEDIAN_SHAPE_FILE

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
                yield inp, 0


@pytest.mark.median_dim
def test_median_dim_bench():
    bench = MedianDimBenchmark(
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# ===========================================================================
# median.dim_values benchmark — out variant of dim median
# ===========================================================================
class MedianDimValuesBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]
    DEFAULT_SHAPE_FILES = MEDIAN_SHAPE_FILE

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = sum([base.shape_utils.size_in_bytes(item) for item in [inp, inp]])
        return io_amount * 1e-9 / (latency * 1e-3)

    def _out_shapes(self, inp, dim):
        ndim = inp.ndim
        _dim = dim % ndim
        out_shape = list(inp.shape)
        out_shape[_dim] = 1
        return out_shape, out_shape

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            if inp.ndim > 1:
                dim = -1
            else:
                dim = 0
            v_shape, i_shape = self._out_shapes(inp, dim)
            values = torch.empty(v_shape, dtype=inp.dtype, device=inp.device)
            indices = torch.empty(i_shape, dtype=torch.long, device=inp.device)
            yield inp, dim, {"values": values, "indices": indices}


@pytest.mark.median_dim_values
def test_median_dim_values_bench():
    bench = MedianDimValuesBenchmark(
        op_name="median_dim_values",
        torch_op=lambda inp, dim, *, values, indices: torch.ops.aten.median.dim_values(
            inp, dim, keepdim=True, values=values, indices=indices
        ),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
