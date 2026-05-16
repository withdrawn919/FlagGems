import pytest
import torch

import flag_gems

from . import base, consts


class ConvTranspose2DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


class ConvTranspose2DBackwardBenchmark(ConvTranspose2DBenchmark):
    pass


BENCH_DTYPES = [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.float32]


def _input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel_h,
        kernel_w,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        with_bias,
    ) = shape
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (input_c, out_c // groups, kernel_h, kernel_w)
    inp = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)
    bias = torch.randn(size=(out_c,), device=device, dtype=dtype) if with_bias else None

    yield inp, weight, bias, {
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation,
    },


def _backward_input_fn(shape, dtype, device):
    for inp, weight, bias, params in _input_fn(shape, dtype, device):
        yield inp, {
            "weight": weight,
            "bias": bias,
            **params,
        }


@pytest.mark.conv_transpose2d
def test_conv_transpose2d():
    torch.backends.cudnn.allow_tf32 = False
    bench = ConvTranspose2DBenchmark(
        input_fn=_input_fn,
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=BENCH_DTYPES,
    )
    bench.set_gems(flag_gems.conv_transpose2d)

    bench.run()


@pytest.mark.conv_transpose2d_backward
def test_conv_transpose2d_backward():
    torch.backends.cudnn.allow_tf32 = False
    bench = ConvTranspose2DBackwardBenchmark(
        input_fn=_backward_input_fn,
        op_name="conv_transpose2d_backward",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=BENCH_DTYPES,
        is_backward=True,
    )
    bench.set_gems(flag_gems.conv_transpose2d)

    bench.run()
