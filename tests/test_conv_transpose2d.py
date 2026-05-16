import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


CONV_TRANSPOSE2D_CASES = [
    ((2, 3, 8, 8), (3, 4, 3, 3), 1, 0, 0, 1, 1),
    ((2, 3, 8, 8), (3, 4, 3, 3), 2, 1, 1, 1, 1),
    ((2, 3, 7, 8), (3, 5, 2, 3), (2, 1), (0, 1), (0, 0), 1, 1),
    ((2, 3, 8, 8), (3, 4, 3, 3), 1, 2, 0, 1, 2),
    ((2, 4, 8, 8), (4, 3, 3, 3), 2, 1, 1, 2, 1),
]


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "input_shape, weight_shape, stride, padding, output_padding, groups, dilation",
    CONV_TRANSPOSE2D_CASES,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("has_bias", [True, False])
def test_conv_transpose2d(
    input_shape,
    weight_shape,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
    dtype,
    has_bias,
):
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    weight = torch.randn(
        weight_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    out_channels = weight_shape[1] * groups
    if has_bias:
        bias = torch.randn(
            (out_channels,), dtype=dtype, device=flag_gems.device, requires_grad=True
        )
    else:
        bias = None

    ref_out = torch.nn.functional.conv_transpose2d(
        utils.to_reference(inp, True),
        utils.to_reference(weight, True),
        bias=utils.to_reference(bias, True) if bias is not None else None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    ).to(dtype)
    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )

    output_reduce_dim = input_shape[1] * weight_shape[2] * weight_shape[3] * groups
    weight_grad_reduce_dim = 2 * input_shape[0] * res_out.shape[2] * res_out.shape[3]
    bias_grad_reduce_dim = weight_grad_reduce_dim

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=output_reduce_dim)

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True) if bias is not None else None
    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=ref_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
    out_grad = torch.randn_like(res_out, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad, True)
    if bias is not None:
        ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
        )
        res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
    else:
        ref_in_grad, ref_weight_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        res_in_grad, res_weight_grad = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=output_reduce_dim
    )
    utils.gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight_grad_reduce_dim
    )
    if bias is not None:
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, dtype, reduce_dim=bias_grad_reduce_dim
        )
