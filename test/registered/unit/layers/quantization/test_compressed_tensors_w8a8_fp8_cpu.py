"""CPU coverage for compressed-tensors W8A8 FP8 linear dispatch."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

import sglang.srt.layers.quantization.fp8_utils as fp8_utils
from sglang.srt.layers.quantization.compressed_tensors import (
    compressed_tensors as ct_config_module,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_fp8 as ct_fp8_module,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear
from sglang.test.test_utils import CustomTestCase


def _has_cpu_fp8_ops() -> bool:
    return hasattr(torch.ops, "sgl_kernel") and hasattr(
        torch.ops.sgl_kernel, "float8_linear_prepack_cpu"
    )


def _make_channel_scheme() -> CompressedTensorsW8A8Fp8:
    weight_quant = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )
    return CompressedTensorsW8A8Fp8(
        weight_quant=weight_quant, is_static_input_scheme=False
    )


def _make_tensor_scheme() -> CompressedTensorsW8A8Fp8:
    weight_quant = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    return CompressedTensorsW8A8Fp8(
        weight_quant=weight_quant, is_static_input_scheme=True
    )


def _make_block_scheme() -> CompressedTensorsW8A8Fp8:
    weight_quant = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=QuantizationStrategy.BLOCK,
        symmetric=True,
        dynamic=False,
        block_structure=[128, 128],
    )
    return CompressedTensorsW8A8Fp8(
        weight_quant=weight_quant, is_static_input_scheme=False
    )


class TestCompressedTensorsW8A8Fp8CPU(CustomTestCase):
    @unittest.skipUnless(_has_cpu_fp8_ops(), "requires sgl_kernel CPU FP8 ops")
    def test_apply_fp8_linear_cpu_uses_prepacked_weight(self):
        torch.manual_seed(0)
        n, k = 64, 128
        x = torch.randn(5, k, dtype=torch.bfloat16)
        weight = torch.randn(n, k, dtype=torch.float32)
        weight_scale = (
            weight.abs().amax(dim=-1, keepdim=True)
            / torch.finfo(torch.float8_e4m3fn).max
        ).clamp(min=torch.finfo(torch.float32).eps)
        weight_fp8 = (weight / weight_scale).to(torch.float8_e4m3fn)
        packed_weight, packed_scale = torch.ops.sgl_kernel.float8_linear_prepack_cpu(
            weight_fp8.contiguous(), weight_scale.contiguous()
        )
        bias = torch.randn(n, dtype=torch.float32)

        with patch.object(fp8_utils, "_is_cpu", True):
            out = apply_fp8_linear(
                x,
                packed_weight,
                packed_scale,
                input_scale=None,
                bias=bias,
                use_per_token_if_dynamic=True,
            )

        x_q, x_scale = torch.ops.sgl_kernel._quantize_fp8e4m3_vec(x, True, None)
        ref = torch.ops.sgl_kernel.float8_linear_cpu(
            x_q, x_scale, packed_weight, packed_scale, bias, x.dtype
        )
        torch.testing.assert_close(out, ref)

    @unittest.skipUnless(_has_cpu_fp8_ops(), "requires sgl_kernel CPU FP8 ops")
    def test_channel_weight_uses_cpu_prepack_and_apply_fp8_linear(self):
        torch.manual_seed(0)
        n, k = 64, 128
        x = torch.randn(2, 3, k, dtype=torch.bfloat16)
        weight = torch.randn(n, k, dtype=torch.float32)
        weight_scale = (
            weight.abs().amax(dim=-1, keepdim=True)
            / torch.finfo(torch.float8_e4m3fn).max
        ).clamp(min=torch.finfo(torch.float32).eps)
        weight_fp8 = (weight / weight_scale).to(torch.float8_e4m3fn)
        bias = torch.randn(n, dtype=torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.logical_widths = [n]

        scheme = _make_channel_scheme()
        with patch.object(ct_fp8_module, "_is_cpu", True), patch.object(
            ct_fp8_module, "_is_cpu_amx_available", True
        ), patch.object(fp8_utils, "_is_cpu", True):
            scheme.process_weights_after_loading(layer)
            out = scheme.apply_weights(layer, x, bias=bias)

        self.assertEqual(layer.weight.dim(), 4)
        self.assertIsNone(layer.input_scale)

        x_2d = x.view(-1, k)
        x_q, x_scale = torch.ops.sgl_kernel._quantize_fp8e4m3_vec(x_2d, True, None)
        ref = torch.ops.sgl_kernel.float8_linear_cpu(
            x_q, x_scale, layer.weight, layer.weight_scale, bias, x.dtype
        ).view(2, 3, n)

        torch.testing.assert_close(out, ref)

    @unittest.skipUnless(_has_cpu_fp8_ops(), "requires sgl_kernel CPU FP8 ops")
    def test_static_tensor_input_scale_uses_cpu_apply_fp8_linear(self):
        torch.manual_seed(1)
        n, k = 64, 128
        x = torch.randn(4, k, dtype=torch.bfloat16)
        weight = torch.randn(n, k, dtype=torch.float32)
        weight_scale = (
            weight.abs().max() / torch.finfo(torch.float8_e4m3fn).max
        ).clamp(min=torch.finfo(torch.float32).eps)
        weight_fp8 = (weight / weight_scale).to(torch.float8_e4m3fn)
        bias = torch.randn(n, dtype=torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            weight_scale.reshape(1), requires_grad=False
        )
        layer.input_scale = torch.nn.Parameter(
            torch.tensor([0.05]), requires_grad=False
        )
        layer.logical_widths = [n]

        scheme = _make_tensor_scheme()
        with patch.object(ct_fp8_module, "_is_cpu", True), patch.object(
            ct_fp8_module, "_is_cpu_amx_available", True
        ), patch.object(fp8_utils, "_is_cpu", True):
            scheme.process_weights_after_loading(layer)
            out = scheme.apply_weights(layer, x, bias=bias)

        self.assertEqual(layer.weight.dim(), 4)
        self.assertEqual(layer.input_scale.numel(), 1)

        x_q, x_scale = torch.ops.sgl_kernel._quantize_fp8e4m3_vec(
            x, False, layer.input_scale
        )
        ref = torch.ops.sgl_kernel.float8_linear_cpu(
            x_q, x_scale, layer.weight, layer.weight_scale, bias, x.dtype
        )

        torch.testing.assert_close(out, ref)

    @unittest.skipUnless(_has_cpu_fp8_ops(), "requires sgl_kernel CPU FP8 ops")
    def test_block_weight_uses_cpu_fp8_scaled_mm_with_quant(self):
        torch.manual_seed(2)
        n, k = 128, 256
        block_n, block_k = 128, 128
        x = torch.randn(2, 3, k, dtype=torch.bfloat16)
        x[..., :block_k] *= 0.01
        x[..., block_k:] *= 8.0
        weight = torch.randn(n, k, dtype=torch.float32)
        weight_view = weight.view(n // block_n, block_n, k // block_k, block_k)
        weight_scale = (
            weight_view.abs().amax(dim=(1, 3)) / torch.finfo(torch.float8_e4m3fn).max
        ).clamp(min=torch.finfo(torch.float32).eps)
        weight_fp8 = (
            (weight_view / weight_scale[:, None, :, None])
            .to(torch.float8_e4m3fn)
            .view(n, k)
        )
        bias = torch.randn(n, dtype=torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.logical_widths = [n]
        layer.weight_block_size = [block_n, block_k]

        scheme = _make_block_scheme()
        with patch.object(ct_fp8_module, "_is_cpu", True), patch.object(
            ct_fp8_module, "_is_cpu_amx_available", True
        ):
            scheme.process_weights_after_loading(layer)
            out = scheme.apply_weights(layer, x, bias=bias)

        self.assertEqual(layer.weight.dim(), 4)
        self.assertIsNone(layer.input_scale)

        x_2d = x.view(-1, k)
        x_q, x_scale = torch.ops.sgl_kernel.per_token_group_quant_fp8_cpu(
            x_2d.contiguous(), block_k, 1e-10
        )
        ref = torch.ops.sgl_kernel.float8_linear_cpu(
            x_q, x_scale, layer.weight, layer.weight_scale, bias, x.dtype
        ).view(2, 3, n)

        torch.testing.assert_close(out, ref)

    def test_w8a8_fp8_linear_scheme_is_cpu_gated_by_amx(self):
        config = CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format="float-quantized",
            kv_cache_scheme=None,
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )

        with patch.object(ct_config_module, "_is_cpu", True), patch.object(
            ct_config_module, "_is_cpu_amx_available", True
        ):
            self.assertTrue(
                config._check_scheme_supported(
                    CompressedTensorsW8A8Fp8.get_min_capability(),
                    error=False,
                    allow_cpu=True,
                )
            )
            self.assertFalse(
                config._check_scheme_supported(
                    CompressedTensorsW8A8Fp8.get_min_capability(), error=False
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
