"""CPU coverage for compressed-tensors W8A8 INT8 linear dispatch."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.compressed_tensors import (
    compressed_tensors as ct_config_module,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8 as ct_int8_module,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from sglang.test.test_utils import CustomTestCase


def _has_cpu_int8_ops() -> bool:
    return (
        hasattr(torch.ops, "sgl_kernel")
        and hasattr(torch.ops.sgl_kernel, "per_token_quant_int8_cpu")
        and hasattr(torch.ops.sgl_kernel, "int8_scaled_mm_cpu")
    )


def _make_channel_scheme() -> CompressedTensorsW8A8Int8:
    return CompressedTensorsW8A8Int8(
        strategy=QuantizationStrategy.CHANNEL,
        is_static_input_scheme=False,
        input_symmetric=True,
    )


def _make_tensor_scheme() -> CompressedTensorsW8A8Int8:
    return CompressedTensorsW8A8Int8(
        strategy=QuantizationStrategy.TENSOR,
        is_static_input_scheme=True,
        input_symmetric=True,
    )


class TestCompressedTensorsW8A8Int8CPU(CustomTestCase):
    @unittest.skipUnless(_has_cpu_int8_ops(), "requires sgl_kernel CPU INT8 ops")
    def test_per_token_quant_int8_uses_cpu_kernel(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 128, dtype=torch.bfloat16) / 10

        x_q, x_scale = per_token_quant_int8(x)
        ref_q, ref_scale = torch.ops.sgl_kernel.per_token_quant_int8_cpu(
            x.view(-1, x.shape[-1])
        )

        self.assertEqual(x_q.dtype, torch.uint8)
        self.assertEqual(x_q.shape, x.shape)
        self.assertEqual(x_scale.shape, (*x.shape[:-1], 1))
        torch.testing.assert_close(x_q.view(-1, x.shape[-1]), ref_q)
        torch.testing.assert_close(x_scale.view(-1), ref_scale)

    @unittest.skipUnless(_has_cpu_int8_ops(), "requires sgl_kernel CPU INT8 ops")
    def test_channel_weight_uses_cpu_per_token_quant_int8_and_scaled_mm(self):
        torch.manual_seed(0)
        n, k = 64, 128
        x = torch.randn(2, 3, k, dtype=torch.bfloat16) / 10
        weight = torch.randn(n, k, dtype=torch.float32) / 10
        weight_scale = (
            weight.abs().amax(dim=-1, keepdim=True) / torch.iinfo(torch.int8).max
        ).clamp(min=torch.finfo(torch.float32).eps)
        weight_int8 = torch.round(weight / weight_scale).clamp(-128, 127).to(torch.int8)
        bias = torch.randn(n, dtype=torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_int8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.logical_widths = [n]

        scheme = _make_channel_scheme()
        with patch.object(ct_int8_module, "_is_cpu", True), patch.object(
            ct_int8_module, "_is_cpu_amx_available", True
        ):
            scheme.process_weights_after_loading(layer)
            out = scheme.apply_weights(layer, x, bias=bias)

        self.assertTrue(layer.use_intel_amx_backend)
        self.assertEqual(layer.weight.shape, (n, k + 4))

        x_2d = x.view(-1, k)
        x_q, x_scale = per_token_quant_int8(x_2d)
        ref = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            x_q,
            layer.weight,
            x_scale,
            layer.weight_scale,
            bias,
            x.dtype,
            True,
        ).view(2, 3, n)

        torch.testing.assert_close(out, ref)

    @unittest.skipUnless(_has_cpu_int8_ops(), "requires sgl_kernel CPU INT8 ops")
    def test_tensor_weight_uses_cpu_per_token_quant_int8_and_scaled_mm(self):
        torch.manual_seed(1)
        n, k = 64, 128
        x = torch.randn(4, k, dtype=torch.bfloat16) / 10
        weight = torch.randn(n, k, dtype=torch.float32) / 10
        weight_scale = (weight.abs().max() / torch.iinfo(torch.int8).max).clamp(
            min=torch.finfo(torch.float32).eps
        )
        weight_int8 = torch.round(weight / weight_scale).clamp(-128, 127).to(torch.int8)
        bias = torch.randn(n, dtype=torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_int8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            weight_scale.reshape(1), requires_grad=False
        )
        layer.input_scale = torch.nn.Parameter(
            torch.tensor([0.05]), requires_grad=False
        )
        layer.logical_widths = [n]

        scheme = _make_tensor_scheme()
        with patch.object(ct_int8_module, "_is_cpu", True), patch.object(
            ct_int8_module, "_is_cpu_amx_available", True
        ):
            scheme.process_weights_after_loading(layer)
            out = scheme.apply_weights(layer, x, bias=bias)

        self.assertTrue(layer.use_intel_amx_backend)
        self.assertEqual(layer.weight.shape, (n, k + 4))
        self.assertEqual(layer.weight_scale.numel(), n)
        self.assertEqual(layer.input_scale.numel(), 1)

        x_q, x_scale = per_token_quant_int8(x)
        ref = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            x_q,
            layer.weight,
            x_scale,
            layer.weight_scale,
            bias,
            x.dtype,
            True,
        )

        torch.testing.assert_close(out, ref)

    def test_w8a8_int8_linear_scheme_is_cpu_gated_by_amx(self):
        config = CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format="int-quantized",
            kv_cache_scheme=None,
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )

        with patch.object(ct_config_module, "_is_cpu", True), patch.object(
            ct_config_module, "_is_cpu_amx_available", True
        ):
            self.assertTrue(
                config._check_scheme_supported(
                    CompressedTensorsW8A8Int8.get_min_capability(),
                    error=False,
                    allow_cpu=True,
                )
            )
            self.assertFalse(
                config._check_scheme_supported(
                    CompressedTensorsW8A8Int8.get_min_capability(), error=False
                )
            )

        weight_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
        )
        input_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
        )
        with patch.object(ct_config_module, "_is_cpu", True), patch.object(
            ct_config_module, "_is_cpu_amx_available", True
        ):
            scheme = config._get_scheme_from_parts(
                weight_quant=weight_quant,
                input_quant=input_quant,
                format="int-quantized",
            )
            config._check_scheme_supported(
                scheme.get_min_capability(),
                allow_cpu=isinstance(scheme, CompressedTensorsW8A8Int8),
            )
            self.assertIsInstance(scheme, CompressedTensorsW8A8Int8)


if __name__ == "__main__":
    unittest.main(verbosity=3)
