"""CPU coverage for compressed-tensors W8A16 FP8 linear dispatch."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.quantization.compressed_tensors import (
    compressed_tensors as ct_config_module,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a16_fp8 as ct_fp8_module,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (
    CompressedTensorsW8A16Fp8,
)
from sglang.test.test_utils import CustomTestCase


def _has_cpu_fp8_ops() -> bool:
    return hasattr(torch.ops, "sgl_kernel") and hasattr(
        torch.ops.sgl_kernel, "fp8_scaled_mm_cpu"
    )


class TestCompressedTensorsW8A16Fp8CPU(CustomTestCase):
    @unittest.skipUnless(_has_cpu_fp8_ops(), "requires sgl_kernel CPU FP8 ops")
    def test_apply_weights_uses_cpu_fp8_scaled_mm(self):
        torch.manual_seed(0)
        n, k = 128, 256
        x = torch.randn(3, k, dtype=torch.bfloat16) * 0.1
        weight = torch.randn(n, k, dtype=torch.float32) * 0.1
        weight_scale = (
            weight.abs().amax(dim=-1, keepdim=True)
            / torch.finfo(torch.float8_e4m3fn).max
        ).clamp(min=torch.finfo(torch.float32).eps)
        weight_fp8 = (weight / weight_scale).to(torch.float8_e4m3fn)
        bias = torch.randn(n, dtype=torch.float32) * 0.1

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.logical_widths = [n]
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n

        scheme = CompressedTensorsW8A16Fp8(
            strategy=QuantizationStrategy.CHANNEL, is_static_input_scheme=False
        )
        with patch.object(ct_fp8_module, "_is_cpu", True), patch.object(
            ct_fp8_module, "_is_cpu_amx_available", True
        ):
            scheme.process_weights_after_loading(layer)
            out = scheme.apply_weights(layer, x, bias=bias)

        self.assertEqual(layer.weight.dim(), 2)
        self.assertEqual(layer.weight_scale.shape, (2, 2))
        self.assertEqual(layer.weight_block_size, [64, 128])

        dequant_weight = weight_fp8.float() * weight_scale
        ref = torch.nn.functional.linear(x.float(), dequant_weight.float(), bias).to(
            x.dtype
        )
        rel_err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        self.assertLess(rel_err, 0.10)

    def test_w8a16_fp8_linear_scheme_is_cpu_gated_by_amx(self):
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
                    CompressedTensorsW8A16Fp8.get_min_capability(),
                    error=False,
                    allow_cpu=True,
                )
            )
            self.assertFalse(
                config._check_scheme_supported(
                    CompressedTensorsW8A16Fp8.get_min_capability(), error=False
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
