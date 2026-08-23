"""CPU coverage for NVFP4 compressed-tensors linear quantization."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import sgl_kernel  # noqa: F401
import torch

from sglang.srt.layers.quantization.compressed_tensors import (
    compressed_tensors as ct_config_module,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4 as ct_fp4_module,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
    CompressedTensorsW4A4Fp4,
)
from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
from sglang.test.quant_ref_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
)
from sglang.test.test_utils import CustomTestCase


def _has_cpu_fp4_ops() -> bool:
    return (
        hasattr(torch.ops, "sgl_kernel")
        and hasattr(torch.ops.sgl_kernel, "fp4_quantize_cpu")
        and hasattr(torch.ops.sgl_kernel, "convert_weight_packed")
        and hasattr(torch.ops.sgl_kernel, "weight_packed_linear")
    )


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    return (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / x.abs().max().to(torch.float32)
    ).reshape(())


class TestCompressedTensorsW4A4Fp4CPU(CustomTestCase):
    @unittest.skipUnless(_has_cpu_fp4_ops(), "requires sgl_kernel CPU FP4 ops")
    def test_fp4_quantize_cpu_matches_nvfp4_dequant_reference(self):
        torch.manual_seed(0)
        x = torch.randn(7, 64, dtype=torch.bfloat16) * 0.1
        global_scale = _global_scale(x)

        x_fp4, x_scale = fp4_quantize(x, global_scale)

        self.assertEqual(x_fp4.dtype, torch.uint8)
        self.assertEqual(x_fp4.shape, (7, 32))
        self.assertEqual(x_scale.dtype, torch.uint8)
        self.assertEqual(x_scale.shape, (128, 4))

        x_dequant = dequantize_nvfp4_to_dtype(
            x_fp4, x_scale, global_scale, torch.float32
        )
        rel_err = ((x_dequant - x.float()).norm() / x.float().norm()).item()
        self.assertLess(rel_err, 0.18)

    @unittest.skipUnless(_has_cpu_fp4_ops(), "requires sgl_kernel CPU FP4 ops")
    def test_apply_weights_cpu_uses_fp4_quantize(self):
        torch.manual_seed(1)
        m, n, k = 5, 64, 128
        x = torch.randn(m, k, dtype=torch.bfloat16) * 0.1
        weight = torch.randn(n, k, dtype=torch.bfloat16) * 0.1
        bias = torch.randn(n, dtype=torch.float32) * 0.1
        input_global_scale = _global_scale(x)
        weight_global_scale = _global_scale(weight)
        weight_fp4, weight_scale_swizzled = fp4_quantize(weight, weight_global_scale)
        weight_scale = convert_swizzled_to_linear(
            weight_scale_swizzled.view(torch.float8_e4m3fn), n, k, 16
        )

        layer = torch.nn.Module()
        layer.weight_packed = torch.nn.Parameter(weight_fp4, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.input_global_scale = torch.nn.Parameter(
            input_global_scale, requires_grad=False
        )
        layer.weight_global_scale = torch.nn.Parameter(
            weight_global_scale, requires_grad=False
        )
        layer.logical_widths = [n]
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        layer.params_dtype = torch.bfloat16

        scheme = CompressedTensorsW4A4Fp4()
        with patch.object(ct_fp4_module, "_is_cpu", True), patch.object(
            ct_fp4_module, "_is_cpu_amx_available", True
        ):
            scheme.process_weights_after_loading(layer)
            out = scheme.apply_weights(layer, x, bias=bias)

        x_fp4, x_scale = fp4_quantize(x, input_global_scale)
        x_dequant = dequantize_nvfp4_to_dtype(
            x_fp4, x_scale, input_global_scale, torch.float32
        )
        weight_dequant = dequantize_nvfp4_to_dtype(
            weight_fp4, weight_scale_swizzled, weight_global_scale, torch.float32
        )
        ref = torch.nn.functional.linear(x_dequant, weight_dequant, bias).to(x.dtype)
        rel_err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        self.assertLess(rel_err, 0.05)

    def test_w4a4_fp4_linear_scheme_is_cpu_gated_by_amx(self):
        config = CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format="nvfp4-pack-quantized",
            kv_cache_scheme=None,
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )

        with patch.object(ct_config_module, "_is_cpu", True), patch.object(
            ct_config_module, "_is_cpu_amx_available", True
        ):
            self.assertTrue(
                config._check_scheme_supported(
                    CompressedTensorsW4A4Fp4.get_min_capability(),
                    error=False,
                    allow_cpu=True,
                )
            )
            self.assertFalse(
                config._check_scheme_supported(
                    CompressedTensorsW4A4Fp4.get_min_capability(), error=False
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
