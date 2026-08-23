"""CPU coverage for compressed-tensors WNA16 GPTQ Marlin dispatch."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch
from compressed_tensors.quantization import QuantizationStrategy
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.quantization.compressed_tensors import (
    compressed_tensors as ct_config_module,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16 as ct_wna16_module,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    CompressedTensorsWNA16,
)
from sglang.srt.layers.quantization.marlin_utils import MarlinLinearLayerConfig
from sglang.srt.layers.quantization.utils import gptq_quantize_weights, pack_cols
from sglang.srt.utils import cpu_has_amx_support
from sglang.test.test_utils import CustomTestCase


def _has_cpu_wna16_ops() -> bool:
    return (
        hasattr(torch.ops, "sgl_kernel")
        and hasattr(torch.ops.sgl_kernel, "convert_weight_packed")
        and hasattr(torch.ops.sgl_kernel, "weight_packed_linear")
    )


class TestCompressedTensorsWNA16CPU(CustomTestCase):
    @unittest.skipUnless(
        _has_cpu_wna16_ops() and cpu_has_amx_support(),
        "requires sgl_kernel CPU packed GEMM ops and AMX support",
    )
    def test_apply_gptq_marlin_linear_cpu_matches_dequant_reference(self):
        for num_bits, quant_type in (
            (4, scalar_types.uint4b8),
            (8, scalar_types.uint8b128),
        ):
            with self.subTest(num_bits=num_bits):
                self._run_apply_gptq_marlin_linear_cpu_case(num_bits, quant_type)

    def _run_apply_gptq_marlin_linear_cpu_case(self, num_bits, quant_type):
        torch.manual_seed(100 + num_bits)
        m, k, n, group_size = 5, 128, 64, 128
        weight = torch.randn(k, n, dtype=torch.bfloat16) / 10
        weight_ref, qweight, weight_scale, _, _ = gptq_quantize_weights(
            weight, quant_type, group_size, act_order=False
        )
        compressed_weight = pack_cols(qweight.t().contiguous(), num_bits, n, k)

        layer = torch.nn.Module()
        layer.weight_packed = torch.nn.Parameter(compressed_weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            weight_scale.t().contiguous(), requires_grad=False
        )

        scheme = CompressedTensorsWNA16(
            strategy=QuantizationStrategy.GROUP.value,
            num_bits=num_bits,
            group_size=group_size,
            symmetric=True,
            actorder=None,
        )
        scheme.kernel_config = MarlinLinearLayerConfig(
            full_weight_shape=(k, n),
            partition_weight_shape=(k, n),
            weight_type=quant_type,
            act_type=torch.bfloat16,
            group_size=group_size,
            zero_points=False,
            has_g_idx=False,
        )

        x = torch.randn(m, k, dtype=torch.bfloat16)
        bias = torch.randn(n, dtype=torch.float32)

        with patch.object(ct_wna16_module, "_is_cpu", True), patch.object(
            ct_wna16_module, "_is_cpu_amx_available", True
        ):
            scheme.process_weights_after_loading(layer)
            out = scheme.apply_weights(layer, x, bias)

        ref = torch.matmul(x, weight_ref).to(torch.bfloat16)
        ref.add_(bias.to(torch.bfloat16))
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    def test_wna16_linear_scheme_is_cpu_gated_by_amx(self):
        config = CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format="pack-quantized",
            kv_cache_scheme=None,
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )

        with patch.object(ct_config_module, "_is_cpu", True), patch.object(
            ct_config_module, "_is_cpu_amx_available", True
        ):
            self.assertTrue(
                config._check_scheme_supported(
                    CompressedTensorsWNA16.get_min_capability(),
                    error=False,
                    allow_cpu=True,
                )
            )
            self.assertFalse(
                config._check_scheme_supported(
                    CompressedTensorsWNA16.get_min_capability(), error=False
                )
            )


if __name__ == "__main__":
    unittest.main()
