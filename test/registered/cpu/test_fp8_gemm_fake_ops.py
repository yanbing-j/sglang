import unittest

# TODO: use interface in cpu.py
import sgl_kernel  # noqa: F401
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang.srt.model_executor.cpu_graph_runner import register_fake_ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFP8GemmFakeOps(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        register_fake_ops(tp_size=1)

    def test_float8_linear_fake_ops(self):
        with FakeTensorMode():
            input = torch.empty(2, 512, dtype=torch.float8_e4m3fn)
            input_scales = torch.empty(2, dtype=torch.float32)
            weight = torch.empty(4, 1, 128, 32, dtype=torch.float8_e4m3fn)
            weight_scales = torch.empty(4, 1, 32, dtype=torch.float32)

            out = torch.ops.sgl_kernel.float8_linear_cpu(
                input,
                input_scales,
                weight,
                weight_scales,
                None,
                torch.bfloat16,
            )
            fused_out = torch.ops.sgl_kernel.fp8_scaled_mm_with_quant(
                input.to(torch.bfloat16),
                None,
                True,
                weight,
                weight_scales,
                None,
                torch.bfloat16,
            )

        self.assertEqual(out.shape, (2, 128))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(fused_out.shape, (2, 128))
        self.assertEqual(fused_out.dtype, torch.bfloat16)

    def test_quantize_fp8_fake_ops(self):
        with FakeTensorMode():
            input = torch.empty(2, 512, dtype=torch.bfloat16)
            per_row_quant, per_row_scale = torch.ops.sgl_kernel._quantize_fp8e4m3_vec(
                input,
                True,
                None,
            )
            per_tensor_quant, per_tensor_scale = torch.ops.sgl_kernel.quantize_fp8e4m3(
                input,
                False,
                torch.empty(1, dtype=torch.float32),
            )

        self.assertEqual(per_row_quant.shape, (2, 512))
        self.assertEqual(per_row_quant.dtype, torch.float8_e4m3fn)
        self.assertEqual(per_row_scale.shape, (2,))
        self.assertEqual(per_row_scale.dtype, torch.float32)
        self.assertEqual(per_tensor_quant.shape, (2, 512))
        self.assertEqual(per_tensor_quant.dtype, torch.float8_e4m3fn)
        self.assertEqual(per_tensor_scale.shape, (1,))
        self.assertEqual(per_tensor_scale.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
