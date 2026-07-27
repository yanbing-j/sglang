import unittest

# TODO: use interface in cpu.py
import sgl_kernel  # noqa: F401
import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import (
    MXFP4QuantizeUtil,
    convert_weight,
    native_w8a8_per_token_matmul,
    parametrize,
    per_token_quant_int8,
    precision,
    unpack_and_dequant_awq,
    unpack_and_dequant_gptq,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    static_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

torch.manual_seed(1234)


class Mod(nn.Module):
    def __init__(self, input_channel, output_channel, has_bias):
        super(Mod, self).__init__()
        self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

    def forward(self, x):
        return self.linear(x)


class TestGemm(CustomTestCase):

    @parametrize(
        M=[1, 101],
        N=[16, 32 * 13],
        K=[32 * 16],
        has_bias=[False, True],
        dim=[2, 3, 4, 5],
    )
    def test_bf16_gemm(self, M, N, K, has_bias, dim):
        mat1 = torch.randn(M, K, dtype=torch.bfloat16)
        mat2 = torch.randn(N, K, dtype=torch.bfloat16)
        if dim == 3:
            mat1 = mat1.unsqueeze(0).repeat(2, 1, 1)
        if dim == 4:
            mat1 = mat1.unsqueeze(0).unsqueeze(0).repeat(2, 2, 1, 1)
        if dim == 5:
            mat1 = mat1.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(2, 2, 2, 1, 1)

        ref = torch.matmul(mat1.float(), mat2.float().t())
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)
            ref.add_(bias.bfloat16())

        ref = ref.bfloat16()

        out = torch.ops.sgl_kernel.weight_packed_linear(
            mat1, mat2, bias if has_bias else None, False
        )

        packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(mat2)
        out2 = torch.ops.sgl_kernel.weight_packed_linear(
            mat1, packed_mat2, bias if has_bias else None, True
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref, out2, atol=atol, rtol=rtol)

    @parametrize(M=[1, 4, 5, 101], N=[16, 32 * 13], K=[32 * 16], has_bias=[False, True])
    def test_fp16_gemm(self, M, N, K, has_bias):
        # M <= 4 takes the AVX10.2 tinygemm, above it brgemm
        mat1 = torch.randn(M, K, dtype=torch.float16)
        mat2 = torch.randn(N, K, dtype=torch.float16)

        ref = torch.matmul(mat1.float(), mat2.float().t())
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)
            ref.add_(bias.half())

        ref = ref.half()

        out = torch.ops.sgl_kernel.weight_packed_linear(
            mat1, mat2, bias if has_bias else None, False
        )

        packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(mat2)
        out2 = torch.ops.sgl_kernel.weight_packed_linear(
            mat1, packed_mat2, bias if has_bias else None, True
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref, out2, atol=atol, rtol=rtol)

    @parametrize(
        M=[1, 8, 32, 1024],
        N=[12, 1],
        K=[32 * 16],
        has_bias=[False, True],
        use_post_sigmul=[False, True],
    )
    def bf16_gemm_with_small_oc(self, M, N, K, has_bias, use_post_sigmul):
        use_post_sigmul = use_post_sigmul and N == 1
        mat_mul = (
            None if not use_post_sigmul else torch.randn(M, 2 * K, dtype=torch.bfloat16)
        )
        mat1 = torch.randn(M, K, dtype=torch.bfloat16)
        mat2 = torch.randn(N, K, dtype=torch.bfloat16)

        ref = torch.nn.functional.linear(mat1, mat2)
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)
            ref.add_(bias)
        if use_post_sigmul:
            ref = torch.nn.functional.sigmoid(ref) * mat_mul
            out = torch.ops.sgl_kernel.fused_linear_sigmoid_mul(
                mat1,
                torch.ops.sgl_kernel.convert_weight_packed(mat2),
                bias if has_bias else None,
                True,
                mat_mul if use_post_sigmul else None,
            )
        else:
            out = torch.ops.sgl_kernel.weight_packed_linear(
                mat1,
                torch.ops.sgl_kernel.convert_weight_packed(mat2),
                bias if has_bias else None,
                True,
            )
        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @parametrize(M=[2, 128], N=[32 * 12], K=[32 * 17], has_bias=[False, True])
    def test_int8_gemm(self, M, N, K, has_bias):
        dtype = torch.bfloat16
        A = torch.randn((M, K), dtype=dtype) / 10
        Aq, As = per_token_quant_int8(A)

        factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        B = (torch.rand((N, K), dtype=torch.float32) - 0.5) * 2
        Bq = (B * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
        Bs = torch.rand(N) * factor_for_scale

        bias = torch.randn(N) if has_bias else None
        ref_out = native_w8a8_per_token_matmul(Aq, Bq, As, Bs, bias, dtype)

        atol = rtol = precision[ref_out.dtype]

        Aq2, As2 = torch.ops.sgl_kernel.per_token_quant_int8_cpu(A)
        out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            Aq2, Bq, As2, Bs, bias if has_bias else None, torch.bfloat16, False
        )
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

        # test the fused version
        fused_out = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            A, Bq, Bs, bias if has_bias else None, torch.bfloat16, False
        )
        torch.testing.assert_close(ref_out, fused_out, atol=atol, rtol=rtol)

    @parametrize(M=[1, 11], N=[128, 224], K=[512, 576], has_bias=[False, True])
    def test_fp8_gemm(self, M, N, K, has_bias):
        prepack = True
        chunk = False
        scale_block_size_N = 64
        scale_block_size_K = 128
        assert scale_block_size_N <= N
        assert scale_block_size_K <= K
        dtype = torch.bfloat16

        model = Mod(K, N, has_bias).eval()
        if chunk:
            data = torch.randn(M, K + 6, dtype=dtype).narrow(1, 0, K)
        else:
            data = torch.randn(M, K, dtype=dtype)

        weight = model.linear.weight  # (N, K)

        if has_bias:
            bias = model.linear.bias

        fp8_weight, scales, dq_weight = convert_weight(
            weight, [scale_block_size_N, scale_block_size_K], dtype
        )

        if has_bias:
            ref = torch.matmul(data.to(dtype), dq_weight.T) + bias.to(dtype)
        else:
            ref = torch.matmul(data.to(dtype), dq_weight.T)

        if prepack:
            fp8_weight = torch.ops.sgl_kernel.convert_weight_packed(fp8_weight)

        out = torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
            data,
            fp8_weight,
            scales,
            [scale_block_size_N, scale_block_size_K],
            bias if has_bias else None,
            data.dtype,
            prepack,
        )
        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @parametrize(M=[1, 11], N=[128, 224], K=[512, 576], has_bias=[False, True])
    def test_mxfp4_gemm(self, M, N, K, has_bias):
        prepack = True
        dtype = torch.bfloat16

        A = torch.randn((M, K), dtype=dtype) / 10

        # we randomly generate Bq and Bs, then dequantize it to BFloat16 as reference
        Bq = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        Bs = torch.randint(126, 127, (N, K // 32), dtype=torch.uint8)

        Bdq = MXFP4QuantizeUtil.dequantize(Bq, dtype, Bs)

        B_packed = torch.ops.sgl_kernel.convert_weight_packed(Bq)
        Bs_packed = torch.ops.sgl_kernel.convert_scale_packed(Bs)

        bias = torch.randn(N) if has_bias else None

        ref = torch.matmul(A.float(), Bdq.float().t()).bfloat16()
        if bias is not None:
            ref.add_(bias.view(1, -1))

        out = torch.ops.sgl_kernel.mxfp4_scaled_mm_cpu(
            A, B_packed, Bs_packed, bias, prepack
        )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @parametrize(
        M=[1, 32], N=[4096], K=[4096], group_size=[128], has_bias=[False, True]
    )
    def test_int4_awq_gemm(self, M, N, K, group_size, has_bias):
        awq_weight = torch.randint(-128, 128, (K, N // 8)).to(torch.int)
        awq_zero = torch.randint(0, 10, (K // group_size, N // 8)).to(torch.int)
        awq_scales = torch.rand(int(K // group_size), N).to(torch.bfloat16)
        bf16_weight, _ = unpack_and_dequant_awq(
            awq_weight, awq_zero, awq_scales, 4, 128
        )
        if has_bias:
            bias = torch.rand(bf16_weight.shape[0]).to(torch.float)
        else:
            bias = None
        x = torch.rand(M, bf16_weight.size(-1)).to(torch.bfloat16)
        ref_res = torch.nn.functional.linear(
            x, bf16_weight, bias=bias.to(torch.bfloat16) if has_bias else None
        )

        packed_weight, packed_zero, packed_scales = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                awq_weight, awq_zero, awq_scales, 0
            )
        )
        target_res = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
            x,
            packed_weight,
            packed_zero,
            packed_scales,
            bias,
        )

        atol = rtol = precision[ref_res.dtype]
        torch.testing.assert_close(ref_res, target_res, atol=atol, rtol=rtol)

    @parametrize(M=[1, 11], N=[128, 224], K=[512, 576], has_bias=[False, True])
    def test_fp8_w8a8_gemm(self, M, N, K, has_bias):
        """Test fp8_scaled_mm_with_quant matches unfused _quantize_fp8e4m3_vec + float8_linear_cpu.

        We use the kernel's own quantization op as the reference to avoid mismatches
        from different fp32->fp8 rounding conventions (custom AVX512 intrinsic vs PyTorch).
        """
        dtype = torch.bfloat16
        fp8_max = 448.0
        eps = torch.finfo(torch.float32).eps

        act = torch.randn(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=torch.float32)

        # Per-channel weight quantization
        weight_scale = (weight.abs().amax(dim=-1, keepdim=True) / fp8_max).clamp(
            min=eps
        )
        weight_fp8 = (
            (weight / weight_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        )

        # Pack weight for CPU (float8_linear_prepack_cpu expects [N, K] weight and [N, 1] scale)
        packed_weight, packed_scales = torch.ops.sgl_kernel.float8_linear_prepack_cpu(
            weight_fp8, weight_scale.to(torch.float32)
        )

        bias = torch.randn(N, dtype=torch.float32) if has_bias else None

        # Reference: use the kernel's own act quantization op + float8_linear_cpu (unfused path).
        # This ensures both ref and out use the same fp32->fp8 rounding implementation.
        act_fp8, act_scale = torch.ops.sgl_kernel._quantize_fp8e4m3_vec(act, True, None)
        ref = torch.ops.sgl_kernel.float8_linear_cpu(
            act_fp8, act_scale, packed_weight, packed_scales, bias, dtype
        )

        # Fused path: quantize activation internally then run GEMM
        out = torch.ops.sgl_kernel.fp8_scaled_mm_with_quant(
            act,
            None,  # act_scales=None: dynamic per-token
            True,  # channelwise=True: per-token (PER_ROW)
            packed_weight,
            packed_scales,
            bias,
            dtype,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_per_token_group_quant_fp8_cpu(self):
        dtype = torch.bfloat16
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        eps = 1e-10

        for shape, group_size in [
            ((3, 128), 64),
            ((2, 3, 256), 128),
            ((2, 3, 130), 65),
        ]:
            x = torch.randn(shape, dtype=dtype).contiguous()
            quantized, scale = per_token_group_quant_fp8(x, group_size, eps=eps)

            expected_scale_shape = (*shape[:-1], shape[-1] // group_size)
            self.assertEqual(quantized.shape, x.shape)
            self.assertEqual(scale.shape, expected_scale_shape)
            self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
            self.assertEqual(scale.dtype, torch.float32)
            self.assertEqual(quantized.device.type, "cpu")
            self.assertEqual(scale.device.type, "cpu")

            x_grouped = x.float().reshape(-1, group_size)
            expected_scale = (
                x_grouped.abs().amax(dim=-1).clamp_min(eps) / fp8_max
            ).reshape(expected_scale_shape)
            expected_quantized = (
                (x_grouped / expected_scale.reshape(-1, 1))
                .clamp(-fp8_max, fp8_max)
                .reshape(shape)
                .to(torch.float8_e4m3fn)
            )

            torch.testing.assert_close(scale, expected_scale)
            torch.testing.assert_close(
                quantized.float(), expected_quantized.float(), rtol=0.20, atol=2.0
            )

    def test_scaled_fp8_quant_cpu(self):
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = (torch.randn(5, 19, dtype=dtype) * 13).contiguous()

            quantized, scale = scaled_fp8_quant(x)
            expected_scale = x.float().abs().amax().clamp_min(1e-12) / fp8_max
            expected_quantized = (
                (x.float() / expected_scale)
                .clamp(-fp8_max, fp8_max)
                .to(torch.float8_e4m3fn)
            )
            self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
            self.assertEqual(scale.dtype, torch.float32)
            self.assertEqual(quantized.shape, x.shape)
            torch.testing.assert_close(scale, expected_scale.reshape(1))
            torch.testing.assert_close(quantized.float(), expected_quantized.float())

            quantized_static, returned_scale = scaled_fp8_quant(x, scale)
            self.assertIs(returned_scale, scale)
            torch.testing.assert_close(quantized_static.float(), expected_quantized.float())

            quantized_padded, scale_padded = scaled_fp8_quant(
                x, None, num_token_padding=8
            )
            self.assertEqual(quantized_padded.shape, (8, 19))
            torch.testing.assert_close(scale_padded, scale)
            torch.testing.assert_close(quantized_padded[: x.shape[0]].float(), quantized.float())

            per_token_quantized, per_token_scale = scaled_fp8_quant(
                x, None, use_per_token_if_dynamic=True
            )
            expected_per_token_scale = (
                x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / fp8_max
            )
            expected_per_token_quantized = (
                (x.float() / expected_per_token_scale)
                .clamp(-fp8_max, fp8_max)
                .to(torch.float8_e4m3fn)
            )
            torch.testing.assert_close(per_token_scale, expected_per_token_scale)
            torch.testing.assert_close(per_token_quantized.float(), expected_per_token_quantized.float())

    def test_mxfp8_group_quantize_cpu(self):
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        def ceil_to_ue8m0_byte(x):
            bits = x.abs().float().view(torch.int32)
            exp = (bits >> 23) & 0xFF
            mantissa = bits & 0x7FFFFF
            exp = exp + (mantissa != 0).to(torch.int32)
            return exp.clamp(1, 254).to(torch.uint8)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = (torch.randn(5, 96, dtype=dtype) / 4).contiguous()
            quantized, scale_u8 = mxfp8_group_quantize(x)

            x_grouped = x.float().view(x.shape[0], x.shape[1] // 32, 32)
            expected_scale_u8 = ceil_to_ue8m0_byte(
                x_grouped.abs().amax(dim=-1) / fp8_max
            )
            expected_scale = (expected_scale_u8.to(torch.int32) << 23).view(
                torch.float32
            )
            expected_quantized = (
                (x_grouped / expected_scale.unsqueeze(-1))
                .clamp(-fp8_max, fp8_max)
                .to(torch.float8_e4m3fn)
                .view_as(x)
            )

            self.assertEqual(quantized.shape, x.shape)
            self.assertEqual(scale_u8.shape, (x.shape[0], x.shape[1] // 32))
            self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
            self.assertEqual(scale_u8.dtype, torch.uint8)
            torch.testing.assert_close(scale_u8, expected_scale_u8)
            torch.testing.assert_close(
                quantized.float(), expected_quantized.float(), rtol=0.20, atol=2.0
            )

    def test_static_quant_fp8_cpu(self):
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = (torch.randn(5, 19, dtype=dtype) * 13).contiguous()
            scale = x.float().abs().amax().clamp_min(1e-12).reshape(1) / fp8_max
            expected_quantized = (
                (x.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            )

            quantized, returned_scale = static_quant_fp8(x, scale, repeat_scale=False)
            self.assertEqual(quantized.shape, x.shape)
            self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
            self.assertIs(returned_scale, scale)
            torch.testing.assert_close(
                quantized.float(), expected_quantized.float(), rtol=0.20, atol=2.0
            )

            quantized, repeated_scale = static_quant_fp8(x, scale, repeat_scale=True)
            self.assertEqual(repeated_scale.shape, (x.shape[0], 1))
            torch.testing.assert_close(repeated_scale, scale.expand(x.shape[0], 1))
            torch.testing.assert_close(
                quantized.float(), expected_quantized.float(), rtol=0.20, atol=2.0
            )

    @parametrize(
        M=[1, 32], N=[4096], K=[4096], group_size=[128], has_bias=[False, True]
    )
    def test_int4_gptq_gemm(self, M, N, K, group_size, has_bias):
        torch.manual_seed(127)
        gptq_weight = torch.randint(-128, 128, (K // 8, N)).to(torch.int)
        gptq_zero = torch.randint(0, 10, (K // group_size, N // 8)).to(torch.int)
        gptq_scales = torch.rand(int(K // group_size), N).to(torch.bfloat16) // 10

        bf16_weight = unpack_and_dequant_gptq(gptq_weight, gptq_zero, gptq_scales)
        if has_bias:
            bias = torch.rand(bf16_weight.shape[0]).to(torch.float)
        else:
            bias = None
        x = torch.rand(M, bf16_weight.size(-1)).to(torch.bfloat16)
        ref_res = torch.nn.functional.linear(
            x, bf16_weight, bias=bias.to(torch.bfloat16) if has_bias else None
        )

        packed_weight, packed_zero, packed_scales = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                gptq_weight, gptq_zero, gptq_scales, 1
            )
        )
        target_res = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
            x,
            packed_weight,
            packed_zero,
            packed_scales,
            bias,
        )

        atol = rtol = precision[ref_res.dtype]
        torch.testing.assert_close(ref_res, target_res, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
