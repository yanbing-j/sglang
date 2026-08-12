// CPU GGUF quantized GEMM with AVX512 optimization.
//
// Supports Q8_0 and Q4_0 quantization types (the most common GGUF types).
// Other types fall back to Python-level numpy dequantization.
//
// GGML block layouts:
//   Q8_0: [d: fp16 (2 B)] [qs: 32 x int8 (32 B)] = 34 B / 32 elements
//   Q4_0: [d: fp16 (2 B)] [qs: 16 x uint8 (16 B)] = 18 B / 32 elements
//         where qs[i] = (elem[2i] + 8) | ((elem[2i+1] + 8) << 4)
//
// Dispatch:
//   M == 1 (decode): fused dequant + dot, no weight materialization
//   M  > 1 (prefill): parallel AVX512 dequant to bf16, then AT mm

#include <torch/all.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "common.h"
#include "vec.h"

#if defined(CPU_CAPABILITY_AVX512)

static constexpr int GGUF_Q8_0_BLOCK = 32;
static constexpr int GGUF_Q8_0_BYTES = 34;  // 2 (fp16 d) + 32 (int8 qs)

static constexpr int GGUF_Q4_0_BLOCK = 32;
static constexpr int GGUF_Q4_0_BYTES = 18;  // 2 (fp16 d) + 16 (packed nibbles)

static constexpr int GGUF_QTYPE_Q4_0 = 2;
static constexpr int GGUF_QTYPE_Q8_0 = 8;

// --------------------------------------------------------------------------
// Scalar fp16 → fp32 helper
// --------------------------------------------------------------------------
inline float load_fp16_to_fp32(const uint8_t* ptr) {
  uint16_t h;
  memcpy(&h, ptr, 2);
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

// --------------------------------------------------------------------------
// Decode Q8_0 block (32 elements) → fp32[32]
// --------------------------------------------------------------------------
inline void q8_0_block_to_fp32(const uint8_t* block, float* out) {
  const float scale = load_fp16_to_fp32(block);
  const __m512 vs = _mm512_set1_ps(scale);
  // Load two 128-bit halves of 32 int8 values
  __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2));
  __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 18));
  // Sign-extend int8 → int32, convert to fp32, multiply by scale
  _mm512_storeu_ps(out, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), vs));
  _mm512_storeu_ps(out + 16, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), vs));
}

// --------------------------------------------------------------------------
// Decode Q4_0 block (32 elements) → fp32[32]
//
// Q4_0 nibble layout:
//   qs[i] = (elem[2i] + 8) | ((elem[2i+1] + 8) << 4)
// After unpack: lo nibbles = even elements, hi nibbles = odd elements.
// Need to interleave back to sequential order [0,1,2,...,31].
// --------------------------------------------------------------------------
inline void q4_0_block_to_fp32(const uint8_t* block, float* out) {
  const float scale = load_fp16_to_fp32(block);
  const __m512 vs = _mm512_set1_ps(scale);
  const __m512i voff = _mm512_set1_epi32(8);

  __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2));
  // Zero-extend 16 bytes → 16 uint16
  __m256i p16 = _mm256_cvtepu8_epi16(packed);
  // ggml Q4_0 layout: qs[j] = lo_nibble(elem[j]) | hi_nibble(elem[j+16])
  // lo[0..15] = elements 0..15 (first half of block output)
  // hi[0..15] = elements 16..31 (second half of block output)
  // No interleaving needed.
  __m256i lo = _mm256_and_si256(p16, _mm256_set1_epi16(0x0F));
  __m256i hi = _mm256_srli_epi16(p16, 4);

  __m512i i32_lo = _mm512_sub_epi32(_mm512_cvtepi16_epi32(lo), voff);
  __m512i i32_hi = _mm512_sub_epi32(_mm512_cvtepi16_epi32(hi), voff);
  _mm512_storeu_ps(out, _mm512_mul_ps(_mm512_cvtepi32_ps(i32_lo), vs));
  _mm512_storeu_ps(out + 16, _mm512_mul_ps(_mm512_cvtepi32_ps(i32_hi), vs));
}

// --------------------------------------------------------------------------
// Decode one weight row to bf16 (for the BRGEMM/prefill path).
// IS_Q8: true → Q8_0, false → Q4_0
// --------------------------------------------------------------------------
template <bool IS_Q8>
static void decode_row_to_bf16(const uint8_t* qrow, at::BFloat16* out, int64_t K) {
  constexpr int BLOCK = IS_Q8 ? GGUF_Q8_0_BLOCK : GGUF_Q4_0_BLOCK;
  constexpr int BYTES = IS_Q8 ? GGUF_Q8_0_BYTES : GGUF_Q4_0_BYTES;
  const int64_t nb = K / BLOCK;

  alignas(64) float tmp[32];
  for (int64_t b = 0; b < nb; ++b) {
    if constexpr (IS_Q8)
      q8_0_block_to_fp32(qrow + b * BYTES, tmp);
    else
      q4_0_block_to_fp32(qrow + b * BYTES, tmp);

    __m512 fp0 = _mm512_loadu_ps(tmp);
    __m512 fp1 = _mm512_loadu_ps(tmp + 16);
    // _mm512_cvtne2ps_pbh(a, b): dst[0..15 bf16] = cvt(b), dst[16..31 bf16] = cvt(a)
    // So (fp1, fp0) → [elem0..15 as bf16 | elem16..31 as bf16] ✓
    _mm512_storeu_si512(out + b * BLOCK, (__m512i)_mm512_cvtne2ps_pbh(fp1, fp0));
  }
}

// --------------------------------------------------------------------------
// Fused dequant + dot product against fp32 input (decode path, M=1).
// IS_Q8: true → Q8_0, false → Q4_0
// --------------------------------------------------------------------------
template <bool IS_Q8>
static float gguf_dot_fp32(const uint8_t* qrow, const float* x, int64_t K) {
  constexpr int BLOCK = IS_Q8 ? GGUF_Q8_0_BLOCK : GGUF_Q4_0_BLOCK;
  constexpr int BYTES = IS_Q8 ? GGUF_Q8_0_BYTES : GGUF_Q4_0_BYTES;
  const int64_t nb = K / BLOCK;

  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();

  if constexpr (IS_Q8) {
    for (int64_t b = 0; b < nb; ++b) {
      const __m512 vs = _mm512_set1_ps(load_fp16_to_fp32(qrow + b * BYTES));
      __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qrow + b * BYTES + 2));
      __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qrow + b * BYTES + 18));
      __m512 q0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), vs);
      __m512 q1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), vs);
      acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + b * BLOCK), q0, acc0);
      acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + b * BLOCK + 16), q1, acc1);
    }
  } else {
    const __m512i voff = _mm512_set1_epi32(8);
    for (int64_t b = 0; b < nb; ++b) {
      const __m512 vs = _mm512_set1_ps(load_fp16_to_fp32(qrow + b * BYTES));
      __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qrow + b * BYTES + 2));
      __m256i p16 = _mm256_cvtepu8_epi16(packed);
      // lo → elements 0..15, hi → elements 16..31 (no interleaving needed)
      __m256i lo = _mm256_and_si256(p16, _mm256_set1_epi16(0x0F));
      __m256i hi = _mm256_srli_epi16(p16, 4);
      __m512 q0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepi16_epi32(lo), voff)), vs);
      __m512 q1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepi16_epi32(hi), voff)), vs);
      acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + b * BLOCK), q0, acc0);
      acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + b * BLOCK + 16), q1, acc1);
    }
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

// --------------------------------------------------------------------------
// GEMM dispatch for a given quantization type.
// Decode path (M==1): parallel vec-dot, no weight materialization.
// BRGEMM path (M > 1): parallel dequant to bf16 → at::mm (oneDNN/MKL).
// --------------------------------------------------------------------------
template <bool IS_Q8>
static at::Tensor gguf_mul_mat_impl(
    const at::Tensor& x,        // [M, K] bf16/fp16/fp32, contiguous
    const at::Tensor& qweight,  // [N, packed_cols] uint8, contiguous
    int64_t N,
    int64_t K) {
  constexpr int BYTES = IS_Q8 ? GGUF_Q8_0_BYTES : GGUF_Q4_0_BYTES;
  constexpr int BLOCK = IS_Q8 ? GGUF_Q8_0_BLOCK : GGUF_Q4_0_BLOCK;

  const int64_t M = x.numel() / K;
  const int64_t packed_cols = qweight.size(1);
  const uint8_t* qw = reinterpret_cast<const uint8_t*>(qweight.data_ptr());
  const at::ScalarType out_dtype = x.scalar_type();

  // ---- Decode path (M == 1): fused dequant + dot, no weight buffer ----
  if (M == 1) {
    // Convert input to fp32 for the dot kernel
    auto x_fp32 = x.reshape({K}).to(at::kFloat).contiguous();
    const float* x_ptr = x_fp32.data_ptr<float>();

    auto out = at::empty({1, N}, at::TensorOptions().dtype(at::kFloat));
    float* out_ptr = out.data_ptr<float>();

    at::parallel_for(0, N, 0, [&](int64_t n0, int64_t n1) {
      for (int64_t n = n0; n < n1; ++n) {
        out_ptr[n] = gguf_dot_fp32<IS_Q8>(qw + n * packed_cols, x_ptr, K);
      }
    });

    return out.reshape({1, N}).to(out_dtype);
  }

  // ---- BRGEMM path (M > 1): parallel dequant to bf16 + at::mm ----
  auto B_bf16 = at::empty({N, K}, at::TensorOptions().dtype(at::kBFloat16));
  at::BFloat16* B_ptr = B_bf16.data_ptr<at::BFloat16>();

  at::parallel_for(0, N, 0, [&](int64_t n0, int64_t n1) {
    for (int64_t n = n0; n < n1; ++n) {
      decode_row_to_bf16<IS_Q8>(qw + n * packed_cols, B_ptr + n * K, K);
    }
  });

  // at::linear(x_bf16, B_bf16) = x_bf16 @ B_bf16.T
  auto x_bf16 = x.to(at::kBFloat16).reshape({M, K});
  return at::linear(x_bf16, B_bf16).to(out_dtype).reshape(
      at::IntArrayRef({M, N}));
}

#endif  // CPU_CAPABILITY_AVX512

// --------------------------------------------------------------------------
// Public entry point registered as sgl_kernel::gguf_mul_mat_cpu
// --------------------------------------------------------------------------
at::Tensor gguf_mul_mat_cpu(
    const at::Tensor& x,        // [*, K] any float dtype
    const at::Tensor& qweight,  // [N, packed_cols] uint8
    int64_t qtype,
    int64_t N,
    int64_t K) {
#if defined(CPU_CAPABILITY_AVX512)
  TORCH_CHECK(x.device().is_cpu(), "gguf_mul_mat_cpu: x must be CPU tensor");
  TORCH_CHECK(qweight.device().is_cpu(), "gguf_mul_mat_cpu: qweight must be CPU tensor");
  TORCH_CHECK(x.is_contiguous(), "gguf_mul_mat_cpu: x must be contiguous");
  TORCH_CHECK(qweight.is_contiguous(), "gguf_mul_mat_cpu: qweight must be contiguous");
  TORCH_CHECK(K % 32 == 0, "gguf_mul_mat_cpu: K must be divisible by 32");

  if (qtype == GGUF_QTYPE_Q8_0) return gguf_mul_mat_impl<true>(x, qweight, N, K);
  if (qtype == GGUF_QTYPE_Q4_0) return gguf_mul_mat_impl<false>(x, qweight, N, K);
#endif
  TORCH_CHECK(
      false,
      "gguf_mul_mat_cpu: unsupported qtype=",
      qtype,
      " (only Q8_0=8 and Q4_0=2 are supported with AVX512)");
}
