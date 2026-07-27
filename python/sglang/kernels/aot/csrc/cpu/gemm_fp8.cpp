#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [data0, data1] = load_float_vec2(input + d);
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

template <typename scalar_t>
inline void copy_add_stub(
    scalar_t* __restrict__ out, const float* __restrict__ input, const float* __restrict__ bias, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [data0, data1] = load_float_vec2(input + d);
    auto [bias0, bias1] = load_float_vec2(bias + d);
    bVec out_vec = convert_from_float_ext<scalar_t>(data0 + bias0, data1 + bias1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + bias[d]);
  }
}
template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int size, float scale) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec vscale = fVec(scale);

  int d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [data0, data1] = load_float_vec2(input + d);
    bVec out_vec = convert_from_float_ext<scalar_t>(data0 * vscale, data1 * vscale);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * scale);
  }
}

template <>
inline void
copy_add_stub(float* __restrict__ out, const float* __restrict__ input, const float* __restrict__ bias, int64_t size) {
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = fVec::size();

  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data = fVec::loadu(input + d) + fVec::loadu(bias + d);
    data.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = input[d] + bias[d];
  }
}

inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const at::Float8_e4m3fn* __restrict__ packed_B,
    int64_t N,
    int64_t K,
    int64_t ldb,
    int64_t ldb_tmp,
    float scale) {
#if defined(CPU_CAPABILITY_AVX512)
  // [K/2, N, 2]
  const int64_t K2 = K >> 1;
  const int64_t ldb2 = ldb;  // ldb * 2 >> 1;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(packed_B);
  const __m512 vexp = _mm512_castsi512_ps(_mm512_set1_epi32(kFP8_BIAS));
  const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(scale), vexp);

  constexpr int BLOCK_N = block_size_n();
  static_assert(BLOCK_N == 32);

  // prefetch distance
  constexpr int PREFETCH_SIZE_K = 64;

#pragma GCC unroll 4
  for (int64_t k = 0; k < K2; ++k) {
    __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2);
    if constexpr (PREFETCH_SIZE_K > 0) {
      _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2, _MM_HINT_T0);
    }

    __m256i b8_0 = _mm512_extracti32x8_epi32(b8, 0);
    __m256i b8_1 = _mm512_extracti32x8_epi32(b8, 1);

    __m512bh bf16_0 = CVT_FP8_TO_BF16_EXT(b8_0);
    __m512bh bf16_1 = CVT_FP8_TO_BF16_EXT(b8_1);

    // Apply scale
    __m512 f0_lo = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_0, 0));
    __m512 f0_hi = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_0, 1));
    __m512 f1_lo = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_1, 0));
    __m512 f1_hi = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_1, 1));

    f0_lo = _mm512_mul_ps(f0_lo, vd);
    f0_hi = _mm512_mul_ps(f0_hi, vd);
    f1_lo = _mm512_mul_ps(f1_lo, vd);
    f1_hi = _mm512_mul_ps(f1_hi, vd);

    bf16_0 = _mm512_cvtne2ps_pbh(f0_hi, f0_lo);
    bf16_1 = _mm512_cvtne2ps_pbh(f1_hi, f1_lo);

    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 0, (__m512i)bf16_0);
    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 32, (__m512i)bf16_1);
  }
#else
  TORCH_CHECK(false, "unpack_B: scalar path not implemented!");
#endif
}

inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const at::Float8_e4m3fn* __restrict__ packed_B,
    int N,
    int K,
    int ldb,
    int ldb_tmp) {
#if defined(CPU_CAPABILITY_AVX512)
  // [K/2, N, 2]
  const int K2 = K >> 1;
  const int ldb2 = ldb;  // ldb * 2 >> 1;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(packed_B);

  // prefetch distance
  constexpr int PREFETCH_SIZE_K = 64;
#pragma GCC unroll 4
  for (int k = 0; k < K2; ++k) {
    __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2);
    if constexpr (PREFETCH_SIZE_K > 0) {
      _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2, _MM_HINT_T0);
    }

    __m256i b8_0 = _mm512_extracti32x8_epi32(b8, 0);
    __m256i b8_1 = _mm512_extracti32x8_epi32(b8, 1);

    __m512bh bf16_0 = CVT_FP8_TO_BF16(b8_0);
    __m512bh bf16_1 = CVT_FP8_TO_BF16(b8_1);
    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 0, (__m512i)bf16_0);
    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 32, (__m512i)bf16_1);
  }
#else
  TORCH_CHECK(false, "unpack_B: scalar path not implemented!");
#endif
}

// mxfp4
inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const uint8_t* __restrict__ packed_B,
    int64_t N,
    int64_t K,
    int64_t ldb,
    int64_t ldb_tmp,
    const uint8_t* __restrict__ scale) {
#if defined(CPU_CAPABILITY_AVX512)
  // [K/2, N, 2]
  const int64_t K2 = K >> 1;
  const int64_t ldb2 = ldb;                                           // ldb * 2 >> 1;
  const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(packed_B);  // 2 * 4 bit = 8 bit

  constexpr int BLOCK_N = block_size_n();
  static_assert(BLOCK_N == 32);

  // prefetch distance
  constexpr int PREFETCH_SIZE_K = 64;

  // exponent bias 127
  const __m512i off = _mm512_set1_epi16(0x7F);

  // load 32 bytes only once for each block
  __m256i s8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scale));
  __m512i s16 = _mm512_slli_epi16(_mm512_sub_epi16(_mm512_cvtepu8_epi16(s8), off), 0x7);

  // holds Nx2(64) scales, interleaved as 2 belongs to K dimension
  // e.g. vs0: { s0,  s0,  s1,  s1, ..., s15, s15}
  //      vs1: {s16, s16, s17, s17, ..., s31, s31}
  auto [vscale0, vscale1] = transpose_2x32_16bit(s16, s16);

#pragma GCC unroll 4
  for (int64_t k = 0; k < K2; ++k) {
    __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + k * ldb2));
    if constexpr (PREFETCH_SIZE_K > 0) {
      _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2, _MM_HINT_T0);
    }
    auto [vb0, vb1] = CVT_MXFP4_TO_BF16(b4, vscale0, vscale1);

    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 0, (__m512i)vb0);
    _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + 32, (__m512i)vb1);
  }
#else
  TORCH_CHECK(false, "unpack_B: scalar path not implemented!");
#endif
}

template <typename scalar_t, typename packed_t, typename param_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const float* __restrict__ bias,
      const param_t* __restrict__ scale,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t block_size_K) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2 {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      scalar_t* __restrict__ C,
      float scale,
      int K,
      int lda,
      int ldb,
      int ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};
#if defined(CPU_CAPABILITY_AVX512)
template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, at::Float8_e4m3fn, float, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      const float* __restrict__ bias,
      const float* __restrict__ scale,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t block_size_K) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    const int64_t KB = div_up(K, (int64_t)BLOCK_K);

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 64;
    constexpr int PREFETCH_SIZE_KB = 1;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];
    __m512 vsum[ROWS * COLS];

    // block quant scale
    __m512 vscale;

    const __m512 vexp = _mm512_castsi512_ps(_mm512_set1_epi32(kFP8_BIAS));

    auto loadc = [&](auto i) {
      constexpr int col = i % COLS;
      if constexpr (has_bias) {
        vc[i] = _mm512_loadu_ps(bias + col * 16);
      } else {
        vc[i] = _mm512_setzero_ps();
      }
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;  // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(a_ptr + row * lda2 + k + PREFETCH_SIZE_K, _MM_HINT_T0);
        }
      }
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2 + col * 16);
          if constexpr (PREFETCH_SIZE_K > 0) {
            _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          }
          vb[col + 0] = CVT_FP8_TO_BF16_EXT(_mm512_extracti32x8_epi32(b8, 0));
          vb[col + 1] = CVT_FP8_TO_BF16_EXT(_mm512_extracti32x8_epi32(b8, 1));
        }
      }
      vsum[i] = _mm512_dpbf16_ps(vsum[i], va, vb[col]);
    };

    constexpr int64_t BLOCK_K2 = BLOCK_K >> 1;
    for (int64_t kb = 0; kb < KB; ++kb) {
      int64_t kb_start = kb * BLOCK_K2;
      int64_t kb_end = std::min(K >> 1, kb_start + BLOCK_K2);
      // 1. load scale vector
      vscale = _mm512_set1_ps(scale[kb]);
      vscale = _mm512_mul_ps(vscale, vexp);
      if constexpr (PREFETCH_SIZE_KB > 0) {
        _mm_prefetch(scale + kb + PREFETCH_SIZE_KB, _MM_HINT_T0);
      }
      // 2. zero vsum for each block
      Unroll<ROWS * COLS>{}([&](auto i) { vsum[i] = _mm512_setzero_ps(); });
      // 3. accumulate across each block
      for (int k = kb_start; k < kb_end; ++k) {
        Unroll<ROWS * COLS>{}(compute, k);
      }
      // 4. apply scale
      Unroll<ROWS * COLS>{}([&](auto i) { vc[i] = _mm512_fmadd_ps(vsum[i], vscale, vc[i]); });
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2,4 use 512bit store
      if constexpr (col % 2 == 0) {
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(vc[row * COLS + col + 1], vc[row * COLS + col])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};

template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn2<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      float scale,
      int K,
      int lda,
      int ldb,
      int ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 64;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    const __m512 vscale = _mm512_set1_ps(scale);

    auto loadc = [&](auto i) { vc[i] = _mm512_setzero_ps(); };
    Unroll<ROWS * COLS>{}(loadc);

    const int K2 = K >> 1;
    const int lda2 = lda >> 1;
    const int ldb2 = ldb;  // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2 + col * 16);
          if constexpr (PREFETCH_SIZE_K > 0) {
            _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          }
          vb[col + 0] = CVT_FP8_TO_BF16(_mm512_extracti32x8_epi32(b8, 0));
          vb[col + 1] = CVT_FP8_TO_BF16(_mm512_extracti32x8_epi32(b8, 1));
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    for (int k = 0; k < K2; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        __m512 vc0 = _mm512_mul_ps(vc[row * COLS + col + 0], vscale);
        __m512 vc1 = _mm512_mul_ps(vc[row * COLS + col + 1], vscale);
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)), (__m512i)(_mm512_cvtne2ps_pbh(vc1, vc0)));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};

template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, uint8_t, uint8_t, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const uint8_t* __restrict__ B,
      at::BFloat16* __restrict__ C,
      const float* __restrict__ bias,
      const uint8_t* __restrict__ scale,
      int K,
      int lda,
      int ldb,
      int ldc,
      int64_t block_size_K) {
    // mxfp4 supports only group size of 32
    // expect weight packed in 32-way, vnni2 format Nx2(64)
    assert(block_size_K == 32);
    assert(BLOCK_N == 32);

    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 64;
    constexpr int PREFETCH_SIZE_KB = 1;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];

    // holds Nx2(64) scales, interleaved as 2 belongs to K dimension
    // e.g. vs0: { s0,  s0,  s1,  s1, ..., s15, s15}
    //      vs1: {s16, s16, s17, s17, ..., s31, s31}
    __m512i vscale[COLS];

    // exponent bias 127
    const __m512i off = _mm512_set1_epi16(0x7F);

    auto loadc = [&](auto i) {
      constexpr int col = i % COLS;
      if constexpr (has_bias) {
        vc[i] = _mm512_loadu_ps(bias + col * 16);
      } else {
        vc[i] = _mm512_setzero_ps();
      }
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb;  // ldb * 2 >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(B);

    auto compute = [&](auto i, int k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(a_ptr + row * lda2 + k + PREFETCH_SIZE_K, _MM_HINT_T0);
        }
      }
      if constexpr (row == 0) {
        // load 32 * 2 (64) int4 at a time
        if constexpr (col % 2 == 0) {
          __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + k * ldb2 + col * 16));
          if constexpr (PREFETCH_SIZE_K > 0) {
            _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
          }
          std::tie(vb[col + 0], vb[col + 1]) = CVT_MXFP4_TO_BF16(b4, vscale[col + 0], vscale[col + 1]);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };

    for (int64_t k = 0; k < K2; ++k) {
      // update scales every 16x2 K
      if ((k & 15) == 0) {
        __m256i s8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scale + (k >> 4) * 32));
        __m512i s16 = _mm512_slli_epi16(_mm512_sub_epi16(_mm512_cvtepu8_epi16(s8), off), 0x7);
        std::tie(vscale[0], vscale[1]) = transpose_2x32_16bit(s16, s16);
      }
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2,4 use 512bit store
      if constexpr (col % 2 == 0) {
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(vc[row * COLS + col + 1], vc[row * COLS + col])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                                   \
  tinygemm_kernel_nn<scalar_t, packed_t, param_t, has_bias, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                                             \
      B + nb_start * 2,                                                               \
      C + mb_start * ldc + nb_start,                                                  \
      has_bias ? bias + nb_start : nullptr,                                           \
      scale,                                                                          \
      K,                                                                              \
      lda,                                                                            \
      ldb,                                                                            \
      ldc,                                                                            \
      block_size_K);

#define LAUNCH_TINYGEMM_KERNEL_NN2(MB_SIZE, NB_SIZE)      \
  tinygemm_kernel_nn2<scalar_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, B + nb_start * 2, C + mb_start * ldc + nb_start, scale, K, lda, ldb, ldc);

template <typename scalar_t, typename packed_t, typename param_t, bool has_bias>
struct brgemm {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const param_t* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc,
      bool do_unpack = true) {
    TORCH_CHECK(false, "struct brgemm: primary template not implemented!");
  }
};
template <typename scalar_t>
struct brgemm2 {};

template <bool has_bias>
struct brgemm<at::BFloat16, at::Float8_e4m3fn, float, has_bias> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const float* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc,
      bool do_unpack = true) {
    constexpr int BLOCK_N = block_size_n();

    // [K, BLOCK_N] -> [K / 2, BLOCK_N * 2]
    const int ldb_tmp = BLOCK_N;

    if (do_unpack) {
      for (int k = 0; k < K; k += BLOCK_K) {
        int kb_size = std::min(BLOCK_K, K - k);

        int idx = k >> 7;  // k / BLOCK_K where BLOCK_K = 128
        unpack_B(Btmp + k * ldb_tmp, B + k * ldb, N, kb_size, ldb, ldb_tmp, scale[idx]);
      }
    }

    at::native::cpublas::brgemm(M, N, K, lda, ldb_tmp, BLOCK_N, /* add_C */ false, A, Btmp, Ctmp);

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }
    }
  }
};

template <>
struct brgemm2<at::BFloat16> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      float scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    constexpr int BLOCK_N = block_size_n();

    // [BLOCK_K, BLOCK_N] -> [BLOCK_K / 2, BLOCK_N * 2]
    const int ldb_tmp = block_size_n();

    // accumulate across K per BLOCK_K
    for (int k = 0; k < K; k += BLOCK_K) {
      int kb_size = std::min(BLOCK_K, K - k);
      unpack_B(Btmp, B + k * ldb, N, kb_size, ldb, ldb_tmp);

      const bool add_C = (k != 0);
      at::native::cpublas::brgemm(M, N, kb_size, lda, ldb_tmp, BLOCK_N, add_C, A + k, Btmp, Ctmp);
    }

    // copy from Ctmp to C and mul scale
    for (int m = 0; m < M; ++m) {
      copy_mul_stub(C + m * ldc, Ctmp + m * BLOCK_N, N, scale);
    }
  }
};

template <bool has_bias>
struct brgemm<at::BFloat16, uint8_t, uint8_t, has_bias> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const uint8_t* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const uint8_t* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc,
      bool do_unpack = true) {
    constexpr int BLOCK_N = block_size_n();

    // [K, BLOCK_N] -> [K / 2, BLOCK_N * 2]
    const int ldb_tmp = BLOCK_N;

    if (do_unpack) {
      // group size 32 for mxfp4
      for (int k = 0; k < K; k += 32) {
        unpack_B(Btmp + k * ldb_tmp, B + k * (ldb >> 1), N, 32, ldb, ldb_tmp, scale + (k >> 5) * BLOCK_N);
      }
    }

    at::native::cpublas::brgemm(M, N, K, lda, ldb_tmp, BLOCK_N, /* add_C */ false, A, Btmp, Ctmp);

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }
    }
  }
};

template <typename scalar_t, typename packed_t, typename param_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const packed_t* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const param_t* __restrict__ scale,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K,
    bool do_unpack = true) {
  if (brg) {
    brgemm<scalar_t, packed_t, param_t, has_bias>::apply(
        A, B, C, Btmp, Ctmp, bias, scale, M, N, K, lda, ldb, ldc, do_unpack);
    return;
  }

  // pattern: 1-4-16
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, 32);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, 32);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, 32);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, 32);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t>
void tinygemm_kernel2(
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    float scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  if (brg) {
    brgemm2<scalar_t>::apply(A, B, C, Btmp, Ctmp, scale, M, N, K, lda, ldb, ldc);
    return;
  }

  // pattern: 1-8-8
  if (M == 1) {
    constexpr int64_t BLOCK_N = 128;
    const int64_t NB = div_up(N, BLOCK_N);
    int64_t mb_start = 0;

    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (nb_size >> 4) {
        case 2:
          LAUNCH_TINYGEMM_KERNEL_NN2(1, 32);
          break;
        case 4:
          LAUNCH_TINYGEMM_KERNEL_NN2(1, 64);
          break;
        case 6:
          LAUNCH_TINYGEMM_KERNEL_NN2(1, 96);
          break;
        case 8:
          LAUNCH_TINYGEMM_KERNEL_NN2(1, 128);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, 1x", "nb_size");
      }
    }
    return;
  }

  // pattern: 1-4-16
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN2(1, 32);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NN2(1, 64);
          break;
        // mb_size = 2
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN2(2, 32);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NN2(2, 64);
          break;
        // mb_size = 3
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN2(3, 32);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NN2(3, 64);
          break;
        // mb_size = 4
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN2(4, 32);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NN2(4, 64);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

// NB: fp8/fp4 scaled mm kernel implementation
//
//        scalar_t     packed_t     param_t
//   FP8    BF16         FP8         FP32
//  MXFP4   BF16          U8           U8
//
template <typename scalar_t, typename packed_t, typename param_t, typename func_t>
void fp_scaled_mm_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const packed_t* __restrict__ mat2,
    const param_t* __restrict__ scales2,
    const float* __restrict__ bias,
    scalar_t* __restrict__ buffer,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    int64_t block_size_N,
    int64_t block_size_K,
    int64_t buffer_size_per_thread,
    const func_t& scale_offset_per_block) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  const bool use_brgemm = can_use_brgemm<packed_t>(M);

  // use K/2 for mxfp4 and K for fp8
  const int64_t packed_K = get_row_size<packed_t>(K);

  // parallel on [MB, NB]
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
      int tid = get_thread_num();
      scalar_t* __restrict__ Btmp = buffer + tid * buffer_size_per_thread;
      float* __restrict__ Ctmp = (float*)((void*)(Btmp + MAX_CACHE_BLOCK_SIZE * BLOCK_N * K));

      loop_2d<packed_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
        const param_t* scale_ptr = scales2 + scale_offset_per_block(nb);

        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);

        // only do unpacking for the first row
        bool do_unpack = (mb == mb0);

        tinygemm_kernel<scalar_t, packed_t, param_t, has_bias>(
            /*   A            */ mat1 + mb_start * mat1_strideM,
            /*   B            */ mat2 + nb_start * packed_K,  // nb * BLOCK_N * K
            /*   C            */ out + mb_start * out_strideM + nb_start,
            /*   Btmp         */ Btmp + nb_offset * BLOCK_N * K,
            /*   Ctmp         */ Ctmp,
            /*   scale        */ scale_ptr,
            /*   bias         */ bias + nb_start,
            /*   M            */ mb_size,
            /*   N            */ nb_size,
            /*   K            */ K,
            /*   lda          */ mat1_strideM,
            /*   ldb          */ nb_size,
            /*   ldc          */ out_strideM,
            /*   brg          */ use_brgemm,
            /*   block_size_K */ block_size_K,
            /*   do_unpack    */ do_unpack);
      });

      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });
}

}  // anonymous namespace

inline __m128i cvtfp32_fp8e4m3(__m512& src) {
  // cvt 16x32 from fp32 to fp8 e4m3
  const __m512i sign_mask = _mm512_set1_epi32(0x80000000);
  const __m512i fp8_max = _mm512_set1_epi32(UINT32_C(1087) << 20);
  const __m512i denorm_thresh = _mm512_set1_epi32(UINT32_C(121) << 23);
  const __m512i denorm_mask = _mm512_set1_epi32(UINT32_C(141) << 23);
  const __m512i bias_part1 = _mm512_set1_epi32((uint32_t)(7 - 127) << 23);
  const __m512i rounding_bias = _mm512_set1_epi32(0x7FFFF);
  __m512i f_bits = _mm512_castps_si512(src);
  // Extract and save sign
  __m512i sign = _mm512_and_epi32(f_bits, sign_mask);
  f_bits = _mm512_xor_epi32(f_bits, sign);

  // Prepare result containers
  __m512i result = _mm512_setzero_si512();

  // Step 1: Handle case of overflow
  // (f_bits >= fp8_max): set result = 0x7f
  __mmask16 overflow_mask = _mm512_cmpge_epu32_mask(f_bits, fp8_max);
  if (overflow_mask) {
    result = _mm512_mask_set1_epi32(result, overflow_mask, 0x7f);
  }

  // Step 2: Handle small numbers (denormals)
  // Small numbers (f_bits < denorm_thresh)
  __mmask16 denorm_thresh_mask = _mm512_cmplt_epu32_mask(f_bits, denorm_thresh);

  if (denorm_thresh_mask) {
    __m512 small_input = _mm512_castsi512_ps(f_bits);
    __m512 small_denorm = _mm512_add_ps(small_input, _mm512_castsi512_ps(denorm_mask));
    __m512i small_denorm_bits = _mm512_castps_si512(small_denorm);
    __m512i small_result = _mm512_sub_epi32(small_denorm_bits, denorm_mask);
    result = _mm512_mask_mov_epi32(result, denorm_thresh_mask, small_result);
  }

  // Step 3: Handle normal numbers
  __mmask16 normal_mask = ~(overflow_mask | denorm_thresh_mask);

  if (normal_mask) {
    // mant_odd = (f_bits >> 20) & 1
    __m512i mant_odd = _mm512_and_epi32(_mm512_srli_epi32(f_bits, 20), _mm512_set1_epi32(1));
    // f_bits += bias_part1 + rounding_bias
    __m512i rounded = _mm512_add_epi32(f_bits, bias_part1);
    rounded = _mm512_add_epi32(rounded, rounding_bias);
    // Add mant_odd
    rounded = _mm512_add_epi32(rounded, mant_odd);
    // Shift right by 20 bits
    __m512i normal_result = _mm512_srli_epi32(rounded, 20);
    result = _mm512_mask_mov_epi32(result, normal_mask, normal_result);
  }

  // Merge back the sign
  __m512i sign_shifted = _mm512_srli_epi32(sign, 24);
  result = _mm512_or_epi32(result, sign_shifted);

  // Now result is 16 x 32-bit integers, but we only need 8-bit for each
  __m512i packed = _mm512_and_si512(result, _mm512_set1_epi32(0xFF));

  // Narrow 32-bit integers to 8-bit
  return _mm512_cvtepi32_epi8(packed);
}

inline __m512 load_16_as_fp32(const float* __restrict__ input) {
  return _mm512_loadu_ps(input);
}

inline __m512 load_16_as_fp32(const at::Half* __restrict__ input) {
  return CVT_FP16_TO_FP32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input)));
}

inline __m512 load_16_as_fp32(const at::BFloat16* __restrict__ input) {
  __m256i input_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(input_bf16), 16));
}

template <typename scalar_t>
void per_token_group_quant_fp8_cpu_kernel(
    const scalar_t* __restrict__ input_data,
    at::Float8_e4m3fn* __restrict__ quantized_data,
    float* __restrict__ scale_data,
    int64_t num_groups,
    int64_t group_size,
    float eps) {
  constexpr float quant_max = 448.0f;
  constexpr float neg_quant_max = -quant_max;
  const __m512 sign_bit = _mm512_set1_ps(-0.0f);
  const __m512 quant_max_vec = _mm512_set1_ps(quant_max);
  const __m512 neg_quant_max_vec = _mm512_set1_ps(neg_quant_max);

  // Common case: group_size is a multiple of 16 (e.g. 64, 128, 256).
  // Unroll 2x (32 elements per iteration) to improve ILP and reduce loop overhead.
  const bool aligned = (group_size % 32 == 0);

  at::parallel_for(0, num_groups, 0, [&](int64_t begin, int64_t end) {
    for (int64_t group_index = begin; group_index < end; ++group_index) {
      const int64_t group_offset = group_index * group_size;
      __m512 absmax_vec0 = _mm512_setzero_ps();
      __m512 absmax_vec1 = _mm512_setzero_ps();

      int64_t element_index = 0;

      // Pass 1: find absmax — unrolled 2x
      if (aligned) {
        for (; element_index < group_size; element_index += 32) {
          __m512 v0 = load_16_as_fp32(input_data + group_offset + element_index);
          __m512 v1 = load_16_as_fp32(input_data + group_offset + element_index + 16);
          absmax_vec0 = _mm512_max_ps(absmax_vec0, _mm512_andnot_ps(sign_bit, v0));
          absmax_vec1 = _mm512_max_ps(absmax_vec1, _mm512_andnot_ps(sign_bit, v1));
        }
      } else {
        for (; element_index <= group_size - 32; element_index += 32) {
          __m512 v0 = load_16_as_fp32(input_data + group_offset + element_index);
          __m512 v1 = load_16_as_fp32(input_data + group_offset + element_index + 16);
          absmax_vec0 = _mm512_max_ps(absmax_vec0, _mm512_andnot_ps(sign_bit, v0));
          absmax_vec1 = _mm512_max_ps(absmax_vec1, _mm512_andnot_ps(sign_bit, v1));
        }
        for (; element_index <= group_size - 16; element_index += 16) {
          __m512 v0 = load_16_as_fp32(input_data + group_offset + element_index);
          absmax_vec0 = _mm512_max_ps(absmax_vec0, _mm512_andnot_ps(sign_bit, v0));
        }
      }
      absmax_vec0 = _mm512_max_ps(absmax_vec0, absmax_vec1);
      float absmax = _mm512_reduce_max_ps(absmax_vec0);
      for (; element_index < group_size; ++element_index) {
        absmax = std::max(absmax, std::abs(static_cast<float>(input_data[group_offset + element_index])));
      }

      const float scale = std::max(absmax, eps) / quant_max;
      const float inv_scale = 1.0f / scale;
      scale_data[group_index] = scale;

      const __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
      element_index = 0;

      // Pass 2: scale + clamp + convert to fp8 — unrolled 2x
      if (aligned) {
        for (; element_index < group_size; element_index += 32) {
          __m512 v0 = load_16_as_fp32(input_data + group_offset + element_index);
          __m512 v1 = load_16_as_fp32(input_data + group_offset + element_index + 16);
          v0 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v0, inv_scale_vec), neg_quant_max_vec), quant_max_vec);
          v1 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v1, inv_scale_vec), neg_quant_max_vec), quant_max_vec);
          __m128i fp8_0 = cvtfp32_fp8e4m3(v0);
          __m128i fp8_1 = cvtfp32_fp8e4m3(v1);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(quantized_data + group_offset + element_index), fp8_0);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(quantized_data + group_offset + element_index + 16), fp8_1);
        }
      } else {
        for (; element_index <= group_size - 32; element_index += 32) {
          __m512 v0 = load_16_as_fp32(input_data + group_offset + element_index);
          __m512 v1 = load_16_as_fp32(input_data + group_offset + element_index + 16);
          v0 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v0, inv_scale_vec), neg_quant_max_vec), quant_max_vec);
          v1 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v1, inv_scale_vec), neg_quant_max_vec), quant_max_vec);
          __m128i fp8_0 = cvtfp32_fp8e4m3(v0);
          __m128i fp8_1 = cvtfp32_fp8e4m3(v1);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(quantized_data + group_offset + element_index), fp8_0);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(quantized_data + group_offset + element_index + 16), fp8_1);
        }
        for (; element_index <= group_size - 16; element_index += 16) {
          __m512 v0 = load_16_as_fp32(input_data + group_offset + element_index);
          v0 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v0, inv_scale_vec), neg_quant_max_vec), quant_max_vec);
          __m128i fp8_0 = cvtfp32_fp8e4m3(v0);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(quantized_data + group_offset + element_index), fp8_0);
        }
        if (element_index < group_size) {
          alignas(64) float tail_values[16] = {0.0f};
          const int64_t tail_size = group_size - element_index;
          for (int64_t tail_index = 0; tail_index < tail_size; ++tail_index) {
            const float value = static_cast<float>(input_data[group_offset + element_index + tail_index]) * inv_scale;
            tail_values[tail_index] = std::clamp(value, neg_quant_max, quant_max);
          }
          __m512 tail_vec = _mm512_load_ps(tail_values);
          __m128i tail_fp8_vec = cvtfp32_fp8e4m3(tail_vec);
          alignas(16) uint8_t tail_fp8[16];
          _mm_store_si128(reinterpret_cast<__m128i*>(tail_fp8), tail_fp8_vec);
          uint8_t* quantized_tail = reinterpret_cast<uint8_t*>(quantized_data + group_offset + element_index);
          for (int64_t tail_index = 0; tail_index < tail_size; ++tail_index) {
            quantized_tail[tail_index] = tail_fp8[tail_index];
          }
        }
      }
    }
  });
}

std::tuple<at::Tensor, at::Tensor>
per_token_group_quant_fp8_cpu(const at::Tensor& input, int64_t group_size, double eps) {
  TORCH_CHECK(input.device().is_cpu(), "per_token_group_quant_fp8_cpu: input must be a CPU tensor");
  TORCH_CHECK(input.is_contiguous(), "per_token_group_quant_fp8_cpu: input must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "per_token_group_quant_fp8_cpu: input must have at least 2 dimensions");
  TORCH_CHECK(group_size > 0, "per_token_group_quant_fp8_cpu: group_size must be positive");
  TORCH_CHECK(
      input.size(-1) % group_size == 0,
      "per_token_group_quant_fp8_cpu: input last dimension must be divisible by group_size");
  TORCH_CHECK(eps > 0.0, "per_token_group_quant_fp8_cpu: eps must be positive");

  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(
      input_dtype == at::kBFloat16 || input_dtype == at::kHalf || input_dtype == at::kFloat,
      "per_token_group_quant_fp8_cpu: input must be bfloat16, float16, or float32");

  auto scale_sizes = input.sizes().vec();
  scale_sizes.back() = input.size(-1) / group_size;
  auto scale = at::empty(scale_sizes, input.options().dtype(at::kFloat));
  auto quantized = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));

  const int64_t num_groups = input.numel() / group_size;
  if (num_groups > 0) {
    switch (input_dtype) {
      case at::kFloat:
        per_token_group_quant_fp8_cpu_kernel<float>(
            input.data_ptr<float>(),
            quantized.data_ptr<at::Float8_e4m3fn>(),
            scale.data_ptr<float>(),
            num_groups,
            group_size,
            static_cast<float>(eps));
        break;
      case at::kHalf:
        per_token_group_quant_fp8_cpu_kernel<at::Half>(
            input.data_ptr<at::Half>(),
            quantized.data_ptr<at::Float8_e4m3fn>(),
            scale.data_ptr<float>(),
            num_groups,
            group_size,
            static_cast<float>(eps));
        break;
      case at::kBFloat16:
        per_token_group_quant_fp8_cpu_kernel<at::BFloat16>(
            input.data_ptr<at::BFloat16>(),
            quantized.data_ptr<at::Float8_e4m3fn>(),
            scale.data_ptr<float>(),
            num_groups,
            group_size,
            static_cast<float>(eps));
        break;
      default:
        TORCH_CHECK(false, "per_token_group_quant_fp8_cpu: unsupported input dtype");
    }
  }

  return std::make_tuple(quantized, scale);
}
template <typename scalar_t>
void quantize_fp8e4m3_cpu_kernel(
    const scalar_t* __restrict__ input_data,
    at::Float8_e4m3fn* __restrict__ output_data,
    int64_t num_elements,
    float scale) {
  constexpr float quant_max = 448.0f;
  constexpr float neg_quant_max = -quant_max;
  const float inv_scale = 1.0f / scale;
  const __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
  const __m512 quant_max_vec = _mm512_set1_ps(quant_max);
  const __m512 neg_quant_max_vec = _mm512_set1_ps(neg_quant_max);

  at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
    int64_t element_index = begin;
    for (; element_index <= end - 32; element_index += 32) {
      __m512 v0 = load_16_as_fp32(input_data + element_index);
      __m512 v1 = load_16_as_fp32(input_data + element_index + 16);
      v0 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v0, inv_scale_vec), neg_quant_max_vec), quant_max_vec);
      v1 = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v1, inv_scale_vec), neg_quant_max_vec), quant_max_vec);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data + element_index), cvtfp32_fp8e4m3(v0));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data + element_index + 16), cvtfp32_fp8e4m3(v1));
    }
    for (; element_index <= end - 16; element_index += 16) {
      __m512 value_vec = load_16_as_fp32(input_data + element_index);
      __m512 clamped_vec = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(value_vec, inv_scale_vec), neg_quant_max_vec), quant_max_vec);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data + element_index), cvtfp32_fp8e4m3(clamped_vec));
    }

    if (element_index < end) {
      alignas(64) float tail_values[16] = {0.0f};
      const int64_t tail_size = end - element_index;
      for (int64_t tail_index = 0; tail_index < tail_size; ++tail_index) {
        const float value = static_cast<float>(input_data[element_index + tail_index]) * inv_scale;
        tail_values[tail_index] = std::clamp(value, neg_quant_max, quant_max);
      }
      __m512 tail_vec = _mm512_load_ps(tail_values);
      __m128i tail_fp8_vec = cvtfp32_fp8e4m3(tail_vec);
      alignas(16) uint8_t tail_fp8[16];
      _mm_store_si128(reinterpret_cast<__m128i*>(tail_fp8), tail_fp8_vec);
      uint8_t* output_tail = reinterpret_cast<uint8_t*>(output_data + element_index);
      for (int64_t tail_index = 0; tail_index < tail_size; ++tail_index) {
        output_tail[tail_index] = tail_fp8[tail_index];
      }
    }
  });
}

template <typename scalar_t>
float dynamic_per_tensor_scaled_fp8_quant_cpu_kernel(
    const scalar_t* __restrict__ input_data, at::Float8_e4m3fn* __restrict__ output_data, int64_t num_elements) {
  constexpr float quant_max = 448.0f;
  constexpr float eps = 1.0e-12f;
  const __m512 sign_bit = _mm512_set1_ps(-0.0f);

  // Parallel absmax reduction: each thread computes a local max, then merge.
  int num_threads = at::get_num_threads();
  std::vector<float> per_thread_absmax(num_threads, 0.0f);
  at::parallel_for(0, num_elements, 4096, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    __m512 absmax_vec0 = _mm512_setzero_ps();
    __m512 absmax_vec1 = _mm512_setzero_ps();
    int64_t i = begin;
    for (; i <= end - 32; i += 32) {
      absmax_vec0 = _mm512_max_ps(absmax_vec0, _mm512_andnot_ps(sign_bit, load_16_as_fp32(input_data + i)));
      absmax_vec1 = _mm512_max_ps(absmax_vec1, _mm512_andnot_ps(sign_bit, load_16_as_fp32(input_data + i + 16)));
    }
    absmax_vec0 = _mm512_max_ps(absmax_vec0, absmax_vec1);
    for (; i <= end - 16; i += 16) {
      absmax_vec0 = _mm512_max_ps(absmax_vec0, _mm512_andnot_ps(sign_bit, load_16_as_fp32(input_data + i)));
    }
    float local_absmax = _mm512_reduce_max_ps(absmax_vec0);
    for (; i < end; ++i) {
      local_absmax = std::max(local_absmax, std::abs(static_cast<float>(input_data[i])));
    }
    per_thread_absmax[tid] = local_absmax;
  });
  float absmax = *std::max_element(per_thread_absmax.begin(), per_thread_absmax.end());

  const float scale = std::max(absmax, eps) / quant_max;
  quantize_fp8e4m3_cpu_kernel(input_data, output_data, num_elements, scale);
  return scale;
}

std::tuple<at::Tensor, at::Tensor> scaled_fp8_quant_cpu(
    const at::Tensor& input,
    const std::optional<at::Tensor>& scale_opt,
    int64_t num_token_padding,
    bool use_per_token_if_dynamic) {
  TORCH_CHECK(input.device().is_cpu(), "scaled_fp8_quant_cpu: input must be a CPU tensor");
  TORCH_CHECK(input.is_contiguous(), "scaled_fp8_quant_cpu: input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "scaled_fp8_quant_cpu: input must be 2D");
  TORCH_CHECK(num_token_padding >= 0, "scaled_fp8_quant_cpu: num_token_padding must be non-negative");

  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(
      input_dtype == at::kBFloat16 || input_dtype == at::kHalf || input_dtype == at::kFloat,
      "scaled_fp8_quant_cpu: input must be bfloat16, float16, or float32");

  const int64_t num_rows = input.size(0);
  const int64_t row_size = input.size(1);
  const int64_t output_rows = std::max(num_rows, num_token_padding);
  auto output = at::empty({output_rows, row_size}, input.options().dtype(at::kFloat8_e4m3fn));

  at::Tensor scale;
  if (scale_opt.has_value()) {
    scale = scale_opt.value();
    TORCH_CHECK(scale.device().is_cpu(), "scaled_fp8_quant_cpu: scale must be a CPU tensor");
    TORCH_CHECK(scale.numel() == 1, "scaled_fp8_quant_cpu: static scale must be scalar");
    const float scale_value = scale.item<float>();
    switch (input_dtype) {
      case at::kFloat:
        quantize_fp8e4m3_cpu_kernel(
            input.data_ptr<float>(), output.data_ptr<at::Float8_e4m3fn>(), input.numel(), scale_value);
        break;
      case at::kHalf:
        quantize_fp8e4m3_cpu_kernel(
            input.data_ptr<at::Half>(), output.data_ptr<at::Float8_e4m3fn>(), input.numel(), scale_value);
        break;
      case at::kBFloat16:
        quantize_fp8e4m3_cpu_kernel(
            input.data_ptr<at::BFloat16>(), output.data_ptr<at::Float8_e4m3fn>(), input.numel(), scale_value);
        break;
      default:
        TORCH_CHECK(false, "scaled_fp8_quant_cpu: unsupported input dtype");
    }
  } else if (use_per_token_if_dynamic) {
    scale = at::empty({output_rows, 1}, input.options().dtype(at::kFloat));
    switch (input_dtype) {
      case at::kFloat:
        per_token_group_quant_fp8_cpu_kernel(
            input.data_ptr<float>(), output.data_ptr<at::Float8_e4m3fn>(), scale.data_ptr<float>(), num_rows, row_size, 1e-12f);
        break;
      case at::kHalf:
        per_token_group_quant_fp8_cpu_kernel(
            input.data_ptr<at::Half>(),
            output.data_ptr<at::Float8_e4m3fn>(),
            scale.data_ptr<float>(),
            num_rows,
            row_size,
            1e-12f);
        break;
      case at::kBFloat16:
        per_token_group_quant_fp8_cpu_kernel(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::Float8_e4m3fn>(),
            scale.data_ptr<float>(),
            num_rows,
            row_size,
            1e-12f);
        break;
      default:
        TORCH_CHECK(false, "scaled_fp8_quant_cpu: unsupported input dtype");
    }
  } else {
    scale = at::empty({1}, input.options().dtype(at::kFloat));
    float scale_value = 0.0f;
    switch (input_dtype) {
      case at::kFloat:
        scale_value = dynamic_per_tensor_scaled_fp8_quant_cpu_kernel(
            input.data_ptr<float>(), output.data_ptr<at::Float8_e4m3fn>(), input.numel());
        break;
      case at::kHalf:
        scale_value = dynamic_per_tensor_scaled_fp8_quant_cpu_kernel(
            input.data_ptr<at::Half>(), output.data_ptr<at::Float8_e4m3fn>(), input.numel());
        break;
      case at::kBFloat16:
        scale_value = dynamic_per_tensor_scaled_fp8_quant_cpu_kernel(
            input.data_ptr<at::BFloat16>(), output.data_ptr<at::Float8_e4m3fn>(), input.numel());
        break;
      default:
        TORCH_CHECK(false, "scaled_fp8_quant_cpu: unsupported input dtype");
    }
    scale.fill_(scale_value);
  }

  return std::make_tuple(output, scale);
}

inline uint8_t ceil_to_ue8m0_byte(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t exponent = (bits >> 23) & 0xFF;
  const uint32_t mantissa = bits & 0x7FFFFF;
  exponent += mantissa != 0;
  exponent = std::min<uint32_t>(std::max<uint32_t>(exponent, 1), 254);
  return static_cast<uint8_t>(exponent);
}

inline float ue8m0_byte_to_float(uint8_t scale) {
  const uint32_t bits = static_cast<uint32_t>(scale) << 23;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename scalar_t>
void mxfp8_group_quantize_cpu_kernel(
    const scalar_t* __restrict__ input_data,
    at::Float8_e4m3fn* __restrict__ output_data,
    uint8_t* __restrict__ scale_data,
    int64_t num_groups) {
  constexpr int64_t group_size = 32;
  constexpr float quant_max = 448.0f;
  constexpr float neg_quant_max = -quant_max;
  const __m512 sign_bit = _mm512_set1_ps(-0.0f);
  const __m512 quant_max_vec = _mm512_set1_ps(quant_max);
  const __m512 neg_quant_max_vec = _mm512_set1_ps(neg_quant_max);

  at::parallel_for(0, num_groups, 0, [&](int64_t begin, int64_t end) {
    for (int64_t group_index = begin; group_index < end; ++group_index) {
      const int64_t group_offset = group_index * group_size;
      __m512 value_vec0 = load_16_as_fp32(input_data + group_offset);
      __m512 value_vec1 = load_16_as_fp32(input_data + group_offset + 16);
      __m512 abs_vec0 = _mm512_andnot_ps(sign_bit, value_vec0);
      __m512 abs_vec1 = _mm512_andnot_ps(sign_bit, value_vec1);
      const float absmax = _mm512_reduce_max_ps(_mm512_max_ps(abs_vec0, abs_vec1));
      const uint8_t scale_byte = ceil_to_ue8m0_byte(absmax / quant_max);
      const float inv_scale = 1.0f / ue8m0_byte_to_float(scale_byte);
      scale_data[group_index] = scale_byte;

      const __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
      __m512 scaled_vec0 = _mm512_mul_ps(value_vec0, inv_scale_vec);
      __m512 scaled_vec1 = _mm512_mul_ps(value_vec1, inv_scale_vec);
      __m512 clamped_vec0 = _mm512_min_ps(_mm512_max_ps(scaled_vec0, neg_quant_max_vec), quant_max_vec);
      __m512 clamped_vec1 = _mm512_min_ps(_mm512_max_ps(scaled_vec1, neg_quant_max_vec), quant_max_vec);
      __m128i fp8_vec0 = cvtfp32_fp8e4m3(clamped_vec0);
      __m128i fp8_vec1 = cvtfp32_fp8e4m3(clamped_vec1);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data + group_offset), fp8_vec0);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data + group_offset + 16), fp8_vec1);
    }
  });
}

std::tuple<at::Tensor, at::Tensor> mxfp8_group_quantize_cpu(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_cpu(), "mxfp8_group_quantize_cpu: input must be a CPU tensor");
  TORCH_CHECK(input.is_contiguous(), "mxfp8_group_quantize_cpu: input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "mxfp8_group_quantize_cpu: input must be 2D");
  TORCH_CHECK(input.size(1) % 32 == 0, "mxfp8_group_quantize_cpu: input K dimension must be divisible by 32");

  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(
      input_dtype == at::kBFloat16 || input_dtype == at::kHalf || input_dtype == at::kFloat,
      "mxfp8_group_quantize_cpu: input must be bfloat16, float16, or float32");

  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));
  auto scale = at::empty({input.size(0), input.size(1) / 32}, input.options().dtype(at::kByte));
  const int64_t num_groups = input.numel() / 32;

  if (num_groups > 0) {
    switch (input_dtype) {
      case at::kFloat:
        mxfp8_group_quantize_cpu_kernel(
            input.data_ptr<float>(), output.data_ptr<at::Float8_e4m3fn>(), scale.data_ptr<uint8_t>(), num_groups);
        break;
      case at::kHalf:
        mxfp8_group_quantize_cpu_kernel(
            input.data_ptr<at::Half>(), output.data_ptr<at::Float8_e4m3fn>(), scale.data_ptr<uint8_t>(), num_groups);
        break;
      case at::kBFloat16:
        mxfp8_group_quantize_cpu_kernel(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<at::Float8_e4m3fn>(),
            scale.data_ptr<uint8_t>(),
            num_groups);
        break;
      default:
        TORCH_CHECK(false, "mxfp8_group_quantize_cpu: unsupported input dtype");
    }
  }

  return std::make_tuple(output, scale);
}
// tinygemm interface
template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const float* __restrict__ Bbias,
    const float* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K,
    bool do_unpack) {
  if (Bbias != nullptr) {
    tinygemm_kernel<scalar_t, at::Float8_e4m3fn, float, true>(
        A, B, C, Btmp, Ctmp, scale, Bbias, M, N, K, lda, ldb, ldc, brg, block_size_K, do_unpack);
    return;
  }
  tinygemm_kernel<scalar_t, at::Float8_e4m3fn, float, false>(
      A, B, C, Btmp, Ctmp, scale, nullptr, M, N, K, lda, ldb, ldc, brg, block_size_K, do_unpack);
}

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    float scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg) {
  tinygemm_kernel2<scalar_t>(A, B, C, Btmp, Ctmp, scale, M, N, K, lda, ldb, ldc, brg);
}

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const float* __restrict__ Bbias,
    const uint8_t* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K,
    bool do_unpack) {
  if (Bbias != nullptr) {
    tinygemm_kernel<scalar_t, uint8_t, uint8_t, true>(
        A, B, C, Btmp, Ctmp, scale, Bbias, M, N, K, lda, ldb, ldc, brg, block_size_K, do_unpack);
    return;
  }
  tinygemm_kernel<scalar_t, uint8_t, uint8_t, false>(
      A, B, C, Btmp, Ctmp, scale, nullptr, M, N, K, lda, ldb, ldc, brg, block_size_K, do_unpack);
}

// tinygemm interface
template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B,
    float* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    const float* __restrict__ Bbias,
    const float* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K,
    bool do_unpack) {
  if (Bbias != nullptr) {
    tinygemm_kernel<scalar_t, at::Float8_e4m3fn, float, true>(
        A, B, C, Btmp, scale, Bbias, M, N, K, lda, ldb, ldc, brg, block_size_K, do_unpack);
    return;
  }
  tinygemm_kernel<scalar_t, at::Float8_e4m3fn, float, false>(
      A, B, C, Btmp, scale, nullptr, M, N, K, lda, ldb, ldc, brg, block_size_K, do_unpack);
}

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    const float* __restrict__ Bbias,
    const uint8_t* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K,
    bool do_unpack) {
  if (Bbias != nullptr) {
    tinygemm_kernel<scalar_t, uint8_t, uint8_t, true>(
        A, B, C, Btmp, scale, Bbias, M, N, K, lda, ldb, ldc, brg, block_size_K, do_unpack);
    return;
  }
  tinygemm_kernel<scalar_t, uint8_t, uint8_t, false>(
      A, B, C, Btmp, scale, nullptr, M, N, K, lda, ldb, ldc, brg, block_size_K, do_unpack);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE_A, TYPE_B, TYPE_S) \
  template void tinygemm_kernel<TYPE_A>(                      \
      const TYPE_A* __restrict__ A,                           \
      const TYPE_B* __restrict__ B,                           \
      TYPE_A* __restrict__ C,                                 \
      TYPE_A* __restrict__ Btmp,                              \
      float* __restrict__ Ctmp,                               \
      const float* __restrict__ Bbias,                        \
      const TYPE_S* __restrict__ scale,                       \
      int64_t M,                                              \
      int64_t N,                                              \
      int64_t K,                                              \
      int64_t lda,                                            \
      int64_t ldb,                                            \
      int64_t ldc,                                            \
      bool brg,                                               \
      int64_t block_size_K,                                   \
      bool do_unpack)

INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16, at::Float8_e4m3fn, float);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half, at::Float8_e4m3fn, float);
INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16, uint8_t, uint8_t);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half, uint8_t, uint8_t);

#define INSTANTIATE_TINYGEMM_TEMPLATE2(TYPE)   \
  template void tinygemm_kernel<TYPE>(         \
      const TYPE* __restrict__ A,              \
      const at::Float8_e4m3fn* __restrict__ B, \
      TYPE* __restrict__ C,                    \
      TYPE* __restrict__ Btmp,                 \
      float* __restrict__ Ctmp,                \
      float scale,                             \
      int64_t M,                               \
      int64_t N,                               \
      int64_t K,                               \
      int64_t lda,                             \
      int64_t ldb,                             \
      int64_t ldc,                             \
      bool brg)

INSTANTIATE_TINYGEMM_TEMPLATE2(at::BFloat16);

inline const float* get_bias_data(const std::optional<at::Tensor>& bias, int64_t N) {
  if (bias.has_value()) {
    const auto& bias_ref = bias.value();
    CHECK_EQ(bias_ref.size(0), N);
    return bias_ref.data_ptr<float>();
  }
  return nullptr;
}

// FP8 and MXFP4 WoQ uses the same pattern:
//   Btmp : [T, BLOCK_N * K]
//   Ctmp : [T, BLOCK_M * BLOCK_N]
inline at::Tensor alloc_thread_buffer(const at::TensorOptions& options, int64_t K) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  int num_threads = at::get_num_threads();
  int64_t size_per_thread = MAX_CACHE_BLOCK_SIZE * BLOCK_N * K + BLOCK_M * BLOCK_N * 2;
  return at::empty({num_threads, size_per_thread}, options);
}

at::Tensor fp8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    std::vector<int64_t> block_size,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales2);
  TORCH_CHECK(scales2.scalar_type() == at::kFloat, "fp8_scaled_mm_cpu: expect scales2 to be float32.");

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat2.size(1);

  CHECK_EQ(mat1.size(1), K);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  TORCH_CHECK(block_size.size() == 2, "fp8_scaled_mm_cpu: expect block_size.size() to be 2.");
  int64_t block_size_N = block_size[0];
  int64_t block_size_K = block_size[1];

  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(block_size_N % BLOCK_N == 0, "fp8_scaled_mm_cpu: expect block_size_N to be multiples of BLOCK_N");
  TORCH_CHECK(block_size_K == BLOCK_K, "fp8_scaled_mm_cpu: expect block_size_K equals to BLOCK_K");
  CHECK_EQ(scales2.size(0), div_up(N, block_size_N));
  CHECK_EQ(scales2.size(1), div_up(K, block_size_K));

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf, "fp8_scaled_mm_cpu: expect A to be bfloat16 or half.");
  TORCH_CHECK(st == out_dtype, "fp8_scaled_mm_cpu: expect A has same dtype with out_dtype.");
  TORCH_CHECK(mat2.scalar_type() == at::kFloat8_e4m3fn, "fp8_scaled_mm_cpu: expect mat2 to be fp8_e4m3.");
  TORCH_CHECK(scales2.scalar_type() == at::kFloat, "fp8_scaled_mm_cpu: expect scales to be float32.");
  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  auto buffer = alloc_thread_buffer(mat1.options(), K);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_dtype, "fp8_scaled_mm_kernel_impl", [&] {
    // used for lambda computing scale offset for each block
    //   fp8 block gemm sale shape: [N/128, K/128]
    //   for each block: [1, K/128]
    const int64_t scale_size_K = div_up(K, block_size_K);
    const int64_t blocks_n_per_group = block_size_N / BLOCK_N;

    fp_scaled_mm_kernel_impl<scalar_t, at::Float8_e4m3fn, float>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<at::Float8_e4m3fn>(),
        scales2.data_ptr<float>(),
        get_bias_data(bias, N),
        buffer.data_ptr<scalar_t>(),
        M,
        N,
        K,
        mat1.stride(0),
        out.stride(0),
        block_size_N,
        block_size_K,
        buffer.size(-1),
        [&](int64_t nb) { return (nb / blocks_n_per_group) * scale_size_K; });
  });

  return out;
}

// mat1 : [M, K] bfloat16
// mat2 : [N, K / 2] uint8, actual layout: [N / BLOCK_N, K / 2, BLOCK_N, 2]
// scales2: [N, K / G], actual layout: [N / BLOCK_N, K / G, BLOCK_N]
at::Tensor mxfp4_scaled_mm_cpu(
    at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales2, const std::optional<at::Tensor>& bias, bool is_vnni) {
  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales2);

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat2.size(1) * 2;

  // mxfp4 supports only group size of 32 (2^5)
  constexpr int64_t group_size = 32;
  constexpr int64_t BLOCK_N = block_size_n();

  CHECK_EQ(mat1.size(1), K);
  CHECK_EQ(scales2.numel(), N * K >> 5);

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf, "mxfp4_scaled_mm_cpu: expect A to be bfloat16 or half.");
  TORCH_CHECK(mat2.scalar_type() == at::kByte, "mxfp4_scaled_mm_cpu: expect mat2 to be uint8.");
  TORCH_CHECK(scales2.scalar_type() == at::kByte, "mxfp4_scaled_mm_cpu: expect scales to be uint8.");
  auto out = at::empty({M, N}, mat1.options());

  auto buffer = alloc_thread_buffer(mat1.options(), K);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "mxfp4_scaled_mm_kernel_impl", [&] {
    // used for lambda computing scale offset for each block
    //   mxfp4 block gemm sale shape: [N/BLOCK_N, K/32, BLOCK_N]
    //   for each block: [K/32, BLOCK_N]
    const int64_t s_strideN = (K >> 5) * BLOCK_N;

    fp_scaled_mm_kernel_impl<scalar_t, uint8_t, uint8_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<uint8_t>(),
        scales2.data_ptr<uint8_t>(),
        get_bias_data(bias, N),
        buffer.data_ptr<scalar_t>(),
        M,
        N,
        K,
        mat1.stride(0),
        out.stride(0),
        /* block_size_N */ 1,
        /* block_size_K */ group_size,
        buffer.size(-1),
        [&](int64_t nb) { return nb * s_strideN; });
  });

  return out;
}
