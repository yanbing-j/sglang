#include "common.h"
#include "vec.h"
#include "gemm.h"

namespace {

//   silu :    shape          leading dimension
//  input0  [m_size, BLOCK_N]    BLOCK_N
//  input1  [m_size, BLOCK_N]    BLOCK_N
//  output  [M * topk, N]          N
template <typename scalar_t, int BLOCK_N>
inline void silu_and_mul(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input0,  // x: x0, x1
    const scalar_t* __restrict__ input1,  // y: y0, y1
    int64_t m_size,
    int64_t N) {

  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  const bVec one = bVec(1.f);

  // no remainder
  for (int64_t m = 0; m < m_size; ++m) {
    scalar_t* __restrict__ out = output + m * N;
    const scalar_t* __restrict__ x = input0 + m * BLOCK_N;
    const scalar_t* __restrict__ y = input1 + m * BLOCK_N;

    for (int64_t d = 0; d < BLOCK_N; d += bVec::size()) {
      // fVec x0 = fVec::loadu(x + d);
      // fVec x1 = fVec::loadu(x + d + fVec::size());
      // fVec y0 = fVec::loadu(y + d);
      // fVec y1 = fVec::loadu(y + d + fVec::size());
      bVec x_ = bVec::loadu(x + d);
      bVec y_ = bVec::loadu(y + d);
      // silu
      x_ = x_ / (one + x_.neg().exp_u20());
      // x1 = x1 / (one + x1.neg().exp_u20());
      // mul
      x_ = x_ * y_;
      // x1 = x1 * y1;
      // convert
      // bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
      x_.store(out + d);
    }
  }
}

// out = input + input2 * scale
template <typename scalar_t>
inline void add_mul_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ input2, float scale, int64_t size) {

  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  std::cout<<"bvecsize "<<kVecSize<<" fvecsize "<<fVec::size()<<"\n";
  const fVec s_vec = fVec(scale);
  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    bVec x_bvec = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x_bvec);

    bVec y_bvec = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y_bvec);

    x0 = x0 + y0 * s_vec;
    x1 = x1 + y1 * s_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + float(input2[d]) * scale);
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d);
    fVec data1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

template <typename scalar_t>
inline void copy_add_stub(scalar_t* __restrict__ out, const float* __restrict__ input, const float* __restrict__ bias, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) + fVec::loadu(bias + d);
    fVec data1 = fVec::loadu(input + d + fVec::size()) + fVec::loadu(bias + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + bias[d]);
  }
}

inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const at::Float8_e4m3fn* __restrict__ packed_B,
    int N,
    int K,
    int ldb,
    int ldb_tmp,
    float scale) {
  // [K/2, N, 2]
  const int K2 = K >> 1;
  const int ldb2 = ldb; // ldb * 2 >> 1;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(packed_B);
  const __m512 vd = _mm512_set1_ps(scale);

  constexpr int BLOCK_N = block_size_n();
  // static_assert(BLOCK_N == 32);

  for (int k = 0; k < K2; ++k) {
    for (int n = 0; n < N; n += 64) { // BLOCK_N = 32
        __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2 + n);

        __m256i b8_0 = _mm512_extracti32x8_epi32(b8, 0);
        __m256i b8_1 = _mm512_extracti32x8_epi32(b8, 1);

        __m512bh bf16_0 = CVT_FP8_TO_BF16(b8_0);
        __m512bh bf16_1 = CVT_FP8_TO_BF16(b8_1);

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

        _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + n * 2 + 0, (__m512i)bf16_0);
        _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + n * 2 + 32, (__m512i)bf16_1);
    }
  }
}

template <typename scalar_t, typename packed_t, bool has_bias>
struct brgemm {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const float* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    TORCH_CHECK(false, "struct brgemm: primary template not implemented!");
  }
};

template <typename scalar_t, bool has_bias>
struct brgemm<scalar_t, scalar_t, has_bias> {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const float* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    UNUSED(scale);

    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(
        M, N, K, lda, ldb, BLOCK_N, /* add_C */ false, A, B, Ctmp);

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

template <bool has_bias>
struct brgemm<at::BFloat16, at::Float8_e4m3fn, has_bias> {
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
      int ldc) {
    constexpr int BLOCK_N = block_size_n();

    // [BLOCK_K, BLOCK_N] -> [BLOCK_K / 2, BLOCK_N * 2]
    const int ldb_tmp = block_size_n();

    static_assert(BLOCK_K == 128);

    // accumulate across K per BLOCK_K
    for (int k = 0; k < K; k += BLOCK_K) {
      int kb_size = std::min(BLOCK_K, K - k);

      int idx = k >> 7; // k / BLOCK_K where BLOCK_K = 128
      unpack_B(Btmp, B + k * ldb, N, kb_size, ldb, ldb_tmp, scale[idx]);

      const bool add_C = (k != 0);
      at::native::cpublas::brgemm(
          M, N, kb_size, lda, ldb_tmp, BLOCK_N, add_C, A + k, Btmp, Ctmp);
    }

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

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K) {

  if (brg) {
    brgemm<scalar_t, at::Float8_e4m3fn, has_bias>::apply(
        A, B, C, Btmp, Ctmp, bias, scale, M, N, K, lda, ldb, ldc);
    return;
  }

  // TODO: add the support for use_brgemm = false;
  TORCH_CHECK(false, "use_brgemm = false is not supported yet");
}

}

template <typename scalar_t>
void shared_expert_fp8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const at::Float8_e4m3fn* __restrict__ packed_w1,
    const at::Float8_e4m3fn* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    int64_t block_size_N,
    int64_t block_size_K,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K) {

  // handle 2 tiles per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  // print the above values
  std::cout<<"BLOCK_M "<<BLOCK_M<<" BLOCK_N "<<BLOCK_N<<"\n";

   // stage 1: intermediate_cache1 = silu(hidden_states @ w1)
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(2 * N, BLOCK_N);
  // print the above values
  std::cout<<"MB "<<MB<<" NB "<<NB<<"\n";

   int64_t scale_size_N = div_up(2 * N, block_size_N);
   int64_t scale_size_K = div_up(K, block_size_K);
   int64_t blocks_n_per_group = block_size_N / BLOCK_N;
  // print the above values
  std::cout<<"scale_size_N "<<scale_size_N<<" scale_size_K "<<scale_size_K<<"\n";
  // print the above values
  std::cout<<"blocks_n_per_group "<<blocks_n_per_group<<"\n";

  // TODO: add the support for use_brgemm = false;
  // use avx512-bf16 when a) M is small; b) dtype is bfloat16, otherwise use amx
  const bool use_brgemm = can_use_brgemm<at::Float8_e4m3fn>(M);

  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  const int64_t stride_n = K;
  int64_t mat1_strideM = K;
  int64_t out_strideM = 2 * N;


  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  // at::parallel_for(0, MB * NB, 102400, [&](int64_t begin, int64_t end) {
    // get local pointers
    // int tid = at::get_thread_num();
    // float* __restrict__ C0_tmp = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    // float* __restrict__ C1_tmp = C0_tmp + BLOCK_M * BLOCK_N;
    alignas(64) scalar_t Btmp[BLOCK_N * BLOCK_K];
    alignas(64) scalar_t my_output[M * 2 * N];
    alignas(64) scalar_t output_ic1[M * N];
    alignas(64) scalar_t output_ic2[M * K];
    // alignas(64) scalar_t C0[BLOCK_M * BLOCK_N];
    // alignas(64) scalar_t C1[BLOCK_M * BLOCK_N];
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];
    alignas(64) scalar_t C0[BLOCK_M * BLOCK_N];
    alignas(64) scalar_t C1[BLOCK_M * BLOCK_N];

    for (int64_t i = 0; i < MB * NB; ++i) {
      printf("i = %ld\n", i);
      int64_t mb = i / NB;
      int64_t nb = i % NB;
      // print mb nb
      printf("mb = %ld, nb = %ld\n", mb, nb);

      // nb0 from top half and nb1 from bottom half
      // int64_t nb0 = nb, nb1 = nb + NB;
      // const float* scale_ptr_nb0 = w1s + (nb0 / blocks_n_per_group) * scale_size_K;
      // const float* scale_ptr_nb1 = w1s + (nb1 / blocks_n_per_group) * scale_size_K;
      const float* scale_ptr = w1s + (nb / blocks_n_per_group) * scale_size_K;
      int64_t mb_start = mb * BLOCK_M;
      int64_t mb_size = std::min(M - mb_start, BLOCK_M);
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(2 * N - nb_start, BLOCK_N);
      // print the above values
      printf("m_size = %ld, nb_size = %ld\n", mb_size, nb_size);
      // // print scale_ptr_nb0 and scale_ptr_nb1
      // printf("scale_ptr_nb0 = %f, scale_ptr_nb1 = %f\n", *scale_ptr_nb0, *scale_ptr_nb1);

      // A shape [m_size, K]
      const scalar_t* A = input + mb * BLOCK_M * K;

      // 1.b gemm: C0 = A @ B0
      tinygemm_kernel<scalar_t, false>(
        /*   A                  */ input + mb_start * mat1_strideM,
        /*   B                  */ packed_w1 + nb_start * K,
        /*   C                  */ my_output + mb_start * out_strideM + nb_start,
        /*   Btmp               */ Btmp,
        /*   Ctmp               */ Ctmp,
        /*   scale              */ scale_ptr,
        /*   bias               */ nullptr,
        /*   M                  */ mb_size,
        /*   N                  */ nb_size,
        /*   K                  */ K,
        /*   lda                */ mat1_strideM,
        /*   ldb                */ nb_size,
        /*   ldc                */ out_strideM,
        /*   brg                */ use_brgemm,
        /*   block_size_K       */ block_size_K);
        // for (int64_t m = 0;m < mb_size;m++) {
        //   for(int64_t d = 0;d < BLOCK_N;d++){
        //     output[mb_start*out_strideM + nb_start + m*BLOCK_N+d] = C0[m*BLOCK_N + d];
        //   }
        // }
      // 1.c gemm: C1 = A @ B1
      // tinygemm_kernel<scalar_t, false>(
      // /*   A                  */ A,
      // /*   B                  */ packed_w1 + nb1 * BLOCK_N * stride_n,
      // /*   C                  */ output + mb * BLOCK_M * N + nb1 * BLOCK_N,
      // /*   Btmp               */ Btmp,
      // /*   Ctmp               */ Ctmp,
      // /*   scale              */ scale_ptr_nb1,
      // /*   bias               */ nullptr,
      // /*   M                  */ m_size,
      // /*   N                  */ n_size,
      // /*   K                  */ K,
      // /*   lda                */ K,
      // /*   ldb                */ n_size,
      // /*   ldc                */ BLOCK_N,
      // /*   brg                */ use_brgemm,
      // /*   block_size_K       */ block_size_K);

      // 1.d silu and mul
    // silu_and_mul<scalar_t, BLOCK_N>(
    //     ic1 + mb * BLOCK_M * N + nb * BLOCK_N,
    //     C0,
    //     C1,
    //     m_size,
    //     N);
    
    }
    alignas(64) scalar_t out[M * N];
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
  
    const bVec one = bVec(1.f);
    for (int64_t m = 0; m < M; m++) {
      for (int64_t d = 0; d < N;d+=bVec::size()) {
        bVec x_ = bVec::loadu(my_output + m * 2 * N + d);
        bVec y_ = bVec::loadu(my_output + m * 2 * N + N + d);
        x_ = x_ / (one + x_.neg().exp_u20());
        // mul
        x_ = x_ * y_;
        // convert
        x_.store(output_ic1 + m * N + d);
      }
    }
    
    // for (int64_t m = 0; m < M; m++) {
    //   for (int64_t n = 0;n < N * 2; n++) {
    //     if (n < N) {
    //       out[m * N + n] = output[m * 2 * N + n];
    //     }
    //   }
    // }
// }
// // });

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [K, N] as [OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_oc = N;
  mat1_strideM = IC;
  out_strideM = OC;
  scale_size_N = div_up(K, block_size_N);
  scale_size_K = div_up(N, block_size_K);
  blocks_n_per_group = block_size_N / BLOCK_N;

  // parallel on [MB2, NB2]
  // at::parallel_for(0, MB2 * NB2, 102400, [&](int64_t begin, int64_t end) {
    // get local pointers
    // int tid = at::get_thread_num();
    // we won't be using C1 for gemm2
    alignas(64) scalar_t Btmp2[BLOCK_K * BLOCK_N];
    alignas(64) scalar_t my_output2[M * K];
    alignas(64) scalar_t C2[BLOCK_M * BLOCK_K];
    alignas(64) float Ctmp2[BLOCK_M * BLOCK_K];
    // float* __restrict__ Cx_tmp = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;

    for (int64_t i = 0; i < MB2 * NB2; ++i) {
      int64_t mb = i / NB2;
      int64_t nb = i % NB2;
      const float* scale_ptr_2 = w2s + (nb / blocks_n_per_group) * scale_size_K;
      int64_t mb_start = mb * BLOCK_M;
      int64_t mb_size = std::min(M - mb_start, BLOCK_M);
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(OC - nb_start, BLOCK_N);

      // A shape [m_size, IC]
      // const scalar_t* __restrict__ A = ic1 + mb * BLOCK_M * N;

      // 2.a gemm: C = A @ B
      tinygemm_kernel<scalar_t, false>(
        /*   A                  */ output_ic1 + mb_start * mat1_strideM,
        /*   B                  */ packed_w2 + nb_start * N,
        /*   C                  */ C2,//output_ic2 + mb_start * out_strideM + nb_start,
        /*   Btmp               */ Btmp2,
        /*   Ctmp               */ Ctmp2,
        /*   scale              */ scale_ptr_2,
        /*   bias               */ nullptr,
        /*   M                  */ mb_size,
        /*   N                  */ nb_size,
        /*   K                  */ IC,
        /*   lda                */ mat1_strideM,
        /*   ldb                */ nb_size,
        /*   ldc                */ BLOCK_N,
        /*   brg                */ use_brgemm,
        /*   block_size_K       */ block_size_K);

  //       // 2.b copy from C to output and add fused_experts_out
      scalar_t* __restrict__ out = output + mb_start * out_strideM + nb_start;
      const scalar_t* __restrict__ fused_out = fused_experts_out + mb_start * out_strideM + nb_start;
      for (int64_t m = 0; m < mb_size; ++m) {
        add_mul_stub(out + m * K, C2 + m * BLOCK_N, fused_out + m * K, routed_scaling_factor, nb_size);
      }

      }
    //   for (int64_t m = 0; m < M; ++m) {
    //     add_mul_stub(output + m * mat1_strideM, output_ic2 + m * mat1_strideM, fused_experts_out + m * mat1_strideM, routed_scaling_factor, mat1_strideM);
    // }

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  // });

}

#define INSTANTIATE_SHARED_EXPERT_FP8_TEMPLATE(TYPE)                                        \
  template void shared_expert_fp8_kernel_impl<TYPE> (                                       \
      TYPE* __restrict__ output, TYPE* __restrict__ ic1,                                    \
      float* __restrict__ C_tmp, const TYPE* __restrict__ input,                             \
      const at::Float8_e4m3fn* __restrict__ packed_w1,                                      \
      const at::Float8_e4m3fn* __restrict__ packed_w2,                                      \
      const float* __restrict__ w1s, const float* __restrict__ w2s,                         \
      int64_t block_size_N, int64_t block_size_K,                                           \
      const TYPE* __restrict__ fused_experts_out,                                           \
      float routed_scaling_factor, int64_t M, int64_t N, int64_t K)

INSTANTIATE_SHARED_EXPERT_FP8_TEMPLATE(at::BFloat16);
INSTANTIATE_SHARED_EXPERT_FP8_TEMPLATE(at::Half);
