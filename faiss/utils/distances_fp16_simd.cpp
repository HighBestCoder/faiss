/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances_fp16.h>

#include <cassert>
#include <cmath>
#include <cstring>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/fp16.h>

#ifdef __SSE3__
#include <immintrin.h>
#endif

namespace faiss {

/*********************************************************
 * Reference (scalar) implementations
 *********************************************************/

namespace {

// FP16 scalar reference implementations

float fp16vec_inner_product_ref(
        const uint16_t* x,
        const uint16_t* y,
        size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        res += decode_fp16(x[i]) * decode_fp16(y[i]);
    }
    return res;
}

float fp16vec_L2sqr_ref(const uint16_t* x, const uint16_t* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float diff = decode_fp16(x[i]) - decode_fp16(y[i]);
        res += diff * diff;
    }
    return res;
}

float fp16vec_norm_L2sqr_ref(const uint16_t* x, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float v = decode_fp16(x[i]);
        res += v * v;
    }
    return res;
}

// BF16 scalar reference implementations

float bf16vec_inner_product_ref(
        const uint16_t* x,
        const uint16_t* y,
        size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        res += decode_bf16(x[i]) * decode_bf16(y[i]);
    }
    return res;
}

float bf16vec_L2sqr_ref(const uint16_t* x, const uint16_t* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float diff = decode_bf16(x[i]) - decode_bf16(y[i]);
        res += diff * diff;
    }
    return res;
}

float bf16vec_norm_L2sqr_ref(const uint16_t* x, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float v = decode_bf16(x[i]);
        res += v * v;
    }
    return res;
}

} // anonymous namespace

/*********************************************************
 * Tier 1: AVX-512F + F16C implementations
 *
 * FP16: load 16 x uint16 -> _mm256_loadu_si256
 *       convert to 16 x float32 -> _mm512_cvtph_ps (F16C + AVX512)
 *       compute with FMA -> _mm512_fmadd_ps
 *
 * BF16: load 16 x uint16 -> _mm256_loadu_si256
 *       zero-extend to 32-bit -> _mm512_cvtepu16_epi32
 *       shift left 16 -> _mm512_slli_epi32
 *       reinterpret as float -> _mm512_castsi512_ps
 *       compute with FMA -> _mm512_fmadd_ps
 *
 * Processes 16 elements per iteration (2x throughput vs AVX2).
 * Horizontal sum uses _mm512_reduce_add_ps.
 *********************************************************/

#if defined(__AVX512F__) && defined(__F16C__)

namespace {

/// Load 16 FP16 values and convert to 16 FP32 values in __m512
FAISS_ALWAYS_INLINE __m512 load_fp16_to_fp32_avx512(const uint16_t* p) {
    __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm512_cvtph_ps(h);
}

/// Load 16 BF16 values and convert to 16 FP32 values in __m512
/// BF16 is the upper 16 bits of FP32, so: zero-extend to 32 bits, shift left 16
FAISS_ALWAYS_INLINE __m512 load_bf16_to_fp32_avx512(const uint16_t* p) {
    __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    __m512i wide = _mm512_cvtepu16_epi32(h);
    wide = _mm512_slli_epi32(wide, 16);
    return _mm512_castsi512_ps(wide);
}

/// Load 8 FP16 values and convert to 8 FP32 values in __m256 (for tail handling)
FAISS_ALWAYS_INLINE __m256 load_fp16_to_fp32(const uint16_t* p) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_cvtph_ps(h);
}

/// Load 8 BF16 values and convert to 8 FP32 values in __m256 (for tail handling)
FAISS_ALWAYS_INLINE __m256 load_bf16_to_fp32(const uint16_t* p) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m256i wide = _mm256_cvtepu16_epi32(h);
    wide = _mm256_slli_epi32(wide, 16);
    return _mm256_castsi256_ps(wide);
}

/// Horizontal sum of 8 floats in a __m256
inline float horizontal_sum_avx2(__m256 v) {
    __m128 v0 =
            _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    __m128 v1 = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 3, 2));
    __m128 v2 = _mm_add_ps(v0, v1);
    __m128 v3 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 v4 = _mm_add_ps(v2, v3);
    return _mm_cvtss_f32(v4);
}

} // anonymous namespace

/*********************************************************
 * FP16 SIMD implementations (AVX-512F + F16C)
 *********************************************************/

#if defined(__AVX512FP16__)

/*
 * SPR fp16 mixed-precision: process 32 fp16 per iteration with two
 * independent fp32 accumulators. Math is FP32 throughout (cvtph_ps lossless,
 * no precision change vs the 16/iter AVX512 path); the win is purely
 * microarchitectural — twice the loads-per-iteration amortized against the
 * same horizontal reduce, and 2-wide accumulator chain that fills both FMA
 * ports of SPR.
 *
 * Bit-equivalence vs the 16/iter path: NOT bit-identical because pair
 * association of the FMAs differs (acc0 covers even 16-blocks, acc1 covers
 * odd ones). FP rounding stays in eps range; recall impact negligible.
 */

float fp16vec_inner_product(const uint16_t* x, const uint16_t* y, size_t d) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    size_t i = 0;

    // Main loop: 32 fp16/iter via two cvtph_ps.
    for (; i + 32 <= d; i += 32) {
        __m512 xf0 = load_fp16_to_fp32_avx512(x + i);
        __m512 yf0 = load_fp16_to_fp32_avx512(y + i);
        __m512 xf1 = load_fp16_to_fp32_avx512(x + i + 16);
        __m512 yf1 = load_fp16_to_fp32_avx512(y + i + 16);
        acc0 = _mm512_fmadd_ps(xf0, yf0, acc0);
        acc1 = _mm512_fmadd_ps(xf1, yf1, acc1);
    }

    // 16-element tail.
    if (i + 16 <= d) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);
        __m512 yf = load_fp16_to_fp32_avx512(y + i);
        acc0 = _mm512_fmadd_ps(xf, yf, acc0);
        i += 16;
    }

    float res = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));

    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 yf = load_fp16_to_fp32(y + i);
        __m256 prod = _mm256_mul_ps(xf, yf);
        res += horizontal_sum_avx2(prod);
        i += 8;
    }

    for (; i < d; i++) {
        res += decode_fp16(x[i]) * decode_fp16(y[i]);
    }
    return res;
}

float fp16vec_L2sqr(const uint16_t* x, const uint16_t* y, size_t d) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 32 <= d; i += 32) {
        __m512 xf0 = load_fp16_to_fp32_avx512(x + i);
        __m512 yf0 = load_fp16_to_fp32_avx512(y + i);
        __m512 xf1 = load_fp16_to_fp32_avx512(x + i + 16);
        __m512 yf1 = load_fp16_to_fp32_avx512(y + i + 16);
        __m512 d0 = _mm512_sub_ps(xf0, yf0);
        __m512 d1 = _mm512_sub_ps(xf1, yf1);
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
    }

    if (i + 16 <= d) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);
        __m512 yf = load_fp16_to_fp32_avx512(y + i);
        __m512 diff = _mm512_sub_ps(xf, yf);
        acc0 = _mm512_fmadd_ps(diff, diff, acc0);
        i += 16;
    }

    float res = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));

    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 yf = load_fp16_to_fp32(y + i);
        __m256 diff = _mm256_sub_ps(xf, yf);
        __m256 sq = _mm256_mul_ps(diff, diff);
        res += horizontal_sum_avx2(sq);
        i += 8;
    }

    for (; i < d; i++) {
        float diff = decode_fp16(x[i]) - decode_fp16(y[i]);
        res += diff * diff;
    }
    return res;
}

float fp16vec_norm_L2sqr(const uint16_t* x, size_t d) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 32 <= d; i += 32) {
        __m512 xf0 = load_fp16_to_fp32_avx512(x + i);
        __m512 xf1 = load_fp16_to_fp32_avx512(x + i + 16);
        acc0 = _mm512_fmadd_ps(xf0, xf0, acc0);
        acc1 = _mm512_fmadd_ps(xf1, xf1, acc1);
    }

    if (i + 16 <= d) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);
        acc0 = _mm512_fmadd_ps(xf, xf, acc0);
        i += 16;
    }

    float res = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));

    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 sq = _mm256_mul_ps(xf, xf);
        res += horizontal_sum_avx2(sq);
        i += 8;
    }

    for (; i < d; i++) {
        float v = decode_fp16(x[i]);
        res += v * v;
    }
    return res;
}

#else // !__AVX512FP16__

float fp16vec_inner_product(const uint16_t* x, const uint16_t* y, size_t d) {
    __m512 sum512 = _mm512_setzero_ps();
    size_t i = 0;

    // Main loop: process 16 FP16 elements per iteration
    for (; i + 16 <= d; i += 16) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);
        __m512 yf = load_fp16_to_fp32_avx512(y + i);
        sum512 = _mm512_fmadd_ps(xf, yf, sum512);
    }

    float res = _mm512_reduce_add_ps(sum512);

    // Handle 8-element tail with AVX2
    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 yf = load_fp16_to_fp32(y + i);
        __m256 prod = _mm256_mul_ps(xf, yf);
        res += horizontal_sum_avx2(prod);
        i += 8;
    }

    // Handle remaining scalar tail
    for (; i < d; i++) {
        res += decode_fp16(x[i]) * decode_fp16(y[i]);
    }
    return res;
}

float fp16vec_L2sqr(const uint16_t* x, const uint16_t* y, size_t d) {
    __m512 sum512 = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= d; i += 16) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);
        __m512 yf = load_fp16_to_fp32_avx512(y + i);
        __m512 diff = _mm512_sub_ps(xf, yf);
        sum512 = _mm512_fmadd_ps(diff, diff, sum512);
    }

    float res = _mm512_reduce_add_ps(sum512);

    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 yf = load_fp16_to_fp32(y + i);
        __m256 diff = _mm256_sub_ps(xf, yf);
        __m256 sq = _mm256_mul_ps(diff, diff);
        res += horizontal_sum_avx2(sq);
        i += 8;
    }

    for (; i < d; i++) {
        float diff = decode_fp16(x[i]) - decode_fp16(y[i]);
        res += diff * diff;
    }
    return res;
}

float fp16vec_norm_L2sqr(const uint16_t* x, size_t d) {
    __m512 sum512 = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= d; i += 16) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);
        sum512 = _mm512_fmadd_ps(xf, xf, sum512);
    }

    float res = _mm512_reduce_add_ps(sum512);

    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 sq = _mm256_mul_ps(xf, xf);
        res += horizontal_sum_avx2(sq);
        i += 8;
    }

    for (; i < d; i++) {
        float v = decode_fp16(x[i]);
        res += v * v;
    }
    return res;
}

#endif // __AVX512FP16__

void fp16vec_L2sqr_ny(
        float* dis,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        dis[j] = fp16vec_L2sqr(x, y + j * d, d);
    }
}

void fp16vec_inner_products_ny(
        float* ip,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        ip[j] = fp16vec_inner_product(x, y + j * d, d);
    }
}

void fp16vec_inner_product_batch_4(
        const uint16_t* __restrict x,
        const uint16_t* __restrict y0,
        const uint16_t* __restrict y1,
        const uint16_t* __restrict y2,
        const uint16_t* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    // 4 independent accumulators for ILP
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);

        sum0 = _mm512_fmadd_ps(xf, load_fp16_to_fp32_avx512(y0 + i), sum0);
        sum1 = _mm512_fmadd_ps(xf, load_fp16_to_fp32_avx512(y1 + i), sum1);
        sum2 = _mm512_fmadd_ps(xf, load_fp16_to_fp32_avx512(y2 + i), sum2);
        sum3 = _mm512_fmadd_ps(xf, load_fp16_to_fp32_avx512(y3 + i), sum3);
    }

    dis0 = _mm512_reduce_add_ps(sum0);
    dis1 = _mm512_reduce_add_ps(sum1);
    dis2 = _mm512_reduce_add_ps(sum2);
    dis3 = _mm512_reduce_add_ps(sum3);

    // Handle 8-element tail with AVX2
    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 prod0 = _mm256_mul_ps(xf, load_fp16_to_fp32(y0 + i));
        __m256 prod1 = _mm256_mul_ps(xf, load_fp16_to_fp32(y1 + i));
        __m256 prod2 = _mm256_mul_ps(xf, load_fp16_to_fp32(y2 + i));
        __m256 prod3 = _mm256_mul_ps(xf, load_fp16_to_fp32(y3 + i));
        dis0 += horizontal_sum_avx2(prod0);
        dis1 += horizontal_sum_avx2(prod1);
        dis2 += horizontal_sum_avx2(prod2);
        dis3 += horizontal_sum_avx2(prod3);
        i += 8;
    }

    // Handle remaining scalar tail
    for (; i < d; i++) {
        float xv = decode_fp16(x[i]);
        dis0 += xv * decode_fp16(y0[i]);
        dis1 += xv * decode_fp16(y1[i]);
        dis2 += xv * decode_fp16(y2[i]);
        dis3 += xv * decode_fp16(y3[i]);
    }
}

void fp16vec_L2sqr_batch_4(
        const uint16_t* __restrict x,
        const uint16_t* __restrict y0,
        const uint16_t* __restrict y1,
        const uint16_t* __restrict y2,
        const uint16_t* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);

        __m512 diff0 = _mm512_sub_ps(xf, load_fp16_to_fp32_avx512(y0 + i));
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);

        __m512 diff1 = _mm512_sub_ps(xf, load_fp16_to_fp32_avx512(y1 + i));
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);

        __m512 diff2 = _mm512_sub_ps(xf, load_fp16_to_fp32_avx512(y2 + i));
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);

        __m512 diff3 = _mm512_sub_ps(xf, load_fp16_to_fp32_avx512(y3 + i));
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
    }

    dis0 = _mm512_reduce_add_ps(sum0);
    dis1 = _mm512_reduce_add_ps(sum1);
    dis2 = _mm512_reduce_add_ps(sum2);
    dis3 = _mm512_reduce_add_ps(sum3);

    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);

        __m256 d0 = _mm256_sub_ps(xf, load_fp16_to_fp32(y0 + i));
        __m256 d1 = _mm256_sub_ps(xf, load_fp16_to_fp32(y1 + i));
        __m256 d2 = _mm256_sub_ps(xf, load_fp16_to_fp32(y2 + i));
        __m256 d3 = _mm256_sub_ps(xf, load_fp16_to_fp32(y3 + i));
        dis0 += horizontal_sum_avx2(_mm256_mul_ps(d0, d0));
        dis1 += horizontal_sum_avx2(_mm256_mul_ps(d1, d1));
        dis2 += horizontal_sum_avx2(_mm256_mul_ps(d2, d2));
        dis3 += horizontal_sum_avx2(_mm256_mul_ps(d3, d3));
        i += 8;
    }

    for (; i < d; i++) {
        float xv = decode_fp16(x[i]);
        float dd0 = xv - decode_fp16(y0[i]);
        float dd1 = xv - decode_fp16(y1[i]);
        float dd2 = xv - decode_fp16(y2[i]);
        float dd3 = xv - decode_fp16(y3[i]);
        dis0 += dd0 * dd0;
        dis1 += dd1 * dd1;
        dis2 += dd2 * dd2;
        dis3 += dd3 * dd3;
    }
}

/*********************************************************
 * BF16 SIMD implementations (AVX-512F)
 *
 * Standard path: load 16 x BF16, zero-extend, shift left 16,
 * reinterpret as FP32, then FMA.
 *
 * When __AVX512BF16__ is available (Sapphire Rapids / Genoa),
 * the inner product can use the native VDPBF16PS instruction
 * via _mm512_dpbf16_ps, which processes 32 BF16 elements
 * per accumulate (two pairs of 16 BF16 → 16 FP32 fused).
 *********************************************************/

#if defined(__AVX512BF16__)

/*
 * Native AVX-512 BF16 inner product using VDPBF16PS.
 *
 * _mm512_dpbf16_ps(acc, a, b) computes:
 *   acc[i] += a[2*i] * b[2*i] + a[2*i+1] * b[2*i+1]
 * where a,b are vectors of BF16 pairs and acc is FP32.
 * Each call processes 32 BF16 values into 16 FP32 accumulators.
 *
 * We process 32 BF16 elements per iteration when possible,
 * falling back to the standard convert-and-FMA path for tails.
 */

float bf16vec_inner_product(const uint16_t* x, const uint16_t* y, size_t d) {
    __m512 sum512 = _mm512_setzero_ps();
    size_t i = 0;

    // Main loop: process 32 BF16 elements per iteration with VDPBF16PS
    for (; i + 32 <= d; i += 32) {
        __m512bh xbf = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(x + i));
        __m512bh ybf = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(y + i));
        sum512 = _mm512_dpbf16_ps(sum512, xbf, ybf);
    }

    // 16-element tail with VDPBF16PS
    if (i + 16 <= d) {
        // Use zero-masked load for the remaining 16 BF16 elements:
        // Load 16 BF16 values into the low half of a 512-bit register,
        // zero the high half, and use VDPBF16PS.
        __m256i xh = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x + i));
        __m256i yh = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(y + i));
        __m512i x512 = _mm512_castsi256_si512(xh);
        __m512i y512 = _mm512_castsi256_si512(yh);
        // Zero the high 256 bits
        x512 = _mm512_inserti64x4(x512, _mm256_setzero_si256(), 1);
        y512 = _mm512_inserti64x4(y512, _mm256_setzero_si256(), 1);
        sum512 = _mm512_dpbf16_ps(sum512, (__m512bh)x512, (__m512bh)y512);
        i += 16;
    }

    float res = _mm512_reduce_add_ps(sum512);

    // 8-element tail with AVX2 convert path
    if (i + 8 <= d) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 yf = load_bf16_to_fp32(y + i);
        __m256 prod = _mm256_mul_ps(xf, yf);
        res += horizontal_sum_avx2(prod);
        i += 8;
    }

    // Scalar tail
    for (; i < d; i++) {
        res += decode_bf16(x[i]) * decode_bf16(y[i]);
    }
    return res;
}

#else // !__AVX512BF16__

float bf16vec_inner_product(const uint16_t* x, const uint16_t* y, size_t d) {
    __m512 sum512 = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= d; i += 16) {
        __m512 xf = load_bf16_to_fp32_avx512(x + i);
        __m512 yf = load_bf16_to_fp32_avx512(y + i);
        sum512 = _mm512_fmadd_ps(xf, yf, sum512);
    }

    float res = _mm512_reduce_add_ps(sum512);

    if (i + 8 <= d) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 yf = load_bf16_to_fp32(y + i);
        __m256 prod = _mm256_mul_ps(xf, yf);
        res += horizontal_sum_avx2(prod);
        i += 8;
    }

    for (; i < d; i++) {
        res += decode_bf16(x[i]) * decode_bf16(y[i]);
    }
    return res;
}

#endif // __AVX512BF16__ (bf16vec_inner_product)

#if defined(__AVX512BF16__)

/*
 * SPR / Genoa BF16 L2-squared via VDPBF16PS norm decomposition:
 *   ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * <x, y>
 * Three FP32 accumulators (sx, sy, sxy), each fed by _mm512_dpbf16_ps over
 * 32 BF16 elements per iteration. This is NOT bit-identical to the explicit
 * (x-y)^2 path (FP rounding ordering differs), but FP32 accumulator keeps
 * relative error in the eps range (~1e-7); recall impact is negligible.
 */
float bf16vec_L2sqr(const uint16_t* x, const uint16_t* y, size_t d) {
    __m512 sxy = _mm512_setzero_ps();
    __m512 sxx = _mm512_setzero_ps();
    __m512 syy = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 32 <= d; i += 32) {
        __m512bh xb = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(x + i));
        __m512bh yb = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(y + i));
        sxy = _mm512_dpbf16_ps(sxy, xb, yb);
        sxx = _mm512_dpbf16_ps(sxx, xb, xb);
        syy = _mm512_dpbf16_ps(syy, yb, yb);
    }

    if (i + 16 <= d) {
        __m256i xh = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x + i));
        __m256i yh = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(y + i));
        __m512i x512 = _mm512_inserti64x4(
                _mm512_castsi256_si512(xh), _mm256_setzero_si256(), 1);
        __m512i y512 = _mm512_inserti64x4(
                _mm512_castsi256_si512(yh), _mm256_setzero_si256(), 1);
        sxy = _mm512_dpbf16_ps(sxy, (__m512bh)x512, (__m512bh)y512);
        sxx = _mm512_dpbf16_ps(sxx, (__m512bh)x512, (__m512bh)x512);
        syy = _mm512_dpbf16_ps(syy, (__m512bh)y512, (__m512bh)y512);
        i += 16;
    }

    float nx = _mm512_reduce_add_ps(sxx);
    float ny = _mm512_reduce_add_ps(syy);
    float dxy = _mm512_reduce_add_ps(sxy);

    if (i + 8 <= d) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 yf = load_bf16_to_fp32(y + i);
        nx += horizontal_sum_avx2(_mm256_mul_ps(xf, xf));
        ny += horizontal_sum_avx2(_mm256_mul_ps(yf, yf));
        dxy += horizontal_sum_avx2(_mm256_mul_ps(xf, yf));
        i += 8;
    }

    for (; i < d; i++) {
        float xv = decode_bf16(x[i]);
        float yv = decode_bf16(y[i]);
        nx += xv * xv;
        ny += yv * yv;
        dxy += xv * yv;
    }
    return nx + ny - 2.0f * dxy;
}

/*
 * SPR BF16 self-dot via VDPBF16PS: identical math to current path but
 * processes 32 BF16/iter instead of 16 (FP32 accumulator preserved).
 */
float bf16vec_norm_L2sqr(const uint16_t* x, size_t d) {
    __m512 sum512 = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 32 <= d; i += 32) {
        __m512bh xb = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(x + i));
        sum512 = _mm512_dpbf16_ps(sum512, xb, xb);
    }

    if (i + 16 <= d) {
        __m256i xh = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x + i));
        __m512i x512 = _mm512_inserti64x4(
                _mm512_castsi256_si512(xh), _mm256_setzero_si256(), 1);
        sum512 = _mm512_dpbf16_ps(sum512, (__m512bh)x512, (__m512bh)x512);
        i += 16;
    }

    float res = _mm512_reduce_add_ps(sum512);

    if (i + 8 <= d) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 sq = _mm256_mul_ps(xf, xf);
        res += horizontal_sum_avx2(sq);
        i += 8;
    }

    for (; i < d; i++) {
        float v = decode_bf16(x[i]);
        res += v * v;
    }
    return res;
}

#else // !__AVX512BF16__

float bf16vec_L2sqr(const uint16_t* x, const uint16_t* y, size_t d) {
    __m512 sum512 = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= d; i += 16) {
        __m512 xf = load_bf16_to_fp32_avx512(x + i);
        __m512 yf = load_bf16_to_fp32_avx512(y + i);
        __m512 diff = _mm512_sub_ps(xf, yf);
        sum512 = _mm512_fmadd_ps(diff, diff, sum512);
    }

    float res = _mm512_reduce_add_ps(sum512);

    if (i + 8 <= d) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 yf = load_bf16_to_fp32(y + i);
        __m256 diff = _mm256_sub_ps(xf, yf);
        __m256 sq = _mm256_mul_ps(diff, diff);
        res += horizontal_sum_avx2(sq);
        i += 8;
    }

    for (; i < d; i++) {
        float diff = decode_bf16(x[i]) - decode_bf16(y[i]);
        res += diff * diff;
    }
    return res;
}

float bf16vec_norm_L2sqr(const uint16_t* x, size_t d) {
    __m512 sum512 = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= d; i += 16) {
        __m512 xf = load_bf16_to_fp32_avx512(x + i);
        sum512 = _mm512_fmadd_ps(xf, xf, sum512);
    }

    float res = _mm512_reduce_add_ps(sum512);

    if (i + 8 <= d) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 sq = _mm256_mul_ps(xf, xf);
        res += horizontal_sum_avx2(sq);
        i += 8;
    }

    for (; i < d; i++) {
        float v = decode_bf16(x[i]);
        res += v * v;
    }
    return res;
}

#endif // __AVX512BF16__ (bf16vec_L2sqr / norm_L2sqr)

void bf16vec_L2sqr_ny(
        float* dis,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        dis[j] = bf16vec_L2sqr(x, y + j * d, d);
    }
}

void bf16vec_inner_products_ny(
        float* ip,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        ip[j] = bf16vec_inner_product(x, y + j * d, d);
    }
}

#if defined(__AVX512BF16__)

void bf16vec_inner_product_batch_4(
        const uint16_t* __restrict x,
        const uint16_t* __restrict y0,
        const uint16_t* __restrict y1,
        const uint16_t* __restrict y2,
        const uint16_t* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= d; i += 32) {
        __m512bh xb = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(x + i));
        sum0 = _mm512_dpbf16_ps(sum0, xb,
                (__m512bh)_mm512_loadu_si512(
                        reinterpret_cast<const void*>(y0 + i)));
        sum1 = _mm512_dpbf16_ps(sum1, xb,
                (__m512bh)_mm512_loadu_si512(
                        reinterpret_cast<const void*>(y1 + i)));
        sum2 = _mm512_dpbf16_ps(sum2, xb,
                (__m512bh)_mm512_loadu_si512(
                        reinterpret_cast<const void*>(y2 + i)));
        sum3 = _mm512_dpbf16_ps(sum3, xb,
                (__m512bh)_mm512_loadu_si512(
                        reinterpret_cast<const void*>(y3 + i)));
    }

    if (i + 16 <= d) {
        __m256i xh = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x + i));
        __m512i x512 = _mm512_inserti64x4(
                _mm512_castsi256_si512(xh), _mm256_setzero_si256(), 1);
        auto load_zextend = [](const uint16_t* p) {
            __m256i h = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(p));
            return _mm512_inserti64x4(
                    _mm512_castsi256_si512(h), _mm256_setzero_si256(), 1);
        };
        sum0 = _mm512_dpbf16_ps(
                sum0, (__m512bh)x512, (__m512bh)load_zextend(y0 + i));
        sum1 = _mm512_dpbf16_ps(
                sum1, (__m512bh)x512, (__m512bh)load_zextend(y1 + i));
        sum2 = _mm512_dpbf16_ps(
                sum2, (__m512bh)x512, (__m512bh)load_zextend(y2 + i));
        sum3 = _mm512_dpbf16_ps(
                sum3, (__m512bh)x512, (__m512bh)load_zextend(y3 + i));
        i += 16;
    }

    dis0 = _mm512_reduce_add_ps(sum0);
    dis1 = _mm512_reduce_add_ps(sum1);
    dis2 = _mm512_reduce_add_ps(sum2);
    dis3 = _mm512_reduce_add_ps(sum3);

    if (i + 8 <= d) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 prod0 = _mm256_mul_ps(xf, load_bf16_to_fp32(y0 + i));
        __m256 prod1 = _mm256_mul_ps(xf, load_bf16_to_fp32(y1 + i));
        __m256 prod2 = _mm256_mul_ps(xf, load_bf16_to_fp32(y2 + i));
        __m256 prod3 = _mm256_mul_ps(xf, load_bf16_to_fp32(y3 + i));
        dis0 += horizontal_sum_avx2(prod0);
        dis1 += horizontal_sum_avx2(prod1);
        dis2 += horizontal_sum_avx2(prod2);
        dis3 += horizontal_sum_avx2(prod3);
        i += 8;
    }

    for (; i < d; i++) {
        float xv = decode_bf16(x[i]);
        dis0 += xv * decode_bf16(y0[i]);
        dis1 += xv * decode_bf16(y1[i]);
        dis2 += xv * decode_bf16(y2[i]);
        dis3 += xv * decode_bf16(y3[i]);
    }
}

/*
 * SPR L2 batch_4 via norm decomposition. Computes ||x||^2 once then
 * ||y_k||^2 + ||x||^2 - 2<x,y_k> per k. Uses 9 _mm512_dpbf16_ps per 32
 * elements (sxx + 4 syy + 4 sxy) vs current 4 sub+fmadd per 16 (= 8 per 32).
 * Wins are from larger lane width (32 vs 16) and load amortization.
 */
void bf16vec_L2sqr_batch_4(
        const uint16_t* __restrict x,
        const uint16_t* __restrict y0,
        const uint16_t* __restrict y1,
        const uint16_t* __restrict y2,
        const uint16_t* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    __m512 sxx = _mm512_setzero_ps();
    __m512 sxy0 = _mm512_setzero_ps();
    __m512 sxy1 = _mm512_setzero_ps();
    __m512 sxy2 = _mm512_setzero_ps();
    __m512 sxy3 = _mm512_setzero_ps();
    __m512 syy0 = _mm512_setzero_ps();
    __m512 syy1 = _mm512_setzero_ps();
    __m512 syy2 = _mm512_setzero_ps();
    __m512 syy3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= d; i += 32) {
        __m512bh xb = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(x + i));
        __m512bh y0b = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(y0 + i));
        __m512bh y1b = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(y1 + i));
        __m512bh y2b = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(y2 + i));
        __m512bh y3b = (__m512bh)_mm512_loadu_si512(
                reinterpret_cast<const void*>(y3 + i));
        sxx = _mm512_dpbf16_ps(sxx, xb, xb);
        sxy0 = _mm512_dpbf16_ps(sxy0, xb, y0b);
        sxy1 = _mm512_dpbf16_ps(sxy1, xb, y1b);
        sxy2 = _mm512_dpbf16_ps(sxy2, xb, y2b);
        sxy3 = _mm512_dpbf16_ps(sxy3, xb, y3b);
        syy0 = _mm512_dpbf16_ps(syy0, y0b, y0b);
        syy1 = _mm512_dpbf16_ps(syy1, y1b, y1b);
        syy2 = _mm512_dpbf16_ps(syy2, y2b, y2b);
        syy3 = _mm512_dpbf16_ps(syy3, y3b, y3b);
    }

    float nxx = _mm512_reduce_add_ps(sxx);
    float ny0 = _mm512_reduce_add_ps(syy0);
    float ny1 = _mm512_reduce_add_ps(syy1);
    float ny2 = _mm512_reduce_add_ps(syy2);
    float ny3 = _mm512_reduce_add_ps(syy3);
    float dx0 = _mm512_reduce_add_ps(sxy0);
    float dx1 = _mm512_reduce_add_ps(sxy1);
    float dx2 = _mm512_reduce_add_ps(sxy2);
    float dx3 = _mm512_reduce_add_ps(sxy3);

    // Initialize outputs with the 32-wide decomposition contribution; tails
    // accumulate on top.
    dis0 = nxx + ny0 - 2.0f * dx0;
    dis1 = nxx + ny1 - 2.0f * dx1;
    dis2 = nxx + ny2 - 2.0f * dx2;
    dis3 = nxx + ny3 - 2.0f * dx3;

    // 16-element and 8-element tails: use the explicit (x-y)^2 form for the
    // remaining lanes; mathematically identical to the decomposition since
    // we just add to the same dis_k accumulators.
    if (i + 16 <= d) {
        __m512 xf = load_bf16_to_fp32_avx512(x + i);
        __m512 d0 = _mm512_sub_ps(xf, load_bf16_to_fp32_avx512(y0 + i));
        __m512 d1 = _mm512_sub_ps(xf, load_bf16_to_fp32_avx512(y1 + i));
        __m512 d2 = _mm512_sub_ps(xf, load_bf16_to_fp32_avx512(y2 + i));
        __m512 d3 = _mm512_sub_ps(xf, load_bf16_to_fp32_avx512(y3 + i));
        dis0 += _mm512_reduce_add_ps(_mm512_mul_ps(d0, d0));
        dis1 += _mm512_reduce_add_ps(_mm512_mul_ps(d1, d1));
        dis2 += _mm512_reduce_add_ps(_mm512_mul_ps(d2, d2));
        dis3 += _mm512_reduce_add_ps(_mm512_mul_ps(d3, d3));
        i += 16;
    }

    if (i + 8 <= d) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 dd0 = _mm256_sub_ps(xf, load_bf16_to_fp32(y0 + i));
        __m256 dd1 = _mm256_sub_ps(xf, load_bf16_to_fp32(y1 + i));
        __m256 dd2 = _mm256_sub_ps(xf, load_bf16_to_fp32(y2 + i));
        __m256 dd3 = _mm256_sub_ps(xf, load_bf16_to_fp32(y3 + i));
        dis0 += horizontal_sum_avx2(_mm256_mul_ps(dd0, dd0));
        dis1 += horizontal_sum_avx2(_mm256_mul_ps(dd1, dd1));
        dis2 += horizontal_sum_avx2(_mm256_mul_ps(dd2, dd2));
        dis3 += horizontal_sum_avx2(_mm256_mul_ps(dd3, dd3));
        i += 8;
    }

    for (; i < d; i++) {
        float xv = decode_bf16(x[i]);
        float dd0 = xv - decode_bf16(y0[i]);
        float dd1 = xv - decode_bf16(y1[i]);
        float dd2 = xv - decode_bf16(y2[i]);
        float dd3 = xv - decode_bf16(y3[i]);
        dis0 += dd0 * dd0;
        dis1 += dd1 * dd1;
        dis2 += dd2 * dd2;
        dis3 += dd3 * dd3;
    }
}

#else // !__AVX512BF16__ — batch_4 fallback paths

void bf16vec_inner_product_batch_4(
        const uint16_t* __restrict x,
        const uint16_t* __restrict y0,
        const uint16_t* __restrict y1,
        const uint16_t* __restrict y2,
        const uint16_t* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        __m512 xf = load_bf16_to_fp32_avx512(x + i);

        sum0 = _mm512_fmadd_ps(xf, load_bf16_to_fp32_avx512(y0 + i), sum0);
        sum1 = _mm512_fmadd_ps(xf, load_bf16_to_fp32_avx512(y1 + i), sum1);
        sum2 = _mm512_fmadd_ps(xf, load_bf16_to_fp32_avx512(y2 + i), sum2);
        sum3 = _mm512_fmadd_ps(xf, load_bf16_to_fp32_avx512(y3 + i), sum3);
    }

    dis0 = _mm512_reduce_add_ps(sum0);
    dis1 = _mm512_reduce_add_ps(sum1);
    dis2 = _mm512_reduce_add_ps(sum2);
    dis3 = _mm512_reduce_add_ps(sum3);

    if (i + 8 <= d) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 prod0 = _mm256_mul_ps(xf, load_bf16_to_fp32(y0 + i));
        __m256 prod1 = _mm256_mul_ps(xf, load_bf16_to_fp32(y1 + i));
        __m256 prod2 = _mm256_mul_ps(xf, load_bf16_to_fp32(y2 + i));
        __m256 prod3 = _mm256_mul_ps(xf, load_bf16_to_fp32(y3 + i));
        dis0 += horizontal_sum_avx2(prod0);
        dis1 += horizontal_sum_avx2(prod1);
        dis2 += horizontal_sum_avx2(prod2);
        dis3 += horizontal_sum_avx2(prod3);
        i += 8;
    }

    for (; i < d; i++) {
        float xv = decode_bf16(x[i]);
        dis0 += xv * decode_bf16(y0[i]);
        dis1 += xv * decode_bf16(y1[i]);
        dis2 += xv * decode_bf16(y2[i]);
        dis3 += xv * decode_bf16(y3[i]);
    }
}

void bf16vec_L2sqr_batch_4(
        const uint16_t* __restrict x,
        const uint16_t* __restrict y0,
        const uint16_t* __restrict y1,
        const uint16_t* __restrict y2,
        const uint16_t* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        __m512 xf = load_bf16_to_fp32_avx512(x + i);

        __m512 diff0 = _mm512_sub_ps(xf, load_bf16_to_fp32_avx512(y0 + i));
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);

        __m512 diff1 = _mm512_sub_ps(xf, load_bf16_to_fp32_avx512(y1 + i));
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);

        __m512 diff2 = _mm512_sub_ps(xf, load_bf16_to_fp32_avx512(y2 + i));
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);

        __m512 diff3 = _mm512_sub_ps(xf, load_bf16_to_fp32_avx512(y3 + i));
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
    }

    dis0 = _mm512_reduce_add_ps(sum0);
    dis1 = _mm512_reduce_add_ps(sum1);
    dis2 = _mm512_reduce_add_ps(sum2);
    dis3 = _mm512_reduce_add_ps(sum3);

    if (i + 8 <= d) {
        __m256 xf = load_bf16_to_fp32(x + i);

        __m256 dd0 = _mm256_sub_ps(xf, load_bf16_to_fp32(y0 + i));
        __m256 dd1 = _mm256_sub_ps(xf, load_bf16_to_fp32(y1 + i));
        __m256 dd2 = _mm256_sub_ps(xf, load_bf16_to_fp32(y2 + i));
        __m256 dd3 = _mm256_sub_ps(xf, load_bf16_to_fp32(y3 + i));
        dis0 += horizontal_sum_avx2(_mm256_mul_ps(dd0, dd0));
        dis1 += horizontal_sum_avx2(_mm256_mul_ps(dd1, dd1));
        dis2 += horizontal_sum_avx2(_mm256_mul_ps(dd2, dd2));
        dis3 += horizontal_sum_avx2(_mm256_mul_ps(dd3, dd3));
        i += 8;
    }

    for (; i < d; i++) {
        float xv = decode_bf16(x[i]);
        float dd0 = xv - decode_bf16(y0[i]);
        float dd1 = xv - decode_bf16(y1[i]);
        float dd2 = xv - decode_bf16(y2[i]);
        float dd3 = xv - decode_bf16(y3[i]);
        dis0 += dd0 * dd0;
        dis1 += dd1 * dd1;
        dis2 += dd2 * dd2;
        dis3 += dd3 * dd3;
    }
}

#endif // __AVX512BF16__ (bf16vec_*_batch_4)

/*********************************************************
 * FP16 native-precision implementations (AVX-512_FP16, SPR+)
 *
 * Use __m512h (32 fp16 lanes) and _mm512_fmadd_ph for ~2x throughput vs
 * the FP32-accumulator path. Accumulator is fp16 throughout the inner
 * loop — partial sums saturate at ±65504, and rounding error grows
 * faster than FP32. Public header documents the recall risk; these are
 * for benchmarking, not silent default.
 *
 * Tail handling: 16-element fp16 tail uses one fmadd_ph on the low half
 * with a masked load. 8-element / scalar tails fall back to the FP32
 * path's helpers so we don't carry duplicate scalar code.
 *********************************************************/

#if defined(__AVX512FP16__)

namespace {

/// Reduce a __m512h (32 fp16 lanes) to a scalar fp32 by promoting to fp32
/// before summing — keeps the final scalar accurate even when lane values
/// are near fp16 saturation. Uses two _mm512_cvtxph_ps over the low and
/// high __m256h halves rather than _mm512_reduce_add_ph (which sums in
/// fp16 and can saturate the final scalar).
FAISS_ALWAYS_INLINE float reduce_add_ph_to_fp32(__m512h v) {
    __m256i raw = _mm512_castsi512_si256(_mm512_castph_si512(v));
    __m256i raw_hi = _mm512_extracti64x4_epi64(_mm512_castph_si512(v), 1);
    __m512 lo = _mm512_cvtxph_ps((__m256h)raw);
    __m512 hi = _mm512_cvtxph_ps((__m256h)raw_hi);
    return _mm512_reduce_add_ps(_mm512_add_ps(lo, hi));
}

} // anonymous namespace

float fp16vec_inner_product_native(
        const uint16_t* x,
        const uint16_t* y,
        size_t d) {
    __m512h acc = _mm512_setzero_ph();
    size_t i = 0;

    // Main loop: 32 fp16 lanes per fmadd_ph.
    for (; i + 32 <= d; i += 32) {
        __m512h xh = (__m512h)_mm512_loadu_si512(
                reinterpret_cast<const void*>(x + i));
        __m512h yh = (__m512h)_mm512_loadu_si512(
                reinterpret_cast<const void*>(y + i));
        acc = _mm512_fmadd_ph(xh, yh, acc);
    }

    float res = reduce_add_ph_to_fp32(acc);

    // 16-fp16 tail via FP32 helper to keep partials in FP32.
    if (i + 16 <= d) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);
        __m512 yf = load_fp16_to_fp32_avx512(y + i);
        res += _mm512_reduce_add_ps(_mm512_mul_ps(xf, yf));
        i += 16;
    }
    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 yf = load_fp16_to_fp32(y + i);
        res += horizontal_sum_avx2(_mm256_mul_ps(xf, yf));
        i += 8;
    }
    for (; i < d; i++) {
        res += decode_fp16(x[i]) * decode_fp16(y[i]);
    }
    return res;
}

float fp16vec_L2sqr_native(const uint16_t* x, const uint16_t* y, size_t d) {
    __m512h acc = _mm512_setzero_ph();
    size_t i = 0;

    for (; i + 32 <= d; i += 32) {
        __m512h xh = (__m512h)_mm512_loadu_si512(
                reinterpret_cast<const void*>(x + i));
        __m512h yh = (__m512h)_mm512_loadu_si512(
                reinterpret_cast<const void*>(y + i));
        __m512h diff = _mm512_sub_ph(xh, yh);
        acc = _mm512_fmadd_ph(diff, diff, acc);
    }

    float res = reduce_add_ph_to_fp32(acc);

    if (i + 16 <= d) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);
        __m512 yf = load_fp16_to_fp32_avx512(y + i);
        __m512 diff = _mm512_sub_ps(xf, yf);
        res += _mm512_reduce_add_ps(_mm512_mul_ps(diff, diff));
        i += 16;
    }
    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 yf = load_fp16_to_fp32(y + i);
        __m256 diff = _mm256_sub_ps(xf, yf);
        res += horizontal_sum_avx2(_mm256_mul_ps(diff, diff));
        i += 8;
    }
    for (; i < d; i++) {
        float diff = decode_fp16(x[i]) - decode_fp16(y[i]);
        res += diff * diff;
    }
    return res;
}

float fp16vec_norm_L2sqr_native(const uint16_t* x, size_t d) {
    __m512h acc = _mm512_setzero_ph();
    size_t i = 0;

    for (; i + 32 <= d; i += 32) {
        __m512h xh = (__m512h)_mm512_loadu_si512(
                reinterpret_cast<const void*>(x + i));
        acc = _mm512_fmadd_ph(xh, xh, acc);
    }

    float res = reduce_add_ph_to_fp32(acc);

    if (i + 16 <= d) {
        __m512 xf = load_fp16_to_fp32_avx512(x + i);
        res += _mm512_reduce_add_ps(_mm512_mul_ps(xf, xf));
        i += 16;
    }
    if (i + 8 <= d) {
        __m256 xf = load_fp16_to_fp32(x + i);
        res += horizontal_sum_avx2(_mm256_mul_ps(xf, xf));
        i += 8;
    }
    for (; i < d; i++) {
        float v = decode_fp16(x[i]);
        res += v * v;
    }
    return res;
}

#else // !__AVX512FP16__ — within Tier 1: fall back to FP32 helpers

float fp16vec_inner_product_native(
        const uint16_t* x,
        const uint16_t* y,
        size_t d) {
    return fp16vec_inner_product(x, y, d);
}

float fp16vec_L2sqr_native(const uint16_t* x, const uint16_t* y, size_t d) {
    return fp16vec_L2sqr(x, y, d);
}

float fp16vec_norm_L2sqr_native(const uint16_t* x, size_t d) {
    return fp16vec_norm_L2sqr(x, d);
}

#endif // __AVX512FP16__

/*********************************************************
 * Tier 2: AVX2 + F16C implementations
 *
 * FP16: load 8 x uint16 -> _mm_loadu_si128
 *       convert to 8 x float32 -> _mm256_cvtph_ps (F16C)
 *       compute with FMA -> _mm256_fmadd_ps
 *
 * BF16: load 8 x uint16 -> _mm_loadu_si128
 *       zero-extend to 32-bit -> _mm256_cvtepu16_epi32
 *       shift left 16 -> _mm256_slli_epi32
 *       reinterpret as float -> _mm256_castsi256_ps
 *       compute with FMA -> _mm256_fmadd_ps
 *
 * Processes 8 elements per iteration.
 *********************************************************/

#elif defined(__AVX2__) && defined(__F16C__)

namespace {

/// Horizontal sum of 8 floats in a __m256
inline float horizontal_sum_avx2(__m256 v) {
    // add high and low 128-bit lanes
    __m128 v0 =
            _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    // horizontal sum of 4 floats
    __m128 v1 = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(0, 0, 3, 2));
    __m128 v2 = _mm_add_ps(v0, v1);
    __m128 v3 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 v4 = _mm_add_ps(v2, v3);
    return _mm_cvtss_f32(v4);
}

/// Load 8 FP16 values and convert to 8 FP32 values in __m256
FAISS_ALWAYS_INLINE __m256 load_fp16_to_fp32(const uint16_t* p) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_cvtph_ps(h);
}

/// Load 8 BF16 values and convert to 8 FP32 values in __m256
/// BF16 is the upper 16 bits of FP32, so: zero-extend to 32 bits, shift left 16
FAISS_ALWAYS_INLINE __m256 load_bf16_to_fp32(const uint16_t* p) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m256i wide = _mm256_cvtepu16_epi32(h);
    wide = _mm256_slli_epi32(wide, 16);
    return _mm256_castsi256_ps(wide);
}

} // anonymous namespace

/*********************************************************
 * FP16 SIMD implementations (AVX2 + F16C)
 *********************************************************/

float fp16vec_inner_product(const uint16_t* x, const uint16_t* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    // Main loop: process 8 FP16 elements per iteration
    for (; i + 8 <= d; i += 8) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 yf = load_fp16_to_fp32(y + i);
        sum = _mm256_fmadd_ps(xf, yf, sum);
    }

    float res = horizontal_sum_avx2(sum);

    // Handle tail elements
    for (; i < d; i++) {
        res += decode_fp16(x[i]) * decode_fp16(y[i]);
    }
    return res;
}

float fp16vec_L2sqr(const uint16_t* x, const uint16_t* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 xf = load_fp16_to_fp32(x + i);
        __m256 yf = load_fp16_to_fp32(y + i);
        __m256 diff = _mm256_sub_ps(xf, yf);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    float res = horizontal_sum_avx2(sum);

    for (; i < d; i++) {
        float diff = decode_fp16(x[i]) - decode_fp16(y[i]);
        res += diff * diff;
    }
    return res;
}

float fp16vec_norm_L2sqr(const uint16_t* x, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 xf = load_fp16_to_fp32(x + i);
        sum = _mm256_fmadd_ps(xf, xf, sum);
    }

    float res = horizontal_sum_avx2(sum);

    for (; i < d; i++) {
        float v = decode_fp16(x[i]);
        res += v * v;
    }
    return res;
}

void fp16vec_L2sqr_ny(
        float* dis,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        dis[j] = fp16vec_L2sqr(x, y + j * d, d);
    }
}

void fp16vec_inner_products_ny(
        float* ip,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        ip[j] = fp16vec_inner_product(x, y + j * d, d);
    }
}

void fp16vec_inner_product_batch_4(
        const uint16_t* __restrict x,
        const uint16_t* __restrict y0,
        const uint16_t* __restrict y1,
        const uint16_t* __restrict y2,
        const uint16_t* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    // 4 independent accumulators for ILP
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        // Load x once, convert to FP32
        __m256 xf = load_fp16_to_fp32(x + i);

        // Load each y vector, convert to FP32, accumulate
        __m256 y0f = load_fp16_to_fp32(y0 + i);
        sum0 = _mm256_fmadd_ps(xf, y0f, sum0);

        __m256 y1f = load_fp16_to_fp32(y1 + i);
        sum1 = _mm256_fmadd_ps(xf, y1f, sum1);

        __m256 y2f = load_fp16_to_fp32(y2 + i);
        sum2 = _mm256_fmadd_ps(xf, y2f, sum2);

        __m256 y3f = load_fp16_to_fp32(y3 + i);
        sum3 = _mm256_fmadd_ps(xf, y3f, sum3);
    }

    // Reduce accumulators to scalars
    dis0 = horizontal_sum_avx2(sum0);
    dis1 = horizontal_sum_avx2(sum1);
    dis2 = horizontal_sum_avx2(sum2);
    dis3 = horizontal_sum_avx2(sum3);

    // Handle tail
    for (; i < d; i++) {
        float xv = decode_fp16(x[i]);
        dis0 += xv * decode_fp16(y0[i]);
        dis1 += xv * decode_fp16(y1[i]);
        dis2 += xv * decode_fp16(y2[i]);
        dis3 += xv * decode_fp16(y3[i]);
    }
}

void fp16vec_L2sqr_batch_4(
        const uint16_t* __restrict x,
        const uint16_t* __restrict y0,
        const uint16_t* __restrict y1,
        const uint16_t* __restrict y2,
        const uint16_t* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 xf = load_fp16_to_fp32(x + i);

        __m256 diff0 = _mm256_sub_ps(xf, load_fp16_to_fp32(y0 + i));
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        __m256 diff1 = _mm256_sub_ps(xf, load_fp16_to_fp32(y1 + i));
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        __m256 diff2 = _mm256_sub_ps(xf, load_fp16_to_fp32(y2 + i));
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

        __m256 diff3 = _mm256_sub_ps(xf, load_fp16_to_fp32(y3 + i));
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }

    dis0 = horizontal_sum_avx2(sum0);
    dis1 = horizontal_sum_avx2(sum1);
    dis2 = horizontal_sum_avx2(sum2);
    dis3 = horizontal_sum_avx2(sum3);

    for (; i < d; i++) {
        float xv = decode_fp16(x[i]);
        float d0 = xv - decode_fp16(y0[i]);
        float d1 = xv - decode_fp16(y1[i]);
        float d2 = xv - decode_fp16(y2[i]);
        float d3 = xv - decode_fp16(y3[i]);
        dis0 += d0 * d0;
        dis1 += d1 * d1;
        dis2 += d2 * d2;
        dis3 += d3 * d3;
    }
}

// FP16 native-precision functions: AVX2 tier has no AVX512_FP16, so
// forward to the FP32-accumulator path. Callers can link unconditionally.
float fp16vec_inner_product_native(
        const uint16_t* x,
        const uint16_t* y,
        size_t d) {
    return fp16vec_inner_product(x, y, d);
}

float fp16vec_L2sqr_native(const uint16_t* x, const uint16_t* y, size_t d) {
    return fp16vec_L2sqr(x, y, d);
}

float fp16vec_norm_L2sqr_native(const uint16_t* x, size_t d) {
    return fp16vec_norm_L2sqr(x, d);
}

/*********************************************************
 * BF16 SIMD implementations (AVX2)
 *
 * Pattern: load 8 x uint16 -> _mm_loadu_si128
 *          zero-extend to 32-bit -> _mm256_cvtepu16_epi32
 *          shift left 16 -> _mm256_slli_epi32
 *          reinterpret as float -> _mm256_castsi256_ps
 *          compute with FMA -> _mm256_fmadd_ps
 *********************************************************/

float bf16vec_inner_product(const uint16_t* x, const uint16_t* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 yf = load_bf16_to_fp32(y + i);
        sum = _mm256_fmadd_ps(xf, yf, sum);
    }

    float res = horizontal_sum_avx2(sum);

    for (; i < d; i++) {
        res += decode_bf16(x[i]) * decode_bf16(y[i]);
    }
    return res;
}

float bf16vec_L2sqr(const uint16_t* x, const uint16_t* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 xf = load_bf16_to_fp32(x + i);
        __m256 yf = load_bf16_to_fp32(y + i);
        __m256 diff = _mm256_sub_ps(xf, yf);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    float res = horizontal_sum_avx2(sum);

    for (; i < d; i++) {
        float diff = decode_bf16(x[i]) - decode_bf16(y[i]);
        res += diff * diff;
    }
    return res;
}

float bf16vec_norm_L2sqr(const uint16_t* x, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 xf = load_bf16_to_fp32(x + i);
        sum = _mm256_fmadd_ps(xf, xf, sum);
    }

    float res = horizontal_sum_avx2(sum);

    for (; i < d; i++) {
        float v = decode_bf16(x[i]);
        res += v * v;
    }
    return res;
}

void bf16vec_L2sqr_ny(
        float* dis,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        dis[j] = bf16vec_L2sqr(x, y + j * d, d);
    }
}

void bf16vec_inner_products_ny(
        float* ip,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        ip[j] = bf16vec_inner_product(x, y + j * d, d);
    }
}

void bf16vec_inner_product_batch_4(
        const uint16_t* __restrict x,
        const uint16_t* __restrict y0,
        const uint16_t* __restrict y1,
        const uint16_t* __restrict y2,
        const uint16_t* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 xf = load_bf16_to_fp32(x + i);

        sum0 = _mm256_fmadd_ps(xf, load_bf16_to_fp32(y0 + i), sum0);
        sum1 = _mm256_fmadd_ps(xf, load_bf16_to_fp32(y1 + i), sum1);
        sum2 = _mm256_fmadd_ps(xf, load_bf16_to_fp32(y2 + i), sum2);
        sum3 = _mm256_fmadd_ps(xf, load_bf16_to_fp32(y3 + i), sum3);
    }

    dis0 = horizontal_sum_avx2(sum0);
    dis1 = horizontal_sum_avx2(sum1);
    dis2 = horizontal_sum_avx2(sum2);
    dis3 = horizontal_sum_avx2(sum3);

    for (; i < d; i++) {
        float xv = decode_bf16(x[i]);
        dis0 += xv * decode_bf16(y0[i]);
        dis1 += xv * decode_bf16(y1[i]);
        dis2 += xv * decode_bf16(y2[i]);
        dis3 += xv * decode_bf16(y3[i]);
    }
}

void bf16vec_L2sqr_batch_4(
        const uint16_t* __restrict x,
        const uint16_t* __restrict y0,
        const uint16_t* __restrict y1,
        const uint16_t* __restrict y2,
        const uint16_t* __restrict y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 xf = load_bf16_to_fp32(x + i);

        __m256 diff0 = _mm256_sub_ps(xf, load_bf16_to_fp32(y0 + i));
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        __m256 diff1 = _mm256_sub_ps(xf, load_bf16_to_fp32(y1 + i));
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        __m256 diff2 = _mm256_sub_ps(xf, load_bf16_to_fp32(y2 + i));
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

        __m256 diff3 = _mm256_sub_ps(xf, load_bf16_to_fp32(y3 + i));
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }

    dis0 = horizontal_sum_avx2(sum0);
    dis1 = horizontal_sum_avx2(sum1);
    dis2 = horizontal_sum_avx2(sum2);
    dis3 = horizontal_sum_avx2(sum3);

    for (; i < d; i++) {
        float xv = decode_bf16(x[i]);
        float d0 = xv - decode_bf16(y0[i]);
        float d1 = xv - decode_bf16(y1[i]);
        float d2 = xv - decode_bf16(y2[i]);
        float d3 = xv - decode_bf16(y3[i]);
        dis0 += d0 * d0;
        dis1 += d1 * d1;
        dis2 += d2 * d2;
        dis3 += d3 * d3;
    }
}

#else

/*********************************************************
 * Tier 3: Fallback (scalar) implementations when neither
 * AVX-512F+F16C nor AVX2+F16C is available
 *********************************************************/

float fp16vec_inner_product(const uint16_t* x, const uint16_t* y, size_t d) {
    return fp16vec_inner_product_ref(x, y, d);
}

float fp16vec_L2sqr(const uint16_t* x, const uint16_t* y, size_t d) {
    return fp16vec_L2sqr_ref(x, y, d);
}

float fp16vec_norm_L2sqr(const uint16_t* x, size_t d) {
    return fp16vec_norm_L2sqr_ref(x, d);
}

void fp16vec_L2sqr_ny(
        float* dis,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        dis[j] = fp16vec_L2sqr_ref(x, y + j * d, d);
    }
}

void fp16vec_inner_products_ny(
        float* ip,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        ip[j] = fp16vec_inner_product_ref(x, y + j * d, d);
    }
}

void fp16vec_inner_product_batch_4(
        const uint16_t* x,
        const uint16_t* y0,
        const uint16_t* y1,
        const uint16_t* y2,
        const uint16_t* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    dis0 = fp16vec_inner_product_ref(x, y0, d);
    dis1 = fp16vec_inner_product_ref(x, y1, d);
    dis2 = fp16vec_inner_product_ref(x, y2, d);
    dis3 = fp16vec_inner_product_ref(x, y3, d);
}

void fp16vec_L2sqr_batch_4(
        const uint16_t* x,
        const uint16_t* y0,
        const uint16_t* y1,
        const uint16_t* y2,
        const uint16_t* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    dis0 = fp16vec_L2sqr_ref(x, y0, d);
    dis1 = fp16vec_L2sqr_ref(x, y1, d);
    dis2 = fp16vec_L2sqr_ref(x, y2, d);
    dis3 = fp16vec_L2sqr_ref(x, y3, d);
}

// Scalar-tier fallbacks: forward to FP32 ref path.
float fp16vec_inner_product_native(
        const uint16_t* x,
        const uint16_t* y,
        size_t d) {
    return fp16vec_inner_product_ref(x, y, d);
}

float fp16vec_L2sqr_native(const uint16_t* x, const uint16_t* y, size_t d) {
    return fp16vec_L2sqr_ref(x, y, d);
}

float fp16vec_norm_L2sqr_native(const uint16_t* x, size_t d) {
    return fp16vec_norm_L2sqr_ref(x, d);
}

float bf16vec_inner_product(const uint16_t* x, const uint16_t* y, size_t d) {
    return bf16vec_inner_product_ref(x, y, d);
}

float bf16vec_L2sqr(const uint16_t* x, const uint16_t* y, size_t d) {
    return bf16vec_L2sqr_ref(x, y, d);
}

float bf16vec_norm_L2sqr(const uint16_t* x, size_t d) {
    return bf16vec_norm_L2sqr_ref(x, d);
}

void bf16vec_L2sqr_ny(
        float* dis,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        dis[j] = bf16vec_L2sqr_ref(x, y + j * d, d);
    }
}

void bf16vec_inner_products_ny(
        float* ip,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        ip[j] = bf16vec_inner_product_ref(x, y + j * d, d);
    }
}

void bf16vec_inner_product_batch_4(
        const uint16_t* x,
        const uint16_t* y0,
        const uint16_t* y1,
        const uint16_t* y2,
        const uint16_t* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    dis0 = bf16vec_inner_product_ref(x, y0, d);
    dis1 = bf16vec_inner_product_ref(x, y1, d);
    dis2 = bf16vec_inner_product_ref(x, y2, d);
    dis3 = bf16vec_inner_product_ref(x, y3, d);
}

void bf16vec_L2sqr_batch_4(
        const uint16_t* x,
        const uint16_t* y0,
        const uint16_t* y1,
        const uint16_t* y2,
        const uint16_t* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    dis0 = bf16vec_L2sqr_ref(x, y0, d);
    dis1 = bf16vec_L2sqr_ref(x, y1, d);
    dis2 = bf16vec_L2sqr_ref(x, y2, d);
    dis3 = bf16vec_L2sqr_ref(x, y3, d);
}

#endif // __AVX512F__ && __F16C__ / __AVX2__ && __F16C__

} // namespace faiss
