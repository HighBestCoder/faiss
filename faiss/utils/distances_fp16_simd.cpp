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

#endif // __AVX512BF16__

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
