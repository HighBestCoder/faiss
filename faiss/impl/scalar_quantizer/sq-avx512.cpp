/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

namespace faiss {

namespace scalar_quantizer {

/**********************************************************
 * Codecs
 **********************************************************/

template <>
struct Codec8bit<SIMDLevel::AVX512> : Codec8bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd16float32
    decode_16_components(const uint8_t* code, size_t i) {
        const __m128i c16 = _mm_loadu_si128((__m128i*)(code + i));
        const __m512i i32 = _mm512_cvtepu8_epi32(c16);
        const __m512 f16 = _mm512_cvtepi32_ps(i32);
        const __m512 half_one_255 = _mm512_set1_ps(0.5f / 255.f);
        const __m512 one_255 = _mm512_set1_ps(1.f / 255.f);
        return simd16float32(_mm512_fmadd_ps(f16, one_255, half_one_255));
    }
};

template <>
struct Codec4bit<SIMDLevel::AVX512> : Codec4bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd16float32
    decode_16_components(const uint8_t* code, size_t i) {
        uint64_t c8 = *(uint64_t*)(code + (i >> 1));
        uint64_t mask = 0x0f0f0f0f0f0f0f0f;
        uint64_t c8ev = c8 & mask;
        uint64_t c8od = (c8 >> 4) & mask;

        __m128i c16 =
                _mm_unpacklo_epi8(_mm_set1_epi64x(c8ev), _mm_set1_epi64x(c8od));
        __m256i c8lo = _mm256_cvtepu8_epi32(c16);
        __m256i c8hi = _mm256_cvtepu8_epi32(_mm_srli_si128(c16, 8));
        __m512i i16 = _mm512_castsi256_si512(c8lo);
        i16 = _mm512_inserti32x8(i16, c8hi, 1);
        __m512 f16 = _mm512_cvtepi32_ps(i16);
        const __m512 half_one_255 = _mm512_set1_ps(0.5f / 15.f);
        const __m512 one_255 = _mm512_set1_ps(1.f / 15.f);
        return simd16float32(_mm512_fmadd_ps(f16, one_255, half_one_255));
    }
};

template <>
struct Codec6bit<SIMDLevel::AVX512> : Codec6bit<SIMDLevel::NONE> {
    static FAISS_ALWAYS_INLINE simd16float32
    decode_16_components(const uint8_t* code, size_t i) {
        // pure AVX512 implementation (not necessarily the fastest).
        // see:
        // https://github.com/zilliztech/knowhere/blob/main/thirdparty/faiss/faiss/impl/ScalarQuantizerCodec_avx512.h

        // clang-format off

        // 16 components, 16x6 bit=12 bytes
        const __m128i bit_6v =
                _mm_maskz_loadu_epi8(0b0000111111111111, code + (i >> 2) * 3);
        const __m256i bit_6v_256 = _mm256_broadcast_i32x4(bit_6v);

        // 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F
        // 00          01          02          03
        const __m256i shuffle_mask = _mm256_setr_epi16(
                0xFF00, 0x0100, 0x0201, 0xFF02,
                0xFF03, 0x0403, 0x0504, 0xFF05,
                0xFF06, 0x0706, 0x0807, 0xFF08,
                0xFF09, 0x0A09, 0x0B0A, 0xFF0B);
        const __m256i shuffled = _mm256_shuffle_epi8(bit_6v_256, shuffle_mask);

        // 0: xxxxxxxx xx543210
        // 1: xxxx5432 10xxxxxx
        // 2: xxxxxx54 3210xxxx
        // 3: xxxxxxxx 543210xx
        const __m256i shift_right_v = _mm256_setr_epi16(
                0x0U, 0x6U, 0x4U, 0x2U,
                0x0U, 0x6U, 0x4U, 0x2U,
                0x0U, 0x6U, 0x4U, 0x2U,
                0x0U, 0x6U, 0x4U, 0x2U);
        __m256i shuffled_shifted = _mm256_srlv_epi16(shuffled, shift_right_v);

        // remove unneeded bits
        shuffled_shifted =
                _mm256_and_si256(shuffled_shifted, _mm256_set1_epi16(0x003F));

        // scale
        const __m512 f8 =
                _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(shuffled_shifted));
        const __m512 half_one_255 = _mm512_set1_ps(0.5f / 63.f);
        const __m512 one_255 = _mm512_set1_ps(1.f / 63.f);
        return simd16float32(_mm512_fmadd_ps(f8, one_255, half_one_255));

        // clang-format on
    }
};

/**********************************************************
 * Quantizers (uniform and non-uniform)
 **********************************************************/

template <class Codec>
struct QuantizerTemplate<
        Codec,
        scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
        SIMDLevel::AVX512>
        : QuantizerTemplate<
                  Codec,
                  scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
                  SIMDLevel::NONE> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
                      SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i).f;
        return simd16float32(_mm512_fmadd_ps(
                xi, _mm512_set1_ps(this->vdiff), _mm512_set1_ps(this->vmin)));
    }
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
        SIMDLevel::AVX512>
        : QuantizerTemplate<
                  Codec,
                  scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
                  SIMDLevel::NONE> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
                      SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i).f;
        return simd16float32(_mm512_fmadd_ps(
                xi,
                _mm512_loadu_ps(this->vdiff + i),
                _mm512_loadu_ps(this->vmin + i)));
    }
};

/**********************************************************
 * FP16 Quantizer
 **********************************************************/

template <>
struct QuantizerFP16<SIMDLevel::AVX512> : QuantizerFP16<SIMDLevel::NONE> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i codei = _mm256_loadu_si256((const __m256i*)(code + 2 * i));
        return simd16float32(_mm512_cvtph_ps(codei));
    }
};

/**********************************************************
 * BF16 Quantizer
 **********************************************************/

template <>
struct QuantizerBF16<SIMDLevel::AVX512> : QuantizerBF16<SIMDLevel::NONE> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i code_256i = _mm256_loadu_si256((const __m256i*)(code + 2 * i));
        __m512i code_512i = _mm512_cvtepu16_epi32(code_256i);
        code_512i = _mm512_slli_epi32(code_512i, 16);
        return simd16float32(_mm512_castsi512_ps(code_512i));
    }
};

/**********************************************************
 * 8bit Direct Quantizer
 **********************************************************/

template <>
struct Quantizer8bitDirect<SIMDLevel::AVX512>
        : Quantizer8bitDirect<SIMDLevel::NONE> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m128i x16 = _mm_loadu_si128((__m128i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi32(x16);             // 16 * int32
        return simd16float32(_mm512_cvtepi32_ps(y16));       // 16 * float32
    }
};

/**********************************************************
 * 8bit Direct Signed Quantizer
 **********************************************************/

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::AVX512>
        : Quantizer8bitDirectSigned<SIMDLevel::NONE> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<SIMDLevel::NONE>(d, trained) {
        assert(d % 16 == 0);
    }

    FAISS_ALWAYS_INLINE simd16float32
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m128i x16 = _mm_loadu_si128((__m128i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi32(x16);             // 16 * int32
        __m512i c16 = _mm512_set1_epi32(128);
        __m512i z16 = _mm512_sub_epi32(y16, c16); // subtract 128 from all lanes
        return simd16float32(_mm512_cvtepi32_ps(z16)); // 16 * float32
    }
};

/**********************************************************
 * Similarities (L2 and IP)
 **********************************************************/

template <>
struct SimilarityL2<SIMDLevel::AVX512> {
    static constexpr int simdwidth = 16;
    static constexpr SIMDLevel simd_level = SIMDLevel::AVX512;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y), yi(nullptr) {}

    simd16float32 accu16;

    FAISS_ALWAYS_INLINE void begin_16() {
        accu16.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_16_components(simd16float32 x) {
        simd16float32 yiv(yi);
        yi += 16;
        simd16float32 tmp = yiv - x;
        accu16 = accu16 + tmp * tmp;
    }

    FAISS_ALWAYS_INLINE void add_16_components_2(
            simd16float32 x,
            simd16float32 y_2) {
        simd16float32 tmp = y_2 - x;
        accu16 = accu16 + tmp * tmp;
    }

    FAISS_ALWAYS_INLINE float result_16() {
        return horizontal_add(accu16);
    }
};

template <>
struct SimilarityIP<SIMDLevel::AVX512> {
    static constexpr int simdwidth = 16;
    static constexpr SIMDLevel simd_level = SIMDLevel::AVX512;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    explicit SimilarityIP(const float* y) : y(y), yi(nullptr) {}

    simd16float32 accu16;

    FAISS_ALWAYS_INLINE void begin_16() {
        accu16.clear();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_16_components(simd16float32 x) {
        simd16float32 yiv(yi);
        yi += 16;
        accu16 = accu16 + yiv * x;
    }

    FAISS_ALWAYS_INLINE void add_16_components_2(
            simd16float32 x1,
            simd16float32 x2) {
        accu16 = accu16 + x1 * x2;
    }

    FAISS_ALWAYS_INLINE float result_16() {
        return horizontal_add(accu16);
    }
};

/**********************************************************
 * Distance Computers
 **********************************************************/

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512>
        : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_16();
        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 xi = quant.reconstruct_16_components(code, i);
            sim.add_16_components(xi);
        }
        return sim.result_16();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_16();
        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 x1 = quant.reconstruct_16_components(code1, i);
            simd16float32 x2 = quant.reconstruct_16_components(code2, i);
            sim.add_16_components_2(x1, x2);
        }
        return sim.result_16();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }

    // Optimization: parallel multi-vector distance with independent
    // accumulators. The default DistanceComputer::distances_batch_{4,8}
    // implementations call operator() N times sequentially; each call has only
    // a single dependency chain on its accu16 register, so the FMA pipeline
    // (depth ~4-5 cycles on Ice Lake) is fully exposed and IPC ~ 0.2.
    //
    // By unrolling N codes against the same query in a single loop with N
    // independent accumulators, the FMAs from different codes can be issued
    // back-to-back, hiding the FMA latency and reaching ~1 FMA/cycle. This is
    // the same trick already used by fvec_inner_product_batch_8 in
    // distances_avx512.cpp for IndexFlat; we replicate it here for the
    // ScalarQuantizer DistanceComputer (SQfp16 etc.) which is the HNSW hot
    // path for SQ-quantized indexes.
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        const uint8_t* c0 = codes + idx0 * code_size;
        const uint8_t* c1 = codes + idx1 * code_size;
        const uint8_t* c2 = codes + idx2 * code_size;
        const uint8_t* c3 = codes + idx3 * code_size;

        Similarity sim0(q), sim1(q), sim2(q), sim3(q);
        sim0.begin_16();
        sim1.begin_16();
        sim2.begin_16();
        sim3.begin_16();

        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 v0 = quant.reconstruct_16_components(c0, i);
            simd16float32 v1 = quant.reconstruct_16_components(c1, i);
            simd16float32 v2 = quant.reconstruct_16_components(c2, i);
            simd16float32 v3 = quant.reconstruct_16_components(c3, i);
            sim0.add_16_components(v0);
            sim1.add_16_components(v1);
            sim2.add_16_components(v2);
            sim3.add_16_components(v3);
        }
        dis0 = sim0.result_16();
        dis1 = sim1.result_16();
        dis2 = sim2.result_16();
        dis3 = sim3.result_16();
    }

    void distances_batch_8(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            const idx_t idx4,
            const idx_t idx5,
            const idx_t idx6,
            const idx_t idx7,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3,
            float& dis4,
            float& dis5,
            float& dis6,
            float& dis7) override {
        const uint8_t* c0 = codes + idx0 * code_size;
        const uint8_t* c1 = codes + idx1 * code_size;
        const uint8_t* c2 = codes + idx2 * code_size;
        const uint8_t* c3 = codes + idx3 * code_size;
        const uint8_t* c4 = codes + idx4 * code_size;
        const uint8_t* c5 = codes + idx5 * code_size;
        const uint8_t* c6 = codes + idx6 * code_size;
        const uint8_t* c7 = codes + idx7 * code_size;

        Similarity s0(q), s1(q), s2(q), s3(q);
        Similarity s4(q), s5(q), s6(q), s7(q);
        s0.begin_16();
        s1.begin_16();
        s2.begin_16();
        s3.begin_16();
        s4.begin_16();
        s5.begin_16();
        s6.begin_16();
        s7.begin_16();

        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 v0 = quant.reconstruct_16_components(c0, i);
            simd16float32 v1 = quant.reconstruct_16_components(c1, i);
            simd16float32 v2 = quant.reconstruct_16_components(c2, i);
            simd16float32 v3 = quant.reconstruct_16_components(c3, i);
            simd16float32 v4 = quant.reconstruct_16_components(c4, i);
            simd16float32 v5 = quant.reconstruct_16_components(c5, i);
            simd16float32 v6 = quant.reconstruct_16_components(c6, i);
            simd16float32 v7 = quant.reconstruct_16_components(c7, i);
            s0.add_16_components(v0);
            s1.add_16_components(v1);
            s2.add_16_components(v2);
            s3.add_16_components(v3);
            s4.add_16_components(v4);
            s5.add_16_components(v5);
            s6.add_16_components(v6);
            s7.add_16_components(v7);
        }
        dis0 = s0.result_16();
        dis1 = s1.result_16();
        dis2 = s2.result_16();
        dis3 = s3.result_16();
        dis4 = s4.result_16();
        dis5 = s5.result_16();
        dis6 = s6.result_16();
        dis7 = s7.result_16();
    }
};

/**********************************************************
 * O18: VNNI int8 IP for SQ8_vnni (QT_8bit_vnni, NON_UNIFORM, IP only)
 *
 * Opt-in qtype "SQ8_vnni" — encoding is bit-identical to QT_8bit, but the
 * AVX512 IP DistanceComputer uses VNNI vpdpbusd with a per-query int8-quantized
 * query vector instead of the default fp32 reconstruct + fmadd path. Faster
 * (+30~50% QPS at d=768) but introduces query-side quantization noise that
 * costs a few points of recall on long-traversal HNSW searches; users opt in
 * by selecting "SQ8_vnni" in the index factory.
 *
 * Math (NON_UNIFORM SQ8, IP):
 *   y_recon[i] = (c[i] + 0.5) / 255 * vdiff[i] + vmin[i]
 *   <x, y> = sum_i x[i] * c[i] * vdiff[i]/255           (the dot product)
 *          + sum_i x[i] * (0.5 * vdiff[i]/255 + vmin[i]) (constant per query)
 *
 * In set_query we precompute:
 *   w[i]  = x[i] * vdiff[i] / 255
 *   bias  = sum_i x[i] * (0.5 * vdiff[i]/255 + vmin[i])
 *   s_q   = max_i |w[i]| / 127
 *   q8[i] = saturate_int8(round(w[i] / s_q))
 *
 * Then per code: dot_int = vpdpbusd_reduce(c, q8); return s_q * dot_int + bias.
 *
 * Only IP is implemented here. L2 keeps the existing fp32 path (would need
 * dpwssd on i16 diffs and a different correction; not in this change).
 **********************************************************/
struct DCSQ8VNNI_AVX512 : SQDistanceComputer {
    using Quantizer = QuantizerTemplate<
            Codec8bit<SIMDLevel::AVX512>,
            scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
            SIMDLevel::AVX512>;
    using Similarity = SimilarityIP<SIMDLevel::AVX512>;
    using Sim = Similarity;

    Quantizer quant;

    // Per-query precomputed state (mutable so query_to_code can stay const).
    mutable std::vector<int8_t> q8;
    mutable float scale_q = 1.0f;
    mutable float bias_q = 0.0f;
    mutable bool degenerate = false; // all w[i] == 0 -> dot is 0, return bias

    DCSQ8VNNI_AVX512(size_t d, const std::vector<float>& trained)
            : quant(d, trained), q8(d) {}

    // ---- fallback fp32 paths for the rare "no query set" / symmetric cases.

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_16();
        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 xi = quant.reconstruct_16_components(code, i);
            sim.add_16_components(xi);
        }
        return sim.result_16();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_16();
        for (size_t i = 0; i < quant.d; i += 16) {
            simd16float32 x1 = quant.reconstruct_16_components(code1, i);
            simd16float32 x2 = quant.reconstruct_16_components(code2, i);
            sim.add_16_components_2(x1, x2);
        }
        return sim.result_16();
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    // ---- query-side int8 quantization.

    void set_query(const float* x) final {
        q = x;
        const size_t d = quant.d;
        const float* vmin = quant.vmin;
        const float* vdiff = quant.vdiff;
        const float inv255 = 1.0f / 255.0f;

        // Pass 1: compute w[i] = x[i] * vdiff[i] / 255, bias = sum x[i] *
        // (0.5*vdiff[i]/255 + vmin[i]), and max|w|. Single fused loop.
        std::vector<float> w(d);
        __m512 vmax = _mm512_setzero_ps();
        __m512 vbias = _mm512_setzero_ps();
        const __m512 v_inv255 = _mm512_set1_ps(inv255);
        const __m512 v_half_inv255 = _mm512_set1_ps(0.5f * inv255);
        for (size_t i = 0; i < d; i += 16) {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vd = _mm512_loadu_ps(vdiff + i);
            __m512 vm = _mm512_loadu_ps(vmin + i);
            __m512 vw = _mm512_mul_ps(vx, _mm512_mul_ps(vd, v_inv255));
            _mm512_storeu_ps(w.data() + i, vw);
            // bias term: x[i] * (0.5 * vdiff[i] / 255 + vmin[i])
            __m512 vt = _mm512_fmadd_ps(vd, v_half_inv255, vm);
            vbias = _mm512_fmadd_ps(vx, vt, vbias);
            // max |w|
            __m512 vabs = _mm512_abs_ps(vw);
            vmax = _mm512_max_ps(vmax, vabs);
        }
        bias_q = _mm512_reduce_add_ps(vbias);
        const float maxw = _mm512_reduce_max_ps(vmax);

        if (maxw == 0.0f) {
            // Degenerate query: all w are zero -> dot is always 0; return bias.
            scale_q = 0.0f;
            degenerate = true;
            std::fill(q8.begin(), q8.end(), int8_t{0});
            return;
        }
        degenerate = false;
        scale_q = maxw / 127.0f;
        const float invs = 127.0f / maxw;
        const __m512 v_invs = _mm512_set1_ps(invs);

        // Pass 2: q8[i] = sat_int8(round(w[i] * invs)).
        for (size_t i = 0; i < d; i += 16) {
            __m512 vw = _mm512_loadu_ps(w.data() + i);
            __m512 vs = _mm512_mul_ps(vw, v_invs);
            // round to nearest, saturate to int8 via cvtsepi32_epi8.
            __m512i vi = _mm512_cvtps_epi32(vs); // rounds to nearest (default)
            __m128i vi8 = _mm512_cvtsepi32_epi8(vi); // saturating pack
            _mm_storeu_si128(
                    reinterpret_cast<__m128i*>(q8.data() + i), vi8);
        }
    }

    // ---- core: VNNI dot product over the whole code, returns float distance.

    FAISS_ALWAYS_INLINE int vnni_dot(const uint8_t* code) const {
        const size_t d = quant.d;
        __m512i acc = _mm512_setzero_si512();
        size_t i = 0;
        for (; i + 64 <= d; i += 64) {
            __m512i c = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(code + i));
            __m512i q = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(q8.data() + i));
            // vpdpbusd: u8 * s8 -> i32, accumulating 4 products per lane.
            acc = _mm512_dpbusd_epi32(acc, c, q);
        }
        if (i + 32 <= d) {
            // 32-byte tail (d % 16 == 0 by SIMD requirement, so tail is 0/16/32/48).
            const __mmask64 m32 = (__mmask64{1} << 32) - 1;
            __m512i c = _mm512_maskz_loadu_epi8(m32, code + i);
            __m512i q = _mm512_maskz_loadu_epi8(m32, q8.data() + i);
            acc = _mm512_dpbusd_epi32(acc, c, q);
            i += 32;
        }
        if (i < d) {
            // 16-byte tail.
            const __mmask64 m16 = (__mmask64{1} << 16) - 1;
            __m512i c = _mm512_maskz_loadu_epi8(m16, code + i);
            __m512i q = _mm512_maskz_loadu_epi8(m16, q8.data() + i);
            acc = _mm512_dpbusd_epi32(acc, c, q);
        }
        return _mm512_reduce_add_epi32(acc);
    }

    float query_to_code(const uint8_t* code) const final {
        if (degenerate) {
            return bias_q;
        }
        int dot = vnni_dot(code);
        return scale_q * static_cast<float>(dot) + bias_q;
    }

    // ---- batch APIs: N independent VNNI accumulators sharing the query.

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        if (degenerate) {
            dis0 = dis1 = dis2 = dis3 = bias_q;
            return;
        }
        const uint8_t* c0 = codes + idx0 * code_size;
        const uint8_t* c1 = codes + idx1 * code_size;
        const uint8_t* c2 = codes + idx2 * code_size;
        const uint8_t* c3 = codes + idx3 * code_size;

        const size_t d = quant.d;
        __m512i a0 = _mm512_setzero_si512();
        __m512i a1 = _mm512_setzero_si512();
        __m512i a2 = _mm512_setzero_si512();
        __m512i a3 = _mm512_setzero_si512();

        size_t i = 0;
        for (; i + 64 <= d; i += 64) {
            __m512i q = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(q8.data() + i));
            a0 = _mm512_dpbusd_epi32(
                    a0,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c0 + i)),
                    q);
            a1 = _mm512_dpbusd_epi32(
                    a1,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c1 + i)),
                    q);
            a2 = _mm512_dpbusd_epi32(
                    a2,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c2 + i)),
                    q);
            a3 = _mm512_dpbusd_epi32(
                    a3,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c3 + i)),
                    q);
        }
        for (; i < d; i += 16) {
            const __mmask64 m = (__mmask64{1} << 16) - 1;
            __m512i q = _mm512_maskz_loadu_epi8(m, q8.data() + i);
            a0 = _mm512_dpbusd_epi32(
                    a0, _mm512_maskz_loadu_epi8(m, c0 + i), q);
            a1 = _mm512_dpbusd_epi32(
                    a1, _mm512_maskz_loadu_epi8(m, c1 + i), q);
            a2 = _mm512_dpbusd_epi32(
                    a2, _mm512_maskz_loadu_epi8(m, c2 + i), q);
            a3 = _mm512_dpbusd_epi32(
                    a3, _mm512_maskz_loadu_epi8(m, c3 + i), q);
        }

        dis0 = scale_q * float(_mm512_reduce_add_epi32(a0)) + bias_q;
        dis1 = scale_q * float(_mm512_reduce_add_epi32(a1)) + bias_q;
        dis2 = scale_q * float(_mm512_reduce_add_epi32(a2)) + bias_q;
        dis3 = scale_q * float(_mm512_reduce_add_epi32(a3)) + bias_q;
    }

    void distances_batch_8(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            const idx_t idx4,
            const idx_t idx5,
            const idx_t idx6,
            const idx_t idx7,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3,
            float& dis4,
            float& dis5,
            float& dis6,
            float& dis7) override {
        if (degenerate) {
            dis0 = dis1 = dis2 = dis3 = dis4 = dis5 = dis6 = dis7 = bias_q;
            return;
        }
        const uint8_t* c0 = codes + idx0 * code_size;
        const uint8_t* c1 = codes + idx1 * code_size;
        const uint8_t* c2 = codes + idx2 * code_size;
        const uint8_t* c3 = codes + idx3 * code_size;
        const uint8_t* c4 = codes + idx4 * code_size;
        const uint8_t* c5 = codes + idx5 * code_size;
        const uint8_t* c6 = codes + idx6 * code_size;
        const uint8_t* c7 = codes + idx7 * code_size;

        const size_t d = quant.d;
        __m512i a0 = _mm512_setzero_si512();
        __m512i a1 = _mm512_setzero_si512();
        __m512i a2 = _mm512_setzero_si512();
        __m512i a3 = _mm512_setzero_si512();
        __m512i a4 = _mm512_setzero_si512();
        __m512i a5 = _mm512_setzero_si512();
        __m512i a6 = _mm512_setzero_si512();
        __m512i a7 = _mm512_setzero_si512();

        size_t i = 0;
        for (; i + 64 <= d; i += 64) {
            __m512i q = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(q8.data() + i));
            a0 = _mm512_dpbusd_epi32(
                    a0,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c0 + i)),
                    q);
            a1 = _mm512_dpbusd_epi32(
                    a1,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c1 + i)),
                    q);
            a2 = _mm512_dpbusd_epi32(
                    a2,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c2 + i)),
                    q);
            a3 = _mm512_dpbusd_epi32(
                    a3,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c3 + i)),
                    q);
            a4 = _mm512_dpbusd_epi32(
                    a4,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c4 + i)),
                    q);
            a5 = _mm512_dpbusd_epi32(
                    a5,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c5 + i)),
                    q);
            a6 = _mm512_dpbusd_epi32(
                    a6,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c6 + i)),
                    q);
            a7 = _mm512_dpbusd_epi32(
                    a7,
                    _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(c7 + i)),
                    q);
        }
        for (; i < d; i += 16) {
            const __mmask64 m = (__mmask64{1} << 16) - 1;
            __m512i q = _mm512_maskz_loadu_epi8(m, q8.data() + i);
            a0 = _mm512_dpbusd_epi32(
                    a0, _mm512_maskz_loadu_epi8(m, c0 + i), q);
            a1 = _mm512_dpbusd_epi32(
                    a1, _mm512_maskz_loadu_epi8(m, c1 + i), q);
            a2 = _mm512_dpbusd_epi32(
                    a2, _mm512_maskz_loadu_epi8(m, c2 + i), q);
            a3 = _mm512_dpbusd_epi32(
                    a3, _mm512_maskz_loadu_epi8(m, c3 + i), q);
            a4 = _mm512_dpbusd_epi32(
                    a4, _mm512_maskz_loadu_epi8(m, c4 + i), q);
            a5 = _mm512_dpbusd_epi32(
                    a5, _mm512_maskz_loadu_epi8(m, c5 + i), q);
            a6 = _mm512_dpbusd_epi32(
                    a6, _mm512_maskz_loadu_epi8(m, c6 + i), q);
            a7 = _mm512_dpbusd_epi32(
                    a7, _mm512_maskz_loadu_epi8(m, c7 + i), q);
        }

        dis0 = scale_q * float(_mm512_reduce_add_epi32(a0)) + bias_q;
        dis1 = scale_q * float(_mm512_reduce_add_epi32(a1)) + bias_q;
        dis2 = scale_q * float(_mm512_reduce_add_epi32(a2)) + bias_q;
        dis3 = scale_q * float(_mm512_reduce_add_epi32(a3)) + bias_q;
        dis4 = scale_q * float(_mm512_reduce_add_epi32(a4)) + bias_q;
        dis5 = scale_q * float(_mm512_reduce_add_epi32(a5)) + bias_q;
        dis6 = scale_q * float(_mm512_reduce_add_epi32(a6)) + bias_q;
        dis7 = scale_q * float(_mm512_reduce_add_epi32(a7)) + bias_q;
    }
};

// Factories declared in sq-dispatch.h. Defined here so the dispatch headers
// (also included by non-AVX512 TUs) don't need the full DCSQ8VNNI_AVX512
// definition or the IVFSQScannerIP<...> instantiation.
SQDistanceComputer* make_DCSQ8VNNI_AVX512(
        size_t d,
        const std::vector<float>& trained) {
    return new DCSQ8VNNI_AVX512(d, trained);
}

InvertedListScanner* make_IVFSQ8VNNI_AVX512_scanner(
        int d,
        const std::vector<float>& trained,
        size_t code_size,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    return new IVFSQScannerIP<DCSQ8VNNI_AVX512>(
            d, trained, code_size, store_pairs, sel, by_residual);
}

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::AVX512>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        // compute 16 lanes of 32-bit products (16-bytes) at once for
        // the supported metrics
        __m512i accu = _mm512_setzero_si512();
        constexpr int kLanes = 16;
        for (int i = 0; i < d; i += kLanes) {
            __m128i c1 = _mm_loadu_si128((__m128i*)(code1 + i));
            __m128i c2 = _mm_loadu_si128((__m128i*)(code2 + i));
            __m512i c1i = _mm512_cvtepu8_epi32(c1);
            __m512i c2i = _mm512_cvtepu8_epi32(c2);

            __m512i v;
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                v = _mm512_mullo_epi32(c1i, c2i);
            } else {
                __m512i diff = _mm512_sub_epi32(c1i, c2i);
                v = _mm512_mullo_epi32(diff, diff);
            }
            accu = _mm512_add_epi32(accu, v);
        }
        return _mm512_reduce_add_epi32(accu);
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

#ifdef COMPILE_SIMD_AVX512_SPR
// AVX512_SPR: Sapphire Rapids is a superset of AVX512. Until dedicated
// SPR specializations are written (e.g. AVX512_FP16 native fmadd), reuse
// the AVX512 implementations by inheritance. This produces distinct
// QuantizerXxx<SIMDLevel::AVX512_SPR> symbols so sq_select_quantizer
// /sq_select_distance_computer / sq_select_InvertedListScanner can be
// instantiated for AVX512_SPR via the same sq-dispatch.h template machinery.

template <>
struct Codec8bit<SIMDLevel::AVX512_SPR> : Codec8bit<SIMDLevel::AVX512> {};

template <>
struct Codec4bit<SIMDLevel::AVX512_SPR> : Codec4bit<SIMDLevel::AVX512> {};

template <>
struct Codec6bit<SIMDLevel::AVX512_SPR> : Codec6bit<SIMDLevel::AVX512> {};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
        SIMDLevel::AVX512_SPR>
        : QuantizerTemplate<
                  Codec,
                  scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
                  SIMDLevel::AVX512> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      scalar_quantizer::QuantizerTemplateScaling::UNIFORM,
                      SIMDLevel::AVX512>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate<
        Codec,
        scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
        SIMDLevel::AVX512_SPR>
        : QuantizerTemplate<
                  Codec,
                  scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
                  SIMDLevel::AVX512> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      scalar_quantizer::QuantizerTemplateScaling::NON_UNIFORM,
                      SIMDLevel::AVX512>(d, trained) {}
};

template <>
struct QuantizerFP16<SIMDLevel::AVX512_SPR>
        : QuantizerFP16<SIMDLevel::AVX512> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<SIMDLevel::AVX512>(d, trained) {}
};

template <>
struct QuantizerBF16<SIMDLevel::AVX512_SPR>
        : QuantizerBF16<SIMDLevel::AVX512> {
    QuantizerBF16(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<SIMDLevel::AVX512>(d, trained) {}
};

template <>
struct Quantizer8bitDirect<SIMDLevel::AVX512_SPR>
        : Quantizer8bitDirect<SIMDLevel::AVX512> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<SIMDLevel::AVX512>(d, trained) {}
};

template <>
struct Quantizer8bitDirectSigned<SIMDLevel::AVX512_SPR>
        : Quantizer8bitDirectSigned<SIMDLevel::AVX512> {
    Quantizer8bitDirectSigned(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<SIMDLevel::AVX512>(d, trained) {}
};

template <>
struct SimilarityL2<SIMDLevel::AVX512_SPR> : SimilarityL2<SIMDLevel::AVX512> {
    static constexpr SIMDLevel simd_level = SIMDLevel::AVX512_SPR;
    explicit SimilarityL2(const float* y) : SimilarityL2<SIMDLevel::AVX512>(y) {}
};

template <>
struct SimilarityIP<SIMDLevel::AVX512_SPR> : SimilarityIP<SIMDLevel::AVX512> {
    static constexpr SIMDLevel simd_level = SIMDLevel::AVX512_SPR;
    explicit SimilarityIP(const float* y) : SimilarityIP<SIMDLevel::AVX512>(y) {}
};

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512_SPR>
        : DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512> {
    using Sim = Similarity;
    DCTemplate(size_t d, const std::vector<float>& trained)
            : DCTemplate<Quantizer, Similarity, SIMDLevel::AVX512>(d, trained) {}
};

// SPR DistanceComputerByte: uses VNNI int8 (vpdpbusd) for IP and VNNI int16
// (vpdpwssd) for L2. Both are bit-identical to the AVX512 mullo/sub path
// because all arithmetic is exact integer.
//
// IP for u8 codes: <a,b> = sum a_i*b_i, a_i,b_i in [0,255]. dpbusd takes
// (u8, s8). Reinterpret b' = (b - 128) as s8; then a*b' = a*b - 128*a.
// Correction: dot_unsigned = dot_with_b' + 128 * sum(a). The sum(a) term
// is computed concurrently with a second dpbusd against an all-ones s8.
//
// L2: zero-extend each 32 u8 to 32 u16, take signed diff (range -255..255
// fits i16), feed dpwssd(acc, diff, diff) → 32 i16 pair-products into
// 16 i32 lanes per iteration.
//
// Dispatcher guarantees d % 32 == 0 (sq-dispatch.h QT_8bit_direct guard);
// IP main loop steps 64, with a single 32-byte mask-load tail.
template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::AVX512_SPR>
        : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        if constexpr (Sim::metric_type == METRIC_INNER_PRODUCT) {
            __m512i acc = _mm512_setzero_si512();
            __m512i sum_c1 = _mm512_setzero_si512();
            const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
            const __m512i ones = _mm512_set1_epi8(1);
            int i = 0;
            for (; i + 64 <= d; i += 64) {
                __m512i c1 = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(code1 + i));
                __m512i c2 = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(code2 + i));
                __m512i c2s = _mm512_xor_si512(c2, sign_flip);
                acc = _mm512_dpbusd_epi32(acc, c1, c2s);
                sum_c1 = _mm512_dpbusd_epi32(sum_c1, c1, ones);
            }
            // 32-byte tail (d % 32 == 0 → tail is 0 or 32 bytes).
            if (i < d) {
                const __mmask64 m = (__mmask64{1} << 32) - 1;
                __m512i c1 = _mm512_maskz_loadu_epi8(m, code1 + i);
                __m512i c2 = _mm512_maskz_loadu_epi8(m, code2 + i);
                __m512i c2s = _mm512_xor_si512(c2, sign_flip);
                acc = _mm512_dpbusd_epi32(acc, c1, c2s);
                sum_c1 = _mm512_dpbusd_epi32(sum_c1, c1, ones);
            }
            int dot = _mm512_reduce_add_epi32(acc);
            int sc1 = _mm512_reduce_add_epi32(sum_c1);
            return dot + 128 * sc1;
        } else {
            // L2 via VNNI int16 dpwssd.
            __m512i acc = _mm512_setzero_si512();
            int i = 0;
            for (; i + 32 <= d; i += 32) {
                __m256i c1_8 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(code1 + i));
                __m256i c2_8 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(code2 + i));
                __m512i c1_16 = _mm512_cvtepu8_epi16(c1_8);
                __m512i c2_16 = _mm512_cvtepu8_epi16(c2_8);
                __m512i diff = _mm512_sub_epi16(c1_16, c2_16);
                acc = _mm512_dpwssd_epi32(acc, diff, diff);
            }
            return _mm512_reduce_add_epi32(acc);
        }
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

#endif // COMPILE_SIMD_AVX512_SPR

} // namespace scalar_quantizer
} // namespace faiss

#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX512
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>
#undef THE_LEVEL_TO_DISPATCH

#ifdef COMPILE_SIMD_AVX512_SPR

// AVX512_SPR: Sapphire Rapids is a superset of AVX512. Forward to the AVX512
// implementation until dedicated SPR specializations are written. We can't
// just `#include sq-dispatch.h` again with THE_LEVEL_TO_DISPATCH=AVX512_SPR
// because that header has `#pragma once` (and also defines helper templates
// at namespace scope that must not be redefined). Same forwarder shape as
// the ARM_SVE block in sq-neon.cpp.

namespace faiss {
namespace scalar_quantizer {

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<SIMDLevel::AVX512_SPR>(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    return sq_select_quantizer<SIMDLevel::AVX512>(qtype, d, trained);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
SQDistanceComputer* sq_select_distance_computer<SIMDLevel::AVX512_SPR>(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    return sq_select_distance_computer<SIMDLevel::AVX512>(
            metric, qtype, d, trained);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
InvertedListScanner* sq_select_InvertedListScanner<SIMDLevel::AVX512_SPR>(
        QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    return sq_select_InvertedListScanner<SIMDLevel::AVX512>(
            qtype,
            mt,
            d,
            code_size,
            trained,
            quantizer,
            store_pairs,
            sel,
            by_residual);
}

} // namespace scalar_quantizer
} // namespace faiss

#endif // COMPILE_SIMD_AVX512_SPR

#endif // COMPILE_SIMD_AVX512
