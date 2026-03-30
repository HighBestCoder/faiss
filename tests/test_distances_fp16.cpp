/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <faiss/utils/bf16.h>
#include <faiss/utils/distances_fp16.h>
#include <faiss/utils/fp16.h>

/*********************************************************
 * Helper: generate random float data and encode to FP16/BF16
 *********************************************************/

namespace {

std::vector<uint16_t> make_fp16_vectors(
        size_t n,
        size_t d,
        std::default_random_engine& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> out(n * d);
    for (size_t i = 0; i < n * d; i++) {
        out[i] = faiss::encode_fp16(dist(rng));
    }
    return out;
}

std::vector<uint16_t> make_bf16_vectors(
        size_t n,
        size_t d,
        std::default_random_engine& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> out(n * d);
    for (size_t i = 0; i < n * d; i++) {
        out[i] = faiss::encode_bf16(dist(rng));
    }
    return out;
}

// Scalar reference implementations for verification
float ref_fp16_inner_product(const uint16_t* x, const uint16_t* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        res += faiss::decode_fp16(x[i]) * faiss::decode_fp16(y[i]);
    }
    return res;
}

float ref_fp16_L2sqr(const uint16_t* x, const uint16_t* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float diff = faiss::decode_fp16(x[i]) - faiss::decode_fp16(y[i]);
        res += diff * diff;
    }
    return res;
}

float ref_fp16_norm_L2sqr(const uint16_t* x, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float v = faiss::decode_fp16(x[i]);
        res += v * v;
    }
    return res;
}

float ref_bf16_inner_product(const uint16_t* x, const uint16_t* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        res += faiss::decode_bf16(x[i]) * faiss::decode_bf16(y[i]);
    }
    return res;
}

float ref_bf16_L2sqr(const uint16_t* x, const uint16_t* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float diff = faiss::decode_bf16(x[i]) - faiss::decode_bf16(y[i]);
        res += diff * diff;
    }
    return res;
}

float ref_bf16_norm_L2sqr(const uint16_t* x, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float v = faiss::decode_bf16(x[i]);
        res += v * v;
    }
    return res;
}

} // anonymous namespace

/*********************************************************
 * FP16 basic distance tests
 *********************************************************/

class FP16DistanceTest : public ::testing::TestWithParam<size_t> {};

TEST_P(FP16DistanceTest, InnerProduct) {
    size_t d = GetParam();
    std::default_random_engine rng(42);
    auto x = make_fp16_vectors(1, d, rng);
    auto y = make_fp16_vectors(1, d, rng);

    float result = faiss::fp16vec_inner_product(x.data(), y.data(), d);
    float ref = ref_fp16_inner_product(x.data(), y.data(), d);

    // FP16 accumulation can have small rounding differences vs scalar
    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-5 + 1e-6)
            << "FP16 inner product mismatch at d=" << d;
}

TEST_P(FP16DistanceTest, L2sqr) {
    size_t d = GetParam();
    std::default_random_engine rng(42);
    auto x = make_fp16_vectors(1, d, rng);
    auto y = make_fp16_vectors(1, d, rng);

    float result = faiss::fp16vec_L2sqr(x.data(), y.data(), d);
    float ref = ref_fp16_L2sqr(x.data(), y.data(), d);

    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-5 + 1e-6)
            << "FP16 L2sqr mismatch at d=" << d;
}

TEST_P(FP16DistanceTest, NormL2sqr) {
    size_t d = GetParam();
    std::default_random_engine rng(42);
    auto x = make_fp16_vectors(1, d, rng);

    float result = faiss::fp16vec_norm_L2sqr(x.data(), d);
    float ref = ref_fp16_norm_L2sqr(x.data(), d);

    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-5 + 1e-6)
            << "FP16 norm_L2sqr mismatch at d=" << d;
}

INSTANTIATE_TEST_SUITE_P(
        VariousDimensions,
        FP16DistanceTest,
        // Test various dimensions including:
        // - below 8 (scalar tail only)
        // - exact multiples of 8 (AVX2 SIMD only)
        // - non-multiples of 8 (AVX2 SIMD + tail)
        // - exact multiples of 16 (AVX-512 SIMD only)
        // - 16+8 (AVX-512 main + AVX2 tail)
        // - 16+8+k (AVX-512 main + AVX2 tail + scalar tail)
        // - typical vector search dimensions
        ::testing::Values(
                1,
                3,
                7,
                8,
                9,
                15,
                16,
                17,
                24,
                25,
                31,
                32,
                48,
                64,
                128,
                256,
                512));

/*********************************************************
 * FP16 batch-4 tests
 *********************************************************/

TEST(FP16Batch4Test, InnerProductBatch4) {
    std::default_random_engine rng(123);

    for (size_t d : {8, 16, 17, 24, 25, 32, 48, 64, 128, 7, 15, 33}) {
        auto x = make_fp16_vectors(1, d, rng);
        auto y = make_fp16_vectors(4, d, rng);

        float dis0, dis1, dis2, dis3;
        faiss::fp16vec_inner_product_batch_4(
                x.data(),
                y.data() + 0 * d,
                y.data() + 1 * d,
                y.data() + 2 * d,
                y.data() + 3 * d,
                d,
                dis0,
                dis1,
                dis2,
                dis3);

        float ref0 = ref_fp16_inner_product(x.data(), y.data() + 0 * d, d);
        float ref1 = ref_fp16_inner_product(x.data(), y.data() + 1 * d, d);
        float ref2 = ref_fp16_inner_product(x.data(), y.data() + 2 * d, d);
        float ref3 = ref_fp16_inner_product(x.data(), y.data() + 3 * d, d);

        EXPECT_NEAR(dis0, ref0, std::abs(ref0) * 1e-5 + 1e-6)
                << "d=" << d << " idx=0";
        EXPECT_NEAR(dis1, ref1, std::abs(ref1) * 1e-5 + 1e-6)
                << "d=" << d << " idx=1";
        EXPECT_NEAR(dis2, ref2, std::abs(ref2) * 1e-5 + 1e-6)
                << "d=" << d << " idx=2";
        EXPECT_NEAR(dis3, ref3, std::abs(ref3) * 1e-5 + 1e-6)
                << "d=" << d << " idx=3";
    }
}

TEST(FP16Batch4Test, L2sqrBatch4) {
    std::default_random_engine rng(123);

    for (size_t d : {8, 16, 17, 24, 25, 32, 48, 64, 128, 7, 15, 33}) {
        auto x = make_fp16_vectors(1, d, rng);
        auto y = make_fp16_vectors(4, d, rng);

        float dis0, dis1, dis2, dis3;
        faiss::fp16vec_L2sqr_batch_4(
                x.data(),
                y.data() + 0 * d,
                y.data() + 1 * d,
                y.data() + 2 * d,
                y.data() + 3 * d,
                d,
                dis0,
                dis1,
                dis2,
                dis3);

        float ref0 = ref_fp16_L2sqr(x.data(), y.data() + 0 * d, d);
        float ref1 = ref_fp16_L2sqr(x.data(), y.data() + 1 * d, d);
        float ref2 = ref_fp16_L2sqr(x.data(), y.data() + 2 * d, d);
        float ref3 = ref_fp16_L2sqr(x.data(), y.data() + 3 * d, d);

        EXPECT_NEAR(dis0, ref0, std::abs(ref0) * 1e-5 + 1e-6)
                << "d=" << d << " idx=0";
        EXPECT_NEAR(dis1, ref1, std::abs(ref1) * 1e-5 + 1e-6)
                << "d=" << d << " idx=1";
        EXPECT_NEAR(dis2, ref2, std::abs(ref2) * 1e-5 + 1e-6)
                << "d=" << d << " idx=2";
        EXPECT_NEAR(dis3, ref3, std::abs(ref3) * 1e-5 + 1e-6)
                << "d=" << d << " idx=3";
    }
}

/*********************************************************
 * FP16 ny tests
 *********************************************************/

TEST(FP16NyTest, L2sqrNy) {
    std::default_random_engine rng(456);

    for (size_t d : {8, 16, 17, 24, 25, 32, 48, 64, 7, 33}) {
        for (size_t ny : {1, 5, 10, 20}) {
            auto x = make_fp16_vectors(1, d, rng);
            auto y = make_fp16_vectors(ny, d, rng);

            std::vector<float> dis(ny);
            faiss::fp16vec_L2sqr_ny(dis.data(), x.data(), y.data(), d, ny);

            for (size_t j = 0; j < ny; j++) {
                float ref = ref_fp16_L2sqr(x.data(), y.data() + j * d, d);
                EXPECT_NEAR(dis[j], ref, std::abs(ref) * 1e-5 + 1e-6)
                        << "d=" << d << " ny=" << ny << " j=" << j;
            }
        }
    }
}

TEST(FP16NyTest, InnerProductsNy) {
    std::default_random_engine rng(456);

    for (size_t d : {8, 16, 17, 24, 25, 32, 48, 64, 7, 33}) {
        for (size_t ny : {1, 5, 10, 20}) {
            auto x = make_fp16_vectors(1, d, rng);
            auto y = make_fp16_vectors(ny, d, rng);

            std::vector<float> ip(ny);
            faiss::fp16vec_inner_products_ny(
                    ip.data(), x.data(), y.data(), d, ny);

            for (size_t j = 0; j < ny; j++) {
                float ref =
                        ref_fp16_inner_product(x.data(), y.data() + j * d, d);
                EXPECT_NEAR(ip[j], ref, std::abs(ref) * 1e-5 + 1e-6)
                        << "d=" << d << " ny=" << ny << " j=" << j;
            }
        }
    }
}

/*********************************************************
 * BF16 basic distance tests
 *********************************************************/

class BF16DistanceTest : public ::testing::TestWithParam<size_t> {};

TEST_P(BF16DistanceTest, InnerProduct) {
    size_t d = GetParam();
    std::default_random_engine rng(42);
    auto x = make_bf16_vectors(1, d, rng);
    auto y = make_bf16_vectors(1, d, rng);

    float result = faiss::bf16vec_inner_product(x.data(), y.data(), d);
    float ref = ref_bf16_inner_product(x.data(), y.data(), d);

    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-5 + 1e-6)
            << "BF16 inner product mismatch at d=" << d;
}

TEST_P(BF16DistanceTest, L2sqr) {
    size_t d = GetParam();
    std::default_random_engine rng(42);
    auto x = make_bf16_vectors(1, d, rng);
    auto y = make_bf16_vectors(1, d, rng);

    float result = faiss::bf16vec_L2sqr(x.data(), y.data(), d);
    float ref = ref_bf16_L2sqr(x.data(), y.data(), d);

    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-5 + 1e-6)
            << "BF16 L2sqr mismatch at d=" << d;
}

TEST_P(BF16DistanceTest, NormL2sqr) {
    size_t d = GetParam();
    std::default_random_engine rng(42);
    auto x = make_bf16_vectors(1, d, rng);

    float result = faiss::bf16vec_norm_L2sqr(x.data(), d);
    float ref = ref_bf16_norm_L2sqr(x.data(), d);

    EXPECT_NEAR(result, ref, std::abs(ref) * 1e-5 + 1e-6)
            << "BF16 norm_L2sqr mismatch at d=" << d;
}

INSTANTIATE_TEST_SUITE_P(
        VariousDimensions,
        BF16DistanceTest,
        // Same dimension coverage as FP16 tests
        ::testing::Values(
                1,
                3,
                7,
                8,
                9,
                15,
                16,
                17,
                24,
                25,
                31,
                32,
                48,
                64,
                128,
                256,
                512));

/*********************************************************
 * BF16 batch-4 tests
 *********************************************************/

TEST(BF16Batch4Test, InnerProductBatch4) {
    std::default_random_engine rng(789);

    for (size_t d : {8, 16, 17, 24, 25, 32, 48, 64, 128, 7, 15, 33}) {
        auto x = make_bf16_vectors(1, d, rng);
        auto y = make_bf16_vectors(4, d, rng);

        float dis0, dis1, dis2, dis3;
        faiss::bf16vec_inner_product_batch_4(
                x.data(),
                y.data() + 0 * d,
                y.data() + 1 * d,
                y.data() + 2 * d,
                y.data() + 3 * d,
                d,
                dis0,
                dis1,
                dis2,
                dis3);

        float ref0 = ref_bf16_inner_product(x.data(), y.data() + 0 * d, d);
        float ref1 = ref_bf16_inner_product(x.data(), y.data() + 1 * d, d);
        float ref2 = ref_bf16_inner_product(x.data(), y.data() + 2 * d, d);
        float ref3 = ref_bf16_inner_product(x.data(), y.data() + 3 * d, d);

        EXPECT_NEAR(dis0, ref0, std::abs(ref0) * 1e-5 + 1e-6)
                << "d=" << d << " idx=0";
        EXPECT_NEAR(dis1, ref1, std::abs(ref1) * 1e-5 + 1e-6)
                << "d=" << d << " idx=1";
        EXPECT_NEAR(dis2, ref2, std::abs(ref2) * 1e-5 + 1e-6)
                << "d=" << d << " idx=2";
        EXPECT_NEAR(dis3, ref3, std::abs(ref3) * 1e-5 + 1e-6)
                << "d=" << d << " idx=3";
    }
}

TEST(BF16Batch4Test, L2sqrBatch4) {
    std::default_random_engine rng(789);

    for (size_t d : {8, 16, 17, 24, 25, 32, 48, 64, 128, 7, 15, 33}) {
        auto x = make_bf16_vectors(1, d, rng);
        auto y = make_bf16_vectors(4, d, rng);

        float dis0, dis1, dis2, dis3;
        faiss::bf16vec_L2sqr_batch_4(
                x.data(),
                y.data() + 0 * d,
                y.data() + 1 * d,
                y.data() + 2 * d,
                y.data() + 3 * d,
                d,
                dis0,
                dis1,
                dis2,
                dis3);

        float ref0 = ref_bf16_L2sqr(x.data(), y.data() + 0 * d, d);
        float ref1 = ref_bf16_L2sqr(x.data(), y.data() + 1 * d, d);
        float ref2 = ref_bf16_L2sqr(x.data(), y.data() + 2 * d, d);
        float ref3 = ref_bf16_L2sqr(x.data(), y.data() + 3 * d, d);

        EXPECT_NEAR(dis0, ref0, std::abs(ref0) * 1e-5 + 1e-6)
                << "d=" << d << " idx=0";
        EXPECT_NEAR(dis1, ref1, std::abs(ref1) * 1e-5 + 1e-6)
                << "d=" << d << " idx=1";
        EXPECT_NEAR(dis2, ref2, std::abs(ref2) * 1e-5 + 1e-6)
                << "d=" << d << " idx=2";
        EXPECT_NEAR(dis3, ref3, std::abs(ref3) * 1e-5 + 1e-6)
                << "d=" << d << " idx=3";
    }
}

/*********************************************************
 * BF16 ny tests
 *********************************************************/

TEST(BF16NyTest, L2sqrNy) {
    std::default_random_engine rng(101);

    for (size_t d : {8, 16, 17, 24, 25, 32, 48, 64, 7, 33}) {
        for (size_t ny : {1, 5, 10, 20}) {
            auto x = make_bf16_vectors(1, d, rng);
            auto y = make_bf16_vectors(ny, d, rng);

            std::vector<float> dis(ny);
            faiss::bf16vec_L2sqr_ny(dis.data(), x.data(), y.data(), d, ny);

            for (size_t j = 0; j < ny; j++) {
                float ref = ref_bf16_L2sqr(x.data(), y.data() + j * d, d);
                EXPECT_NEAR(dis[j], ref, std::abs(ref) * 1e-5 + 1e-6)
                        << "d=" << d << " ny=" << ny << " j=" << j;
            }
        }
    }
}

TEST(BF16NyTest, InnerProductsNy) {
    std::default_random_engine rng(101);

    for (size_t d : {8, 16, 17, 24, 25, 32, 48, 64, 7, 33}) {
        for (size_t ny : {1, 5, 10, 20}) {
            auto x = make_bf16_vectors(1, d, rng);
            auto y = make_bf16_vectors(ny, d, rng);

            std::vector<float> ip(ny);
            faiss::bf16vec_inner_products_ny(
                    ip.data(), x.data(), y.data(), d, ny);

            for (size_t j = 0; j < ny; j++) {
                float ref =
                        ref_bf16_inner_product(x.data(), y.data() + j * d, d);
                EXPECT_NEAR(ip[j], ref, std::abs(ref) * 1e-5 + 1e-6)
                        << "d=" << d << " ny=" << ny << " j=" << j;
            }
        }
    }
}

/*********************************************************
 * Edge case: d=0 should return 0
 *********************************************************/

TEST(FP16EdgeCases, ZeroDimension) {
    uint16_t dummy = 0;
    EXPECT_FLOAT_EQ(faiss::fp16vec_inner_product(&dummy, &dummy, 0), 0.0f);
    EXPECT_FLOAT_EQ(faiss::fp16vec_L2sqr(&dummy, &dummy, 0), 0.0f);
    EXPECT_FLOAT_EQ(faiss::fp16vec_norm_L2sqr(&dummy, 0), 0.0f);
    EXPECT_FLOAT_EQ(faiss::bf16vec_inner_product(&dummy, &dummy, 0), 0.0f);
    EXPECT_FLOAT_EQ(faiss::bf16vec_L2sqr(&dummy, &dummy, 0), 0.0f);
    EXPECT_FLOAT_EQ(faiss::bf16vec_norm_L2sqr(&dummy, 0), 0.0f);
}

TEST(FP16EdgeCases, IdenticalVectors) {
    // L2sqr of identical vectors should be 0
    std::default_random_engine rng(999);
    auto x = make_fp16_vectors(1, 128, rng);

    float l2 = faiss::fp16vec_L2sqr(x.data(), x.data(), 128);
    EXPECT_FLOAT_EQ(l2, 0.0f);

    auto bx = make_bf16_vectors(1, 128, rng);
    float bl2 = faiss::bf16vec_L2sqr(bx.data(), bx.data(), 128);
    EXPECT_FLOAT_EQ(bl2, 0.0f);
}
