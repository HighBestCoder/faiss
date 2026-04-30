/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Distance functions for FP16 and BF16 half-precision vectors.
 *
 * Vectors are stored as uint16_t arrays (FP16 or BF16 encoded).
 * All computations are done in FP32 after conversion.
 * The actual implementations are in distances_fp16_simd.cpp.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/impl/platform_macros.h>

namespace faiss {

/*********************************************************
 * FP16 distance computations
 * Vectors stored as IEEE 754 half-precision (binary16)
 *********************************************************/

/// Squared L2 distance between two FP16 vectors
float fp16vec_L2sqr(const uint16_t* x, const uint16_t* y, size_t d);

/// Inner product of two FP16 vectors
float fp16vec_inner_product(const uint16_t* x, const uint16_t* y, size_t d);

/// Squared norm of an FP16 vector
float fp16vec_norm_L2sqr(const uint16_t* x, size_t d);

/// Compute ny L2sqr distances between x and a set of contiguous y vectors
void fp16vec_L2sqr_ny(
        float* dis,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny);

/// Compute ny inner products between x and a set of contiguous y vectors
void fp16vec_inner_products_ny(
        float* ip,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny);

/// Special version of inner product that computes 4 distances
/// between x and yi, which is performance oriented.
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
        float& dis3);

/// Special version of L2sqr that computes 4 distances
/// between x and yi, which is performance oriented.
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
        float& dis3);

/*********************************************************
 * FP16 native-precision distance computations (research / SPR only)
 *
 * These variants use AVX-512_FP16 (Sapphire Rapids+) with a __m512h
 * accumulator and _mm512_fmadd_ph: 32 fp16 lanes per FMA, ~2x the
 * throughput of the FP32-accumulator path on supported hardware.
 *
 * RECALL RISK: accumulator stays in fp16 throughout the inner loop, so
 * partial sums saturate at ±65504 and rounding error accumulates faster
 * than the FP32 path. For high-d vectors or vectors with magnitude
 * variation these can produce visibly different distances vs the
 * mixed-precision path. Always A/B against fp16vec_inner_product()
 * before using in production.
 *
 * On non-SPR builds these forward to the standard FP32-accumulator
 * implementations (so callers can link unconditionally).
 *********************************************************/

/// Inner product with native fp16 accumulator (SPR / AVX-512_FP16)
float fp16vec_inner_product_native(
        const uint16_t* x,
        const uint16_t* y,
        size_t d);

/// Squared L2 distance with native fp16 accumulator (SPR / AVX-512_FP16)
float fp16vec_L2sqr_native(const uint16_t* x, const uint16_t* y, size_t d);

/// Squared norm with native fp16 accumulator (SPR / AVX-512_FP16)
float fp16vec_norm_L2sqr_native(const uint16_t* x, size_t d);

/*********************************************************
 * BF16 distance computations
 * Vectors stored as bfloat16 (upper 16 bits of FP32)
 *********************************************************/

/// Squared L2 distance between two BF16 vectors
float bf16vec_L2sqr(const uint16_t* x, const uint16_t* y, size_t d);

/// Inner product of two BF16 vectors
float bf16vec_inner_product(const uint16_t* x, const uint16_t* y, size_t d);

/// Squared norm of a BF16 vector
float bf16vec_norm_L2sqr(const uint16_t* x, size_t d);

/// Compute ny L2sqr distances between x and a set of contiguous y vectors
void bf16vec_L2sqr_ny(
        float* dis,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny);

/// Compute ny inner products between x and a set of contiguous y vectors
void bf16vec_inner_products_ny(
        float* ip,
        const uint16_t* x,
        const uint16_t* y,
        size_t d,
        size_t ny);

/// Special version of inner product that computes 4 distances
/// between x and yi, which is performance oriented.
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
        float& dis3);

/// Special version of L2sqr that computes 4 distances
/// between x and yi, which is performance oriented.
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
        float& dis3);

} // namespace faiss
