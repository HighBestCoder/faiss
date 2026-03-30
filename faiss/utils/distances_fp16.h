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
