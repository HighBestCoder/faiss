/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHNSWCompressed.h>

namespace faiss {

IndexHNSWCompressed::IndexHNSWCompressed() = default;

IndexHNSWCompressed::IndexHNSWCompressed(
        int d,
        int M,
        size_t block_size,
        size_t cache_size,
        MetricType metric)
        : IndexHNSW(
                  new IndexFlatCompressed(d, block_size, cache_size, metric),
                  M) {
    own_fields = true;
}

IndexHNSWCompressed::IndexHNSWCompressed(
        int d,
        int M,
        std::unique_ptr<CompressionCodec> codec,
        size_t block_size,
        size_t cache_size,
        MetricType metric)
        : IndexHNSW(
                  new IndexFlatCompressed(
                          d,
                          std::move(codec),
                          block_size,
                          cache_size,
                          metric),
                  M) {
    own_fields = true;
}

IndexHNSWCompressedLZ4::IndexHNSWCompressedLZ4(
        int d,
        int M,
        int lz4_acceleration,
        size_t block_size,
        size_t cache_size,
        MetricType metric)
        : IndexHNSWCompressed(
                  d,
                  M,
                  std::make_unique<LZ4Codec>(lz4_acceleration),
                  block_size,
                  cache_size,
                  metric) {}

IndexHNSWCompressedZSTD::IndexHNSWCompressedZSTD(
        int d,
        int M,
        int zstd_level,
        size_t block_size,
        size_t cache_size,
        MetricType metric)
        : IndexHNSWCompressed(
                  d,
                  M,
                  std::make_unique<ZSTDCodec>(zstd_level),
                  block_size,
                  cache_size,
                  metric) {}

} // namespace faiss
