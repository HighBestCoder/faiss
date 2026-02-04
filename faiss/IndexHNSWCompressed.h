/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexFlatCompressed.h>
#include <faiss/IndexHNSW.h>

namespace faiss {

struct IndexHNSWCompressed : IndexHNSW {
    IndexHNSWCompressed();

    IndexHNSWCompressed(
            int d,
            int M = 32,
            size_t block_size = 32,
            size_t cache_size = 16,
            MetricType metric = METRIC_L2);

    IndexHNSWCompressed(
            int d,
            int M,
            std::unique_ptr<CompressionCodec> codec,
            size_t block_size = 32,
            size_t cache_size = 16,
            MetricType metric = METRIC_L2);

    IndexFlatCompressed* get_compressed_storage() {
        return dynamic_cast<IndexFlatCompressed*>(storage);
    }

    const IndexFlatCompressed* get_compressed_storage() const {
        return dynamic_cast<const IndexFlatCompressed*>(storage);
    }
};

struct IndexHNSWCompressedLZ4 : IndexHNSWCompressed {
    IndexHNSWCompressedLZ4() = default;

    IndexHNSWCompressedLZ4(
            int d,
            int M = 32,
            int lz4_acceleration = 1,
            size_t block_size = 32,
            size_t cache_size = 16,
            MetricType metric = METRIC_L2);
};

struct IndexHNSWCompressedZSTD : IndexHNSWCompressed {
    IndexHNSWCompressedZSTD() = default;

    IndexHNSWCompressedZSTD(
            int d,
            int M = 32,
            int zstd_level = 3,
            size_t block_size = 32,
            size_t cache_size = 16,
            MetricType metric = METRIC_L2);
};

} // namespace faiss
