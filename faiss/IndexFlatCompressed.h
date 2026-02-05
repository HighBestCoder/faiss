/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

/**
 * Compression codec interface for IndexFlatCompressed.
 * Allows pluggable compression algorithms (LZ4, ZSTD, etc.)
 */
struct CompressionCodec {
    virtual ~CompressionCodec() = default;

    /// Returns the maximum compressed size for a given input size
    virtual size_t max_compressed_size(size_t input_size) const = 0;

    /// Compress data. Returns actual compressed size.
    virtual size_t compress(
            const uint8_t* input,
            size_t input_size,
            uint8_t* output,
            size_t output_capacity) const = 0;

    /// Decompress data. Returns actual decompressed size.
    virtual size_t decompress(
            const uint8_t* input,
            size_t input_size,
            uint8_t* output,
            size_t output_capacity) const = 0;

    /// Get codec name for logging/debugging
    virtual const char* name() const = 0;
};

/**
 * LZ4 compression codec - fast compression/decompression
 */
struct LZ4Codec : CompressionCodec {
    int acceleration = 1; // 1 = default, higher = faster but worse ratio

    explicit LZ4Codec(int acceleration = 1);
    ~LZ4Codec() override = default;

    size_t max_compressed_size(size_t input_size) const override;
    size_t compress(
            const uint8_t* input,
            size_t input_size,
            uint8_t* output,
            size_t output_capacity) const override;
    size_t decompress(
            const uint8_t* input,
            size_t input_size,
            uint8_t* output,
            size_t output_capacity) const override;
    const char* name() const override {
        return "LZ4";
    }
};

/**
 * ZSTD compression codec - better ratio, slower
 */
struct ZSTDCodec : CompressionCodec {
    int compression_level = 3; // 1-22, default 3

    explicit ZSTDCodec(int level = 3);
    ~ZSTDCodec() override = default;

    size_t max_compressed_size(size_t input_size) const override;
    size_t compress(
            const uint8_t* input,
            size_t input_size,
            uint8_t* output,
            size_t output_capacity) const override;
    size_t decompress(
            const uint8_t* input,
            size_t input_size,
            uint8_t* output,
            size_t output_capacity) const override;
    const char* name() const override {
        return "ZSTD";
    }
};

/**
 * No-op codec for testing/comparison
 */
struct NoopCodec : CompressionCodec {
    size_t max_compressed_size(size_t input_size) const override {
        return input_size;
    }
    size_t compress(
            const uint8_t* input,
            size_t input_size,
            uint8_t* output,
            size_t output_capacity) const override;
    size_t decompress(
            const uint8_t* input,
            size_t input_size,
            uint8_t* output,
            size_t output_capacity) const override;
    const char* name() const override {
        return "NOOP";
    }
};

/**
 * Thread-local decompression cache for IndexFlatCompressed.
 * Uses LRU eviction to cache recently accessed vectors.
 */
struct DecompressionCache {
    struct CacheEntry {
        idx_t vector_id = -1;
        std::vector<float> data;
        uint64_t last_access = 0;
    };

    size_t num_entries;
    size_t vector_size_floats; // number of floats per vector
    std::vector<CacheEntry> entries;
    uint64_t access_counter = 0;

    // Statistics
    mutable std::atomic<uint64_t> hits{0};
    mutable std::atomic<uint64_t> misses{0};

    DecompressionCache(size_t num_entries, size_t vector_size_floats);

    /// Try to get a cached vector. Returns nullptr if not cached.
    const float* get(idx_t vector_id);

    /// Get an entry to write to (evicts LRU if needed).
    /// Returns pointer to data buffer and sets the vector_id.
    float* prepare_entry(idx_t vector_id);

    /// Get cache hit ratio
    double hit_ratio() const;

    /// Reset statistics
    void reset_stats();
};

/**
 * Compressed vector storage for flat indexes.
 *
 * Stores each vector individually compressed using ZSTD (by default).
 * Supports optional thread-local LRU caching to amortize decompression cost.
 *
 * Key parameters:
 * - use_cache: whether to enable LRU cache (default: true)
 * - cache_size: number of vectors to cache per thread (default: 64)
 * - codec: compression algorithm (ZSTD by default for single-vector compression)
 */
struct IndexFlatCompressed : Index {
    // Unique instance ID for thread-local cache keying
    // Using a static counter to ensure uniqueness even after destruction/recreation
    static std::atomic<uint64_t> next_instance_id;
    uint64_t instance_id;

    // Compression parameters
    bool use_cache = true;    // whether to use LRU cache
    size_t cache_size = 64;   // vectors per thread cache (only used if use_cache=true)
    std::unique_ptr<CompressionCodec> codec;

    // Original vector dimension info
    size_t code_size; // bytes per vector = sizeof(float) * d

    // Compressed storage - one entry per vector
    std::vector<std::vector<uint8_t>> compressed_vectors;

    mutable std::vector<DecompressionCache*> thread_cache_ptrs;
    mutable std::mutex cache_mutex;

    // Statistics
    mutable std::atomic<uint64_t> total_decompressions{0};
    mutable std::atomic<uint64_t> total_bytes_decompressed{0};

    /// Construct with dimension and optional parameters
    explicit IndexFlatCompressed(
            idx_t d,
            bool use_cache = true,
            size_t cache_size = 64,
            MetricType metric = METRIC_L2);

    /// Construct with custom codec
    IndexFlatCompressed(
            idx_t d,
            std::unique_ptr<CompressionCodec> codec,
            bool use_cache = true,
            size_t cache_size = 64,
            MetricType metric = METRIC_L2);

    ~IndexFlatCompressed() override;

    // Index interface
    void add(idx_t n, const float* x) override;
    void reset() override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    void reconstruct(idx_t key, float* recons) const override;

    /// Get distance computer for use with HNSW
    DistanceComputer* get_distance_computer() const override;

    /// Decompress a single vector
    /// If use_cache=true, uses LRU cache
    /// If use_cache=false, decompresses directly into thread-local buffer
    const float* get_vector(idx_t i) const;

    /// Decompress a vector into provided buffer (no cache used)
    void decompress_vector(idx_t i, float* output) const;

    /// Get compression statistics
    struct CompressionStats {
        size_t original_size;
        size_t compressed_size;
        double compression_ratio;  // original_size / compressed_size (>1 means good compression)
        size_t num_vectors;
        double avg_compressed_size;
        double min_compressed_size;
        double max_compressed_size;
    };
    CompressionStats get_compression_stats() const;

    /// Get cache statistics (only meaningful if use_cache=true)
    struct CacheStats {
        uint64_t total_hits;
        uint64_t total_misses;
        double hit_ratio;
        uint64_t total_decompressions;
        uint64_t total_bytes_decompressed;
    };
    CacheStats get_cache_stats() const;

    /// Reset all statistics
    void reset_stats() const;

    /// Set codec (must be called before adding vectors)
    void set_codec(std::unique_ptr<CompressionCodec> new_codec);

    /// Enable/disable cache (can be changed at runtime)
    void set_use_cache(bool enabled);

private:
    DecompressionCache* get_thread_cache() const;

    void compress_vector(const float* vector);
};

/**
 * Distance computer for IndexFlatCompressed.
 * Handles decompression transparently.
 */
struct CompressedDistanceComputer : DistanceComputer {
    const IndexFlatCompressed& storage;
    const float* query = nullptr;
    MetricType metric_type;

    explicit CompressedDistanceComputer(const IndexFlatCompressed& storage);

    void set_query(const float* x) override;
    float operator()(idx_t i) override;
    float symmetric_dis(idx_t i, idx_t j) override;

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override;
};

} // namespace faiss
