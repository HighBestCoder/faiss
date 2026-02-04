/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlatCompressed.h>

#include <lz4.h>
#include <zstd.h>

#include <algorithm>
#include <cstring>
#include <thread>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/distances.h>

namespace faiss {

LZ4Codec::LZ4Codec(int acceleration) : acceleration(acceleration) {}

size_t LZ4Codec::max_compressed_size(size_t input_size) const {
    return LZ4_compressBound(input_size);
}

size_t LZ4Codec::compress(
        const uint8_t* input,
        size_t input_size,
        uint8_t* output,
        size_t output_capacity) const {
    int compressed_size = LZ4_compress_fast(
            reinterpret_cast<const char*>(input),
            reinterpret_cast<char*>(output),
            input_size,
            output_capacity,
            acceleration);
    FAISS_THROW_IF_NOT_MSG(compressed_size > 0, "LZ4 compression failed");
    return compressed_size;
}

size_t LZ4Codec::decompress(
        const uint8_t* input,
        size_t input_size,
        uint8_t* output,
        size_t output_capacity) const {
    int decompressed_size = LZ4_decompress_safe(
            reinterpret_cast<const char*>(input),
            reinterpret_cast<char*>(output),
            input_size,
            output_capacity);
    FAISS_THROW_IF_NOT_MSG(decompressed_size > 0, "LZ4 decompression failed");
    return decompressed_size;
}

ZSTDCodec::ZSTDCodec(int level) : compression_level(level) {}

size_t ZSTDCodec::max_compressed_size(size_t input_size) const {
    return ZSTD_compressBound(input_size);
}

size_t ZSTDCodec::compress(
        const uint8_t* input,
        size_t input_size,
        uint8_t* output,
        size_t output_capacity) const {
    size_t compressed_size = ZSTD_compress(
            output, output_capacity, input, input_size, compression_level);
    FAISS_THROW_IF_NOT_MSG(!ZSTD_isError(compressed_size), "ZSTD compression failed");
    return compressed_size;
}

size_t ZSTDCodec::decompress(
        const uint8_t* input,
        size_t input_size,
        uint8_t* output,
        size_t output_capacity) const {
    size_t decompressed_size =
            ZSTD_decompress(output, output_capacity, input, input_size);
    FAISS_THROW_IF_NOT_MSG(!ZSTD_isError(decompressed_size), "ZSTD decompression failed");
    return decompressed_size;
}

size_t NoopCodec::compress(
        const uint8_t* input,
        size_t input_size,
        uint8_t* output,
        size_t /* output_capacity */) const {
    std::memcpy(output, input, input_size);
    return input_size;
}

size_t NoopCodec::decompress(
        const uint8_t* input,
        size_t input_size,
        uint8_t* output,
        size_t /* output_capacity */) const {
    std::memcpy(output, input, input_size);
    return input_size;
}

DecompressionCache::DecompressionCache(
        size_t num_entries,
        size_t block_size_floats)
        : num_entries(num_entries), block_size_floats(block_size_floats) {
    entries.resize(num_entries);
    for (auto& entry : entries) {
        entry.data.resize(block_size_floats);
    }
}

const float* DecompressionCache::get(idx_t block_id) {
    for (auto& entry : entries) {
        if (entry.block_id == block_id) {
            entry.last_access = ++access_counter;
            hits++;
            return entry.data.data();
        }
    }
    misses++;
    return nullptr;
}

float* DecompressionCache::prepare_entry(idx_t block_id) {
    size_t lru_idx = 0;
    uint64_t min_access = entries[0].last_access;

    for (size_t i = 1; i < entries.size(); i++) {
        if (entries[i].last_access < min_access) {
            min_access = entries[i].last_access;
            lru_idx = i;
        }
    }

    entries[lru_idx].block_id = block_id;
    entries[lru_idx].last_access = ++access_counter;
    return entries[lru_idx].data.data();
}

double DecompressionCache::hit_ratio() const {
    uint64_t total = hits + misses;
    return total > 0 ? static_cast<double>(hits) / total : 0.0;
}

void DecompressionCache::reset_stats() {
    hits = 0;
    misses = 0;
}

IndexFlatCompressed::IndexFlatCompressed(
        idx_t d,
        size_t block_size,
        size_t cache_size,
        MetricType metric)
        : Index(d, metric),
          block_size(block_size),
          cache_size(cache_size),
          codec(std::make_unique<LZ4Codec>()),
          code_size(sizeof(float) * d) {
    is_trained = true;
}

IndexFlatCompressed::IndexFlatCompressed(
        idx_t d,
        std::unique_ptr<CompressionCodec> codec,
        size_t block_size,
        size_t cache_size,
        MetricType metric)
        : Index(d, metric),
          block_size(block_size),
          cache_size(cache_size),
          codec(std::move(codec)),
          code_size(sizeof(float) * d) {
    is_trained = true;
}

IndexFlatCompressed::~IndexFlatCompressed() = default;

void IndexFlatCompressed::set_codec(std::unique_ptr<CompressionCodec> new_codec) {
    FAISS_THROW_IF_NOT_MSG(ntotal == 0, "Cannot change codec after adding vectors");
    codec = std::move(new_codec);
}

void IndexFlatCompressed::compress_block(const float* vectors, size_t num_vectors) {
    size_t input_size = num_vectors * code_size;
    size_t max_output_size = codec->max_compressed_size(input_size);

    size_t current_offset = compressed_data.size();
    compressed_data.resize(current_offset + max_output_size);

    size_t actual_size = codec->compress(
            reinterpret_cast<const uint8_t*>(vectors),
            input_size,
            compressed_data.data() + current_offset,
            max_output_size);

    compressed_data.resize(current_offset + actual_size);
    block_offsets.push_back(current_offset);
    block_comp_sizes.push_back(actual_size);
}

void IndexFlatCompressed::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    idx_t remaining = n;
    const float* ptr = x;

    if (last_block_vectors > 0 && last_block_vectors < block_size) {
        FAISS_THROW_MSG("Partial block handling not yet implemented - add in multiples of block_size");
    }

    while (remaining > 0) {
        size_t to_add = std::min(static_cast<size_t>(remaining), block_size);
        compress_block(ptr, to_add);
        remaining -= to_add;
        ptr += to_add * d;
        ntotal += to_add;

        if (to_add < block_size) {
            last_block_vectors = to_add;
        }
    }
}

void IndexFlatCompressed::reset() {
    compressed_data.clear();
    block_offsets.clear();
    block_comp_sizes.clear();
    last_block_vectors = 0;
    ntotal = 0;

    std::lock_guard<std::mutex> lock(cache_mutex);
    thread_caches.clear();
}

void IndexFlatCompressed::decompress_block(idx_t block_id, float* output) const {
    FAISS_THROW_IF_NOT(block_id >= 0 && static_cast<size_t>(block_id) < block_offsets.size());

    size_t offset = block_offsets[block_id];
    size_t comp_size = block_comp_sizes[block_id];

    bool is_last = (static_cast<size_t>(block_id) == block_offsets.size() - 1);
    size_t num_vectors = (is_last && last_block_vectors > 0)
                                 ? last_block_vectors
                                 : block_size;
    size_t output_size = num_vectors * code_size;

    codec->decompress(
            compressed_data.data() + offset,
            comp_size,
            reinterpret_cast<uint8_t*>(output),
            output_size);

    total_decompressions++;
    total_bytes_decompressed += output_size;
}

DecompressionCache* IndexFlatCompressed::get_thread_cache() const {
    thread_local size_t my_cache_idx = SIZE_MAX;

    if (my_cache_idx == SIZE_MAX) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        my_cache_idx = thread_caches.size();
        thread_caches.push_back(
                std::make_unique<DecompressionCache>(cache_size, block_size * d));
    }

    return thread_caches[my_cache_idx].get();
}

const float* IndexFlatCompressed::get_block(idx_t block_id) const {
    DecompressionCache* cache = get_thread_cache();

    const float* cached = cache->get(block_id);
    if (cached) {
        return cached;
    }

    float* buffer = cache->prepare_entry(block_id);
    decompress_block(block_id, buffer);
    return buffer;
}

const float* IndexFlatCompressed::get_vector(idx_t i) const {
    FAISS_THROW_IF_NOT(i >= 0 && i < ntotal);
    idx_t block_id = i / block_size;
    idx_t offset_in_block = i % block_size;
    const float* block = get_block(block_id);
    return block + offset_in_block * d;
}

void IndexFlatCompressed::reconstruct(idx_t key, float* recons) const {
    const float* vec = get_vector(key);
    std::memcpy(recons, vec, code_size);
}

void IndexFlatCompressed::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    const IDSelector* sel = params ? params->sel : nullptr;

#pragma omp parallel for if (n > 1)
    for (idx_t i = 0; i < n; i++) {
        const float* query = x + i * d;
        float* D = distances + i * k;
        idx_t* I = labels + i * k;

        std::vector<std::pair<float, idx_t>> results;
        results.reserve(ntotal);

        for (idx_t j = 0; j < ntotal; j++) {
            if (sel && !sel->is_member(j)) {
                continue;
            }
            const float* vec = get_vector(j);
            float dis = (metric_type == METRIC_L2)
                                ? fvec_L2sqr(query, vec, d)
                                : fvec_inner_product(query, vec, d);
            results.emplace_back(dis, j);
        }

        if (metric_type == METRIC_INNER_PRODUCT) {
            std::partial_sort(
                    results.begin(),
                    results.begin() + std::min(static_cast<size_t>(k), results.size()),
                    results.end(),
                    [](auto& a, auto& b) { return a.first > b.first; });
        } else {
            std::partial_sort(
                    results.begin(),
                    results.begin() + std::min(static_cast<size_t>(k), results.size()),
                    results.end(),
                    [](auto& a, auto& b) { return a.first < b.first; });
        }

        for (idx_t j = 0; j < k; j++) {
            if (static_cast<size_t>(j) < results.size()) {
                D[j] = results[j].first;
                I[j] = results[j].second;
            } else {
                D[j] = (metric_type == METRIC_INNER_PRODUCT) ? -1e10 : 1e10;
                I[j] = -1;
            }
        }
    }
}

IndexFlatCompressed::CompressionStats IndexFlatCompressed::get_compression_stats()
        const {
    CompressionStats stats;
    stats.original_size = ntotal * code_size;
    stats.compressed_size = compressed_data.size();
    stats.compression_ratio = stats.compressed_size > 0
            ? static_cast<double>(stats.original_size) / stats.compressed_size
            : 0.0;
    stats.num_blocks = block_offsets.size();
    stats.avg_block_compressed_size = stats.num_blocks > 0
            ? static_cast<double>(stats.compressed_size) / stats.num_blocks
            : 0.0;
    return stats;
}

IndexFlatCompressed::CacheStats IndexFlatCompressed::get_cache_stats() const {
    CacheStats stats = {0, 0, 0.0, 0, 0};

    std::lock_guard<std::mutex> lock(cache_mutex);
    for (const auto& cache : thread_caches) {
        stats.total_hits += cache->hits;
        stats.total_misses += cache->misses;
    }

    uint64_t total = stats.total_hits + stats.total_misses;
    stats.hit_ratio = total > 0
            ? static_cast<double>(stats.total_hits) / total
            : 0.0;
    stats.total_decompressions = total_decompressions;
    stats.total_bytes_decompressed = total_bytes_decompressed;
    return stats;
}

void IndexFlatCompressed::reset_stats() const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (auto& cache : thread_caches) {
        cache->reset_stats();
    }
    total_decompressions = 0;
    total_bytes_decompressed = 0;
}

DistanceComputer* IndexFlatCompressed::get_distance_computer() const {
    return new CompressedDistanceComputer(*this);
}

CompressedDistanceComputer::CompressedDistanceComputer(
        const IndexFlatCompressed& storage)
        : storage(storage), metric_type(storage.metric_type) {}

void CompressedDistanceComputer::set_query(const float* x) {
    query = x;
}

float CompressedDistanceComputer::operator()(idx_t i) {
    const float* vec = storage.get_vector(i);
    if (metric_type == METRIC_L2) {
        return fvec_L2sqr(query, vec, storage.d);
    } else {
        return fvec_inner_product(query, vec, storage.d);
    }
}

float CompressedDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    const float* vec_i = storage.get_vector(i);
    const float* vec_j = storage.get_vector(j);
    if (metric_type == METRIC_L2) {
        return fvec_L2sqr(vec_i, vec_j, storage.d);
    } else {
        return fvec_inner_product(vec_i, vec_j, storage.d);
    }
}

void CompressedDistanceComputer::distances_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    dis0 = (*this)(idx0);
    dis1 = (*this)(idx1);
    dis2 = (*this)(idx2);
    dis3 = (*this)(idx3);
}

} // namespace faiss
