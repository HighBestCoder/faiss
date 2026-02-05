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
#include <limits>
#include <thread>
#include <unordered_map>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/distances.h>

namespace faiss {

std::atomic<uint64_t> IndexFlatCompressed::next_instance_id{1};

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
    FAISS_THROW_IF_NOT_MSG(
            !ZSTD_isError(compressed_size), "ZSTD compression failed");
    return compressed_size;
}

size_t ZSTDCodec::decompress(
        const uint8_t* input,
        size_t input_size,
        uint8_t* output,
        size_t output_capacity) const {
    size_t decompressed_size =
            ZSTD_decompress(output, output_capacity, input, input_size);
    FAISS_THROW_IF_NOT_MSG(
            !ZSTD_isError(decompressed_size), "ZSTD decompression failed");
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
        size_t vector_size_floats)
        : num_entries(num_entries), vector_size_floats(vector_size_floats) {
    entries.resize(num_entries);
    for (auto& entry : entries) {
        entry.data.resize(vector_size_floats);
        entry.vector_id = -1;
        entry.last_access = 0;
    }
}

const float* DecompressionCache::get(idx_t vector_id) {
    for (auto& entry : entries) {
        if (entry.vector_id == vector_id) {
            entry.last_access = ++access_counter;
            hits++;
            return entry.data.data();
        }
    }
    misses++;
    return nullptr;
}

float* DecompressionCache::prepare_entry(idx_t vector_id) {
    size_t lru_idx = 0;
    uint64_t min_access = entries[0].last_access;

    for (size_t i = 1; i < entries.size(); i++) {
        if (entries[i].last_access < min_access) {
            min_access = entries[i].last_access;
            lru_idx = i;
        }
    }

    entries[lru_idx].vector_id = vector_id;
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
        bool use_cache,
        size_t cache_size,
        MetricType metric)
        : Index(d, metric),
          instance_id(next_instance_id++),
          use_cache(use_cache),
          cache_size(cache_size),
          codec(std::make_unique<ZSTDCodec>()),
          code_size(sizeof(float) * d) {
    is_trained = true;
}

IndexFlatCompressed::IndexFlatCompressed(
        idx_t d,
        std::unique_ptr<CompressionCodec> codec,
        bool use_cache,
        size_t cache_size,
        MetricType metric)
        : Index(d, metric),
          instance_id(next_instance_id++),
          use_cache(use_cache),
          cache_size(cache_size),
          codec(std::move(codec)),
          code_size(sizeof(float) * d) {
    is_trained = true;
}

IndexFlatCompressed::~IndexFlatCompressed() = default;

void IndexFlatCompressed::set_codec(
        std::unique_ptr<CompressionCodec> new_codec) {
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0, "Cannot change codec after adding vectors");
    codec = std::move(new_codec);
}

void IndexFlatCompressed::set_use_cache(bool enabled) {
    use_cache = enabled;
    if (!enabled) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        thread_cache_ptrs.clear();
    }
}

void IndexFlatCompressed::compress_vector(const float* vector) {
    size_t max_output_size = codec->max_compressed_size(code_size);
    std::vector<uint8_t> compressed(max_output_size);

    size_t actual_size = codec->compress(
            reinterpret_cast<const uint8_t*>(vector),
            code_size,
            compressed.data(),
            max_output_size);

    compressed.resize(actual_size);
    compressed.shrink_to_fit();
    compressed_vectors.push_back(std::move(compressed));
}

void IndexFlatCompressed::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }

    compressed_vectors.reserve(compressed_vectors.size() + n);

    for (idx_t i = 0; i < n; i++) {
        compress_vector(x + i * d);
    }
    ntotal += n;
}

void IndexFlatCompressed::reset() {
    compressed_vectors.clear();
    ntotal = 0;

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        thread_cache_ptrs.clear();
    }

    total_decompressions = 0;
    total_bytes_decompressed = 0;
}

void IndexFlatCompressed::decompress_vector(idx_t i, float* output) const {
    FAISS_THROW_IF_NOT(i >= 0 && i < ntotal);

    const auto& compressed = compressed_vectors[i];
    codec->decompress(
            compressed.data(),
            compressed.size(),
            reinterpret_cast<uint8_t*>(output),
            code_size);

    total_decompressions++;
    total_bytes_decompressed += code_size;
}

DecompressionCache* IndexFlatCompressed::get_thread_cache() const {
    thread_local std::unordered_map<uint64_t,
                                    std::unique_ptr<DecompressionCache>>
            local_caches;

    auto it = local_caches.find(instance_id);
    if (it != local_caches.end()) {
        return it->second.get();
    }

    size_t effective_cache_size = use_cache ? cache_size : 4;
    auto cache = std::make_unique<DecompressionCache>(effective_cache_size, d);
    DecompressionCache* ptr = cache.get();
    local_caches[instance_id] = std::move(cache);

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        thread_cache_ptrs.push_back(ptr);
    }

    return ptr;
}

const float* IndexFlatCompressed::get_vector(idx_t i) const {
    FAISS_THROW_IF_NOT(i >= 0 && i < ntotal);

    DecompressionCache* cache = get_thread_cache();

    if (use_cache) {
        const float* cached = cache->get(i);
        if (cached) {
            return cached;
        }
    }

    float* buffer = cache->prepare_entry(i);
    decompress_vector(i, buffer);
    return buffer;
}

void IndexFlatCompressed::reconstruct(idx_t key, float* recons) const {
    decompress_vector(key, recons);
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
                    results.begin() +
                            std::min(static_cast<size_t>(k), results.size()),
                    results.end(),
                    [](auto& a, auto& b) { return a.first > b.first; });
        } else {
            std::partial_sort(
                    results.begin(),
                    results.begin() +
                            std::min(static_cast<size_t>(k), results.size()),
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
    stats.num_vectors = ntotal;

    size_t total_compressed = 0;
    size_t min_size = std::numeric_limits<size_t>::max();
    size_t max_size = 0;

    for (const auto& cv : compressed_vectors) {
        total_compressed += cv.size();
        min_size = std::min(min_size, cv.size());
        max_size = std::max(max_size, cv.size());
    }

    stats.compressed_size = total_compressed;
    stats.compression_ratio = total_compressed > 0
            ? static_cast<double>(stats.original_size) / total_compressed
            : 0.0;
    stats.avg_compressed_size = ntotal > 0
            ? static_cast<double>(total_compressed) / ntotal
            : 0.0;
    stats.min_compressed_size = ntotal > 0 ? static_cast<double>(min_size) : 0;
    stats.max_compressed_size = ntotal > 0 ? static_cast<double>(max_size) : 0;

    return stats;
}

IndexFlatCompressed::CacheStats IndexFlatCompressed::get_cache_stats() const {
    CacheStats stats = {0, 0, 0.0, 0, 0};

    std::lock_guard<std::mutex> lock(cache_mutex);
    for (const auto* cache : thread_cache_ptrs) {
        stats.total_hits += cache->hits;
        stats.total_misses += cache->misses;
    }

    uint64_t total = stats.total_hits + stats.total_misses;
    stats.hit_ratio =
            total > 0 ? static_cast<double>(stats.total_hits) / total : 0.0;
    stats.total_decompressions = total_decompressions;
    stats.total_bytes_decompressed = total_bytes_decompressed;
    return stats;
}

void IndexFlatCompressed::reset_stats() const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (auto* cache : thread_cache_ptrs) {
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
    std::vector<float> vec_i_copy(storage.d);
    storage.decompress_vector(i, vec_i_copy.data());
    const float* vec_j = storage.get_vector(j);
    if (metric_type == METRIC_L2) {
        return fvec_L2sqr(vec_i_copy.data(), vec_j, storage.d);
    } else {
        return fvec_inner_product(vec_i_copy.data(), vec_j, storage.d);
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
