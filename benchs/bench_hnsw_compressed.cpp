/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark comparing IndexHNSWFlat vs IndexHNSWCompressed (LZ4/ZSTD)
 *
 * Measures:
 * - Build time
 * - Search QPS (queries per second)
 * - Recall@10
 * - Memory usage
 * - Compression ratio
 *
 * Tests across multiple dimensions and configurations.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>

#ifdef FAISS_ENABLE_COMPRESSED_STORAGE
#include <faiss/IndexFlatCompressed.h>
#include <faiss/IndexHNSWCompressed.h>
#endif

#include <omp.h>
#include <sys/resource.h>

namespace {

// Get current memory usage in bytes
size_t get_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<size_t>(usage.ru_maxrss) * 1024; // Convert KB to bytes
}

// Generate random float vectors
std::vector<float> generate_random_vectors(size_t n, size_t d, unsigned seed = 42) {
    std::vector<float> data(n * d);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < n * d; i++) {
        data[i] = dist(rng);
    }

    // Normalize vectors
    for (size_t i = 0; i < n; i++) {
        float norm = 0;
        for (size_t j = 0; j < d; j++) {
            norm += data[i * d + j] * data[i * d + j];
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < d; j++) {
            data[i * d + j] /= norm;
        }
    }

    return data;
}

// Compute ground truth using brute force
std::vector<faiss::idx_t> compute_ground_truth(
        const float* queries,
        const float* database,
        size_t nq,
        size_t nb,
        size_t d,
        size_t k) {
    std::vector<faiss::idx_t> gt(nq * k);

#pragma omp parallel for
    for (size_t q = 0; q < nq; q++) {
        const float* query = queries + q * d;
        std::vector<std::pair<float, faiss::idx_t>> distances(nb);

        for (size_t i = 0; i < nb; i++) {
            const float* vec = database + i * d;
            float dist = 0;
            for (size_t j = 0; j < d; j++) {
                float diff = query[j] - vec[j];
                dist += diff * diff;
            }
            distances[i] = {dist, static_cast<faiss::idx_t>(i)};
        }

        std::partial_sort(
                distances.begin(),
                distances.begin() + k,
                distances.end(),
                [](auto& a, auto& b) { return a.first < b.first; });

        for (size_t i = 0; i < k; i++) {
            gt[q * k + i] = distances[i].second;
        }
    }

    return gt;
}

// Compute recall@k
double compute_recall(
        const faiss::idx_t* results,
        const faiss::idx_t* ground_truth,
        size_t nq,
        size_t k) {
    size_t correct = 0;
    for (size_t q = 0; q < nq; q++) {
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < k; j++) {
                if (results[q * k + i] == ground_truth[q * k + j]) {
                    correct++;
                    break;
                }
            }
        }
    }
    return static_cast<double>(correct) / (nq * k);
}

struct BenchmarkResult {
    std::string index_type;
    size_t dimension;
    size_t block_size;
    size_t cache_size;
    double build_time_sec;
    double search_qps;
    double recall_at_10;
    size_t memory_bytes;
    double compression_ratio;
    double cache_hit_ratio;
};

void print_header() {
    std::cout << std::setw(20) << "Index Type"
              << std::setw(8) << "Dim"
              << std::setw(8) << "Block"
              << std::setw(8) << "Cache"
              << std::setw(12) << "Build(s)"
              << std::setw(12) << "QPS"
              << std::setw(10) << "Recall@10"
              << std::setw(14) << "Memory(MB)"
              << std::setw(12) << "Comp.Ratio"
              << std::setw(12) << "CacheHit%"
              << std::endl;
    std::cout << std::string(116, '-') << std::endl;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::setw(20) << r.index_type
              << std::setw(8) << r.dimension
              << std::setw(8) << r.block_size
              << std::setw(8) << r.cache_size
              << std::setw(12) << std::fixed << std::setprecision(3) << r.build_time_sec
              << std::setw(12) << std::fixed << std::setprecision(1) << r.search_qps
              << std::setw(10) << std::fixed << std::setprecision(4) << r.recall_at_10
              << std::setw(14) << std::fixed << std::setprecision(2)
              << (r.memory_bytes / (1024.0 * 1024.0))
              << std::setw(12) << std::fixed << std::setprecision(3) << r.compression_ratio
              << std::setw(12) << std::fixed << std::setprecision(2)
              << (r.cache_hit_ratio * 100)
              << std::endl;
}

BenchmarkResult benchmark_hnsw_flat(
        const float* train_data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb,
        size_t nq,
        size_t d,
        size_t k,
        int M,
        int efConstruction,
        int efSearch) {
    BenchmarkResult result;
    result.index_type = "HNSWFlat";
    result.dimension = d;
    result.block_size = 0;
    result.cache_size = 0;
    result.cache_hit_ratio = 0;

    size_t mem_before = get_memory_usage();

    // Build
    auto build_start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, train_data);
    auto build_end = std::chrono::high_resolution_clock::now();
    result.build_time_sec = std::chrono::duration<double>(build_end - build_start).count();

    size_t mem_after = get_memory_usage();
    result.memory_bytes = mem_after - mem_before;

    // Compression ratio (1.0 = no compression)
    result.compression_ratio = 1.0;

    // Search
    index.hnsw.efSearch = efSearch;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    // Warmup
    index.search(nq / 10 + 1, queries, k, distances.data(), labels.data());

    auto search_start = std::chrono::high_resolution_clock::now();
    index.search(nq, queries, k, distances.data(), labels.data());
    auto search_end = std::chrono::high_resolution_clock::now();

    double search_time = std::chrono::duration<double>(search_end - search_start).count();
    result.search_qps = nq / search_time;

    // Recall
    result.recall_at_10 = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

#ifdef FAISS_ENABLE_COMPRESSED_STORAGE

BenchmarkResult benchmark_hnsw_compressed_lz4(
        const float* train_data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb,
        size_t nq,
        size_t d,
        size_t k,
        int M,
        int efConstruction,
        int efSearch,
        size_t block_size,
        size_t cache_size) {
    BenchmarkResult result;
    result.index_type = "HNSW+LZ4";
    result.dimension = d;
    result.block_size = block_size;
    result.cache_size = cache_size;

    size_t mem_before = get_memory_usage();

    // Build
    auto build_start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWCompressedLZ4 index(d, M, 1, block_size, cache_size);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, train_data);
    auto build_end = std::chrono::high_resolution_clock::now();
    result.build_time_sec = std::chrono::duration<double>(build_end - build_start).count();

    size_t mem_after = get_memory_usage();
    result.memory_bytes = mem_after - mem_before;

    // Compression ratio
    auto* storage = index.get_compressed_storage();
    auto comp_stats = storage->get_compression_stats();
    result.compression_ratio = comp_stats.compression_ratio;

    // Search
    index.hnsw.efSearch = efSearch;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    // Warmup
    storage->reset_stats();
    index.search(nq / 10 + 1, queries, k, distances.data(), labels.data());

    storage->reset_stats();
    auto search_start = std::chrono::high_resolution_clock::now();
    index.search(nq, queries, k, distances.data(), labels.data());
    auto search_end = std::chrono::high_resolution_clock::now();

    double search_time = std::chrono::duration<double>(search_end - search_start).count();
    result.search_qps = nq / search_time;

    // Cache stats
    auto cache_stats = storage->get_cache_stats();
    result.cache_hit_ratio = cache_stats.hit_ratio;

    // Recall
    result.recall_at_10 = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

BenchmarkResult benchmark_hnsw_compressed_zstd(
        const float* train_data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb,
        size_t nq,
        size_t d,
        size_t k,
        int M,
        int efConstruction,
        int efSearch,
        size_t block_size,
        size_t cache_size,
        int zstd_level) {
    BenchmarkResult result;
    result.index_type = "HNSW+ZSTD" + std::to_string(zstd_level);
    result.dimension = d;
    result.block_size = block_size;
    result.cache_size = cache_size;

    size_t mem_before = get_memory_usage();

    // Build
    auto build_start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWCompressedZSTD index(d, M, zstd_level, block_size, cache_size);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, train_data);
    auto build_end = std::chrono::high_resolution_clock::now();
    result.build_time_sec = std::chrono::duration<double>(build_end - build_start).count();

    size_t mem_after = get_memory_usage();
    result.memory_bytes = mem_after - mem_before;

    // Compression ratio
    auto* storage = index.get_compressed_storage();
    auto comp_stats = storage->get_compression_stats();
    result.compression_ratio = comp_stats.compression_ratio;

    // Search
    index.hnsw.efSearch = efSearch;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    // Warmup
    storage->reset_stats();
    index.search(nq / 10 + 1, queries, k, distances.data(), labels.data());

    storage->reset_stats();
    auto search_start = std::chrono::high_resolution_clock::now();
    index.search(nq, queries, k, distances.data(), labels.data());
    auto search_end = std::chrono::high_resolution_clock::now();

    double search_time = std::chrono::duration<double>(search_end - search_start).count();
    result.search_qps = nq / search_time;

    // Cache stats
    auto cache_stats = storage->get_cache_stats();
    result.cache_hit_ratio = cache_stats.hit_ratio;

    // Recall
    result.recall_at_10 = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

#endif // FAISS_ENABLE_COMPRESSED_STORAGE

} // anonymous namespace

int main(int argc, char* argv[]) {
    // Parameters
    size_t nb = 100000;        // Number of database vectors (use 1M for full benchmark)
    size_t nq = 1000;          // Number of queries
    size_t k = 10;             // Top-k
    int M = 32;                // HNSW M parameter
    int efConstruction = 40;   // HNSW construction parameter
    int efSearch = 64;         // HNSW search parameter

    // Parse command line
    if (argc > 1) {
        nb = std::atoi(argv[1]);
    }
    if (argc > 2) {
        nq = std::atoi(argv[2]);
    }

    std::cout << "=== HNSW Compressed Storage Benchmark ===" << std::endl;
    std::cout << "Database size: " << nb << " vectors" << std::endl;
    std::cout << "Queries: " << nq << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "HNSW M: " << M << ", efConstruction: " << efConstruction
              << ", efSearch: " << efSearch << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << std::endl;

    // Dimensions to test
    std::vector<size_t> dimensions = {128, 256, 512, 1024};

    // Block sizes to test
    std::vector<size_t> block_sizes = {16, 32, 64, 128};

    // Cache sizes to test
    std::vector<size_t> cache_sizes = {8, 16, 32};

    for (size_t d : dimensions) {
        std::cout << "\n=== Dimension: " << d << " ===" << std::endl;

        // Generate data
        std::cout << "Generating " << nb << " random vectors of dimension " << d << "..." << std::endl;
        auto database = generate_random_vectors(nb, d, 42);
        auto queries = generate_random_vectors(nq, d, 123);

        // Compute ground truth
        std::cout << "Computing ground truth..." << std::endl;
        auto ground_truth = compute_ground_truth(
                queries.data(), database.data(), nq, nb, d, k);

        std::cout << std::endl;
        print_header();

        // Baseline: HNSW Flat
        auto baseline = benchmark_hnsw_flat(
                database.data(),
                queries.data(),
                ground_truth.data(),
                nb, nq, d, k,
                M, efConstruction, efSearch);
        print_result(baseline);

#ifdef FAISS_ENABLE_COMPRESSED_STORAGE
        // Test different block sizes with default cache
        for (size_t block_size : block_sizes) {
            auto result_lz4 = benchmark_hnsw_compressed_lz4(
                    database.data(),
                    queries.data(),
                    ground_truth.data(),
                    nb, nq, d, k,
                    M, efConstruction, efSearch,
                    block_size, 16);
            print_result(result_lz4);
        }

        // Test different cache sizes with default block size
        for (size_t cache_size : cache_sizes) {
            if (cache_size == 16) continue; // Already tested above
            auto result_lz4 = benchmark_hnsw_compressed_lz4(
                    database.data(),
                    queries.data(),
                    ground_truth.data(),
                    nb, nq, d, k,
                    M, efConstruction, efSearch,
                    32, cache_size);
            print_result(result_lz4);
        }

        // Test ZSTD with different compression levels
        for (int level : {1, 3, 9}) {
            auto result_zstd = benchmark_hnsw_compressed_zstd(
                    database.data(),
                    queries.data(),
                    ground_truth.data(),
                    nb, nq, d, k,
                    M, efConstruction, efSearch,
                    32, 16, level);
            print_result(result_zstd);
        }
#else
        std::cout << "\nNote: Compressed storage not enabled. "
                  << "Rebuild with -DFAISS_ENABLE_COMPRESSED_STORAGE=ON" << std::endl;
#endif
    }

    std::cout << "\n=== Benchmark Complete ===" << std::endl;

    return 0;
}
