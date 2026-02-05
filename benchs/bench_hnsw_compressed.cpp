/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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

size_t get_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<size_t>(usage.ru_maxrss) * 1024;
}

std::vector<float> generate_random_vectors(size_t n, size_t d, unsigned seed = 42) {
    std::vector<float> data(n * d);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < n * d; i++) {
        data[i] = dist(rng);
    }

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
    bool use_cache;
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
              << std::setw(8) << "Cache"
              << std::setw(10) << "CacheSize"
              << std::setw(12) << "Build(s)"
              << std::setw(12) << "QPS"
              << std::setw(10) << "Recall@10"
              << std::setw(14) << "Memory(MB)"
              << std::setw(12) << "Comp.Ratio"
              << std::setw(12) << "CacheHit%"
              << std::endl;
    std::cout << std::string(118, '-') << std::endl;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::setw(20) << r.index_type
              << std::setw(8) << r.dimension
              << std::setw(8) << (r.use_cache ? "ON" : "OFF")
              << std::setw(10) << r.cache_size
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
    result.use_cache = false;
    result.cache_size = 0;
    result.cache_hit_ratio = 0;

    size_t mem_before = get_memory_usage();

    auto build_start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, train_data);
    auto build_end = std::chrono::high_resolution_clock::now();
    result.build_time_sec = std::chrono::duration<double>(build_end - build_start).count();

    size_t mem_after = get_memory_usage();
    result.memory_bytes = mem_after - mem_before;
    result.compression_ratio = 1.0;

    index.hnsw.efSearch = efSearch;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    index.search(nq / 10 + 1, queries, k, distances.data(), labels.data());

    auto search_start = std::chrono::high_resolution_clock::now();
    index.search(nq, queries, k, distances.data(), labels.data());
    auto search_end = std::chrono::high_resolution_clock::now();

    double search_time = std::chrono::duration<double>(search_end - search_start).count();
    result.search_qps = nq / search_time;
    result.recall_at_10 = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

#ifdef FAISS_ENABLE_COMPRESSED_STORAGE

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
        int zstd_level,
        bool use_cache,
        size_t cache_size) {
    BenchmarkResult result;
    result.index_type = "HNSW+ZSTD" + std::to_string(zstd_level);
    result.dimension = d;
    result.use_cache = use_cache;
    result.cache_size = cache_size;

    size_t mem_before = get_memory_usage();

    auto build_start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWCompressedZSTD index(d, M, zstd_level, use_cache, cache_size);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, train_data);
    auto build_end = std::chrono::high_resolution_clock::now();
    result.build_time_sec = std::chrono::duration<double>(build_end - build_start).count();

    size_t mem_after = get_memory_usage();
    result.memory_bytes = mem_after - mem_before;

    auto* storage = index.get_compressed_storage();
    auto comp_stats = storage->get_compression_stats();
    result.compression_ratio = comp_stats.compression_ratio;

    index.hnsw.efSearch = efSearch;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    storage->reset_stats();
    index.search(nq / 10 + 1, queries, k, distances.data(), labels.data());

    storage->reset_stats();
    auto search_start = std::chrono::high_resolution_clock::now();
    index.search(nq, queries, k, distances.data(), labels.data());
    auto search_end = std::chrono::high_resolution_clock::now();

    double search_time = std::chrono::duration<double>(search_end - search_start).count();
    result.search_qps = nq / search_time;

    auto cache_stats = storage->get_cache_stats();
    result.cache_hit_ratio = cache_stats.hit_ratio;

    result.recall_at_10 = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

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
        bool use_cache,
        size_t cache_size) {
    BenchmarkResult result;
    result.index_type = "HNSW+LZ4";
    result.dimension = d;
    result.use_cache = use_cache;
    result.cache_size = cache_size;

    size_t mem_before = get_memory_usage();

    auto build_start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWCompressedLZ4 index(d, M, 1, use_cache, cache_size);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, train_data);
    auto build_end = std::chrono::high_resolution_clock::now();
    result.build_time_sec = std::chrono::duration<double>(build_end - build_start).count();

    size_t mem_after = get_memory_usage();
    result.memory_bytes = mem_after - mem_before;

    auto* storage = index.get_compressed_storage();
    auto comp_stats = storage->get_compression_stats();
    result.compression_ratio = comp_stats.compression_ratio;

    index.hnsw.efSearch = efSearch;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    storage->reset_stats();
    index.search(nq / 10 + 1, queries, k, distances.data(), labels.data());

    storage->reset_stats();
    auto search_start = std::chrono::high_resolution_clock::now();
    index.search(nq, queries, k, distances.data(), labels.data());
    auto search_end = std::chrono::high_resolution_clock::now();

    double search_time = std::chrono::duration<double>(search_end - search_start).count();
    result.search_qps = nq / search_time;

    auto cache_stats = storage->get_cache_stats();
    result.cache_hit_ratio = cache_stats.hit_ratio;

    result.recall_at_10 = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

#endif

}

int main(int argc, char* argv[]) {
    size_t nb = 100000;
    size_t nq = 1000;
    size_t k = 10;
    int M = 32;
    int efConstruction = 40;
    int efSearch = 64;

    if (argc > 1) {
        nb = std::atoi(argv[1]);
    }
    if (argc > 2) {
        nq = std::atoi(argv[2]);
    }

    std::cout << "=== HNSW Compressed Storage Benchmark (Per-Vector) ===" << std::endl;
    std::cout << "Database size: " << nb << " vectors" << std::endl;
    std::cout << "Queries: " << nq << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "HNSW M: " << M << ", efConstruction: " << efConstruction
              << ", efSearch: " << efSearch << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << std::endl;

    std::vector<size_t> dimensions = {128, 256, 512, 1024};
    std::vector<size_t> cache_sizes = {16, 32, 64, 128};

    for (size_t d : dimensions) {
        std::cout << "\n=== Dimension: " << d << " ===" << std::endl;

        std::cout << "Generating " << nb << " random vectors of dimension " << d << "..." << std::endl;
        auto database = generate_random_vectors(nb, d, 42);
        auto queries = generate_random_vectors(nq, d, 123);

        std::cout << "Computing ground truth..." << std::endl;
        auto ground_truth = compute_ground_truth(
                queries.data(), database.data(), nq, nb, d, k);

        std::cout << std::endl;
        print_header();

        auto baseline = benchmark_hnsw_flat(
                database.data(),
                queries.data(),
                ground_truth.data(),
                nb, nq, d, k,
                M, efConstruction, efSearch);
        print_result(baseline);

#ifdef FAISS_ENABLE_COMPRESSED_STORAGE
        std::cout << "\n--- ZSTD Per-Vector Compression ---" << std::endl;
        print_header();

        for (int level : {1, 3}) {
            auto result_cache_on = benchmark_hnsw_compressed_zstd(
                    database.data(),
                    queries.data(),
                    ground_truth.data(),
                    nb, nq, d, k,
                    M, efConstruction, efSearch,
                    level, true, 64);
            print_result(result_cache_on);

            auto result_cache_off = benchmark_hnsw_compressed_zstd(
                    database.data(),
                    queries.data(),
                    ground_truth.data(),
                    nb, nq, d, k,
                    M, efConstruction, efSearch,
                    level, false, 0);
            print_result(result_cache_off);
        }

        std::cout << "\n--- Cache Size Comparison (ZSTD level=3, cache ON) ---" << std::endl;
        print_header();

        for (size_t cache_size : cache_sizes) {
            auto result = benchmark_hnsw_compressed_zstd(
                    database.data(),
                    queries.data(),
                    ground_truth.data(),
                    nb, nq, d, k,
                    M, efConstruction, efSearch,
                    3, true, cache_size);
            print_result(result);
        }

        std::cout << "\n--- LZ4 Per-Vector Compression ---" << std::endl;
        print_header();

        auto lz4_cache_on = benchmark_hnsw_compressed_lz4(
                database.data(),
                queries.data(),
                ground_truth.data(),
                nb, nq, d, k,
                M, efConstruction, efSearch,
                true, 64);
        print_result(lz4_cache_on);

        auto lz4_cache_off = benchmark_hnsw_compressed_lz4(
                database.data(),
                queries.data(),
                ground_truth.data(),
                nb, nq, d, k,
                M, efConstruction, efSearch,
                false, 0);
        print_result(lz4_cache_off);
#else
        std::cout << "\nNote: Compressed storage not enabled. "
                  << "Rebuild with -DFAISS_ENABLE_COMPRESSED_STORAGE=ON" << std::endl;
#endif
    }

    std::cout << "\n=== Benchmark Complete ===" << std::endl;

    return 0;
}
