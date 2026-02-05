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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/IDSelector.h>

#include <omp.h>

namespace {

struct IDSelectorTombstone : faiss::IDSelector {
    const std::unordered_set<faiss::idx_t>& tombstoned;

    explicit IDSelectorTombstone(
            const std::unordered_set<faiss::idx_t>& tombstoned_set)
            : tombstoned(tombstoned_set) {}

    bool is_member(faiss::idx_t id) const override {
        return tombstoned.find(id) == tombstoned.end();
    }

    ~IDSelectorTombstone() override = default;
};

std::vector<float> generate_random_vectors(size_t n, size_t d, unsigned seed) {
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

double measure_search_qps(
        const faiss::IndexHNSW& index,
        const float* queries,
        size_t nq,
        size_t k,
        const std::unordered_set<faiss::idx_t>& tombstoned,
        int num_runs = 3) {
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    IDSelectorTombstone selector(tombstoned);
    faiss::SearchParameters params;
    params.sel = tombstoned.empty() ? nullptr : &selector;

    index.search(nq, queries, k, distances.data(), labels.data(), &params);

    double total_time = 0;
    for (int run = 0; run < num_runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        index.search(nq, queries, k, distances.data(), labels.data(), &params);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double>(end - start).count();
    }

    return (nq * num_runs) / total_time;
}

}  // namespace

int main(int argc, char* argv[]) {
    size_t base_size = 100000;
    size_t nq = 500;
    size_t d = 256;
    size_t k = 10;
    int M = 32;
    int efConstruction = 40;
    int efSearch = 64;

    std::string output_file = "test2.txt";

    if (argc > 1) {
        base_size = std::atoi(argv[1]);
    }
    if (argc > 2) {
        nq = std::atoi(argv[2]);
    }
    if (argc > 3) {
        d = std::atoi(argv[3]);
    }
    if (argc > 4) {
        output_file = argv[4];
    }

    std::ofstream ofs(output_file);
    auto log = [&](const std::string& msg) {
        std::cout << msg;
        ofs << msg;
    };

    std::ostringstream ss;
    ss << "=== HNSW Fixed Active Vectors Benchmark ===\n";
    ss << "Base size (always active): " << base_size << " vectors\n";
    ss << "Dimension: " << d << "\n";
    ss << "Queries: " << nq << "\n";
    ss << "k: " << k << "\n";
    ss << "HNSW M: " << M << ", efConstruction: " << efConstruction
       << ", efSearch: " << efSearch << "\n";
    ss << "Threads: " << omp_get_max_threads() << "\n\n";
    log(ss.str());

    ss.str("");
    ss << "Generating " << base_size << " base vectors of dimension " << d << "...\n";
    log(ss.str());
    auto base_vectors = generate_random_vectors(base_size, d, 42);

    ss.str("");
    ss << "Generating " << nq << " query vectors...\n";
    log(ss.str());
    auto queries = generate_random_vectors(nq, d, 123);

    ss.str("");
    ss << "Building initial HNSW index with " << base_size << " vectors...\n";
    log(ss.str());

    auto build_start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.add(base_size, base_vectors.data());
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(build_end - build_start).count();

    ss.str("");
    ss << "Index built in " << std::fixed << std::setprecision(2) << build_time << " seconds\n\n";
    log(ss.str());

    index.hnsw.efSearch = efSearch;

    std::unordered_set<faiss::idx_t> tombstoned;

    ss.str("");
    ss << std::setw(18) << "Deleted Count"
       << std::setw(18) << "Total Vectors"
       << std::setw(18) << "Active Vectors"
       << std::setw(15) << "QPS"
       << std::setw(15) << "QPS Ratio" << "\n";
    ss << std::string(84, '-') << "\n";
    log(ss.str());

    double baseline_qps = measure_search_qps(index, queries.data(), nq, k, tombstoned, 5);

    ss.str("");
    ss << std::setw(18) << 0
       << std::setw(18) << base_size
       << std::setw(18) << base_size
       << std::setw(15) << std::fixed << std::setprecision(1) << baseline_qps
       << std::setw(14) << std::fixed << std::setprecision(3) << 1.0 << "x\n";
    log(ss.str());

    std::mt19937 rng(789);
    faiss::idx_t next_id = base_size;

    std::vector<size_t> batch_sizes = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    
    for (size_t batch : batch_sizes) {
        if (batch > base_size) break;

        for (size_t i = 0; i < batch; i++) {
            faiss::idx_t id_to_delete;
            do {
                id_to_delete = rng() % next_id;
            } while (tombstoned.count(id_to_delete) > 0);
            tombstoned.insert(id_to_delete);
        }

        auto new_vectors = generate_random_vectors(batch, d, rng());
        index.add(batch, new_vectors.data());
        next_id += batch;

        size_t total_vectors = next_id;
        size_t active_vectors = total_vectors - tombstoned.size();

        double qps = measure_search_qps(index, queries.data(), nq, k, tombstoned, 5);
        double qps_ratio = qps / baseline_qps;

        ss.str("");
        ss << std::setw(18) << tombstoned.size()
           << std::setw(18) << total_vectors
           << std::setw(18) << active_vectors
           << std::setw(15) << std::fixed << std::setprecision(1) << qps
           << std::setw(14) << std::fixed << std::setprecision(3) << qps_ratio << "x\n";
        log(ss.str());
    }

    size_t cumulative_deleted = tombstoned.size();
    while (cumulative_deleted < base_size) {
        size_t batch = std::min(base_size / 10, base_size - cumulative_deleted);
        if (batch == 0) break;

        for (size_t i = 0; i < batch; i++) {
            faiss::idx_t id_to_delete;
            int attempts = 0;
            do {
                id_to_delete = rng() % next_id;
                attempts++;
                if (attempts > 10000) break;
            } while (tombstoned.count(id_to_delete) > 0);
            if (attempts <= 10000) {
                tombstoned.insert(id_to_delete);
            }
        }

        auto new_vectors = generate_random_vectors(batch, d, rng());
        index.add(batch, new_vectors.data());
        next_id += batch;

        cumulative_deleted = tombstoned.size();
        size_t total_vectors = next_id;
        size_t active_vectors = total_vectors - tombstoned.size();

        double qps = measure_search_qps(index, queries.data(), nq, k, tombstoned, 5);
        double qps_ratio = qps / baseline_qps;

        ss.str("");
        ss << std::setw(18) << tombstoned.size()
           << std::setw(18) << total_vectors
           << std::setw(18) << active_vectors
           << std::setw(15) << std::fixed << std::setprecision(1) << qps
           << std::setw(14) << std::fixed << std::setprecision(3) << qps_ratio << "x\n";
        log(ss.str());
    }

    ss.str("");
    ss << "\n=== Benchmark Complete ===\n";
    log(ss.str());

    ofs.close();
    return 0;
}
