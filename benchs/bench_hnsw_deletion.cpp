/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Benchmark: Impact of deletion ratio on HNSW search performance
 * 
 * Uses a custom IDSelector that holds a reference to a tombstone set,
 * avoiding copy overhead during search.
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
#include <sstream>
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

std::unordered_set<faiss::idx_t> generate_tombstone_set(
        size_t ntotal,
        double delete_ratio,
        unsigned seed) {
    std::unordered_set<faiss::idx_t> tombstoned;
    size_t num_deleted = static_cast<size_t>(ntotal * delete_ratio);

    std::mt19937 rng(seed);
    std::vector<faiss::idx_t> all_ids(ntotal);
    for (size_t i = 0; i < ntotal; i++) {
        all_ids[i] = i;
    }
    std::shuffle(all_ids.begin(), all_ids.end(), rng);

    for (size_t i = 0; i < num_deleted; i++) {
        tombstoned.insert(all_ids[i]);
    }

    return tombstoned;
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
        total_time +=
                std::chrono::duration<double>(end - start).count();
    }

    return (nq * num_runs) / total_time;
}

double measure_recall(
        const faiss::IndexHNSW& index,
        const float* queries,
        const float* database,
        size_t nq,
        size_t nb,
        size_t d,
        size_t k,
        const std::unordered_set<faiss::idx_t>& tombstoned) {
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    IDSelectorTombstone selector(tombstoned);
    faiss::SearchParameters params;
    params.sel = tombstoned.empty() ? nullptr : &selector;

    index.search(nq, queries, k, distances.data(), labels.data(), &params);

    size_t correct = 0;
    size_t total = 0;

#pragma omp parallel for reduction(+ : correct, total)
    for (size_t q = 0; q < nq; q++) {
        const float* query = queries + q * d;

        std::vector<std::pair<float, faiss::idx_t>> gt;
        for (size_t i = 0; i < nb; i++) {
            if (tombstoned.find(i) != tombstoned.end()) {
                continue;
            }
            const float* vec = database + i * d;
            float dist = 0;
            for (size_t j = 0; j < d; j++) {
                float diff = query[j] - vec[j];
                dist += diff * diff;
            }
            gt.emplace_back(dist, i);
        }

        std::partial_sort(
                gt.begin(),
                gt.begin() + std::min(k, gt.size()),
                gt.end(),
                [](auto& a, auto& b) { return a.first < b.first; });

        std::unordered_set<faiss::idx_t> gt_set;
        for (size_t i = 0; i < std::min(k, gt.size()); i++) {
            gt_set.insert(gt[i].second);
        }

        for (size_t i = 0; i < k; i++) {
            if (labels[q * k + i] >= 0 &&
                gt_set.count(labels[q * k + i]) > 0) {
                correct++;
            }
            total++;
        }
    }

    return static_cast<double>(correct) / total;
}

}  // namespace

int main(int argc, char* argv[]) {
    size_t nb = 100000;
    size_t nq = 500;
    size_t d = 256;
    size_t k = 10;
    int M = 32;
    int efConstruction = 40;
    int efSearch = 64;
    std::string output_file = "test1.txt";

    if (argc > 1) {
        nb = std::atoi(argv[1]);
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
    ss << "=== HNSW Deletion Impact Benchmark (Test1: Varying Delete Ratio) ===\n";
    ss << "Database size: " << nb << " vectors\n";
    ss << "Dimension: " << d << "\n";
    ss << "Queries: " << nq << "\n";
    ss << "k: " << k << "\n";
    ss << "HNSW M: " << M << ", efConstruction: " << efConstruction
       << ", efSearch: " << efSearch << "\n";
    ss << "Threads: " << omp_get_max_threads() << "\n\n";
    log(ss.str());

    ss.str("");
    ss << "Generating " << nb << " random vectors of dimension " << d << "...\n";
    log(ss.str());
    auto database = generate_random_vectors(nb, d, 42);

    ss.str("");
    ss << "Generating " << nq << " query vectors...\n";
    log(ss.str());
    auto queries = generate_random_vectors(nq, d, 123);

    ss.str("");
    ss << "Building HNSW index...\n";
    log(ss.str());

    auto build_start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, database.data());
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_time =
            std::chrono::duration<double>(build_end - build_start).count();
    
    ss.str("");
    ss << "Index built in " << std::fixed << std::setprecision(2)
       << build_time << " seconds\n\n";
    log(ss.str());

    index.hnsw.efSearch = efSearch;

    std::vector<double> delete_ratios = {
            0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90};

    ss.str("");
    ss << std::setw(15) << "Delete Ratio" << std::setw(15)
       << "Tombstones" << std::setw(15) << "Active" << std::setw(15) << "QPS" << std::setw(15)
       << "QPS Ratio" << std::setw(15) << "Recall@" << k << "\n";
    ss << std::string(90, '-') << "\n";
    log(ss.str());

    double baseline_qps = 0;

    for (double ratio : delete_ratios) {
        auto tombstoned = generate_tombstone_set(nb, ratio, 456);

        double qps = measure_search_qps(
                index, queries.data(), nq, k, tombstoned, 5);

        if (ratio == 0.0) {
            baseline_qps = qps;
        }

        double qps_ratio = qps / baseline_qps;

        double recall = 0.0;
        if (ratio <= 0.5) {
            recall = measure_recall(
                    index,
                    queries.data(),
                    database.data(),
                    std::min(nq, (size_t)100),
                    nb,
                    d,
                    k,
                    tombstoned);
        }

        size_t active = nb - tombstoned.size();
        ss.str("");
        ss << std::setw(14) << std::fixed << std::setprecision(2)
           << (ratio * 100) << "%" << std::setw(15) << tombstoned.size()
           << std::setw(15) << active
           << std::setw(15) << std::fixed << std::setprecision(1) << qps
           << std::setw(14) << std::fixed << std::setprecision(3)
           << qps_ratio << "x" << std::setw(15) << std::fixed
           << std::setprecision(4) << recall << "\n";
        log(ss.str());
    }

    ss.str("");
    ss << "\n=== Benchmark Complete ===\n";
    log(ss.str());

    ofs.close();
    return 0;
}
