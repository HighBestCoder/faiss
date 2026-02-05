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
#include <deque>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/distances.h>

#include <omp.h>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#ifdef ENABLE_VSAG
#include <vsag/vsag.h>
#endif

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

namespace {

double get_time_sec() {
    return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
}

size_t get_memory_usage_kb() {
#ifdef __linux__
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<size_t>(usage.ru_maxrss);
#else
    return 0;
#endif
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
    for (int64_t q = 0; q < (int64_t)nq; q++) {
        const float* query = queries + q * d;
        std::vector<std::pair<float, faiss::idx_t>> distances(nb);

        for (size_t i = 0; i < nb; i++) {
            const float* vec = database + i * d;
            float dist = faiss::fvec_L2sqr(query, vec, d);
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
        std::vector<faiss::idx_t> gt_set(ground_truth + q * k, ground_truth + (q + 1) * k);
        std::sort(gt_set.begin(), gt_set.end());
        for (size_t i = 0; i < k; i++) {
            if (std::binary_search(gt_set.begin(), gt_set.end(), results[q * k + i])) {
                correct++;
            }
        }
    }
    return static_cast<double>(correct) / (nq * k);
}

struct BenchmarkResult {
    std::string name;
    double build_time_sec;
    double search_time_sec;
    double search_qps;
    double recall;
    size_t memory_kb;
};

void print_header() {
    std::cout << std::setw(35) << std::left << "Index Type"
              << std::setw(12) << std::right << "Build(s)"
              << std::setw(12) << "Search(s)"
              << std::setw(15) << "QPS"
              << std::setw(12) << "Recall@10"
              << std::setw(14) << "Memory(MB)" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::setw(35) << std::left << r.name
              << std::setw(12) << std::right << std::fixed << std::setprecision(3) << r.build_time_sec
              << std::setw(12) << std::setprecision(3) << r.search_time_sec
              << std::setw(15) << std::setprecision(1) << r.search_qps
              << std::setw(12) << std::setprecision(4) << r.recall
              << std::setw(14) << std::setprecision(2) << r.memory_kb / 1024.0
              << std::endl;
}

class IndexCacheAlignedFlat : public faiss::Index {
public:
    size_t d;
    size_t vector_size_bytes;
    size_t aligned_vector_size;
    std::vector<float*> vectors;
    
    explicit IndexCacheAlignedFlat(faiss::idx_t dim)
            : faiss::Index(dim, faiss::METRIC_L2), d(dim) {
        vector_size_bytes = d * sizeof(float);
        aligned_vector_size = ((vector_size_bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
    }
    
    ~IndexCacheAlignedFlat() override {
        for (float* ptr : vectors) {
            if (ptr) {
                free(ptr);
            }
        }
    }
    
    const float* get_vector(faiss::idx_t i) const {
        return vectors[i];
    }
    
    void add(faiss::idx_t n, const float* x) override {
        for (faiss::idx_t i = 0; i < n; i++) {
            float* aligned_ptr = static_cast<float*>(aligned_alloc(CACHE_LINE_SIZE, aligned_vector_size));
            std::memcpy(aligned_ptr, x + i * d, d * sizeof(float));
            vectors.push_back(aligned_ptr);
        }
        ntotal += n;
    }
    
    void search(
            faiss::idx_t,
            const float*,
            faiss::idx_t,
            float*,
            faiss::idx_t*,
            const faiss::SearchParameters* = nullptr) const override {
    }
    
    void reset() override {
        for (float* ptr : vectors) {
            free(ptr);
        }
        vectors.clear();
        ntotal = 0;
    }
    
    faiss::DistanceComputer* get_distance_computer() const override;
};

struct CacheAlignedFlatL2Dis : faiss::DistanceComputer {
    const IndexCacheAlignedFlat& storage;
    size_t d;
    std::vector<float> q_copy;

    explicit CacheAlignedFlatL2Dis(const IndexCacheAlignedFlat& s)
            : storage(s), d(s.d), q_copy(s.d) {}

    void set_query(const float* x) override {
        std::memcpy(q_copy.data(), x, d * sizeof(float));
    }

    float operator()(faiss::idx_t i) override {
        return faiss::fvec_L2sqr(q_copy.data(), storage.get_vector(i), d);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        return faiss::fvec_L2sqr(storage.get_vector(i), storage.get_vector(j), d);
    }
    
    void distances_batch_4(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        faiss::fvec_L2sqr_batch_4(
                q_copy.data(),
                storage.get_vector(idx0),
                storage.get_vector(idx1),
                storage.get_vector(idx2),
                storage.get_vector(idx3),
                d,
                dis0, dis1, dis2, dis3);
    }

    void distances_batch_8(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            const faiss::idx_t idx4,
            const faiss::idx_t idx5,
            const faiss::idx_t idx6,
            const faiss::idx_t idx7,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3,
            float& dis4,
            float& dis5,
            float& dis6,
            float& dis7) override {
        faiss::fvec_L2sqr_batch_8(
                q_copy.data(),
                storage.get_vector(idx0),
                storage.get_vector(idx1),
                storage.get_vector(idx2),
                storage.get_vector(idx3),
                storage.get_vector(idx4),
                storage.get_vector(idx5),
                storage.get_vector(idx6),
                storage.get_vector(idx7),
                d,
                dis0, dis1, dis2, dis3, dis4, dis5, dis6, dis7);
    }
};

faiss::DistanceComputer* IndexCacheAlignedFlat::get_distance_computer() const {
    return new CacheAlignedFlatL2Dis(*this);
}

std::vector<faiss::idx_t> generate_bfs_permutation(const faiss::HNSW& hnsw) {
    size_t ntotal = hnsw.levels.size();
    std::vector<faiss::idx_t> perm;
    perm.reserve(ntotal);
    
    std::vector<bool> visited(ntotal, false);
    std::deque<faiss::HNSW::storage_idx_t> bfs_queue;
    
    if (hnsw.entry_point >= 0) {
        bfs_queue.push_back(hnsw.entry_point);
        visited[hnsw.entry_point] = true;
    }
    
    while (!bfs_queue.empty()) {
        faiss::HNSW::storage_idx_t current = bfs_queue.front();
        bfs_queue.pop_front();
        perm.push_back(current);
        
        size_t begin, end;
        hnsw.neighbor_range(current, 0, &begin, &end);
        
        for (size_t j = begin; j < end; j++) {
            faiss::HNSW::storage_idx_t neighbor = hnsw.neighbors[j];
            if (neighbor >= 0 && !visited[neighbor]) {
                visited[neighbor] = true;
                bfs_queue.push_back(neighbor);
            }
        }
    }
    
    for (size_t i = 0; i < ntotal; i++) {
        if (!visited[i]) {
            perm.push_back(i);
        }
    }
    
    return perm;
}

BenchmarkResult benchmark_faiss_hnsw(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb, size_t nq, size_t d, size_t k,
        int M, int efConstruction, int efSearch) {
    BenchmarkResult result;
    result.name = "FAISS IndexHNSWFlat";

    size_t mem_before = get_memory_usage_kb();
    double t0 = get_time_sec();
    
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;
    result.memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.hnsw.efSearch = efSearch;
    
    index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

    double t2 = get_time_sec();
    index.search(nq, queries, k, distances.data(), labels.data());
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;
    result.recall = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

BenchmarkResult benchmark_faiss_cache_aligned(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb, size_t nq, size_t d, size_t k,
        int M, int efConstruction, int efSearch) {
    BenchmarkResult result;
    result.name = "FAISS CacheAligned";

    size_t mem_before = get_memory_usage_kb();
    double t0 = get_time_sec();
    
    IndexCacheAlignedFlat* storage = new IndexCacheAlignedFlat(d);
    faiss::IndexHNSW index(storage, M);
    index.own_fields = true;
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;
    result.memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.hnsw.efSearch = efSearch;
    
    index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

    double t2 = get_time_sec();
    index.search(nq, queries, k, distances.data(), labels.data());
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;
    result.recall = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

BenchmarkResult benchmark_faiss_graph_reorder(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb, size_t nq, size_t d, size_t k,
        int M, int efConstruction, int efSearch) {
    BenchmarkResult result;
    result.name = "FAISS GraphReorder";

    size_t mem_before = get_memory_usage_kb();
    double t0 = get_time_sec();
    
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    auto perm = generate_bfs_permutation(index.hnsw);
    index.permute_entries(perm.data());
    
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;
    result.memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.hnsw.efSearch = efSearch;
    
    index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

    double t2 = get_time_sec();
    index.search(nq, queries, k, distances.data(), labels.data());
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;

    std::vector<faiss::idx_t> inverse_perm(nb);
    for (size_t i = 0; i < nb; i++) {
        inverse_perm[perm[i]] = i;
    }
    std::vector<faiss::idx_t> remapped_gt(nq * k);
    for (size_t i = 0; i < nq * k; i++) {
        remapped_gt[i] = inverse_perm[ground_truth[i]];
    }
    result.recall = compute_recall(labels.data(), remapped_gt.data(), nq, k);

    return result;
}

#ifdef ENABLE_VSAG
BenchmarkResult benchmark_vsag_hnsw(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb, size_t nq, size_t d, size_t k,
        int M, int efConstruction, int efSearch) {
    BenchmarkResult result;
    result.name = "VSAG HNSW";

    size_t mem_before = get_memory_usage_kb();
    double t0 = get_time_sec();

    std::string build_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": )" + std::to_string(d) + R"(,
        "hnsw": {
            "max_degree": )" + std::to_string(M) + R"(,
            "ef_construction": )" + std::to_string(efConstruction) + R"(
        }
    })";
    
    auto index_result = vsag::Factory::CreateIndex("hnsw", build_params);
    if (!index_result.has_value()) {
        std::cerr << "VSAG: Failed to create index: " << index_result.error().message << std::endl;
        result.name = "VSAG HNSW (FAILED)";
        return result;
    }
    auto index = index_result.value();

    auto ids = new int64_t[nb];
    for (size_t i = 0; i < nb; i++) {
        ids[i] = i;
    }
    
    auto base = vsag::Dataset::Make();
    base->NumElements(nb)->Dim(d)->Ids(ids)->Float32Vectors(const_cast<float*>(data))->Owner(false);
    
    auto build_result = index->Build(base);
    if (!build_result.has_value()) {
        std::cerr << "VSAG: Failed to build index: " << build_result.error().message << std::endl;
        result.name = "VSAG HNSW (FAILED)";
        delete[] ids;
        return result;
    }
    
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;
    result.memory_kb = get_memory_usage_kb() - mem_before;

    std::string search_params = R"({
        "hnsw": {
            "ef_search": )" + std::to_string(efSearch) + R"(
        }
    })";

    std::vector<int64_t> all_results(nq * k);
    
    auto warmup_query = vsag::Dataset::Make();
    warmup_query->NumElements(1)->Dim(d)->Float32Vectors(const_cast<float*>(queries))->Owner(false);
    index->KnnSearch(warmup_query, k, search_params);

    double t2 = get_time_sec();
    
    for (size_t q = 0; q < nq; q++) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(d)->Float32Vectors(const_cast<float*>(queries + q * d))->Owner(false);
        auto knn_result = index->KnnSearch(query, k, search_params);
        if (knn_result.has_value()) {
            auto res = knn_result.value();
            for (size_t i = 0; i < k && i < res->GetDim(); i++) {
                all_results[q * k + i] = res->GetIds()[i];
            }
        }
    }
    
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;

    size_t correct = 0;
    for (size_t q = 0; q < nq; q++) {
        std::vector<faiss::idx_t> gt_set(ground_truth + q * k, ground_truth + (q + 1) * k);
        std::sort(gt_set.begin(), gt_set.end());
        for (size_t i = 0; i < k; i++) {
            if (std::binary_search(gt_set.begin(), gt_set.end(), all_results[q * k + i])) {
                correct++;
            }
        }
    }
    result.recall = static_cast<double>(correct) / (nq * k);

    delete[] ids;
    return result;
}
#endif

}

int main(int argc, char* argv[]) {
    size_t nb = 100000;
    size_t d = 128;
    size_t nq = 2000;
    size_t k = 10;
    int M = 32;
    int efConstruction = 40;
    int efSearch = 64;

    if (argc > 1) nb = std::atoi(argv[1]);
    if (argc > 2) d = std::atoi(argv[2]);
    if (argc > 3) nq = std::atoi(argv[3]);
    if (argc > 4) efSearch = std::atoi(argv[4]);
    if (argc > 5) M = std::atoi(argv[5]);
    if (argc > 6) efConstruction = std::atoi(argv[6]);

    std::cout << "================================================================" << std::endl;
    std::cout << "  FAISS vs VSAG HNSW Comparison Benchmark" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Database size: " << nb << " vectors" << std::endl;
    std::cout << "  Dimension: " << d << std::endl;
    std::cout << "  Queries: " << nq << std::endl;
    std::cout << "  k: " << k << std::endl;
    std::cout << "  HNSW M: " << M << std::endl;
    std::cout << "  HNSW efConstruction: " << efConstruction << std::endl;
    std::cout << "  HNSW efSearch: " << efSearch << std::endl;
    std::cout << "  Threads: " << omp_get_max_threads() << std::endl;
#ifdef ENABLE_VSAG
    std::cout << "  VSAG: ENABLED" << std::endl;
#else
    std::cout << "  VSAG: DISABLED (compile with -DENABLE_VSAG)" << std::endl;
#endif
    std::cout << std::endl;

    std::cout << "Generating " << nb << " random vectors of dimension " << d << "..." << std::endl;
    auto database = generate_random_vectors(nb, d, 42);
    
    std::cout << "Generating " << nq << " query vectors..." << std::endl;
    auto queries = generate_random_vectors(nq, d, 123);
    
    std::cout << "Computing ground truth..." << std::endl;
    auto ground_truth = compute_ground_truth(queries.data(), database.data(), nq, nb, d, k);

    std::cout << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "  Results" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    print_header();

    auto result_faiss = benchmark_faiss_hnsw(
            database.data(), queries.data(), ground_truth.data(),
            nb, nq, d, k, M, efConstruction, efSearch);
    print_result(result_faiss);

    auto result_aligned = benchmark_faiss_cache_aligned(
            database.data(), queries.data(), ground_truth.data(),
            nb, nq, d, k, M, efConstruction, efSearch);
    print_result(result_aligned);

    auto result_reorder = benchmark_faiss_graph_reorder(
            database.data(), queries.data(), ground_truth.data(),
            nb, nq, d, k, M, efConstruction, efSearch);
    print_result(result_reorder);

#ifdef ENABLE_VSAG
    auto result_vsag = benchmark_vsag_hnsw(
            database.data(), queries.data(), ground_truth.data(),
            nb, nq, d, k, M, efConstruction, efSearch);
    print_result(result_vsag);
#endif

    std::cout << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "  Summary" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "FAISS baseline QPS: " << std::fixed << std::setprecision(1) << result_faiss.search_qps << std::endl;
    std::cout << "CacheAligned vs baseline: " << std::setprecision(1) 
              << ((result_aligned.search_qps / result_faiss.search_qps - 1) * 100) << "%" << std::endl;
    std::cout << "GraphReorder vs baseline: " << std::setprecision(1)
              << ((result_reorder.search_qps / result_faiss.search_qps - 1) * 100) << "%" << std::endl;
#ifdef ENABLE_VSAG
    std::cout << "VSAG vs FAISS baseline: " << std::setprecision(1)
              << ((result_vsag.search_qps / result_faiss.search_qps - 1) * 100) << "%" << std::endl;
#endif

    std::cout << std::endl;
    std::cout << "Benchmark complete." << std::endl;

    return 0;
}
