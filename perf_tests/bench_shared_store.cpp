/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexFlatShared.h>
#include <faiss/IndexHNSW.h>

namespace {

struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0;
    Timer() : t0(Clock::now()) {}
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0)
                .count();
    }
};

void generate_random_vectors(
        std::vector<float>& out,
        size_t n,
        size_t d,
        unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    out.resize(n * d);
    for (size_t i = 0; i < n * d; i++) {
        out[i] = dist(rng);
    }
}

} // namespace

int main(int argc, char** argv) {
    size_t nb = 100000;
    size_t d = 128;
    size_t nq = 1000;
    int k = 10;
    int M = 32;
    int efConstruction = 40;
    int efSearch = 64;

    if (argc > 1)
        nb = std::atol(argv[1]);
    if (argc > 2)
        d = std::atol(argv[2]);
    if (argc > 3)
        nq = std::atol(argv[3]);

    printf("=== SharedVectorStore Benchmark ===\n");
    printf("nb=%zu  d=%zu  nq=%zu  k=%d  M=%d  efConstruction=%d  efSearch=%d\n\n",
           nb, d, nq, k, M, efConstruction, efSearch);

    std::vector<float> xb, xq;
    generate_random_vectors(xb, nb, d, 42);
    generate_random_vectors(xq, nq, d, 123);

    // ============================================================
    // 1. Build IndexHNSWFlat (baseline)
    // ============================================================
    printf("--- Building IndexHNSWFlat (baseline) ---\n");
    faiss::IndexHNSWFlat baseline(d, M);
    baseline.hnsw.efConstruction = efConstruction;
    {
        Timer t;
        baseline.add(nb, xb.data());
        printf("  Build time: %.1f ms\n", t.elapsed_ms());
    }
    baseline.hnsw.efSearch = efSearch;

    // ============================================================
    // 2. Build IndexHNSW + IndexFlatShared (our implementation)
    // ============================================================
    printf("--- Building IndexHNSW + IndexFlatShared ---\n");
    auto store = std::make_shared<faiss::SharedVectorStore>(d, d * sizeof(float));
    store->reserve(nb);
    auto* shared_storage = new faiss::IndexFlatShared(store, faiss::METRIC_L2);
    auto* shared_index = new faiss::IndexHNSW(shared_storage, M);
    shared_index->own_fields = true;
    shared_index->hnsw.efConstruction = efConstruction;
    shared_index->is_trained = true;
    {
        Timer t;
        shared_index->add(nb, xb.data());
        printf("  Build time: %.1f ms\n", t.elapsed_ms());
    }
    shared_index->hnsw.efSearch = efSearch;

    // ============================================================
    // 3. Correctness: compare search results
    // ============================================================
    printf("\n--- Correctness Check ---\n");
    std::vector<float> dist_baseline(nq * k), dist_shared(nq * k);
    std::vector<faiss::idx_t> ids_baseline(nq * k), ids_shared(nq * k);

    baseline.search(nq, xq.data(), k, dist_baseline.data(), ids_baseline.data());
    shared_index->search(
            nq, xq.data(), k, dist_shared.data(), ids_shared.data());

    double avg_dist_baseline = 0, avg_dist_shared = 0;
    for (size_t i = 0; i < nq * k; i++) {
        avg_dist_baseline += dist_baseline[i];
        avg_dist_shared += dist_shared[i];
    }
    avg_dist_baseline /= (nq * k);
    avg_dist_shared /= (nq * k);
    printf("  Avg distance (baseline): %.4f\n", avg_dist_baseline);
    printf("  Avg distance (shared)  : %.4f\n", avg_dist_shared);
    double dist_ratio = avg_dist_shared / avg_dist_baseline;
    printf("  Distance quality ratio : %.4f (1.0 = identical quality)\n", dist_ratio);

    if (dist_ratio > 1.05) {
        printf("  *** WARNING: shared index returns >5%% worse distances ***\n");
    } else {
        printf("  PASS: distance quality is comparable\n");
    }

    std::vector<float> recons_shared(d);
    bool reconstruct_ok = true;
    for (size_t i = 0; i < std::min(nb, (size_t)100); i++) {
        shared_index->reconstruct(i, recons_shared.data());
        for (size_t j = 0; j < d; j++) {
            if (recons_shared[j] != xb[i * d + j]) {
                reconstruct_ok = false;
                break;
            }
        }
        if (!reconstruct_ok)
            break;
    }
    printf("  Reconstruct check: %s\n",
           reconstruct_ok ? "PASS" : "FAIL");

    // ============================================================
    // 4. Search QPS benchmark (multiple rounds)
    // ============================================================
    printf("\n--- Search QPS Benchmark ---\n");
    int n_rounds = 5;

    double baseline_total_ms = 0;
    for (int r = 0; r < n_rounds; r++) {
        Timer t;
        baseline.search(
                nq,
                xq.data(),
                k,
                dist_baseline.data(),
                ids_baseline.data());
        baseline_total_ms += t.elapsed_ms();
    }
    double baseline_qps = (nq * n_rounds) / (baseline_total_ms / 1000.0);
    printf("  IndexHNSWFlat       : %.0f QPS  (%.1f ms / %d rounds)\n",
           baseline_qps, baseline_total_ms, n_rounds);

    double shared_total_ms = 0;
    for (int r = 0; r < n_rounds; r++) {
        Timer t;
        shared_index->search(
                nq,
                xq.data(),
                k,
                dist_shared.data(),
                ids_shared.data());
        shared_total_ms += t.elapsed_ms();
    }
    double shared_qps = (nq * n_rounds) / (shared_total_ms / 1000.0);
    printf("  IndexHNSW+Shared    : %.0f QPS  (%.1f ms / %d rounds)\n",
           shared_qps, shared_total_ms, n_rounds);

    double ratio = shared_qps / baseline_qps;
    printf("  Ratio (shared/baseline): %.2f\n", ratio);

    // ============================================================
    // 5. Rebuild benchmark (zero-copy)
    // ============================================================
    printf("\n--- Rebuild Benchmark (zero-copy, 10%% deleted) ---\n");
    size_t n_delete = nb / 10;
    auto* old_shared =
            dynamic_cast<faiss::IndexFlatShared*>(shared_index->storage);
    std::mt19937 del_rng(999);
    std::vector<bool> deleted_set(nb, false);
    size_t deleted_count = 0;
    while (deleted_count < n_delete) {
        size_t idx = del_rng() % nb;
        if (!deleted_set[idx]) {
            deleted_set[idx] = true;
            old_shared->mark_deleted(idx);
            deleted_count++;
        }
    }
    printf("  Marked %zu vectors as deleted\n", deleted_count);
    printf("  Alive: %zu\n", old_shared->count_alive());

    {
        Timer t;
        faiss::IndexHNSW* rebuilt = faiss::build_new_index(
                store, *shared_index, M, efConstruction, faiss::METRIC_L2);
        double rebuild_ms = t.elapsed_ms();
        printf("  Rebuild time: %.1f ms\n", rebuild_ms);

        rebuilt->hnsw.efSearch = efSearch;
        std::vector<float> dist_rebuilt(nq * k);
        std::vector<faiss::idx_t> ids_rebuilt(nq * k);

        double rebuilt_total_ms = 0;
        for (int r = 0; r < n_rounds; r++) {
            Timer t2;
            rebuilt->search(
                    nq,
                    xq.data(),
                    k,
                    dist_rebuilt.data(),
                    ids_rebuilt.data());
            rebuilt_total_ms += t2.elapsed_ms();
        }
        double rebuilt_qps =
                (nq * n_rounds) / (rebuilt_total_ms / 1000.0);
        printf("  Rebuilt search QPS  : %.0f QPS\n", rebuilt_qps);
        printf("  vs baseline ratio   : %.2f\n", rebuilt_qps / baseline_qps);

        delete rebuilt;
    }

    // ============================================================
    // Summary
    // ============================================================
    printf("\n=== Summary ===\n");
    printf("  Baseline (IndexHNSWFlat)   QPS: %.0f\n", baseline_qps);
    printf("  Shared (IndexFlatShared)   QPS: %.0f\n", shared_qps);
    printf("  Ratio: %.2f  (%.1f%% of baseline)\n",
           ratio, ratio * 100);
    if (ratio >= 0.95) {
        printf("  PASS: SharedVectorStore search is within 5%% of baseline\n");
    } else if (ratio >= 0.85) {
        printf("  ACCEPTABLE: SharedVectorStore search is within 15%% of baseline\n");
    } else {
        printf("  NEEDS WORK: SharedVectorStore search is more than 15%% slower\n");
    }

    delete shared_index;

    return 0;
}
