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

#include <faiss/HNSWReorder.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexFlatShared.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>

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

double measure_qps(
        faiss::IndexHNSW& index,
        const float* xq,
        size_t nq,
        int k,
        int n_rounds) {
    std::vector<float> dist(nq * k);
    std::vector<faiss::idx_t> ids(nq * k);
    double total_ms = 0;
    for (int r = 0; r < n_rounds; r++) {
        Timer t;
        index.search(nq, xq, k, dist.data(), ids.data());
        total_ms += t.elapsed_ms();
    }
    return (nq * n_rounds) / (total_ms / 1000.0);
}

bool verify_correctness(
        faiss::IndexHNSW& index,
        size_t d,
        size_t n_check) {
    auto* shared = dynamic_cast<faiss::IndexFlatShared*>(index.storage);
    if (!shared)
        return false;

    n_check = std::min(n_check, (size_t)index.ntotal);
    std::vector<float> recons(d);

    for (size_t i = 0; i < n_check; i++) {
        index.reconstruct(i, recons.data());

        faiss::idx_t store_slot = shared->storage_id_map[i];
        const float* stored = shared->store->get_vector(store_slot);
        for (size_t j = 0; j < d; j++) {
            if (recons[j] != stored[j]) {
                printf("    FAIL at vector %zu dim %zu: recons=%.6f stored=%.6f\n",
                       i, j, recons[j], stored[j]);
                return false;
            }
        }
    }
    return true;
}

const char* strategy_name(faiss::ReorderStrategy s) {
    switch (s) {
        case faiss::ReorderStrategy::BFS:
            return "BFS";
        case faiss::ReorderStrategy::RCM:
            return "RCM";
        case faiss::ReorderStrategy::DFS:
            return "DFS";
        case faiss::ReorderStrategy::CLUSTER:
            return "Cluster";
        case faiss::ReorderStrategy::WEIGHTED:
            return "Weighted";
        default:
            return "Unknown";
    }
}

/// Save/restore HNSW graph + store data for non-destructive reorder testing.
/// This avoids rebuilding the HNSW graph for each reorder strategy.
struct HNSWSnapshot {
    std::vector<int> levels;
    std::vector<size_t> offsets;
    std::vector<faiss::HNSW::storage_idx_t> neighbors;
    faiss::HNSW::storage_idx_t entry_point;
    std::vector<uint8_t> store_codes;
    std::vector<faiss::idx_t> storage_id_map;
    bool is_identity_map;

    void save(const faiss::IndexHNSW& idx) {
        auto* shared =
                dynamic_cast<const faiss::IndexFlatShared*>(idx.storage);
        levels = idx.hnsw.levels;
        offsets = idx.hnsw.offsets;
        neighbors.assign(
                idx.hnsw.neighbors.data(),
                idx.hnsw.neighbors.data() + idx.hnsw.neighbors.size());
        entry_point = idx.hnsw.entry_point;
        store_codes = shared->store->codes;
        storage_id_map = shared->storage_id_map;
        is_identity_map = shared->is_identity_map;
    }

    void restore(faiss::IndexHNSW& idx) const {
        auto* shared = dynamic_cast<faiss::IndexFlatShared*>(idx.storage);
        idx.hnsw.levels = levels;
        idx.hnsw.offsets = offsets;
        std::vector<faiss::HNSW::storage_idx_t> neighbors_copy(neighbors);
        idx.hnsw.neighbors = std::move(neighbors_copy);
        idx.hnsw.entry_point = entry_point;
        shared->store->codes = store_codes;
        shared->store->ntotal_store = storage_id_map.size();
        shared->storage_id_map = storage_id_map;
        shared->is_identity_map = is_identity_map;
        shared->codes = faiss::MaybeOwnedVector<uint8_t>::create_view(
                shared->store->codes.data(),
                shared->store->codes.size(),
                shared->store);
    }
};

} // namespace

int main(int argc, char** argv) {
    size_t nb = 50000;
    size_t d = 128;
    size_t nq = 500;
    int k = 10;
    int M = 16;
    int efConstruction = 40;
    int efSearch = 64;
    int n_rounds = 5;

    if (argc > 1)
        nb = std::atol(argv[1]);
    if (argc > 2)
        d = std::atol(argv[2]);
    if (argc > 3)
        nq = std::atol(argv[3]);

    printf("=== SharedVectorStore Benchmark (with compact + reorder) ===\n");
    printf("nb=%zu  d=%zu  nq=%zu  k=%d  M=%d  efConstruction=%d  efSearch=%d\n\n",
           nb, d, nq, k, M, efConstruction, efSearch);

    std::vector<float> xb, xq;
    generate_random_vectors(xb, nb, d, 42);
    generate_random_vectors(xq, nq, d, 123);

    // ============================================================
    // 1. Baseline: IndexHNSWFlat
    // ============================================================
    printf("--- [1] Building IndexHNSWFlat (baseline) ---\n");
    faiss::IndexHNSWFlat baseline(d, M);
    baseline.hnsw.efConstruction = efConstruction;
    {
        Timer t;
        baseline.add(nb, xb.data());
        printf("  Build time: %.1f ms\n", t.elapsed_ms());
    }
    baseline.hnsw.efSearch = efSearch;
    double baseline_qps = measure_qps(baseline, xq.data(), nq, k, n_rounds);
    printf("  QPS: %.0f\n", baseline_qps);

    // ============================================================
    // 2. Fresh build: IndexHNSW + IndexFlatShared
    // ============================================================
    printf("\n--- [2] Building IndexHNSW + IndexFlatShared (fresh) ---\n");
    auto store =
            std::make_shared<faiss::SharedVectorStore>(d, d * sizeof(float));
    store->reserve(nb);
    auto* shared_storage =
            new faiss::IndexFlatShared(store, faiss::METRIC_L2);
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
    double fresh_qps = measure_qps(*shared_index, xq.data(), nq, k, n_rounds);
    printf("  QPS: %.0f  (%.1f%% of baseline)\n",
           fresh_qps, fresh_qps / baseline_qps * 100);
    printf("  is_identity_map: %s\n",
           shared_storage->is_identity_map ? "true" : "false");

    // ============================================================
    // 3. Correctness check (fresh)
    // ============================================================
    printf("\n--- [3] Correctness Check (fresh) ---\n");
    std::vector<float> dist_baseline(nq * k), dist_shared(nq * k);
    std::vector<faiss::idx_t> ids_baseline(nq * k), ids_shared(nq * k);
    baseline.search(
            nq, xq.data(), k, dist_baseline.data(), ids_baseline.data());
    shared_index->search(
            nq, xq.data(), k, dist_shared.data(), ids_shared.data());

    double avg_bl = 0, avg_sh = 0;
    for (size_t i = 0; i < nq * k; i++) {
        avg_bl += dist_baseline[i];
        avg_sh += dist_shared[i];
    }
    avg_bl /= (nq * k);
    avg_sh /= (nq * k);
    printf("  Avg dist baseline: %.4f  shared: %.4f  ratio: %.4f\n",
           avg_bl, avg_sh, avg_sh / avg_bl);

    // ============================================================
    // 4. Delete 10% -> Rebuild (zero-copy)
    // ============================================================
    printf("\n--- [4] Delete 10%% + Rebuild (zero-copy) ---\n");
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
    printf("  Deleted: %zu  Alive: %zu\n",
           deleted_count, old_shared->count_alive());

    faiss::IndexHNSW* rebuilt;
    {
        Timer t;
        rebuilt = faiss::build_new_index(
                store, *shared_index, M, efConstruction, faiss::METRIC_L2);
        printf("  Rebuild time: %.1f ms\n", t.elapsed_ms());
    }
    rebuilt->hnsw.efSearch = efSearch;

    double rebuilt_qps = measure_qps(*rebuilt, xq.data(), nq, k, n_rounds);
    printf("  QPS (rebuilt, before compact): %.0f  (%.1f%% of baseline)\n",
           rebuilt_qps, rebuilt_qps / baseline_qps * 100);

    auto* rebuilt_shared =
            dynamic_cast<faiss::IndexFlatShared*>(rebuilt->storage);
    printf("  is_identity_map: %s\n",
           rebuilt_shared->is_identity_map ? "true" : "false");

    // ============================================================
    // 5. Compact store
    // ============================================================
    printf("\n--- [5] Compact store ---\n");
    {
        Timer t;
        faiss::compact_store(*rebuilt_shared);
        printf("  Compact time: %.1f ms\n", t.elapsed_ms());
    }
    printf("  is_identity_map: %s\n",
           rebuilt_shared->is_identity_map ? "true" : "false");
    printf("  store.ntotal_store: %zu  index.ntotal: %lld\n",
           store->ntotal_store, (long long)rebuilt->ntotal);
    printf("  Correctness: %s\n",
           verify_correctness(*rebuilt, d, 100) ? "PASS" : "FAIL");

    double compact_qps = measure_qps(*rebuilt, xq.data(), nq, k, n_rounds);
    printf("  QPS (after compact): %.0f  (%.1f%% of baseline)\n",
           compact_qps, compact_qps / baseline_qps * 100);

    // ============================================================
    // 6. Try all reorder strategies using save/restore
    //    to avoid rebuilding the HNSW graph for each strategy.
    // ============================================================
    printf("\n--- [6] Reorder strategies (compact + reorder) ---\n");

    HNSWSnapshot snapshot;
    snapshot.save(*rebuilt);

    faiss::ReorderStrategy strategies[] = {
            faiss::ReorderStrategy::BFS,
            faiss::ReorderStrategy::RCM,
            faiss::ReorderStrategy::DFS,
            faiss::ReorderStrategy::CLUSTER,
            faiss::ReorderStrategy::WEIGHTED,
    };

    double best_qps = compact_qps;
    const char* best_strategy = "None (compact only)";

    for (auto strategy : strategies) {
        snapshot.restore(*rebuilt);

        auto perm = faiss::generate_permutation(
                rebuilt->hnsw, strategy);

        {
            Timer t;
            rebuilt->permute_entries(perm.data());
            printf("  %-10s reorder time: %6.1f ms",
                   strategy_name(strategy), t.elapsed_ms());
        }

        bool ok = verify_correctness(*rebuilt, d, 100);
        double qps =
                measure_qps(*rebuilt, xq.data(), nq, k, n_rounds);
        printf("  QPS: %6.0f  (%.1f%% of baseline)  correct: %s\n",
               qps, qps / baseline_qps * 100, ok ? "Y" : "N");

        if (qps > best_qps) {
            best_qps = qps;
            best_strategy = strategy_name(strategy);
        }
    }

    // ============================================================
    // Summary
    // ============================================================
    printf("\n=== Summary ===\n");
    printf("  Baseline (IndexHNSWFlat)      QPS: %8.0f\n", baseline_qps);
    printf("  Fresh (IndexFlatShared)        QPS: %8.0f  (%.1f%%)\n",
           fresh_qps, fresh_qps / baseline_qps * 100);
    printf("  Rebuilt (before compact)       QPS: %8.0f  (%.1f%%)\n",
           rebuilt_qps, rebuilt_qps / baseline_qps * 100);
    printf("  Rebuilt + compact              QPS: %8.0f  (%.1f%%)\n",
           compact_qps, compact_qps / baseline_qps * 100);
    printf("  Best reorder (%s)          QPS: %8.0f  (%.1f%%)\n",
           best_strategy, best_qps, best_qps / baseline_qps * 100);

    if (best_qps / baseline_qps >= 0.95) {
        printf("  PASS: within 5%% of baseline\n");
    } else if (best_qps / baseline_qps >= 0.85) {
        printf("  ACCEPTABLE: within 15%% of baseline\n");
    } else {
        printf("  NEEDS WORK: more than 15%% slower\n");
    }

    delete rebuilt;
    delete shared_index;

    return 0;
}
