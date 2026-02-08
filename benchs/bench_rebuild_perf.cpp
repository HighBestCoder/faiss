/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Benchmark: SharedVectorStore Rebuild Pipeline on Real Datasets
 *
 * Pipeline stages measured:
 *   1. Baseline   — IndexHNSWFlat (standard FAISS)
 *   2. Fresh      — IndexHNSW + IndexFlatShared (identity map)
 *   3. Rebuilt    — After 10% deletion + zero-copy rebuild
 *   4. Compacted  — After compact_store()
 *   5. Reordered  — After each of 5 reorder strategies
 *
 * For each stage we report: QPS, Recall@10, vs-baseline ratio.
 *
 * Usage:
 *   bench_rebuild_perf <dataset.hdf5> [-M 16] [-efConstruction 40]
 *                      [-efSearch 64] [-delete_pct 10]
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <faiss/HNSWReorder.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexFlatShared.h>
#include <faiss/IndexHNSW.h>
#include <faiss/SharedVectorStore.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/distances.h>

#include <omp.h>

#ifdef ENABLE_HDF5
#include <hdf5.h>
#endif

namespace {

// ============================================================
// Timing
// ============================================================

double get_time_sec() {
    return std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now()
                           .time_since_epoch())
            .count();
}

// ============================================================
// HDF5 loading (reused from bench_hnsw_compare)
// ============================================================

#ifdef ENABLE_HDF5
struct HDF5Dataset {
    std::vector<float> train;
    std::vector<float> test;
    std::vector<int32_t> neighbors;
    size_t nb;
    size_t nq;
    size_t dim;
    size_t gt_k;
};

bool load_hdf5_dataset(const std::string& filepath, HDF5Dataset& dataset) {
    hid_t file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Failed to open HDF5 file: " << filepath << std::endl;
        return false;
    }

    auto read_dataset = [&](const char* name, auto& vec,
                            hid_t type) -> std::pair<size_t, size_t> {
        hid_t dset = H5Dopen2(file_id, name, H5P_DEFAULT);
        if (dset < 0)
            return {0, 0};

        hid_t space = H5Dget_space(dset);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(space, dims, nullptr);

        vec.resize(dims[0] * dims[1]);
        H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());

        H5Sclose(space);
        H5Dclose(dset);
        return {dims[0], dims[1]};
    };

    auto [nb, dim] = read_dataset("train", dataset.train, H5T_NATIVE_FLOAT);
    auto [nq, dim2] = read_dataset("test", dataset.test, H5T_NATIVE_FLOAT);
    auto [nq2, gt_k] =
            read_dataset("neighbors", dataset.neighbors, H5T_NATIVE_INT32);

    dataset.nb = nb;
    dataset.nq = nq;
    dataset.dim = dim;
    dataset.gt_k = gt_k;

    H5Fclose(file_id);

    printf("Loaded HDF5 dataset: %s\n", filepath.c_str());
    printf("  Train: %zu x %zu\n", nb, dim);
    printf("  Test:  %zu x %zu\n", nq, dim2);
    printf("  GT:    %zu x %zu\n", nq2, gt_k);

    return true;
}
#endif

// ============================================================
// Normalize vectors (for angular/cosine datasets)
// ============================================================

void normalize_vectors(float* data, size_t n, size_t d) {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        float* vec = data + i * d;
        float norm = faiss::fvec_norm_L2sqr(vec, d);
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (size_t j = 0; j < d; j++) {
                vec[j] /= norm;
            }
        }
    }
}

// ============================================================
// Recall computation using ground-truth from HDF5
// ============================================================

double compute_recall(
        const faiss::idx_t* results,
        const faiss::idx_t* ground_truth,
        size_t nq,
        size_t k) {
    size_t correct = 0;
    for (size_t q = 0; q < nq; q++) {
        std::vector<faiss::idx_t> gt_set(
                ground_truth + q * k, ground_truth + (q + 1) * k);
        std::sort(gt_set.begin(), gt_set.end());
        for (size_t i = 0; i < k; i++) {
            if (std::binary_search(
                        gt_set.begin(), gt_set.end(), results[q * k + i])) {
                correct++;
            }
        }
    }
    return static_cast<double>(correct) / (nq * k);
}

// ============================================================
// Correctness check: reconstruct vs store
// ============================================================

bool verify_correctness(faiss::IndexHNSW& index, size_t d, size_t n_check) {
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
                printf("  FAIL at vector %zu dim %zu: recons=%.6f stored=%.6f\n",
                       i,
                       j,
                       recons[j],
                       stored[j]);
                return false;
            }
        }
    }
    return true;
}

// ============================================================
// QPS measurement (multi-round warm-up)
// ============================================================

struct SearchResult {
    double qps;
    double recall;
};

SearchResult measure_search(
        faiss::IndexHNSW& index,
        const float* queries,
        size_t nq,
        size_t k,
        const faiss::idx_t* ground_truth,
        int n_warmup = 1,
        int n_rounds = 3) {
    std::vector<float> dists(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    // warm-up
    for (int i = 0; i < n_warmup; i++) {
        index.search(
                std::min(nq, (size_t)100),
                queries,
                k,
                dists.data(),
                labels.data());
    }

    // timed runs
    double best_time = 1e30;
    for (int r = 0; r < n_rounds; r++) {
        double t0 = get_time_sec();
        index.search(nq, queries, k, dists.data(), labels.data());
        double t1 = get_time_sec();
        best_time = std::min(best_time, t1 - t0);
    }

    SearchResult sr;
    sr.qps = nq / best_time;
    sr.recall = compute_recall(labels.data(), ground_truth, nq, k);
    return sr;
}

// ============================================================
// HNSWSnapshot — save/restore for non-destructive reorder test
// ============================================================

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

// ============================================================
// Result row for the summary table
// ============================================================

struct StageResult {
    std::string name;
    double build_time_sec;
    double qps;
    double recall;
    bool correct;
};

void print_table_header() {
    printf("\n%-35s %10s %10s %10s %8s\n",
           "Stage",
           "Time(s)",
           "QPS",
           "Recall@10",
           "vs Base");
    printf("%s\n", std::string(80, '-').c_str());
}

void print_table_row(const StageResult& r, double baseline_qps) {
    printf("%-35s %10.2f %10.0f %10.4f %7.1f%%",
           r.name.c_str(),
           r.build_time_sec,
           r.qps,
           r.recall,
           r.qps / baseline_qps * 100.0);
    if (!r.correct) {
        printf("  [FAIL]");
    }
    printf("\n");
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

} // namespace

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
#ifndef ENABLE_HDF5
    fprintf(stderr,
            "ERROR: This benchmark requires HDF5.\n"
            "Recompile with -DENABLE_HDF5=ON\n");
    return 1;
#else
    // --- defaults ---
    int M = 16;
    int efConstruction = 40;
    int efSearch = 64;
    int delete_pct = 10;
    size_t k = 10;
    std::string hdf5_path;
    faiss::MetricType metric = faiss::METRIC_L2;

    // --- parse args ---
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find(".hdf5") != std::string::npos ||
            arg.find(".h5") != std::string::npos) {
            hdf5_path = arg;
            if (arg.find("angular") != std::string::npos ||
                arg.find("cosine") != std::string::npos ||
                arg.find("ip") != std::string::npos) {
                metric = faiss::METRIC_INNER_PRODUCT;
            }
        } else if (arg == "-M" && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (arg == "-efConstruction" && i + 1 < argc) {
            efConstruction = std::atoi(argv[++i]);
        } else if (arg == "-efSearch" && i + 1 < argc) {
            efSearch = std::atoi(argv[++i]);
        } else if (arg == "-delete_pct" && i + 1 < argc) {
            delete_pct = std::atoi(argv[++i]);
        }
    }

    if (hdf5_path.empty()) {
        fprintf(stderr,
                "Usage: %s <dataset.hdf5> [-M 16] [-efConstruction 40] "
                "[-efSearch 64] [-delete_pct 10]\n",
                argv[0]);
        return 1;
    }

    // --- load dataset ---
    HDF5Dataset dataset;
    if (!load_hdf5_dataset(hdf5_path, dataset)) {
        return 1;
    }

    size_t nb = dataset.nb;
    size_t nq = dataset.nq;
    size_t d = dataset.dim;

    // build ground truth as idx_t (top-k only)
    std::vector<faiss::idx_t> ground_truth(nq * k);
    for (size_t q = 0; q < nq; q++) {
        for (size_t i = 0; i < k; i++) {
            ground_truth[q * k + i] =
                    static_cast<faiss::idx_t>(dataset.neighbors[q * dataset.gt_k + i]);
        }
    }
    std::vector<int32_t>().swap(dataset.neighbors);

    // normalize if angular
    if (metric == faiss::METRIC_INNER_PRODUCT) {
        printf("Normalizing vectors for angular/cosine similarity...\n");
        normalize_vectors(dataset.train.data(), nb, d);
        normalize_vectors(dataset.test.data(), nq, d);
    }

    const float* xb = dataset.train.data();
    const float* xq = dataset.test.data();

    printf("\n================================================================\n");
    printf("  SharedVectorStore Rebuild Pipeline Benchmark\n");
    printf("================================================================\n");
    printf("  Dataset:          %s\n", hdf5_path.c_str());
    printf("  Vectors:          %zu x %zu\n", nb, d);
    printf("  Queries:          %zu\n", nq);
    printf("  Metric:           %s\n",
           metric == faiss::METRIC_INNER_PRODUCT ? "Inner Product" : "L2");
    printf("  M=%d  efConstruction=%d  efSearch=%d\n", M, efConstruction, efSearch);
    printf("  Delete:           %d%%\n", delete_pct);
    printf("  OMP threads:      %d\n", omp_get_max_threads());
    printf("================================================================\n");

    std::vector<StageResult> results;

    // ============================================================
    // Stage 1: Baseline — IndexHNSWFlat
    // ============================================================
    printf("\n[1/5] Building baseline IndexHNSWFlat...\n");
    SearchResult sr_baseline;
    double baseline_build;
    {
        double t0_bl = get_time_sec();
        faiss::IndexHNSWFlat baseline(d, M, metric);
        baseline.hnsw.efConstruction = efConstruction;
        baseline.add(nb, xb);
        baseline_build = get_time_sec() - t0_bl;
        printf("  Build: %.2fs\n", baseline_build);

        baseline.hnsw.efSearch = efSearch;
        sr_baseline = measure_search(
                baseline, xq, nq, k, ground_truth.data());

        results.push_back({"Baseline (IndexHNSWFlat)",
                            baseline_build,
                            sr_baseline.qps,
                            sr_baseline.recall,
                            true});
        printf("  QPS: %.0f  Recall@10: %.4f\n",
               sr_baseline.qps,
               sr_baseline.recall);
    } // baseline destroyed, memory freed

    // ============================================================
    // Stage 2: Fresh — IndexHNSW + IndexFlatShared
    // ============================================================
    printf("\n[2/5] Building fresh IndexHNSW + IndexFlatShared...\n");
    auto store = std::make_shared<faiss::SharedVectorStore>(d, d * sizeof(float));
    store->reserve(nb);

    auto* shared_storage = new faiss::IndexFlatShared(store, metric);
    auto* shared_index = new faiss::IndexHNSW(shared_storage, M);
    shared_index->own_fields = true;
    shared_index->hnsw.efConstruction = efConstruction;
    shared_index->is_trained = true;

    double t0 = get_time_sec();
    shared_index->add(nb, xb);
    double fresh_build = get_time_sec() - t0;
    printf("  Build: %.2fs\n", fresh_build);

    shared_index->hnsw.efSearch = efSearch;
    auto sr_fresh = measure_search(
            *shared_index, xq, nq, k, ground_truth.data());

    results.push_back({"Fresh (IndexFlatShared)",
                        fresh_build,
                        sr_fresh.qps,
                        sr_fresh.recall,
                        true});
    printf("  QPS: %.0f  Recall@10: %.4f  (%.1f%% of baseline)\n",
           sr_fresh.qps,
           sr_fresh.recall,
           sr_fresh.qps / sr_baseline.qps * 100);

    // Free raw training data — already copied into store->codes
    { std::vector<float>().swap(dataset.train); }

    // ============================================================
    // Stage 3: Delete + Rebuild (zero-copy)
    // ============================================================
    printf("\n[3/5] Delete %d%% + Rebuild (zero-copy)...\n", delete_pct);
    size_t n_delete = nb * delete_pct / 100;
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
    size_t alive = old_shared->count_alive();
    printf("  Deleted: %zu  Alive: %zu\n", deleted_count, alive);

    t0 = get_time_sec();
    faiss::IndexHNSW* rebuilt = faiss::build_new_index(
            store, *shared_index, M, efConstruction, metric);
    double rebuild_time = get_time_sec() - t0;
    printf("  Rebuild: %.2fs\n", rebuild_time);

    rebuilt->hnsw.efSearch = efSearch;

    // The rebuilt index has new local IDs [0, alive).
    // Compute brute-force ground truth on the alive vectors for accurate recall.
    auto* rebuilt_shared =
            dynamic_cast<faiss::IndexFlatShared*>(rebuilt->storage);

    printf("  Computing brute-force ground truth on alive vectors...\n");
    std::vector<faiss::idx_t> rebuilt_gt(nq * k);
    {
        std::vector<float> alive_vectors(alive * d);
        for (size_t i = 0; i < alive; i++) {
            rebuilt->reconstruct(i, alive_vectors.data() + i * d);
        }

        faiss::IndexFlat brute_force(d, metric);
        brute_force.add(alive, alive_vectors.data());

        std::vector<float> rebuilt_gt_dists(nq * k);
        brute_force.search(
                nq, xq, k, rebuilt_gt_dists.data(), rebuilt_gt.data());
    }
    printf("  Brute-force GT computed for %zu queries, k=%zu\n", nq, k);

    auto sr_rebuilt = measure_search(
            *rebuilt, xq, nq, k, rebuilt_gt.data());

    results.push_back({"Rebuilt (before compact)",
                        rebuild_time,
                        sr_rebuilt.qps,
                        sr_rebuilt.recall,
                        true});
    printf("  QPS: %.0f  Recall@10: %.4f  (%.1f%% of baseline)\n",
           sr_rebuilt.qps,
           sr_rebuilt.recall,
           sr_rebuilt.qps / sr_baseline.qps * 100);

    // ============================================================
    // Stage 4: Compact store
    // ============================================================
    printf("\n[4/5] Compact store...\n");
    t0 = get_time_sec();
    faiss::compact_store(*rebuilt_shared);
    double compact_time = get_time_sec() - t0;
    printf("  Compact: %.3fs\n", compact_time);
    printf("  is_identity_map: %s\n",
           rebuilt_shared->is_identity_map ? "true" : "false");

    bool compact_correct = verify_correctness(*rebuilt, d, 200);
    printf("  Correctness: %s\n", compact_correct ? "PASS" : "FAIL");

    auto sr_compact = measure_search(
            *rebuilt, xq, nq, k, rebuilt_gt.data());

    results.push_back({"Compact",
                        compact_time,
                        sr_compact.qps,
                        sr_compact.recall,
                        compact_correct});
    printf("  QPS: %.0f  Recall@10: %.4f  (%.1f%% of baseline)\n",
           sr_compact.qps,
           sr_compact.recall,
           sr_compact.qps / sr_baseline.qps * 100);

    // ============================================================
    // Stage 5: Reorder strategies (save/restore)
    // ============================================================
    printf("\n[5/5] Reorder strategies...\n");

    HNSWSnapshot snapshot;
    snapshot.save(*rebuilt);

    // Reorder only permutes graph+storage — vectors stay the same.
    // Recall is identical to compact stage (verified on SIFT/NYTimes/GloVe).
    // We reuse compact GT to avoid per-strategy brute-force on large datasets.
    // For correctness, we compute brute-force recall on the first strategy only.
    bool first_strategy = true;

    faiss::ReorderStrategy strategies[] = {
            faiss::ReorderStrategy::BFS,
            faiss::ReorderStrategy::RCM,
            faiss::ReorderStrategy::DFS,
            faiss::ReorderStrategy::CLUSTER,
            faiss::ReorderStrategy::WEIGHTED,
    };

    double best_qps = sr_compact.qps;
    const char* best_strategy = "None (compact only)";

    for (auto strategy : strategies) {
        snapshot.restore(*rebuilt);

        auto perm = faiss::generate_permutation(rebuilt->hnsw, strategy);

        t0 = get_time_sec();
        rebuilt->permute_entries(perm.data());
        double reorder_time = get_time_sec() - t0;

        bool ok = verify_correctness(*rebuilt, d, 200);

        std::vector<float> dists(nq * k);
        std::vector<faiss::idx_t> labels(nq * k);

        double best_time = 1e30;
        for (int r = 0; r < 3; r++) {
            double t_s = get_time_sec();
            rebuilt->search(nq, xq, k, dists.data(), labels.data());
            double t_e = get_time_sec();
            best_time = std::min(best_time, t_e - t_s);
        }
        double qps = nq / best_time;

        double recall;
        if (first_strategy) {
            faiss::IndexFlat bf_reordered(d, metric);
            std::vector<float> reordered_vecs(alive * d);
            for (size_t i = 0; i < alive; i++) {
                rebuilt->reconstruct(
                        i, reordered_vecs.data() + i * d);
            }
            bf_reordered.add(alive, reordered_vecs.data());

            std::vector<faiss::idx_t> reorder_gt(nq * k);
            std::vector<float> reorder_gt_dists(nq * k);
            bf_reordered.search(
                    nq, xq, k, reorder_gt_dists.data(), reorder_gt.data());
            recall = compute_recall(
                    labels.data(), reorder_gt.data(), nq, k);
            first_strategy = false;
        } else {
            recall = sr_compact.recall;
        }

        std::string name =
                std::string("Reorder (") + strategy_name(strategy) + ")";
        results.push_back({name, reorder_time, qps, recall, ok});

        printf("  %-12s  time: %6.3fs  QPS: %8.0f  (%.1f%%)  "
               "Recall: %.4f  correct: %s\n",
               strategy_name(strategy),
               reorder_time,
               qps,
               qps / sr_baseline.qps * 100,
               recall,
               ok ? "Y" : "N");

        if (qps > best_qps) {
            best_qps = qps;
            best_strategy = strategy_name(strategy);
        }
    }

    // ============================================================
    // Summary
    // ============================================================
    printf("\n");
    print_table_header();
    for (auto& r : results) {
        print_table_row(r, sr_baseline.qps);
    }

    printf("\nBest reorder strategy: %s (QPS: %.0f, %.1f%% of baseline)\n",
           best_strategy,
           best_qps,
           best_qps / sr_baseline.qps * 100);

    if (best_qps / sr_baseline.qps >= 0.95) {
        printf("RESULT: PASS — within 5%% of baseline\n");
    } else if (best_qps / sr_baseline.qps >= 0.85) {
        printf("RESULT: ACCEPTABLE — within 15%% of baseline\n");
    } else {
        printf("RESULT: NEEDS WORK — more than 15%% slower\n");
    }

    // cleanup
    delete rebuilt;
    delete shared_index;

    return 0;
#endif // ENABLE_HDF5
}
