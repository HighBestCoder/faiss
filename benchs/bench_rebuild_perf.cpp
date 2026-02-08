/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Benchmark: SharedVectorStore Rebuild Pipeline on Real Datasets
 *
 * For each delete_pct in {10,20,30,40,50,60,70,80}:
 *   Baseline-B  — overfetch search + post-filter deleted IDs
 *   Baseline-C  — new IndexHNSWFlat with only alive vectors
 *   SharedStore — zero-copy rebuild → compact → best reorder
 *
 * Baseline-A (full, no delete) and Fresh (SharedStore, no delete)
 * are measured once as reference.
 *
 * Memory-optimized for large datasets (e.g. GIST-960, 3.66 GB vectors):
 *   Phase 1: Build baseline_a, run ALL Baseline-B, destroy baseline_a.
 *   Phase 2: Build shared_index from xb, free xb (store IS the data).
 *   Phase 3: Per delete_pct: GT+Baseline-C (shared alive buffer,
 *            freed before SharedStore), then SharedStore pipeline.
 *
 * Usage:
 *   bench_rebuild_perf <dataset.hdf5> [-M 16] [-efConstruction 40]
 *                      [-efSearch 64] [-delete_pct 10]
 *   If -delete_pct is given, only that single percentage is tested.
 *   Otherwise all of {10,20,30,40,50,60,70,80} are tested.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// Recall computation using ground-truth
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
// Overfetch search with post-filter (Baseline-B)
//
// Simulates production deletion handling:
//   search_k = top_k * (1 + delete_ratio * overfetch_factor)
//   then filter out deleted IDs from results.
// ============================================================

struct OverfetchResult {
    double qps;
    double recall;
};

OverfetchResult measure_overfetch_search(
        faiss::IndexHNSW& index,
        const float* queries,
        size_t nq,
        size_t k,
        const std::vector<bool>& deleted_set,
        const faiss::idx_t* ground_truth_orig_ids,
        size_t nb,
        int n_warmup = 1,
        int n_rounds = 3) {
    const double overfetch_factor = 3.0;
    double delete_ratio =
            (double)std::count(deleted_set.begin(), deleted_set.end(), true) /
            (double)deleted_set.size();
    uint32_t search_k = static_cast<uint32_t>(
            k * (1.0 + delete_ratio * overfetch_factor));
    search_k = std::max(search_k, static_cast<uint32_t>(k));
    search_k = std::min(search_k, static_cast<uint32_t>(index.ntotal));

    printf("  Overfetch: top_k=%zu, delete_ratio=%.3f, search_k=%u\n",
           k,
           delete_ratio,
           search_k);

    std::vector<float> dists_overfetch(nq * search_k);
    std::vector<faiss::idx_t> labels_overfetch(nq * search_k);
    std::vector<faiss::idx_t> filtered_labels(nq * k, -1);

    // warm-up
    for (int i = 0; i < n_warmup; i++) {
        size_t nq_warmup = std::min(nq, (size_t)100);
        std::vector<float> wu_dists(nq_warmup * search_k);
        std::vector<faiss::idx_t> wu_labels(nq_warmup * search_k);
        index.search(
                nq_warmup,
                queries,
                search_k,
                wu_dists.data(),
                wu_labels.data());
    }

    // timed runs: measure search + filter together
    double best_time = 1e30;
    for (int r = 0; r < n_rounds; r++) {
        std::fill(filtered_labels.begin(), filtered_labels.end(), -1);

        double t0 = get_time_sec();

        // search with overfetch
        index.search(
                nq,
                queries,
                search_k,
                dists_overfetch.data(),
                labels_overfetch.data());

        // post-filter: remove deleted IDs, keep top-k alive
        for (size_t q = 0; q < nq; q++) {
            size_t out_idx = 0;
            for (uint32_t i = 0; i < search_k && out_idx < k; i++) {
                faiss::idx_t label =
                        labels_overfetch[q * search_k + i];
                if (label >= 0 &&
                    label < static_cast<faiss::idx_t>(nb) &&
                    !deleted_set[label]) {
                    filtered_labels[q * k + out_idx] = label;
                    out_idx++;
                }
            }
        }

        double t1 = get_time_sec();
        best_time = std::min(best_time, t1 - t0);
    }

    OverfetchResult result;
    result.qps = nq / best_time;
    result.recall = compute_recall(
            filtered_labels.data(), ground_truth_orig_ids, nq, k);
    return result;
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

struct SummaryRow {
    int delete_pct;
    double baseline_b_qps, baseline_b_recall;
    double baseline_c_qps, baseline_c_recall;
    double shared_best_qps, shared_best_recall;
    std::string best_strategy;
};

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
    int M = 16;
    int efConstruction = 40;
    int efSearch = 64;
    int single_delete_pct = -1;
    size_t k = 10;
    std::string hdf5_path;
    faiss::MetricType metric = faiss::METRIC_L2;

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
            single_delete_pct = std::atoi(argv[++i]);
        }
    }

    if (hdf5_path.empty()) {
        fprintf(stderr,
                "Usage: %s <dataset.hdf5> [-M 16] [-efConstruction 40] "
                "[-efSearch 64] [-delete_pct 10]\n",
                argv[0]);
        return 1;
    }

    HDF5Dataset dataset;
    if (!load_hdf5_dataset(hdf5_path, dataset)) {
        return 1;
    }

    size_t nb = dataset.nb;
    size_t nq = dataset.nq;
    size_t d = dataset.dim;

    std::vector<faiss::idx_t> ground_truth_all(nq * k);
    for (size_t q = 0; q < nq; q++) {
        for (size_t i = 0; i < k; i++) {
            ground_truth_all[q * k + i] =
                    static_cast<faiss::idx_t>(dataset.neighbors[q * dataset.gt_k + i]);
        }
    }
    std::vector<int32_t>().swap(dataset.neighbors);

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
    printf("  OMP threads:      %d\n", omp_get_max_threads());
    printf("  Vector data:      %.2f GB\n", (double)nb * d * sizeof(float) / (1024.0*1024*1024));
    printf("================================================================\n");

    std::vector<int> delete_pcts = {10, 20, 30, 40, 50, 60, 70, 80};
    if (single_delete_pct != -1) {
        delete_pcts = {single_delete_pct};
    }

    // ================================================================
    // Phase 1: Build baseline_a, measure Baseline-A & ALL Baseline-B,
    //          then destroy baseline_a to free memory.
    //          Peak memory: xb + baseline_a + alive_vecs(GT) ≈ 2*VecData + HNSW
    // ================================================================

    printf("\n[Phase 1] Baseline-A + all Baseline-B measurements\n");
    printf("  Building IndexHNSWFlat (all %zu vectors)...\n", nb);

    SearchResult sr_baseline_a;
    double baseline_a_build;

    struct BaselineBRow {
        int delete_pct;
        double qps, recall;
    };
    std::vector<BaselineBRow> baseline_b_rows;

    // Per-delete_pct data needed later: deleted_set, alive_to_orig, gt
    struct DeletePctData {
        int delete_pct;
        size_t n_delete;
        size_t alive;
        std::vector<bool> deleted_set;
        std::vector<faiss::idx_t> alive_to_orig;
        std::vector<faiss::idx_t> gt_alive_origid;
        std::vector<faiss::idx_t> gt_alive_localid;
    };
    std::vector<DeletePctData> pct_data_vec;

    {
        auto* baseline_a = new faiss::IndexHNSWFlat(d, M, metric);
        double t0_bl = get_time_sec();
        baseline_a->hnsw.efConstruction = efConstruction;
        baseline_a->add(nb, xb);
        baseline_a_build = get_time_sec() - t0_bl;
        printf("  Build: %.2fs\n", baseline_a_build);

        baseline_a->hnsw.efSearch = efSearch;
        sr_baseline_a = measure_search(
                *baseline_a, xq, nq, k, ground_truth_all.data());
        printf("  Baseline-A: QPS: %.0f  Recall@10: %.4f\n",
               sr_baseline_a.qps, sr_baseline_a.recall);

        for (int delete_pct : delete_pcts) {
            DeletePctData pd;
            pd.delete_pct = delete_pct;
            pd.n_delete = nb * delete_pct / 100;
            pd.alive = nb - pd.n_delete;

            std::mt19937 del_rng(999 + delete_pct);
            pd.deleted_set.assign(nb, false);
            {
                size_t cnt = 0;
                while (cnt < pd.n_delete) {
                    size_t idx = del_rng() % nb;
                    if (!pd.deleted_set[idx]) {
                        pd.deleted_set[idx] = true;
                        cnt++;
                    }
                }
            }

            pd.alive_to_orig.reserve(pd.alive);
            for (size_t i = 0; i < nb; i++) {
                if (!pd.deleted_set[i]) {
                    pd.alive_to_orig.push_back(static_cast<faiss::idx_t>(i));
                }
            }

            printf("\n  --- delete_pct = %d%% (deleted: %zu, alive: %zu) ---\n",
                   delete_pct, pd.n_delete, pd.alive);

            // Compute brute-force GT on alive vectors (uses xb which is still alive)
            printf("    Computing brute-force GT on %zu alive vectors...\n", pd.alive);
            pd.gt_alive_origid.resize(nq * k);
            pd.gt_alive_localid.resize(nq * k);
            {
                std::vector<float> alive_vecs(pd.alive * d);
                for (size_t i = 0; i < pd.alive; i++) {
                    memcpy(alive_vecs.data() + i * d,
                           xb + pd.alive_to_orig[i] * d,
                           d * sizeof(float));
                }
                faiss::IndexFlat brute_force(d, metric);
                brute_force.add(pd.alive, alive_vecs.data());
                std::vector<float>().swap(alive_vecs);

                std::vector<float> gt_dists(nq * k);
                brute_force.search(nq, xq, k, gt_dists.data(), pd.gt_alive_localid.data());

                for (size_t i = 0; i < nq * k; i++) {
                    faiss::idx_t local = pd.gt_alive_localid[i];
                    pd.gt_alive_origid[i] = (local >= 0) ? pd.alive_to_orig[local] : -1;
                }
            }

            // Baseline-B: Overfetch + post-filter (uses baseline_a)
            auto overfetch = measure_overfetch_search(
                    *baseline_a, xq, nq, k, pd.deleted_set,
                    pd.gt_alive_origid.data(), nb);

            baseline_b_rows.push_back({delete_pct, overfetch.qps, overfetch.recall});
            printf("    Baseline-B: QPS: %.0f  Recall: %.4f\n",
                   overfetch.qps, overfetch.recall);

            pct_data_vec.push_back(std::move(pd));
        }

        delete baseline_a;
        printf("\n  [Phase 1 done] baseline_a destroyed, memory freed.\n");
    }

    // ================================================================
    // Phase 2: Build shared_index from xb, then free xb.
    //          Peak memory: xb + store + HNSW graph (briefly, then xb freed)
    // ================================================================

    printf("\n[Phase 2] Building SharedVectorStore index...\n");

    auto store = std::make_shared<faiss::SharedVectorStore>(d, d * sizeof(float));
    store->reserve(nb);

    auto* shared_storage = new faiss::IndexFlatShared(store, metric);
    auto* shared_index = new faiss::IndexHNSW(shared_storage, M);
    shared_index->own_fields = true;
    shared_index->hnsw.efConstruction = efConstruction;
    shared_index->is_trained = true;

    SearchResult sr_fresh;
    {
        double t0 = get_time_sec();
        shared_index->add(nb, xb);
        double fresh_build = get_time_sec() - t0;
        printf("  Build: %.2fs\n", fresh_build);

        shared_index->hnsw.efSearch = efSearch;
        sr_fresh = measure_search(
                *shared_index, xq, nq, k, ground_truth_all.data());

        printf("  Fresh: QPS: %.0f  Recall@10: %.4f  (%.1f%% of A)\n",
               sr_fresh.qps, sr_fresh.recall,
               sr_fresh.qps / sr_baseline_a.qps * 100);
    }

    // xb and store->codes are identical (identity mapping after add).
    // Free xb; use store->codes as the vector data source hereafter.
    std::vector<float>().swap(dataset.train);
    xb = nullptr;
    printf("  dataset.train freed (%.2f GB). Using store->codes as vector source.\n",
           (double)nb * d * sizeof(float) / (1024.0*1024*1024));

    // Save original store metadata (NOT codes — codes stay in store->codes).
    // We save a copy of codes only once; compact+reorder modify store->codes
    // in-place via the shared pointer, so we need to restore between iterations.
    std::vector<uint8_t> original_store_codes = store->codes;
    size_t original_ntotal_store = store->ntotal_store;
    std::vector<faiss::idx_t> original_free_list = store->free_list;
    printf("  Store backup saved (%.2f GB).\n",
           (double)original_store_codes.size() / (1024.0*1024*1024));

    // Pointer into the backup for alive vector extraction
    const float* vec_data = reinterpret_cast<const float*>(original_store_codes.data());

    // ================================================================
    // Phase 3: Per delete_pct — Baseline-C, then SharedStore pipeline.
    //          alive_vecs is freed before SharedStore to reduce peak.
    //          Peak memory: store_codes_backup + store->codes + alive_vecs
    //                     = 2*VecData + alive_fraction*VecData
    // ================================================================

    printf("\n[Phase 3] Baseline-C + SharedStore per delete_pct\n");

    std::vector<SummaryRow> summary_rows;

    for (size_t pi = 0; pi < pct_data_vec.size(); pi++) {
        auto& pd = pct_data_vec[pi];
        int delete_pct = pd.delete_pct;
        size_t alive = pd.alive;

        printf("\n--- delete_pct = %d%% (deleted: %zu, alive: %zu) ---\n",
               delete_pct, pd.n_delete, alive);

        // Free store->codes during Baseline-C (not needed until SharedStore restore)
        store->codes.clear();
        store->codes.shrink_to_fit();

        SummaryRow sum_row;
        sum_row.delete_pct = delete_pct;
        sum_row.baseline_b_qps = baseline_b_rows[pi].qps;
        sum_row.baseline_b_recall = baseline_b_rows[pi].recall;

        // Baseline-C: New IndexHNSWFlat with only alive vectors
        {
            std::vector<float> alive_vecs(alive * d);
            for (size_t i = 0; i < alive; i++) {
                memcpy(alive_vecs.data() + i * d,
                       vec_data + pd.alive_to_orig[i] * d,
                       d * sizeof(float));
            }

            double t0_c = get_time_sec();
            faiss::IndexHNSWFlat baseline_c(d, M, metric);
            baseline_c.hnsw.efConstruction = efConstruction;
            baseline_c.add(alive, alive_vecs.data());
            std::vector<float>().swap(alive_vecs);
            double baseline_c_build = get_time_sec() - t0_c;

            baseline_c.hnsw.efSearch = efSearch;
            auto sr_baseline_c = measure_search(
                    baseline_c, xq, nq, k, pd.gt_alive_localid.data());

            sum_row.baseline_c_qps = sr_baseline_c.qps;
            sum_row.baseline_c_recall = sr_baseline_c.recall;

            printf("  Baseline-C: QPS: %.0f  Recall: %.4f  Build: %.2fs\n",
                   sr_baseline_c.qps, sr_baseline_c.recall, baseline_c_build);
        }
        // ^ Scope boundary: alive_vecs freed before SharedStore to fit in RAM

        // SharedStore Pipeline: Rebuild -> Compact -> Best Reorder
        printf("  SharedStore pipeline:\n");

        // Restore store to original layout (compact+reorder corrupt it)
        store->codes = original_store_codes;
        store->ntotal_store = original_ntotal_store;
        store->free_list = original_free_list;

        auto* old_shared =
                dynamic_cast<faiss::IndexFlatShared*>(shared_index->storage);
        old_shared->codes = faiss::MaybeOwnedVector<uint8_t>::create_view(
                store->codes.data(), store->codes.size(), store);

        size_t bitmap_words = (nb + 63) / 64;
        old_shared->deleted_bitmap.assign(bitmap_words, 0);

        for (size_t i = 0; i < nb; i++) {
            if (pd.deleted_set[i]) {
                old_shared->mark_deleted(i);
            }
        }

        double t0 = get_time_sec();
        faiss::IndexHNSW* rebuilt = faiss::build_new_index(
                store, *shared_index, M, efConstruction, metric);
        double rebuild_time = get_time_sec() - t0;

        rebuilt->hnsw.efSearch = efSearch;
        auto sr_rebuilt = measure_search(
                *rebuilt, xq, nq, k, pd.gt_alive_localid.data());

        printf("    Rebuilt:       QPS: %.0f  Recall: %.4f  Build: %.2fs\n",
               sr_rebuilt.qps, sr_rebuilt.recall, rebuild_time);

        auto* rebuilt_shared =
                dynamic_cast<faiss::IndexFlatShared*>(rebuilt->storage);
        faiss::compact_store(*rebuilt_shared);

        auto sr_compact = measure_search(
                *rebuilt, xq, nq, k, pd.gt_alive_localid.data());
        printf("    Compact:       QPS: %.0f  Recall: %.4f\n",
               sr_compact.qps, sr_compact.recall);

        HNSWSnapshot snapshot;
        snapshot.save(*rebuilt);

        faiss::ReorderStrategy strategies[] = {
                faiss::ReorderStrategy::BFS,
                faiss::ReorderStrategy::RCM,
                faiss::ReorderStrategy::DFS,
                faiss::ReorderStrategy::CLUSTER,
                faiss::ReorderStrategy::WEIGHTED,
        };

        double best_qps = sr_compact.qps;
        double best_recall = sr_compact.recall;
        std::string best_strat = "Compact";

        size_t vec_bytes = alive * d * sizeof(float);
        bool skip_reorder_gt = (vec_bytes > 2ULL * 1024 * 1024 * 1024);
        if (skip_reorder_gt) {
            printf("    (Skipping reorder GT verification for memory, using compact recall)\n");
        }
        bool first_strategy = true;

        for (auto strategy : strategies) {
            snapshot.restore(*rebuilt);
            auto perm = faiss::generate_permutation(rebuilt->hnsw, strategy);
            rebuilt->permute_entries(perm.data());

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

            if (qps > best_qps) {
                best_qps = qps;
                best_strat = strategy_name(strategy);
            }

            if (first_strategy && !skip_reorder_gt) {
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
                best_recall = compute_recall(
                        labels.data(), reorder_gt.data(), nq, k);
                first_strategy = false;
            }
        }

        printf("    Best reorder:  QPS: %.0f  Recall: %.4f  [%s]  (%.1f%% of C)\n",
               best_qps, best_recall, best_strat.c_str(),
               best_qps / sum_row.baseline_c_qps * 100);

        sum_row.shared_best_qps = best_qps;
        sum_row.shared_best_recall = best_recall;
        sum_row.best_strategy = best_strat;

        summary_rows.push_back(sum_row);

        delete rebuilt;
    }

    printf("\n=== Cross Delete-Pct Summary ===\n");
    printf("Del%%  | Baseline-B QPS (Recall)  | Baseline-C QPS (Recall)  | SharedStore QPS (Recall) [Strategy]  | vs C%%\n");
    printf("------|--------------------------|--------------------------|--------------------------------------|------\n");

    printf("Ref   | Baseline-A (Full):       QPS: %.0f  Recall: %.4f\n",
           sr_baseline_a.qps, sr_baseline_a.recall);
    printf("Ref   | Fresh (Shared Full):     QPS: %.0f  Recall: %.4f\n",
           sr_fresh.qps, sr_fresh.recall);

    for (const auto& row : summary_rows) {
        printf(" %2d%%  |    %5.0f (%.4f)        |    %5.0f (%.4f)        |    %5.0f (%.4f) [%-9s]        | %5.1f%%\n",
               row.delete_pct,
               row.baseline_b_qps, row.baseline_b_recall,
               row.baseline_c_qps, row.baseline_c_recall,
               row.shared_best_qps, row.shared_best_recall, row.best_strategy.c_str(),
               row.shared_best_qps / row.baseline_c_qps * 100.0);
    }

    delete shared_index;

    return 0;
#endif // ENABLE_HDF5
}
