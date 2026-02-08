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

struct StageResult {
    std::string name;
    double build_time_sec;
    double qps;
    double recall;
    bool correct;
};

struct SummaryRow {
    int delete_pct;
    double baseline_b_qps, baseline_b_recall;
    double baseline_c_qps, baseline_c_recall;
    double shared_best_qps, shared_best_recall;
    std::string best_strategy;
};

void print_table_header() {
    printf("\n%-40s %10s %10s %10s %8s\n",
           "Stage",
           "Time(s)",
           "QPS",
           "Recall@10",
           "vs A");
    printf("%s\n", std::string(85, '-').c_str());
}

void print_table_row(const StageResult& r, double baseline_qps) {
    printf("%-40s %10.2f %10.0f %10.4f %7.1f%%",
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
    int single_delete_pct = -1;
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

    // --- load dataset ---
    HDF5Dataset dataset;
    if (!load_hdf5_dataset(hdf5_path, dataset)) {
        return 1;
    }

    size_t nb = dataset.nb;
    size_t nq = dataset.nq;
    size_t d = dataset.dim;

    // HDF5 ground truth (all N vectors, for Baseline-A)
    std::vector<faiss::idx_t> ground_truth_all(nq * k);
    for (size_t q = 0; q < nq; q++) {
        for (size_t i = 0; i < k; i++) {
            ground_truth_all[q * k + i] =
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
    printf("  OMP threads:      %d\n", omp_get_max_threads());
    printf("================================================================\n");

    // ============================================================
    // Baseline-A: IndexHNSWFlat, all N vectors, no deletion
    // Kept alive for Baseline-B
    // ============================================================
    printf("\n[Reference] Baseline-A: IndexHNSWFlat (all %zu vectors, no deletion)...\n", nb);
    double baseline_a_build;
    SearchResult sr_baseline_a;
    auto* baseline_a = new faiss::IndexHNSWFlat(d, M, metric);
    {
        double t0_bl = get_time_sec();
        baseline_a->hnsw.efConstruction = efConstruction;
        baseline_a->add(nb, xb);
        baseline_a_build = get_time_sec() - t0_bl;
        printf("  Build: %.2fs\n", baseline_a_build);

        baseline_a->hnsw.efSearch = efSearch;
        sr_baseline_a = measure_search(
                *baseline_a, xq, nq, k, ground_truth_all.data());

        printf("  QPS: %.0f  Recall@10: %.4f\n",
               sr_baseline_a.qps,
               sr_baseline_a.recall);
    }

    // ============================================================
    // Fresh: IndexHNSW + IndexFlatShared (all N vectors)
    // Kept alive as the "old index" for the rebuild loop
    // ============================================================
    printf("\n[Reference] Fresh: IndexHNSW + IndexFlatShared (all %zu vectors)...\n", nb);
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

        printf("  QPS: %.0f  Recall@10: %.4f  (%.1f%% of A)\n",
               sr_fresh.qps,
               sr_fresh.recall,
               sr_fresh.qps / sr_baseline_a.qps * 100);
    }

    // Save original store state — compact+reorder modify store->codes in
    // place, so we must restore before the next iteration's build_new_index
    // which reads old_shared->storage_id_map into the original layout.
    std::vector<uint8_t> original_store_codes = store->codes;
    size_t original_ntotal_store = store->ntotal_store;
    std::vector<faiss::idx_t> original_free_list = store->free_list;

    // Loop over delete percentages
    std::vector<int> delete_pcts = {10, 20, 30, 40, 50, 60, 70, 80};
    if (single_delete_pct != -1) {
        delete_pcts = {single_delete_pct};
    }

    std::vector<SummaryRow> summary_rows;

    for (int delete_pct : delete_pcts) {
        // --- Determine deleted set ---
        size_t n_delete = nb * delete_pct / 100;
        std::mt19937 del_rng(999 + delete_pct);
        std::vector<bool> deleted_set(nb, false);
        {
            size_t cnt = 0;
            while (cnt < n_delete) {
                size_t idx = del_rng() % nb;
                if (!deleted_set[idx]) {
                    deleted_set[idx] = true;
                    cnt++;
                }
            }
        }
        size_t alive = nb - n_delete;

        printf("\n--- delete_pct = %d%% (deleted: %zu, alive: %zu) ---\n",
               delete_pct, n_delete, alive);

        std::vector<StageResult> results;

        // --- Build alive ID mapping ---
        std::vector<faiss::idx_t> alive_to_orig;
        alive_to_orig.reserve(alive);
        for (size_t i = 0; i < nb; i++) {
            if (!deleted_set[i]) {
                alive_to_orig.push_back(static_cast<faiss::idx_t>(i));
            }
        }

        // --- Compute brute-force GT on alive vectors ---
        printf("  Computing brute-force ground truth on %zu alive vectors...\n", alive);
        std::vector<faiss::idx_t> gt_alive_origid(nq * k);
        std::vector<faiss::idx_t> gt_alive_localid(nq * k);
        {
            // Use dataset.train (xb) which we kept alive
            std::vector<float> alive_vecs(alive * d);
            for (size_t i = 0; i < alive; i++) {
                memcpy(alive_vecs.data() + i * d,
                       xb + alive_to_orig[i] * d,
                       d * sizeof(float));
            }

            faiss::IndexFlat brute_force(d, metric);
            brute_force.add(alive, alive_vecs.data());

            std::vector<float> gt_dists(nq * k);
            brute_force.search(nq, xq, k, gt_dists.data(), gt_alive_localid.data());

            for (size_t i = 0; i < nq * k; i++) {
                faiss::idx_t local = gt_alive_localid[i];
                gt_alive_origid[i] = (local >= 0) ? alive_to_orig[local] : -1;
            }
        }

        SummaryRow sum_row;
        sum_row.delete_pct = delete_pct;

        // ============================================================
        // Baseline-B: Overfetch + post-filter
        // ============================================================
        {
            auto overfetch = measure_overfetch_search(
                    *baseline_a, xq, nq, k, deleted_set,
                    gt_alive_origid.data(), nb);

            sum_row.baseline_b_qps = overfetch.qps;
            sum_row.baseline_b_recall = overfetch.recall;

            results.push_back({"Baseline-B",
                                0.0, // N/A
                                overfetch.qps,
                                overfetch.recall,
                                true});
            printf("  Baseline-B: QPS: %.0f  Recall: %.4f  (%.1f%% of A)\n",
                   overfetch.qps,
                   overfetch.recall,
                   overfetch.qps / sr_baseline_a.qps * 100);
        }

        // ============================================================
        // Baseline-C: New IndexHNSWFlat with only alive vectors
        // ============================================================
        {
            std::vector<float> alive_vecs(alive * d);
            for (size_t i = 0; i < alive; i++) {
                memcpy(alive_vecs.data() + i * d,
                       xb + alive_to_orig[i] * d,
                       d * sizeof(float));
            }

            double t0_c = get_time_sec();
            faiss::IndexHNSWFlat baseline_c(d, M, metric);
            baseline_c.hnsw.efConstruction = efConstruction;
            baseline_c.add(alive, alive_vecs.data());
            double baseline_c_build = get_time_sec() - t0_c;

            baseline_c.hnsw.efSearch = efSearch;
            auto sr_baseline_c = measure_search(
                    baseline_c, xq, nq, k, gt_alive_localid.data());

            sum_row.baseline_c_qps = sr_baseline_c.qps;
            sum_row.baseline_c_recall = sr_baseline_c.recall;

            results.push_back({"Baseline-C",
                                baseline_c_build,
                                sr_baseline_c.qps,
                                sr_baseline_c.recall,
                                true});
            printf("  Baseline-C: QPS: %.0f  Recall: %.4f  Build: %.2fs\n",
                   sr_baseline_c.qps,
                   sr_baseline_c.recall,
                   baseline_c_build);
        }

        // ============================================================
        // SharedStore Pipeline: Rebuild -> Compact -> Best Reorder
        // ============================================================
        printf("  SharedStore pipeline:\n");
        auto* old_shared =
                dynamic_cast<faiss::IndexFlatShared*>(shared_index->storage);
        
        size_t bitmap_words = (nb + 63) / 64;
        old_shared->deleted_bitmap.assign(bitmap_words, 0);

        for (size_t i = 0; i < nb; i++) {
            if (deleted_set[i]) {
                old_shared->mark_deleted(i);
            }
        }
        
        double t0 = get_time_sec();
        faiss::IndexHNSW* rebuilt = faiss::build_new_index(
                store, *shared_index, M, efConstruction, metric);
        double rebuild_time = get_time_sec() - t0;

        rebuilt->hnsw.efSearch = efSearch;
        auto sr_rebuilt = measure_search(
                *rebuilt, xq, nq, k, gt_alive_localid.data());

        printf("    Rebuilt:       QPS: %.0f  Recall: %.4f  Build: %.2fs\n",
               sr_rebuilt.qps, sr_rebuilt.recall, rebuild_time);

        // Compact
        auto* rebuilt_shared =
                dynamic_cast<faiss::IndexFlatShared*>(rebuilt->storage);
        faiss::compact_store(*rebuilt_shared);
        
        auto sr_compact = measure_search(
                *rebuilt, xq, nq, k, gt_alive_localid.data());
        printf("    Compact:       QPS: %.0f  Recall: %.4f\n",
               sr_compact.qps, sr_compact.recall);

        // Reorder
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

        bool first_strategy = true;

        for (auto strategy : strategies) {
            snapshot.restore(*rebuilt);
            auto perm = faiss::generate_permutation(rebuilt->hnsw, strategy);
            rebuilt->permute_entries(perm.data());

            // Measure QPS
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

            // Reuse recall from compact for subsequent strategies
            if (first_strategy) {
                 // Verify recall by brute-force GT on reordered vectors
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
        
        printf("    Best reorder:  QPS: %.0f  Recall: %.4f  [%s]  (%.1f%% of A)\n",
               best_qps, best_recall, best_strat.c_str(), 
               best_qps / sr_baseline_a.qps * 100);

        sum_row.shared_best_qps = best_qps;
        sum_row.shared_best_recall = best_recall;
        sum_row.best_strategy = best_strat;
        
        summary_rows.push_back(sum_row);

        delete rebuilt;

        store->codes = original_store_codes;
        store->ntotal_store = original_ntotal_store;
        store->free_list = original_free_list;
        old_shared->codes = faiss::MaybeOwnedVector<uint8_t>::create_view(
                store->codes.data(), store->codes.size(), store);
    }

    // --- Final Summary Table ---
    printf("\n=== Cross Delete-Pct Summary ===\n");
    printf("Del%%  | Baseline-B QPS (Recall)  | Baseline-C QPS (Recall)  | SharedStore QPS (Recall) [Strategy]  | vs A%%\n");
    printf("------|--------------------------|--------------------------|--------------------------------------|------\n");

    printf("Ref   | Baseline-A (Full):       QPS: %.0f  Recall: %.4f\n", sr_baseline_a.qps, sr_baseline_a.recall);
    printf("Ref   | Fresh (Shared Full):     QPS: %.0f  Recall: %.4f\n", sr_fresh.qps, sr_fresh.recall);
    
    for (const auto& row : summary_rows) {
        printf(" %2d%%  |    %5.0f (%.4f)        |    %5.0f (%.4f)        |    %5.0f (%.4f) [%-9s]        | %5.1f%%\n",
               row.delete_pct,
               row.baseline_b_qps, row.baseline_b_recall,
               row.baseline_c_qps, row.baseline_c_recall,
               row.shared_best_qps, row.shared_best_recall, row.best_strategy.c_str(),
               row.shared_best_qps / sr_baseline_a.qps * 100.0);
    }

    delete baseline_a;
    delete shared_index;

    return 0;
#endif // ENABLE_HDF5
}
