#include <faiss/index_factory.h>
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexRefine.h>
#include <faiss/HNSWReorder.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/VisitedTable.h>
#include <omp.h>
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ---------- forward declarations ----------
long get_rss_kb();

// ---------- fvecs / ivecs readers ----------

// Get dimension and total vector count from fvecs file without loading data
void fvecs_info(const char* fname, int* d_out, int* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    int d;
    if (fread(&d, sizeof(int), 1, f) != 1) { fclose(f); exit(1); }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fclose(f);
    long vec_size = 4 + d * 4;
    *d_out = d;
    *n_out = (int)(fsize / vec_size);
}

float* fvecs_read(const char* fname, int* d_out, int* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    int d;
    if (fread(&d, sizeof(int), 1, f) != 1) { fclose(f); exit(1); }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    long vec_size = 4 + d * 4;
    int n = fsize / vec_size;
    float* data = new float[n * (long)d];
    for (int i = 0; i < n; i++) {
        int dd;
        if (fread(&dd, sizeof(int), 1, f) != 1) break;
        if ((int)fread(data + i * (long)d, sizeof(float), d, f) != d) break;
    }
    fclose(f);
    *d_out = d;
    *n_out = n;
    return data;
}

int* ivecs_read(const char* fname, int* d_out, int* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    int d;
    if (fread(&d, sizeof(int), 1, f) != 1) { fclose(f); exit(1); }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    long vec_size = 4 + d * 4;
    int n = fsize / vec_size;
    int* data = new int[n * (long)d];
    for (int i = 0; i < n; i++) {
        int dd;
        if (fread(&dd, sizeof(int), 1, f) != 1) break;
        if ((int)fread(data + i * (long)d, sizeof(int), d, f) != d) break;
    }
    fclose(f);
    *d_out = d;
    *n_out = n;
    return data;
}

// Stream-load fvecs and add to index in batches (memory-efficient for large datasets)
void fvecs_stream_add(const char* fname, faiss::Index* index, int batch_size = 100000) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }
    int d;
    if (fread(&d, sizeof(int), 1, f) != 1) { fclose(f); exit(1); }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    long vec_size = 4 + d * 4;
    int n = (int)(fsize / vec_size);

    std::vector<float> batch(batch_size * (long)d);
    int added = 0;
    while (added < n) {
        int cur_batch = std::min(batch_size, n - added);
        for (int i = 0; i < cur_batch; i++) {
            int dd;
            if (fread(&dd, sizeof(int), 1, f) != 1) break;
            if ((int)fread(batch.data() + i * (long)d, sizeof(float), d, f) != d) break;
        }
        // L2 normalize batch
        for (int i = 0; i < cur_batch; i++) {
            float norm = 0;
            for (int j = 0; j < d; j++) {
                norm += batch[i * (long)d + j] * batch[i * (long)d + j];
            }
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (int j = 0; j < d; j++) {
                    batch[i * (long)d + j] /= norm;
                }
            }
        }
        index->add(cur_batch, batch.data());
        added += cur_batch;
        if (added % 1000000 == 0 || added == n) {
            printf("  Added %d / %d vectors (RSS: %ld MB)\n", added, n, get_rss_kb() / 1024);
        }
    }
    fclose(f);
}

// ---------- helpers ----------

void l2_normalize(float* data, int n, int d) {
    for (int i = 0; i < n; i++) {
        float norm = 0;
        for (int j = 0; j < d; j++) {
            norm += data[i * (long)d + j] * data[i * (long)d + j];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (int j = 0; j < d; j++) {
                data[i * (long)d + j] /= norm;
            }
        }
    }
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

long get_rss_kb() {
    std::ifstream ifs("/proc/self/status");
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("VmRSS:") == 0) {
            long kb = 0;
            sscanf(line.c_str(), "VmRSS: %ld", &kb);
            return kb;
        }
    }
    return 0;
}

float compute_recall(const faiss::idx_t* results, const int* gt,
                     int nq, int k_result, int k_gt, int at_k) {
    int hit = 0;
    int total = nq * at_k;
    for (int q = 0; q < nq; q++) {
        const int* gt_row = gt + q * (long)k_gt;
        const faiss::idx_t* res_row = results + q * (long)k_result;
        for (int i = 0; i < at_k && i < k_result; i++) {
            for (int j = 0; j < at_k && j < k_gt; j++) {
                if (res_row[i] == gt_row[j]) {
                    hit++;
                    break;
                }
            }
        }
    }
    return (float)hit / total;
}

void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s <index_factory> [efSearch=64,128,256] [nprobe=32,64,128] [--data <dir>] [--stream]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --stream  Batch-load base vectors (memory-efficient for large datasets)\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s \"HNSW32,Flat\"\n", prog);
    fprintf(stderr, "  %s \"HNSW32,SQfp16\" efSearch=64,128,256\n", prog);
    fprintf(stderr, "  %s \"HNSW32,SQfp16\" --data llm/database/cohere_10m --stream\n", prog);
    exit(1);
}

// ---------- main ----------

int main(int argc, char** argv) {
    if (argc < 2) print_usage(argv[0]);

    const char* index_factory_str = argv[1];

    // Parse search params and options
    std::vector<int> ef_values;
    std::vector<int> nprobe_values;
    std::string data_dir = "llm/database/cohere_medium_1m";
    bool stream_mode = false;
    int ef_construction = 0;
    bool no_reorder = false;  // skip O14 BFS reorder
    bool no_vt_reuse = false; // skip O16 VT reuse

    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "efConstruction=", 15) == 0) {
            ef_construction = atoi(argv[i] + 15);
        } else if (strncmp(argv[i], "efSearch=", 9) == 0) {
            char* p = argv[i] + 9;
            while (*p) {
                ef_values.push_back(atoi(p));
                p = strchr(p, ',');
                if (!p) break;
                p++;
            }
        } else if (strncmp(argv[i], "nprobe=", 7) == 0) {
            char* p = argv[i] + 7;
            while (*p) {
                nprobe_values.push_back(atoi(p));
                p = strchr(p, ',');
                if (!p) break;
                p++;
            }
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--stream") == 0) {
            stream_mode = true;
        } else if (strcmp(argv[i], "--no-reorder") == 0) {
            no_reorder = true;
        } else if (strcmp(argv[i], "--no-vt-reuse") == 0) {
            no_vt_reuse = true;
        }
    }

    // Defaults
    if (ef_values.empty() && nprobe_values.empty()) {
        ef_values = {64, 128, 256};
    }

    std::string base_path = data_dir + "/base.fvecs";
    std::string query_path = data_dir + "/query.fvecs";
    std::string gt_path = data_dir + "/groundtruth.ivecs";

    // Load data and build index
    int d, nb;
    printf("Building index: %s\n", index_factory_str);
    double t0 = elapsed();

    if (stream_mode) {
        // Stream mode: read fvecs header for dim/count, then batch-add
        fvecs_info(base_path.c_str(), &d, &nb);
        printf("  base: %d vectors, %d dim (stream mode)\n", nb, d);

        faiss::Index* idx = faiss::index_factory(d, index_factory_str, faiss::METRIC_INNER_PRODUCT);

        // Set efConstruction if specified
        faiss::IndexHNSW* hnsw_build = dynamic_cast<faiss::IndexHNSW*>(idx);
        if (hnsw_build && ef_construction > 0) {
            hnsw_build->hnsw.efConstruction = ef_construction;
            printf("  efConstruction=%d\n", ef_construction);
        }

        // HNSW doesn't need training; IVF needs training data
        if (!idx->is_trained) {
            printf("  Training index (loading sample)...\n");
            // Load first 1M vectors for training
            int train_n = std::min(nb, 1000000);
            FILE* tf = fopen(base_path.c_str(), "rb");
            std::vector<float> train_data(train_n * (long)d);
            for (int i = 0; i < train_n; i++) {
                int dd;
                fread(&dd, sizeof(int), 1, tf);
                fread(train_data.data() + i * (long)d, sizeof(float), d, tf);
            }
            fclose(tf);
            l2_normalize(train_data.data(), train_n, d);
            idx->train(train_n, train_data.data());
        }

        printf("  Adding vectors in batches...\n");
        fvecs_stream_add(base_path.c_str(), idx);

        double build_time_s = elapsed() - t0;
        printf("  Build time: %.1f s\n", build_time_s);

        // Load query and ground truth
        printf("Loading query vectors...\n");
        int dq, nq;
        float* xq = fvecs_read(query_path.c_str(), &dq, &nq);
        printf("  queries: %d vectors, %d dim\n", nq, dq);
        l2_normalize(xq, nq, dq);

        printf("Loading ground truth...\n");
        int gt_k, ngt;
        int* gt_data = ivecs_read(gt_path.c_str(), &gt_k, &ngt);
        printf("  ground truth: %d queries, k=%d\n", ngt, gt_k);

        // Memory
        long rss_mb = get_rss_kb() / 1024;
        printf("  RSS: %ld MB\n", rss_mb);

        // Search and recall (simplified for stream mode)
        int k_search = 100;
        faiss::IndexHNSW* hnsw_idx = dynamic_cast<faiss::IndexHNSW*>(idx);
        faiss::IndexIVF* ivf_idx = dynamic_cast<faiss::IndexIVF*>(idx);

        // BFS graph reorder (O14)
        std::vector<faiss::idx_t> id_remap;
        if (hnsw_idx && !no_reorder) {
            printf("Applying BFS graph reorder...\n");
            double t_reorder = elapsed();
            auto perm = faiss::generate_permutation(hnsw_idx->hnsw, faiss::ReorderStrategy::BFS);
            hnsw_idx->permute_entries(perm.data());
            id_remap.assign(perm.begin(), perm.end());
            printf("  Reorder time: %.1f s\n", elapsed() - t_reorder);
        }

        // Set search params
        std::vector<int>& sweep_values = !ef_values.empty() ? ef_values : nprobe_values;
        if (sweep_values.empty()) sweep_values.push_back(0);

        for (int sv : sweep_values) {
            if (hnsw_idx && sv > 0) hnsw_idx->hnsw.efSearch = sv;
            if (ivf_idx && sv > 0) ivf_idx->nprobe = sv;

            printf("\n--- %s=%d ---\n", hnsw_idx ? "efSearch" : (ivf_idx ? "nprobe" : "none"), sv);

            // Warmup
            {
                std::vector<float> wd(100 * k_search);
                std::vector<faiss::idx_t> wl(100 * k_search);
                omp_set_num_threads(1);
                idx->search(100, xq, k_search, wd.data(), wl.data());
            }

            // Single-thread with VT reuse
            omp_set_num_threads(1);
            std::vector<float> dists(nq * k_search);
            std::vector<faiss::idx_t> labels(nq * k_search);

            std::unique_ptr<faiss::VisitedTable> vt;
            std::unique_ptr<faiss::SearchParametersHNSW> hnsw_params;
            const faiss::SearchParameters* sp = nullptr;
            if (hnsw_idx && !no_vt_reuse) {
                vt = std::make_unique<faiss::VisitedTable>(idx->ntotal);
                hnsw_params = std::make_unique<faiss::SearchParametersHNSW>();
                hnsw_params->efSearch = hnsw_idx->hnsw.efSearch;
                hnsw_params->visited_table = vt.get();
                sp = hnsw_params.get();
            }

            double t1 = elapsed();
            for (int i = 0; i < nq; i++) {
                idx->search(1, xq + i * d, k_search,
                            dists.data() + i * k_search,
                            labels.data() + i * k_search, sp);
            }
            double qps_1t = nq / (elapsed() - t1);

            // Multi-thread
            omp_set_num_threads(16);
            std::vector<float> dists16(nq * k_search);
            std::vector<faiss::idx_t> labels16(nq * k_search);
            double t2 = elapsed();
            idx->search(nq, xq, k_search, dists16.data(), labels16.data());
            double qps_16t = nq / (elapsed() - t2);

            // Remap IDs
            if (!id_remap.empty()) {
                for (int i = 0; i < nq * k_search; i++) {
                    if (labels[i] >= 0) labels[i] = id_remap[labels[i]];
                    if (labels16[i] >= 0) labels16[i] = id_remap[labels16[i]];
                }
            }

            float recall_10 = compute_recall(labels.data(), gt_data, nq, k_search, gt_k, 10);
            float recall_100 = compute_recall(labels.data(), gt_data, nq, k_search, gt_k, 100);

            printf("  QPS_1T: %.0f | QPS_16T: %.0f | Recall@10: %.4f | Recall@100: %.4f\n",
                   qps_1t, qps_16t, recall_10, recall_100);
        }

        printf("\n=== RESULTS ===\n");
        printf("BUILD_TIME_S=%.1f\n", build_time_s);
        printf("RSS_MB=%ld\n", rss_mb);

        delete idx;
        delete[] xq;
        delete[] gt_data;
        return 0;
    }

    // Normal mode: load everything into memory
    printf("Loading base vectors...\n");
    float* xb = fvecs_read(base_path.c_str(), &d, &nb);
    printf("  base: %d vectors, %d dim\n", nb, d);

    printf("Loading query vectors...\n");
    int dq, nq;
    float* xq = fvecs_read(query_path.c_str(), &dq, &nq);
    printf("  queries: %d vectors, %d dim\n", nq, dq);

    printf("Loading ground truth...\n");
    int gt_k, ngt;
    int* gt = ivecs_read(gt_path.c_str(), &gt_k, &ngt);
    printf("  ground truth: %d queries, k=%d\n", ngt, gt_k);

    // L2 normalize for cosine -> IP
    printf("Normalizing vectors...\n");
    l2_normalize(xb, nb, d);
    l2_normalize(xq, nq, dq);

    // Build index
    printf("Building index: %s\n", index_factory_str);
    faiss::Index* index = faiss::index_factory(d, index_factory_str, faiss::METRIC_INNER_PRODUCT);

    // Set efConstruction if specified
    {
        faiss::IndexHNSW* h = dynamic_cast<faiss::IndexHNSW*>(index);
        if (h && ef_construction > 0) {
            h->hnsw.efConstruction = ef_construction;
            printf("  efConstruction=%d\n", ef_construction);
        }
    }

    if (!index->is_trained) {
        printf("  Training index...\n");
        index->train(nb, xb);
    }

    index->add(nb, xb);
    double build_time = elapsed() - t0;
    printf("  Build time: %.1f s\n", build_time);

    // Free base vectors early to save memory (especially for 10M+ datasets)
    delete[] xb;
    xb = nullptr;

    // Memory
    long rss_kb = get_rss_kb();
    long rss_mb = rss_kb / 1024;
    printf("  RSS: %ld MB\n", rss_mb);

    int k_search = 100;

    // Get HNSW / IVF index pointer
    faiss::IndexHNSW* hnsw_index = dynamic_cast<faiss::IndexHNSW*>(index);
    faiss::IndexIVF* ivf_index = dynamic_cast<faiss::IndexIVF*>(index);

    // BFS graph reorder for HNSW cache locality (O14)
    // perm[new_id] = old_id, so we use perm to map search results back
    std::vector<faiss::idx_t> id_remap;
    if (hnsw_index && !no_reorder) {
        printf("Applying BFS graph reorder...\n");
        double t_reorder = elapsed();
        auto perm = faiss::generate_permutation(
                hnsw_index->hnsw, faiss::ReorderStrategy::BFS);
        hnsw_index->permute_entries(perm.data());
        // perm[new_id] = old_id, use as remap table
        id_remap.assign(perm.begin(), perm.end());
        printf("  Reorder time: %.1f s\n", elapsed() - t_reorder);
    }

    // Determine which params to sweep
    struct ParamSet {
        std::string name;
        int value;
    };
    std::vector<ParamSet> params;

    if (hnsw_index && !ef_values.empty()) {
        for (int v : ef_values) params.push_back({"efSearch", v});
    } else if (ivf_index && !nprobe_values.empty()) {
        for (int v : nprobe_values) params.push_back({"nprobe", v});
    } else {
        params.push_back({"none", 0});
    }

    // Track best result
    int best_idx = -1;
    double best_qps_1t = 0;
    float best_recall_10 = 0;

    // Fallback
    int fallback_idx = -1;
    float fallback_recall_10 = 0;

    struct Result {
        std::string param_str;
        double qps_1t, qps_16t;
        float recall_10, recall_100;
    };
    std::vector<Result> results;

    for (int pi = 0; pi < (int)params.size(); pi++) {
        auto& p = params[pi];

        // Set param
        if (p.name == "efSearch" && hnsw_index) {
            hnsw_index->hnsw.efSearch = p.value;
        } else if (p.name == "nprobe" && ivf_index) {
            ivf_index->nprobe = p.value;
        }

        char param_str[64];
        if (p.name == "none") {
            snprintf(param_str, sizeof(param_str), "none");
        } else {
            snprintf(param_str, sizeof(param_str), "%s=%d", p.name.c_str(), p.value);
        }
        printf("\n--- %s ---\n", param_str);

        // Warmup
        {
            std::vector<float> wd(100 * k_search);
            std::vector<faiss::idx_t> wl(100 * k_search);
            omp_set_num_threads(1);
            index->search(100, xq, k_search, wd.data(), wl.data());
        }

        // Single-thread with VisitedTable reuse for HNSW
        omp_set_num_threads(1);
        std::vector<float> dists_1t(nq * k_search);
        std::vector<faiss::idx_t> labels_1t(nq * k_search);

        // Prepare reusable VisitedTable + SearchParametersHNSW (O16)
        std::unique_ptr<faiss::VisitedTable> vt;
        std::unique_ptr<faiss::SearchParametersHNSW> hnsw_params;
        const faiss::SearchParameters* search_params = nullptr;
        if (hnsw_index && !no_vt_reuse) {
            vt = std::make_unique<faiss::VisitedTable>(index->ntotal);
            hnsw_params = std::make_unique<faiss::SearchParametersHNSW>();
            hnsw_params->efSearch = hnsw_index->hnsw.efSearch;
            hnsw_params->visited_table = vt.get();
            search_params = hnsw_params.get();
        }

        double t1 = elapsed();
        for (int i = 0; i < nq; i++) {
            index->search(1, xq + i * d, k_search,
                          dists_1t.data() + i * k_search,
                          labels_1t.data() + i * k_search,
                          search_params);
        }
        double dur_1t = elapsed() - t1;
        double qps_1t = nq / dur_1t;

        // Multi-thread
        omp_set_num_threads(16);
        std::vector<float> dists_16t(nq * k_search);
        std::vector<faiss::idx_t> labels_16t(nq * k_search);
        double t2 = elapsed();
        index->search(nq, xq, k_search, dists_16t.data(), labels_16t.data());
        double dur_16t = elapsed() - t2;
        double qps_16t = nq / dur_16t;

        // Remap IDs back to original if graph was reordered
        if (!id_remap.empty()) {
            for (int i = 0; i < nq * k_search; i++) {
                if (labels_1t[i] >= 0)
                    labels_1t[i] = id_remap[labels_1t[i]];
            }
            for (int i = 0; i < nq * k_search; i++) {
                if (labels_16t[i] >= 0)
                    labels_16t[i] = id_remap[labels_16t[i]];
            }
        }

        // Recall
        float recall_10 = compute_recall(labels_1t.data(), gt, nq, k_search, gt_k, 10);
        float recall_100 = compute_recall(labels_1t.data(), gt, nq, k_search, gt_k, 100);

        printf("  QPS_1T: %.0f | QPS_16T: %.0f | Recall@10: %.4f | Recall@100: %.4f\n",
               qps_1t, qps_16t, recall_10, recall_100);

        results.push_back({param_str, qps_1t, qps_16t, recall_10, recall_100});

        // Update best (highest QPS with recall >= 95%)
        if (recall_10 >= 0.95 && qps_1t > best_qps_1t) {
            best_idx = pi;
            best_qps_1t = qps_1t;
            best_recall_10 = recall_10;
        }

        // Update fallback (highest recall)
        if (recall_10 > fallback_recall_10) {
            fallback_idx = pi;
            fallback_recall_10 = recall_10;
        }
    }

    // Select winner
    int final_idx = (best_idx >= 0) ? best_idx : fallback_idx;
    auto& r = results[final_idx];

    printf("\n=== RESULTS ===\n");
    printf("BUILD_TIME_S=%.1f\n", build_time);
    printf("RSS_MB=%ld\n", rss_mb);
    printf("QPS_1T=%.0f\n", r.qps_1t);
    printf("QPS_16T=%.0f\n", r.qps_16t);
    printf("RECALL_AT_10=%.4f\n", r.recall_10);
    printf("RECALL_AT_100=%.4f\n", r.recall_100);
    printf("SEARCH_PARAMS=%s\n", r.param_str.c_str());

    // Cleanup
    delete index;
    delete[] xq;
    delete[] gt;

    return 0;
}
