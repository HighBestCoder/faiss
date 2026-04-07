#include <faiss/index_factory.h>
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/IndexIVFRaBitQFastScan.h>
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

// Stream-load fvecs and add to index in batches
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
        // L2 normalize batch for cosine similarity
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
    fprintf(stderr, "Usage: %s <index_factory> [nprobe=32,64,128] [qb=0,4,8] [--data <dir>] [--stream]\n", prog);
    fprintf(stderr, "\nRaBitQ index_factory examples:\n");
    fprintf(stderr, "  \"RaBitQ\"              Flat brute-force, 1-bit\n");
    fprintf(stderr, "  \"RaBitQ4\"             Flat brute-force, 4-bit\n");
    fprintf(stderr, "  \"RaBitQfs\"            FastScan, 1-bit (SIMD batch)\n");
    fprintf(stderr, "  \"RaBitQfs4\"           FastScan, 4-bit\n");
    fprintf(stderr, "  \"IVF4096,RaBitQ\"      IVF + 1-bit RaBitQ\n");
    fprintf(stderr, "  \"IVF4096,RaBitQ4\"     IVF + 4-bit RaBitQ\n");
    fprintf(stderr, "  \"IVF4096,RaBitQfs\"    IVF + FastScan 1-bit\n");
    fprintf(stderr, "  \"IVF4096,RaBitQfs4\"   IVF + FastScan 4-bit\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  nprobe=X,Y,Z    IVF search probe count (sweep values)\n");
    fprintf(stderr, "  qb=X,Y,Z       Query quantization bits: 0,1,2,3,4,6,8 (sweep values)\n");
    fprintf(stderr, "  --data <dir>    Dataset directory (default: llm/database/cohere_medium_1m)\n");
    fprintf(stderr, "  --stream        Batch-load base vectors (for large datasets)\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s \"RaBitQ\" qb=0,4,8\n", prog);
    fprintf(stderr, "  %s \"IVF4096,RaBitQ\" nprobe=32,64,128 qb=4\n", prog);
    fprintf(stderr, "  %s \"IVF4096,RaBitQ\" nprobe=64 --data llm/database/cohere_large_10m --stream\n", prog);
    exit(1);
}

// ---------- main ----------

int main(int argc, char** argv) {
    if (argc < 2) print_usage(argv[0]);

    const char* index_factory_str = argv[1];

    // Parse options
    std::vector<int> nprobe_values;
    std::vector<int> qb_values;
    std::string data_dir = "llm/database/cohere_medium_1m";
    bool stream_mode = false;

    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "nprobe=", 7) == 0) {
            char* p = argv[i] + 7;
            while (*p) {
                nprobe_values.push_back(atoi(p));
                p = strchr(p, ',');
                if (!p) break;
                p++;
            }
        } else if (strncmp(argv[i], "qb=", 3) == 0) {
            char* p = argv[i] + 3;
            while (*p) {
                qb_values.push_back(atoi(p));
                p = strchr(p, ',');
                if (!p) break;
                p++;
            }
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--stream") == 0) {
            stream_mode = true;
        }
    }

    // Defaults
    if (qb_values.empty()) qb_values = {0, 4, 8};
    if (nprobe_values.empty()) nprobe_values = {64};

    std::string base_path = data_dir + "/base.fvecs";
    std::string query_path = data_dir + "/query.fvecs";
    std::string gt_path = data_dir + "/groundtruth.ivecs";

    // Load query and ground truth (always needed)
    printf("Loading query vectors...\n");
    int dq, nq;
    float* xq = fvecs_read(query_path.c_str(), &dq, &nq);
    printf("  queries: %d vectors, %d dim\n", nq, dq);
    l2_normalize(xq, nq, dq);

    printf("Loading ground truth...\n");
    int gt_k, ngt;
    int* gt_data = ivecs_read(gt_path.c_str(), &gt_k, &ngt);
    printf("  ground truth: %d queries, k=%d\n", ngt, gt_k);

    // Build index
    int d, nb;
    printf("\n========================================\n");
    printf("Building index: %s\n", index_factory_str);
    printf("========================================\n");
    double t0 = elapsed();

    faiss::Index* index = nullptr;

    if (stream_mode) {
        fvecs_info(base_path.c_str(), &d, &nb);
        printf("  base: %d vectors, %d dim (stream mode)\n", nb, d);

        index = faiss::index_factory(d, index_factory_str, faiss::METRIC_INNER_PRODUCT);

        // Train if needed (IVF requires training)
        if (!index->is_trained) {
            int train_n = std::min(nb, 1000000);
            printf("  Training with %d vectors...\n", train_n);
            FILE* tf = fopen(base_path.c_str(), "rb");
            std::vector<float> train_data(train_n * (long)d);
            for (int i = 0; i < train_n; i++) {
                int dd;
                fread(&dd, sizeof(int), 1, tf);
                fread(train_data.data() + i * (long)d, sizeof(float), d, tf);
            }
            fclose(tf);
            l2_normalize(train_data.data(), train_n, d);
            index->train(train_n, train_data.data());
            printf("  Training done.\n");
        }

        printf("  Adding vectors in batches...\n");
        fvecs_stream_add(base_path.c_str(), index);
    } else {
        printf("Loading base vectors...\n");
        float* xb = fvecs_read(base_path.c_str(), &d, &nb);
        printf("  base: %d vectors, %d dim\n", nb, d);

        printf("Normalizing vectors...\n");
        l2_normalize(xb, nb, d);

        index = faiss::index_factory(d, index_factory_str, faiss::METRIC_INNER_PRODUCT);

        if (!index->is_trained) {
            printf("  Training index...\n");
            index->train(nb, xb);
            printf("  Training done.\n");
        }

        printf("  Adding %d vectors...\n", nb);
        index->add(nb, xb);
        delete[] xb;
    }

    double build_time_s = elapsed() - t0;
    printf("  Build time: %.1f s\n", build_time_s);

    long rss_mb = get_rss_kb() / 1024;
    printf("  RSS: %ld MB\n", rss_mb);

    // Detect index types for parameter setting
    faiss::IndexIVF* ivf_index = dynamic_cast<faiss::IndexIVF*>(index);
    faiss::IndexRaBitQ* rbq_index = dynamic_cast<faiss::IndexRaBitQ*>(index);
    faiss::IndexRaBitQFastScan* rbqfs_index = dynamic_cast<faiss::IndexRaBitQFastScan*>(index);
    faiss::IndexIVFRaBitQ* ivfrbq_index = dynamic_cast<faiss::IndexIVFRaBitQ*>(index);

    int k_search = 100;

    // Build param sweep grid
    struct ParamSet {
        int nprobe;
        int qb;
        std::string label;
    };
    std::vector<ParamSet> param_grid;

    if (ivf_index) {
        for (int np : nprobe_values) {
            for (int q : qb_values) {
                char buf[128];
                snprintf(buf, sizeof(buf), "nprobe=%d,qb=%d", np, q);
                param_grid.push_back({np, q, buf});
            }
        }
    } else {
        // Flat RaBitQ: only qb matters
        for (int q : qb_values) {
            char buf[64];
            snprintf(buf, sizeof(buf), "qb=%d", q);
            param_grid.push_back({0, q, buf});
        }
    }

    // Results
    struct Result {
        std::string label;
        double qps_1t, qps_16t;
        float recall_10, recall_100;
    };
    std::vector<Result> results;

    for (auto& ps : param_grid) {
        printf("\n--- %s ---\n", ps.label.c_str());

        // Set nprobe for IVF
        if (ivf_index && ps.nprobe > 0) {
            ivf_index->nprobe = ps.nprobe;
        }

        // Set qb on the index
        if (rbq_index) {
            rbq_index->qb = ps.qb;
        } else if (rbqfs_index) {
            rbqfs_index->qb = ps.qb;
        } else if (ivfrbq_index) {
            ivfrbq_index->qb = ps.qb;
        }

        // Warmup
        {
            std::vector<float> wd(100 * k_search);
            std::vector<faiss::idx_t> wl(100 * k_search);
            omp_set_num_threads(1);
            index->search(100, xq, k_search, wd.data(), wl.data());
        }

        // Single-thread: one query at a time
        omp_set_num_threads(1);
        std::vector<float> dists_1t(nq * k_search);
        std::vector<faiss::idx_t> labels_1t(nq * k_search);

        double t1 = elapsed();
        for (int i = 0; i < nq; i++) {
            index->search(1, xq + i * d, k_search,
                          dists_1t.data() + i * k_search,
                          labels_1t.data() + i * k_search);
        }
        double qps_1t = nq / (elapsed() - t1);

        // Multi-thread: batch of nq
        omp_set_num_threads(16);
        std::vector<float> dists_16t(nq * k_search);
        std::vector<faiss::idx_t> labels_16t(nq * k_search);
        double t2 = elapsed();
        index->search(nq, xq, k_search, dists_16t.data(), labels_16t.data());
        double qps_16t = nq / (elapsed() - t2);

        // Recall (use single-thread results)
        float recall_10 = compute_recall(labels_1t.data(), gt_data, nq, k_search, gt_k, 10);
        float recall_100 = compute_recall(labels_1t.data(), gt_data, nq, k_search, gt_k, 100);

        printf("  QPS_1T: %.0f | QPS_16T: %.0f | Recall@10: %.4f | Recall@100: %.4f\n",
               qps_1t, qps_16t, recall_10, recall_100);

        results.push_back({ps.label, qps_1t, qps_16t, recall_10, recall_100});
    }

    // Summary table
    printf("\n========================================\n");
    printf("SUMMARY: %s\n", index_factory_str);
    printf("========================================\n");
    printf("%-25s %10s %10s %10s %10s\n", "Params", "QPS_1T", "QPS_16T", "R@10", "R@100");
    printf("%-25s %10s %10s %10s %10s\n", "-------------------------", "----------", "----------", "----------", "----------");
    for (auto& r : results) {
        printf("%-25s %10.0f %10.0f %9.2f%% %9.2f%%\n",
               r.label.c_str(), r.qps_1t, r.qps_16t,
               r.recall_10 * 100, r.recall_100 * 100);
    }

    // Find best config
    int best_idx = -1;
    double best_qps = 0;
    int fallback_idx = 0;
    float fallback_recall = 0;

    for (int i = 0; i < (int)results.size(); i++) {
        if (results[i].recall_10 >= 0.95 && results[i].qps_1t > best_qps) {
            best_idx = i;
            best_qps = results[i].qps_1t;
        }
        if (results[i].recall_10 > fallback_recall) {
            fallback_idx = i;
            fallback_recall = results[i].recall_10;
        }
    }

    int final_idx = (best_idx >= 0) ? best_idx : fallback_idx;
    auto& r = results[final_idx];

    printf("\n=== BEST RESULT ===\n");
    printf("BUILD_TIME_S=%.1f\n", build_time_s);
    printf("RSS_MB=%ld\n", rss_mb);
    printf("QPS_1T=%.0f\n", r.qps_1t);
    printf("QPS_16T=%.0f\n", r.qps_16t);
    printf("RECALL_AT_10=%.4f\n", r.recall_10);
    printf("RECALL_AT_100=%.4f\n", r.recall_100);
    printf("SEARCH_PARAMS=%s\n", r.label.c_str());

    delete index;
    delete[] xq;
    delete[] gt_data;

    return 0;
}
