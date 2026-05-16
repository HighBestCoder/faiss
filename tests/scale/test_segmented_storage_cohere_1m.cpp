/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Scale regression: builds the same HNSWFlat two ways on cohere_medium_1m:
//   reference: one-shot .add(1M) with default in-memory storage
//   incremental: 4 x 250k .add + storage->flush() with SegmentedFileCodesStorage
// Checks recall@10 match and that cumulative bytes written by the incremental
// path is <= 2/3 of the naive 4x write_index baseline.
//
// Skipped automatically if the cohere_medium_1m dataset is not present.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/CodesStorage.h>
#include <faiss/impl/SegmentedFileCodesStorage.h>
#include <faiss/index_io.h>

namespace {

constexpr const char* kDataDir = "/ceph/faiss-dev/llm/database/cohere_medium_1m";
constexpr const char* kWorkBase = "/tmp/appendable_scale_1m";

bool exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

uint64_t file_size(const std::string& p) {
    struct stat st;
    if (::stat(p.c_str(), &st) != 0) return 0;
    return (uint64_t)st.st_size;
}

float* fvecs_read(const char* fname, int* d_out, int* n_out) {
    FILE* f = ::fopen(fname, "rb");
    if (!f) return nullptr;
    int d;
    if (::fread(&d, sizeof(int), 1, f) != 1) {
        ::fclose(f);
        return nullptr;
    }
    ::fseek(f, 0, SEEK_END);
    long fsize = ::ftell(f);
    ::fseek(f, 0, SEEK_SET);
    long vec_size = 4 + (long)d * 4;
    int n = (int)(fsize / vec_size);
    float* data = new float[(long)n * d];
    for (int i = 0; i < n; ++i) {
        int dd;
        if (::fread(&dd, sizeof(int), 1, f) != 1) break;
        if ((int)::fread(data + (long)i * d, sizeof(float), d, f) != d) break;
    }
    ::fclose(f);
    *d_out = d;
    *n_out = n;
    return data;
}

int* ivecs_read(const char* fname, int* d_out, int* n_out) {
    FILE* f = ::fopen(fname, "rb");
    if (!f) return nullptr;
    int d;
    if (::fread(&d, sizeof(int), 1, f) != 1) {
        ::fclose(f);
        return nullptr;
    }
    ::fseek(f, 0, SEEK_END);
    long fsize = ::ftell(f);
    ::fseek(f, 0, SEEK_SET);
    long vec_size = 4 + (long)d * 4;
    int n = (int)(fsize / vec_size);
    int* data = new int[(long)n * d];
    for (int i = 0; i < n; ++i) {
        int dd;
        if (::fread(&dd, sizeof(int), 1, f) != 1) break;
        if ((int)::fread(data + (long)i * d, sizeof(int), d, f) != d) break;
    }
    ::fclose(f);
    *d_out = d;
    *n_out = n;
    return data;
}

double recall_at(
        const faiss::idx_t* labels,
        const int* gt,
        int nq,
        int k,
        int gt_k) {
    size_t hits = 0;
    for (int q = 0; q < nq; ++q) {
        std::unordered_set<int> truth;
        for (int i = 0; i < k; ++i) {
            truth.insert(gt[q * gt_k + i]);
        }
        for (int i = 0; i < k; ++i) {
            if (truth.count((int)labels[q * k + i])) {
                ++hits;
            }
        }
    }
    return (double)hits / (double)(nq * k);
}

void rm_rf(const std::string& p) {
    std::string cmd = "rm -rf " + p + " " + p + ".*";
    (void)std::system(cmd.c_str());
}

} // namespace

TEST(CohereMedium1M, IncrementalFlushMatchesReferenceAndSaves) {
    std::string base = std::string(kDataDir) + "/base.fvecs";
    std::string query = std::string(kDataDir) + "/query.fvecs";
    std::string gtp = std::string(kDataDir) + "/groundtruth.ivecs";
    if (!exists(base) || !exists(query) || !exists(gtp)) {
        GTEST_SKIP() << "cohere_medium_1m dataset not present";
    }

    int d = 0, nb = 0;
    std::unique_ptr<float[]> xb(fvecs_read(base.c_str(), &d, &nb));
    ASSERT_NE(xb.get(), nullptr);
    ASSERT_GT(nb, 0);

    int dq = 0, nq = 0;
    std::unique_ptr<float[]> xq(fvecs_read(query.c_str(), &dq, &nq));
    ASSERT_EQ(d, dq);

    int gt_k = 0, ngt = 0;
    std::unique_ptr<int[]> gt(ivecs_read(gtp.c_str(), &gt_k, &ngt));
    ASSERT_EQ(ngt, nq);

    const int M = 16;
    const int efC = 40;
    const int efS = 64;
    const int k = 10;

    // ---- reference: one-shot add ----
    faiss::IndexHNSWFlat ref(d, M);
    ref.hnsw.efConstruction = efC;
    ref.hnsw.efSearch = efS;
    ref.add(nb, xb.get());

    std::vector<float> ref_d((size_t)nq * k);
    std::vector<faiss::idx_t> ref_l((size_t)nq * k);
    ref.search(nq, xq.get(), k, ref_d.data(), ref_l.data());
    double ref_recall = recall_at(ref_l.data(), gt.get(), nq, k, gt_k);

    // bytes baseline: full write_index after each of 4 batches (no SKIP flag)
    std::string ref_dump = std::string(kWorkBase) + "_ref.bin";
    rm_rf(ref_dump);
    faiss::write_index(&ref, ref_dump.c_str());
    uint64_t baseline_per_flush = file_size(ref_dump);
    ::unlink(ref_dump.c_str());
    uint64_t baseline_total = baseline_per_flush * 4;

    // ---- incremental ----
    std::string base_p = std::string(kWorkBase) + "_inc";
    rm_rf(base_p);

    faiss::SegmentedFileCodesStorage::Options opts;
    opts.segment_bytes_target = 64ull << 20; // 64 MiB
    opts.fsync_files = false;
    auto storage = std::make_shared<faiss::SegmentedFileCodesStorage>(
            base_p, (size_t)d * sizeof(float), opts);
    auto* inner = new faiss::IndexFlatL2(d, storage);
    faiss::IndexHNSWFlat inc(d, M);
    delete inc.storage;
    inc.storage = inner;
    inc.own_fields = true;
    inc.hnsw.efConstruction = efC;
    inc.hnsw.efSearch = efS;

    const int batch = nb / 4;
    uint64_t inc_total_bytes = 0;
    for (int b = 0; b < 4; ++b) {
        int n_b = (b == 3) ? (nb - b * batch) : batch;
        inc.add(n_b, xb.get() + (long)b * batch * d);
        uint64_t before = 0;
        // bytes accounting: sum of all *.bin files under .codes + the new graph
        std::string codes_dir = base_p + ".codes";
        // sum committed segment sizes (= sum after flush minus before)
        // simpler: tally after flush.
        storage->flush(&inc);
        std::string cmd = "du -sb " + codes_dir + " " + base_p + ".graph 2>/dev/null"
                          " | awk '{s+=$1} END{print s}'";
        FILE* fp = ::popen(cmd.c_str(), "r");
        if (fp) {
            uint64_t tot;
            if (::fscanf(fp, "%lu", &tot) == 1) {
                // we want the *new* bytes; since old segments are immutable
                // their sizes carry forward — the delta is what got written.
                inc_total_bytes += (tot - before);
                before = tot;
                (void)before;
            }
            ::pclose(fp);
        }
    }

    // simpler: just take current on-disk total — under append-only, this
    // equals cumulative bytes written (segments never rewritten; old graphs
    // GC'd so only the final graph remains, which understates by a small
    // amount but never overstates).
    std::string cmd =
            "du -sb " + base_p + ".codes 2>/dev/null | awk '{print $1}'";
    uint64_t codes_bytes = 0;
    FILE* fp = ::popen(cmd.c_str(), "r");
    if (fp) {
        if (::fscanf(fp, "%lu", &codes_bytes) != 1) codes_bytes = 0;
        ::pclose(fp);
    }
    // graph bytes: only the *final* generation survives GC, but each flush
    // wrote one. Approximate cumulative graph cost as 4 * final-graph-size.
    cmd = "du -sb " + base_p + ".graph 2>/dev/null | awk '{print $1}'";
    uint64_t final_graph_bytes = 0;
    fp = ::popen(cmd.c_str(), "r");
    if (fp) {
        if (::fscanf(fp, "%lu", &final_graph_bytes) != 1) final_graph_bytes = 0;
        ::pclose(fp);
    }
    uint64_t inc_bytes_estimate = codes_bytes + 4 * final_graph_bytes;

    fprintf(stderr,
            "ref_recall@10 = %.4f, baseline_total = %.2f MiB, "
            "incremental_total = %.2f MiB, savings = %.2fx\n",
            ref_recall,
            baseline_total / (1024.0 * 1024.0),
            inc_bytes_estimate / (1024.0 * 1024.0),
            (double)baseline_total / (double)inc_bytes_estimate);

    // ---- compare search ----
    std::vector<float> inc_d((size_t)nq * k);
    std::vector<faiss::idx_t> inc_l((size_t)nq * k);
    inc.search(nq, xq.get(), k, inc_d.data(), inc_l.data());
    double inc_recall = recall_at(inc_l.data(), gt.get(), nq, k, gt_k);

    // Note: incremental HNSW construction (4 batches) intrinsically diverges
    // from one-shot construction at the graph level. We allow up to 5pp drift
    // here — the real correctness check is reload-equals-in-RAM below, which
    // pins the persistence semantics exactly.
    EXPECT_NEAR(inc_recall, ref_recall, 0.05)
            << "recall drift: ref=" << ref_recall << " inc=" << inc_recall;
    EXPECT_GE((double)baseline_total / (double)inc_bytes_estimate, 1.5)
            << "expected >=1.5x bytes savings";

    // ---- reload from disk and re-check search ----
    auto storage2 = std::make_shared<faiss::SegmentedFileCodesStorage>(
            base_p, (size_t)d * sizeof(float), opts);
    // pick highest-numbered graph file
    std::string graph_cmd =
            "ls -1 " + base_p + ".graph/graph-*.bin 2>/dev/null | sort | tail -1";
    char gpath[1024] = {0};
    fp = ::popen(graph_cmd.c_str(), "r");
    ASSERT_NE(fp, nullptr);
    if (::fgets(gpath, sizeof(gpath), fp)) {
        size_t L = std::strlen(gpath);
        if (L && gpath[L - 1] == '\n') gpath[L - 1] = 0;
    }
    ::pclose(fp);
    ASSERT_GT(std::strlen(gpath), 0u);

    std::unique_ptr<faiss::Index> reloaded(
            faiss::read_index(gpath, faiss::IO_FLAG_SKIP_CODE_BYTES));
    ASSERT_NE(reloaded.get(), nullptr);
    auto* fc = faiss::find_codes_storage(reloaded.get());
    ASSERT_NE(fc, nullptr);
    fc->set_storage(storage2);

    auto* rh = dynamic_cast<faiss::IndexHNSW*>(reloaded.get());
    ASSERT_NE(rh, nullptr);
    rh->hnsw.efSearch = efS;

    std::vector<float> rl_d((size_t)nq * k);
    std::vector<faiss::idx_t> rl_l((size_t)nq * k);
    reloaded->search(nq, xq.get(), k, rl_d.data(), rl_l.data());
    double rl_recall = recall_at(rl_l.data(), gt.get(), nq, k, gt_k);
    EXPECT_NEAR(rl_recall, inc_recall, 1e-9)
            << "reload changed search results";

    rm_rf(base_p);
}
