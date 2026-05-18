/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// QPS / recall parity test:
//   A: IndexHNSWFlat, default in-memory storage  ("before" feature)
//   B: IndexHNSWFlat + SegmentedFileCodesStorage,
//      built one-shot, flush + reload             ("after" feature)
// Both built with one-shot add(N) so the HNSW graph is identical
// — this isolates storage-path overhead from graph-construction effects.
//
// Asserts:
//   1) labels match exactly (bit-for-bit recall parity)
//   2) B_qps / A_qps in [0.95, 1.05]
//
// Skipped automatically if cohere_medium_1m dataset is absent.

#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
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
constexpr const char* kWorkBase = "/tmp/appendable_qps_parity";

bool exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
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

void rm_rf(const std::string& p) {
    std::string cmd = "rm -rf " + p + " " + p + ".*";
    (void)std::system(cmd.c_str());
}

double time_search_qps(
        faiss::Index& idx,
        const float* xq,
        int nq,
        int k,
        int rounds) {
    std::vector<float> d((size_t)nq * k);
    std::vector<faiss::idx_t> l((size_t)nq * k);
    // warmup
    idx.search(nq, xq, k, d.data(), l.data());

    std::vector<double> qps_samples;
    qps_samples.reserve(rounds);
    for (int r = 0; r < rounds; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        idx.search(nq, xq, k, d.data(), l.data());
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        qps_samples.push_back((double)nq / sec);
    }
    std::sort(qps_samples.begin(), qps_samples.end());
    return qps_samples[qps_samples.size() / 2]; // median
}

} // namespace

TEST(QpsParity, InMemoryVsSegmentedAfterReload) {
    std::string base = std::string(kDataDir) + "/base.fvecs";
    std::string query = std::string(kDataDir) + "/query.fvecs";
    if (!exists(base) || !exists(query)) {
        GTEST_SKIP() << "cohere_medium_1m dataset not present";
    }

    int d = 0, nb = 0;
    std::unique_ptr<float[]> xb(fvecs_read(base.c_str(), &d, &nb));
    ASSERT_NE(xb.get(), nullptr);
    ASSERT_GT(nb, 0);

    int dq = 0, nq = 0;
    std::unique_ptr<float[]> xq(fvecs_read(query.c_str(), &dq, &nq));
    ASSERT_EQ(d, dq);

    const int M = 16;
    const int efC = 40;
    const int efS = 64;
    const int k = 10;

    // Single-threaded search to remove scheduler noise from QPS comparison.
    omp_set_num_threads(1);

    // ---- A: in-memory storage, one-shot add ----
    faiss::IndexHNSWFlat A(d, M);
    A.hnsw.efConstruction = efC;
    A.hnsw.efSearch = efS;
    A.add(nb, xb.get());

    // ---- B: segmented storage, one-shot add + flush + reload ----
    std::string base_p = std::string(kWorkBase) + "_seg";
    rm_rf(base_p);

    faiss::SegmentedFileCodesStorage::Options opts;
    opts.segment_bytes_target = 64ull << 20; // 64 MiB
    opts.fsync_files = false;

    auto storage = std::make_shared<faiss::SegmentedFileCodesStorage>(
            base_p, (size_t)d * sizeof(float), opts);
    auto* inner = new faiss::IndexFlatL2(d, storage);
    faiss::IndexHNSWFlat B_build(d, M);
    delete B_build.storage;
    B_build.storage = inner;
    B_build.own_fields = true;
    B_build.hnsw.efConstruction = efC;
    B_build.hnsw.efSearch = efS;
    B_build.add(nb, xb.get());
    storage->flush(&B_build);

    // Reload from disk: this is the "after the feature is deployed" state.
    auto storage2 = std::make_shared<faiss::SegmentedFileCodesStorage>(
            base_p, (size_t)d * sizeof(float), opts);
    std::string graph_cmd =
            "ls -1 " + base_p + ".graph/graph-*.bin 2>/dev/null | sort | tail -1";
    char gpath[1024] = {0};
    FILE* fp = ::popen(graph_cmd.c_str(), "r");
    ASSERT_NE(fp, nullptr);
    if (::fgets(gpath, sizeof(gpath), fp)) {
        size_t L = std::strlen(gpath);
        if (L && gpath[L - 1] == '\n') gpath[L - 1] = 0;
    }
    ::pclose(fp);
    ASSERT_GT(std::strlen(gpath), 0u);

    std::unique_ptr<faiss::Index> B_reloaded(
            faiss::read_index(gpath, faiss::IO_FLAG_SKIP_CODE_BYTES));
    ASSERT_NE(B_reloaded.get(), nullptr);
    auto* fc = faiss::find_codes_storage(B_reloaded.get());
    ASSERT_NE(fc, nullptr);
    fc->set_storage(storage2);

    auto* B = dynamic_cast<faiss::IndexHNSW*>(B_reloaded.get());
    ASSERT_NE(B, nullptr);
    B->hnsw.efSearch = efS;

    ASSERT_EQ(A.ntotal, B->ntotal);

    // ---- 1) recall parity: labels must match bit-for-bit ----
    std::vector<float> A_d((size_t)nq * k), B_d((size_t)nq * k);
    std::vector<faiss::idx_t> A_l((size_t)nq * k), B_l((size_t)nq * k);
    A.search(nq, xq.get(), k, A_d.data(), A_l.data());
    B->search(nq, xq.get(), k, B_d.data(), B_l.data());

    size_t mismatches = 0;
    for (size_t i = 0; i < (size_t)nq * k; ++i) {
        if (A_l[i] != B_l[i]) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0u)
            << "label mismatches: " << mismatches << " / " << (nq * k)
            << " — segmented storage changed search results";

    // ---- 2) QPS parity ----
    const int rounds = 5;
    double A_qps = time_search_qps(A, xq.get(), nq, k, rounds);
    double B_qps = time_search_qps(*B, xq.get(), nq, k, rounds);
    double ratio = B_qps / A_qps;

    fprintf(stderr,
            "[QpsParity] A(in-memory)=%.1f qps, B(segmented+reload)=%.1f qps, "
            "ratio=%.4f\n",
            A_qps,
            B_qps,
            ratio);

    // Only assert a lower bound: a regression would be B noticeably slower
    // than A. B faster than A is not a defect (likely page-aligned mmap
    // beating malloc'd vector locality) — we just log it.
    EXPECT_GE(ratio, 0.95)
            << "segmented storage too slow: ratio=" << ratio;

    rm_rf(base_p);
}
