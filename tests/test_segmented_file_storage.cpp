/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <faiss/impl/SegmentedFileCodesStorage.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

namespace {
constexpr const char* kBase = "/tmp/appendable_test";

std::string fresh_basepath(const char* name) {
    ::mkdir(kBase, 0755);
    std::string p = std::string(kBase) + "/" + name;
    std::string cmd = "rm -rf " + p + " " + p + ".*";
    (void)std::system(cmd.c_str());
    return p;
}

std::vector<uint8_t> rand_bytes(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    std::mt19937 rng(seed);
    for (auto& b : v) {
        b = (uint8_t)(rng() & 0xFF);
    }
    return v;
}
} // namespace

TEST(SegmentedFileStorage, EmptyOnFreshBasepath) {
    auto p = fresh_basepath("empty");
    faiss::SegmentedFileCodesStorage s(p, 16);
    EXPECT_EQ(s.num_codes(), 0u);
    EXPECT_EQ(s.committed_bytes(), 0u);
    EXPECT_EQ(s.num_committed_segments(), 0u);
}

TEST(SegmentedFileStorage, AppendInRamWithoutFlush) {
    auto p = fresh_basepath("ram");
    faiss::SegmentedFileCodesStorage s(p, 8);
    auto a = rand_bytes(80, 1);
    s.append(10, a.data());
    EXPECT_EQ(s.num_codes(), 10u);
    EXPECT_EQ(s.committed_bytes(), 0u);
    auto v = s.try_view();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->nbytes, 80u);
    EXPECT_EQ(0, std::memcmp(v->data, a.data(), 80));
}

TEST(SegmentedFileStorage, FlushFullThenReload) {
    auto p = fresh_basepath("full");
    faiss::SegmentedFileCodesStorage::Options opts;
    opts.segment_bytes_target = 256;
    opts.fsync_files = false;

    auto s = std::make_shared<faiss::SegmentedFileCodesStorage>(p, 16, opts);
    faiss::IndexFlatL2 idx(4, s);

    std::vector<float> data(40 * 4);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = float(i);
    }
    idx.add(40, data.data());

    s->flush(&idx);

    EXPECT_GT(s->num_committed_segments(), 1u);
    EXPECT_EQ(s->committed_bytes(), 640u);

    auto s2 = std::make_shared<faiss::SegmentedFileCodesStorage>(p, 16, opts);
    EXPECT_EQ(s2->num_codes(), 40u);
    auto v = s2->try_view();
    EXPECT_EQ(0, std::memcmp(v->data, s->try_view()->data, 640));
}

namespace {
uint64_t mtime_ns(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        return 0;
    }
#ifdef __linux__
    return (uint64_t)st.st_mtim.tv_sec * 1000000000ULL + st.st_mtim.tv_nsec;
#else
    return (uint64_t)st.st_mtime * 1000000000ULL;
#endif
}
} // namespace

TEST(SegmentedFileStorage, ImmutableSegmentsAcrossFlushes) {
    auto p = fresh_basepath("immut");
    faiss::SegmentedFileCodesStorage::Options opts;
    opts.segment_bytes_target = 256;
    opts.fsync_files = false;

    auto s = std::make_shared<faiss::SegmentedFileCodesStorage>(p, 16, opts);
    faiss::IndexFlatL2 idx(4, s);

    std::vector<float> a(20 * 4);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = float(i);
    }
    idx.add(20, a.data());
    s->flush(&idx);

    std::string seg0 = p + ".codes/seg-00000000.bin";
    std::string seg1 = p + ".codes/seg-00000001.bin";
    uint64_t mt0 = mtime_ns(seg0);
    uint64_t mt1 = mtime_ns(seg1);
    ASSERT_GT(mt0, 0u);
    ASSERT_GT(mt1, 0u);

    ::usleep(20000);

    std::vector<float> b(30 * 4);
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = float(1000 + i);
    }
    idx.add(30, b.data());
    s->flush(&idx);

    EXPECT_EQ(mt0, mtime_ns(seg0));
    EXPECT_EQ(mt1, mtime_ns(seg1));
    EXPECT_EQ(s->num_committed_segments(), 4u);
}

#include <faiss/IndexHNSW.h>
#include <faiss/impl/CodesStorage.h>

namespace {
std::vector<float> rand_floats(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : v) x = dist(rng);
    return v;
}

void compare_search(faiss::Index& a, faiss::Index& b, size_t d) {
    const size_t nq = 32, k = 10;
    auto q = rand_floats(nq * d, 99);
    std::vector<float> da(nq * k), db(nq * k);
    std::vector<faiss::idx_t> la(nq * k), lb(nq * k);
    a.search(nq, q.data(), k, da.data(), la.data());
    b.search(nq, q.data(), k, db.data(), lb.data());
    for (size_t i = 0; i < nq * k; ++i) {
        EXPECT_EQ(la[i], lb[i]) << "label mismatch at " << i;
    }
}
} // namespace

TEST(SegmentedFileStorage, HnswFlatTwoBatchFlushReload) {
    auto p = fresh_basepath("hnsw_flat");
    const size_t d = 32;
    const size_t batch = 4000;

    faiss::SegmentedFileCodesStorage::Options opts;
    opts.segment_bytes_target = 256 * 1024;
    opts.fsync_files = false;

    auto storage = std::make_shared<faiss::SegmentedFileCodesStorage>(
            p, d * sizeof(float), opts);
    auto* inner = new faiss::IndexFlatL2(d, storage);
    faiss::IndexHNSWFlat hnsw(d, 16);
    delete hnsw.storage;
    hnsw.storage = inner;
    hnsw.own_fields = true;
    hnsw.hnsw.efConstruction = 40;

    auto first = rand_floats(batch * d, 1);
    auto second = rand_floats(batch * d, 2);

    hnsw.add(batch, first.data());
    storage->flush(&hnsw);

    std::string seg0 = p + ".codes/seg-00000000.bin";
    uint64_t mt_before = mtime_ns(seg0);
    ASSERT_GT(mt_before, 0u);
    ::usleep(2000);

    hnsw.add(batch, second.data());
    storage->flush(&hnsw);

    EXPECT_EQ(mt_before, mtime_ns(seg0))
            << "seg-00000000.bin was rewritten on incremental flush";

    auto storage2 = std::make_shared<faiss::SegmentedFileCodesStorage>(
            p, d * sizeof(float), opts);
    std::unique_ptr<faiss::Index> reloaded(faiss::read_index(
            (p + ".graph/graph-00000002.bin").c_str(),
            faiss::IO_FLAG_SKIP_CODE_BYTES));
    ASSERT_NE(reloaded.get(), nullptr);
    auto* fc = faiss::find_codes_storage(reloaded.get());
    ASSERT_NE(fc, nullptr);
    fc->set_storage(storage2);

    EXPECT_EQ(reloaded->ntotal, hnsw.ntotal);
    compare_search(hnsw, *reloaded, d);
}

#include <faiss/IndexPQ.h>

TEST(SegmentedFileStorage, HnswPqTwoBatchFlushReload) {
    auto p = fresh_basepath("hnsw_pq");
    const size_t d = 32;
    const size_t batch = 4000;

    faiss::SegmentedFileCodesStorage::Options opts;
    opts.segment_bytes_target = 32 * 1024;
    opts.fsync_files = false;

    auto storage = std::make_shared<faiss::SegmentedFileCodesStorage>(
            p, 8, opts);
    auto* inner = new faiss::IndexPQ(d, 8, 8);
    inner->set_storage(storage);
    faiss::IndexHNSWPQ hnsw(d, 8, 16);
    delete hnsw.storage;
    hnsw.storage = inner;
    hnsw.own_fields = true;

    auto first = rand_floats(batch * d, 1);
    auto second = rand_floats(batch * d, 2);

    inner->train(batch, first.data());
    inner->pq.compute_sdc_table();
    hnsw.is_trained = true;
    hnsw.add(batch, first.data());
    storage->flush(&hnsw);

    std::string seg0 = p + ".codes/seg-00000000.bin";
    uint64_t mt_before = mtime_ns(seg0);
    ASSERT_GT(mt_before, 0u);
    ::usleep(2000);

    hnsw.add(batch, second.data());
    storage->flush(&hnsw);

    EXPECT_EQ(mt_before, mtime_ns(seg0));

    auto storage2 = std::make_shared<faiss::SegmentedFileCodesStorage>(
            p, 8, opts);
    std::unique_ptr<faiss::Index> reloaded(faiss::read_index(
            (p + ".graph/graph-00000002.bin").c_str(),
            faiss::IO_FLAG_SKIP_CODE_BYTES));
    ASSERT_NE(reloaded.get(), nullptr);
    auto* fc = faiss::find_codes_storage(reloaded.get());
    ASSERT_NE(fc, nullptr);
    fc->set_storage(storage2);

    EXPECT_EQ(reloaded->ntotal, hnsw.ntotal);
    compare_search(hnsw, *reloaded, d);
}
