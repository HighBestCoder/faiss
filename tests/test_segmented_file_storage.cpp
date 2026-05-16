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
