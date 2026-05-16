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
