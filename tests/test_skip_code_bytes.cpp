/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace {
std::vector<float> rand_data(size_t n, size_t d, uint32_t seed) {
    std::vector<float> v(n * d);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}
} // namespace

TEST(SkipCodeBytes, FlatL2WriteOmitsCodes) {
    constexpr size_t d = 8, n = 64;
    faiss::IndexFlatL2 idx(d);
    auto data = rand_data(n, d, 1);
    idx.add(n, data.data());

    faiss::VectorIOWriter wr_full;
    faiss::write_index(&idx, &wr_full, 0);
    faiss::VectorIOWriter wr_slim;
    faiss::write_index(&idx, &wr_slim, faiss::IO_FLAG_SKIP_CODE_BYTES);

    EXPECT_GT(wr_full.data.size(), wr_slim.data.size());
    EXPECT_EQ(
            wr_full.data.size() - wr_slim.data.size(),
            n * d * sizeof(float));
}

TEST(SkipCodeBytes, PQWriteOmitsCodes) {
    constexpr size_t d = 8, n = 256;
    faiss::IndexPQ idx(d, 4, 4);
    auto data = rand_data(n, d, 2);
    idx.train(n, data.data());
    idx.add(n, data.data());

    faiss::VectorIOWriter wr_full;
    faiss::write_index(&idx, &wr_full, 0);
    faiss::VectorIOWriter wr_slim;
    faiss::write_index(&idx, &wr_slim, faiss::IO_FLAG_SKIP_CODE_BYTES);

    EXPECT_GT(wr_full.data.size(), wr_slim.data.size());
    EXPECT_EQ(
            wr_full.data.size() - wr_slim.data.size(),
            idx.codes.size());
}

TEST(SkipCodeBytes, FlatL2RoundtripSlim) {
    constexpr size_t d = 8, n = 64;
    faiss::IndexFlatL2 idx(d);
    auto data = rand_data(n, d, 3);
    idx.add(n, data.data());

    faiss::VectorIOWriter wr;
    faiss::write_index(&idx, &wr, faiss::IO_FLAG_SKIP_CODE_BYTES);

    faiss::VectorIOReader rd;
    rd.data = wr.data;
    std::unique_ptr<faiss::Index> loaded(
            faiss::read_index(&rd, faiss::IO_FLAG_SKIP_CODE_BYTES));
    auto* lf = dynamic_cast<faiss::IndexFlat*>(loaded.get());
    ASSERT_NE(lf, nullptr);
    EXPECT_EQ(lf->ntotal, idx.ntotal);
    EXPECT_EQ(lf->codes.size(), 0u);
}

TEST(SkipCodeBytes, PQRoundtripSlim) {
    constexpr size_t d = 8, n = 256;
    faiss::IndexPQ idx(d, 4, 4);
    auto data = rand_data(n, d, 4);
    idx.train(n, data.data());
    idx.add(n, data.data());

    faiss::VectorIOWriter wr;
    faiss::write_index(&idx, &wr, faiss::IO_FLAG_SKIP_CODE_BYTES);
    faiss::VectorIOReader rd;
    rd.data = wr.data;
    std::unique_ptr<faiss::Index> loaded(
            faiss::read_index(&rd, faiss::IO_FLAG_SKIP_CODE_BYTES));
    auto* lp = dynamic_cast<faiss::IndexPQ*>(loaded.get());
    ASSERT_NE(lp, nullptr);
    EXPECT_EQ(lp->ntotal, idx.ntotal);
    EXPECT_EQ(lp->codes.size(), 0u);
    EXPECT_EQ(lp->pq.centroids.size(), idx.pq.centroids.size());
}
