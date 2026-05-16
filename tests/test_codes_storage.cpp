/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/impl/CodesStorage.h>
#include <faiss/impl/FaissException.h>

TEST(InMemoryCodesStorage, AppendThenView) {
    faiss::InMemoryCodesStorage s(4);
    std::vector<uint8_t> a = {1, 2, 3, 4, 5, 6, 7, 8};   // 2 codes
    std::vector<uint8_t> b = {9, 10, 11, 12};            // 1 code
    s.append(2, a.data());
    s.append(1, b.data());

    EXPECT_EQ(s.num_codes(), 3u);
    EXPECT_EQ(s.code_size(), 4u);

    auto v = s.try_view();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->nbytes, 12u);
    std::vector<uint8_t> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    EXPECT_EQ(0, std::memcmp(v->data, expected.data(), 12));
}

TEST(InMemoryCodesStorage, ResetClears) {
    faiss::InMemoryCodesStorage s(2);
    std::vector<uint8_t> a = {1, 2, 3, 4};
    s.append(2, a.data());
    s.reset();
    EXPECT_EQ(s.num_codes(), 0u);
    EXPECT_EQ(s.try_view()->nbytes, 0u);
}

TEST(InMemoryCodesStorage, Permute) {
    faiss::InMemoryCodesStorage s(2);
    std::vector<uint8_t> a = {0xAA, 0xAA, 0xBB, 0xBB, 0xCC, 0xCC};
    s.append(3, a.data());
    faiss::idx_t perm[3] = {2, 0, 1};   // new[i] = old[perm[i]]
    s.permute(perm);
    auto v = s.try_view();
    std::vector<uint8_t> expected = {0xCC, 0xCC, 0xAA, 0xAA, 0xBB, 0xBB};
    EXPECT_EQ(0, std::memcmp(v->data, expected.data(), 6));
}

TEST(InMemoryCodesStorage, AdoptBuffer) {
    std::vector<uint8_t> bytes = {1, 2, 3, 4, 5, 6};
    faiss::InMemoryCodesStorage s(3, std::move(bytes));
    EXPECT_EQ(s.num_codes(), 2u);
    auto v = s.try_view();
    EXPECT_EQ(v->nbytes, 6u);
    EXPECT_EQ(v->data[0], 1u);
}

TEST(CodesStorageBase, FlushThrowsByDefault) {
    faiss::InMemoryCodesStorage s(4);
    EXPECT_FALSE(s.supports_flush());
    EXPECT_THROW(s.flush(nullptr), faiss::FaissException);
}

TEST(IndexFlatCodesStorage, DefaultStorageIsInMemory) {
    faiss::IndexFlatL2 idx(8);
    ASSERT_NE(idx.storage, nullptr);
    EXPECT_TRUE(idx.storage->has_resident_view());
    EXPECT_EQ(idx.storage->code_size(), 8u * sizeof(float));
}

TEST(IndexFlatCodesStorage, AddRoutesThroughStorage) {
    faiss::IndexFlatL2 idx(4);
    std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 8};
    idx.add(2, v.data());
    EXPECT_EQ(idx.ntotal, 2);
    EXPECT_EQ(idx.storage->num_codes(), 2u);
    auto sv = idx.storage->try_view();
    EXPECT_EQ(idx.codes.data(), sv->data);
    EXPECT_EQ(idx.codes.size(), sv->nbytes);
}

TEST(IndexFlatCodesStorage, SetStorageReplacesAndRebinds) {
    faiss::IndexFlatL2 idx(4);
    std::vector<float> v = {1, 2, 3, 4};
    idx.add(1, v.data());

    auto fresh = std::make_shared<faiss::InMemoryCodesStorage>(
            4 * sizeof(float));
    std::vector<float> v2 = {9, 8, 7, 6, 5, 4, 3, 2};
    fresh->append(2, reinterpret_cast<const uint8_t*>(v2.data()));
    idx.set_storage(fresh);
    EXPECT_EQ(idx.storage.get(), fresh.get());
    EXPECT_EQ(idx.codes.size(), 2u * 4u * sizeof(float));
    EXPECT_EQ(
            idx.codes.data(),
            reinterpret_cast<const uint8_t*>(fresh->try_view()->data));
}
