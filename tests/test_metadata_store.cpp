/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/impl/MetadataStore.h>

TEST(InMemoryMetadataStore, LoadCommitRoundtrip) {
    faiss::InMemoryMetadataStore store;
    faiss::AppendableMetadata m;
    EXPECT_FALSE(store.load(m));

    m.inner_fourcc = 0x46784932;
    m.ntotal = 1000;
    m.code_size = 32;
    m.segment_bytes_target = 1 << 20;
    m.segment_sizes = {16384, 16384};
    m.graph_generation = 7;
    m.graph_file_size = 4242;
    store.commit(m);

    faiss::AppendableMetadata loaded;
    ASSERT_TRUE(store.load(loaded));
    EXPECT_EQ(loaded.inner_fourcc, m.inner_fourcc);
    EXPECT_EQ(loaded.ntotal, m.ntotal);
    EXPECT_EQ(loaded.segment_sizes, m.segment_sizes);
    EXPECT_EQ(loaded.graph_generation, 7u);
}
