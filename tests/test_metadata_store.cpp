/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include <faiss/impl/FaissException.h>
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

namespace {
std::string tmp_path() {
    ::mkdir("/tmp/appendable_test", 0755);
    return "/tmp/appendable_test/meta_roundtrip.json";
}
} // namespace

TEST(JsonFileMetadataStore, RoundtripOnDisk) {
    auto p = tmp_path();
    ::unlink(p.c_str());

    {
        faiss::JsonFileMetadataStore w(p, /*fsync_on_commit*/ false);
        faiss::AppendableMetadata m;
        m.inner_fourcc = 0x46784932;
        m.ntotal = 100;
        m.code_size = 12;
        m.segment_bytes_target = 1024;
        m.segment_sizes = {1024, 176};
        m.graph_generation = 3;
        m.graph_file_size = 7777;
        w.commit(m);
    }

    faiss::JsonFileMetadataStore r(p, false);
    faiss::AppendableMetadata loaded;
    ASSERT_TRUE(r.load(loaded));
    EXPECT_EQ(loaded.inner_fourcc, 0x46784932u);
    EXPECT_EQ(loaded.ntotal, 100u);
    EXPECT_EQ(loaded.code_size, 12u);
    EXPECT_EQ(loaded.segment_sizes.size(), 2u);
    EXPECT_EQ(loaded.segment_sizes[0], 1024u);
    EXPECT_EQ(loaded.segment_sizes[1], 176u);
    EXPECT_EQ(loaded.graph_generation, 3u);
    EXPECT_EQ(loaded.graph_file_size, 7777u);
}

TEST(JsonFileMetadataStore, LoadReturnsFalseIfMissing) {
    faiss::JsonFileMetadataStore r(
            "/tmp/appendable_test/does_not_exist.json", false);
    faiss::AppendableMetadata loaded;
    EXPECT_FALSE(r.load(loaded));
}

TEST(JsonFileMetadataStore, CorruptFileThrows) {
    auto p = tmp_path() + ".corrupt";
    FILE* fp = ::fopen(p.c_str(), "w");
    std::fputs("{not valid json", fp);
    std::fclose(fp);
    faiss::JsonFileMetadataStore r(p, false);
    faiss::AppendableMetadata loaded;
    EXPECT_THROW(r.load(loaded), faiss::FaissException);
}
