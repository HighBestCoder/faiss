/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace faiss {

struct AppendableMetadata {
    uint32_t format_version = 1;
    uint32_t inner_fourcc = 0;
    uint64_t ntotal = 0;
    uint64_t code_size = 0;
    uint64_t segment_bytes_target = 0;
    std::vector<uint64_t> segment_sizes; // sum == ntotal * code_size
    uint64_t graph_generation = 0;
    uint64_t graph_file_size = 0;
};

class MetadataStore {
   public:
    virtual ~MetadataStore() = default;
    /// Returns false if no commit point has ever been stored.
    virtual bool load(AppendableMetadata& out) = 0;
    /// Atomic commit of new metadata. Implementations must use rename(2)
    /// or equivalent so that a crash leaves either prev or new fully
    /// readable, never a torn record.
    virtual void commit(const AppendableMetadata& m) = 0;
};

class InMemoryMetadataStore : public MetadataStore {
   public:
    bool load(AppendableMetadata& out) override;
    void commit(const AppendableMetadata& m) override;

   private:
    bool has_ = false;
    AppendableMetadata state_;
};

class JsonFileMetadataStore : public MetadataStore {
   public:
    /// `path` is the destination file; the implementation also uses
    /// `<path>.tmp` and fsyncs the parent directory after rename.
    explicit JsonFileMetadataStore(
            std::string path,
            bool fsync_on_commit = true);
    bool load(AppendableMetadata& out) override;
    void commit(const AppendableMetadata& m) override;

   private:
    std::string path_;
    bool fsync_;
};

} // namespace faiss
