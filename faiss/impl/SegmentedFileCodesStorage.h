/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <faiss/impl/CodesStorage.h>
#include <faiss/impl/MetadataStore.h>

namespace faiss {

/// Crash-injection hook (testing only):
///   Setting environment variable FAISS_APPENDABLE_KILL_AFTER to one of:
///     seg_write_partial   (after any segment file is written, before next)
///     seg_rename          (after a segment .tmp is renamed to .bin)
///     graph_write         (after graph .tmp is written, before rename)
///     graph_rename        (after graph .tmp is renamed, before meta commit)
///     meta_commit         (after meta is committed, before GC)
///     gc                  (after GC runs)
///   causes flush() to ::_exit(137) at that phase. The on-disk state at
///   each kill point must remain readable by a fresh constructor — see
///   tests/test_segmented_file_storage_crash.cpp.
class SegmentedFileCodesStorage : public CodesStorage {
   public:
    struct Options {
        size_t segment_bytes_target = 256ull << 20; // 256 MiB soft cap
        bool fsync_files = true;
    };

    /// On construction, if `basepath` already has committed metadata,
    /// hydrate the in-RAM buffer by reading every segment file listed
    /// in segment_sizes. Otherwise start empty.
    SegmentedFileCodesStorage(
            std::string basepath,
            size_t code_size,
            Options opts);
    SegmentedFileCodesStorage(std::string basepath, size_t code_size)
            : SegmentedFileCodesStorage(
                      std::move(basepath),
                      code_size,
                      Options{}) {}

    /// Inject an alternative metadata backend (for tests).
    SegmentedFileCodesStorage(
            std::string basepath,
            size_t code_size,
            std::unique_ptr<MetadataStore> meta_store,
            Options opts);

    ~SegmentedFileCodesStorage() override;

    size_t code_size() const override {
        return code_size_;
    }
    size_t num_codes() const override {
        return buffer_.size() / code_size_;
    }

    void append(size_t n, const uint8_t* src) override;
    void reset() override;
    void permute(const idx_t* perm) override;

    std::optional<CodesView> try_view() const override;
    bool has_resident_view() const override {
        return true;
    }
    bool supports_flush() const override {
        return true;
    }

    void flush(const Index* idx) override;

    // --- introspection (mainly for tests) ---
    const std::string& basepath() const {
        return basepath_;
    }
    size_t num_committed_segments() const {
        return last_committed_.segment_sizes.size();
    }
    uint64_t committed_bytes() const;

   private:
    void hydrate_();
    void acquire_lock_();
    void release_lock_();
    void maybe_kill_(const char* phase) const;

    std::string basepath_;
    size_t code_size_;
    Options opts_;
    std::unique_ptr<MetadataStore> meta_store_;
    AppendableMetadata last_committed_;

    // Resident contiguous buffer. Form-1 driver: search reads through this.
    std::vector<uint8_t> buffer_;

    int lock_fd_ = -1;
};

} // namespace faiss
