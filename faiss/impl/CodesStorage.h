/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/maybe_owned_vector.h>

namespace faiss {

/// Contiguous view of codes exposed by a Form-1 (resident) CodesStorage.
/// MVP drivers always populate `data`/`nbytes` (single contiguous buffer);
/// the `segments` field is reserved for a future MmapSegmentedFileCodesStorage.
struct CodesView {
    const uint8_t* data = nullptr;
    size_t nbytes = 0;

    struct Segment {
        const uint8_t* base = nullptr;
        size_t byte_offset = 0;
        size_t byte_size = 0;
    };
    std::vector<Segment> segments;
};

/// Storage abstraction for the byte stream behind IndexFlatCodes::codes.
///
/// Inherits MaybeOwnedVectorOwner so that IndexFlatCodes::codes can be a
/// MaybeOwnedVector view into the driver's buffer with proper lifetime.
///
/// Two storage shapes are explicitly recognised:
///  - Form 1 (resident): try_view() returns a contiguous buffer; search uses
///    codes.data()+i*code_size with zero virtual dispatch.
///  - Form 2 (remote): try_view() returns nullopt; consumers must call
///    gather(). MVP ships only Form-1 drivers, but the interface admits Form-2.
class CodesStorage : public MaybeOwnedVectorOwner {
   public:
    ~CodesStorage() override = default;

    // --- meta ---
    virtual size_t code_size() const = 0;
    virtual size_t num_codes() const = 0;

    // --- write path (all drivers) ---
    virtual void append(size_t n, const uint8_t* src) = 0;
    virtual void reset() = 0;
    /// `perm` of length num_codes(); new[i] = old[perm[i]].
    virtual void permute(const idx_t* perm) = 0;

    // --- read path: Form 1 ---
    /// Form-1 drivers MUST return a populated CodesView.
    /// Form-2 drivers MUST return std::nullopt.
    virtual std::optional<CodesView> try_view() const = 0;

    // --- read path: Form 2 ---
    /// Default impl uses try_view() + memcpy; Form-2 drivers must override.
    virtual void gather(size_t n, const idx_t* ids, uint8_t* dst) const;

    // --- persistence (optional) ---
    /// Incrementally commit current state to backing store.
    /// `idx` is the owning IndexFlatCodes-bearing index; SegmentedFile uses
    /// it to write the shell (graph/codebook). Default throws.
    virtual void flush(const Index* idx);

    // --- capabilities ---
    virtual bool has_resident_view() const = 0;
    virtual bool supports_flush() const { return false; }
};

/// In-memory codes buffer. Default storage for every IndexFlatCodes derived
/// class; bit-identical behaviour to pre-storage FAISS.
class InMemoryCodesStorage : public CodesStorage {
   public:
    explicit InMemoryCodesStorage(size_t code_size);
    /// Take ownership of an already-populated byte buffer (used to migrate
    /// the post-read_index state into a storage wrapper without copying).
    InMemoryCodesStorage(size_t code_size, std::vector<uint8_t>&& bytes);

    size_t code_size() const override { return code_size_; }
    size_t num_codes() const override { return buffer_.size() / code_size_; }

    void append(size_t n, const uint8_t* src) override;
    void reset() override;
    void permute(const idx_t* perm) override;

    std::optional<CodesView> try_view() const override;
    bool has_resident_view() const override { return true; }

    /// Exposed so IndexFlatCodes can rebind its `codes` view after mutation.
    uint8_t* mutable_data() { return buffer_.data(); }
    const std::vector<uint8_t>& buffer() const { return buffer_; }

   private:
    size_t code_size_;
    std::vector<uint8_t> buffer_;
};

} // namespace faiss
