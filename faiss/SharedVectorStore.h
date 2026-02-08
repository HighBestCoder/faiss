/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#endif

#include <faiss/Index.h>
#include <faiss/impl/maybe_owned_vector.h>

namespace faiss {

/// Shared vector data store. Holds the single copy of vector data that can
/// be shared across multiple IndexFlatShared instances via shared_ptr.
///
/// Slots have two states:
///   1. Allocated — contains vector data, may be used by an index
///   2. Available — in free_list, can be allocated to new vectors
///
/// Deletion is managed externally by IndexFlatShared's deleted_bitmap.
/// SharedVectorStore is a pure data container and does not track deletion.
///
/// State transitions:
///   allocated → available : reclaim_deleted_slots() after rebuild+swap
///   available → allocated : allocate_slot() when adding new vectors
///   (append)  → allocated : allocate_slot() when free_list is empty
struct SharedVectorStore : MaybeOwnedVectorOwner {
    /// Vector data (owned). size = ntotal_store * code_size
    std::vector<uint8_t> codes;

    /// Reusable slot stack (LIFO). Filled by reclaim_deleted_slots().
    /// allocate_slot() pops from here first.
    std::vector<idx_t> free_list;

    /// Total allocated slot count (includes slots in free_list)
    size_t ntotal_store = 0;

    /// Bytes per vector = sizeof(float) * d
    size_t code_size;

    /// Vector dimension
    size_t d;

    SharedVectorStore(size_t d, size_t code_size);

    /// Reclaim deleted slots from an external bitmap into free_list.
    /// Precondition: only one IndexFlatShared references this store.
    /// @param deleted_bitmap  bitmap where bit i=1 means slot i is deleted
    /// @param ntotal          number of slots covered by the bitmap
    void reclaim_deleted_slots(
            const std::vector<uint64_t>& deleted_bitmap,
            size_t ntotal);

    /// Allocate a slot and write vector data into it.
    /// Reuses from free_list first; appends to end if free_list is empty.
    /// @return the allocated store slot index
    idx_t allocate_slot(const float* vec);

    void reserve(size_t n) {
        codes.reserve(n * code_size);
    }

    /// Get raw code pointer at slot i
    const uint8_t* get_code(idx_t i) const {
        return codes.data() + i * code_size;
    }

    /// Get float vector pointer at slot i
    const float* get_vector(idx_t i) const {
        return reinterpret_cast<const float*>(get_code(i));
    }

    /// Hint the OS to back codes memory with transparent huge pages.
    /// Call after bulk allocation or compaction when the buffer is stable.
    /// No-op on non-Linux platforms or if madvise fails.
    void enable_hugepages() {
#ifdef __linux__
        if (codes.empty())
            return;
        void* ptr = codes.data();
        size_t len = codes.size();
        // Align down to page boundary (4KB)
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t aligned = addr & ~(uintptr_t(4095));
        size_t aligned_len = len + (addr - aligned);
        // MADV_HUGEPAGE hints the kernel to use THP for this range
        madvise(reinterpret_cast<void*>(aligned), aligned_len, MADV_HUGEPAGE);
#endif
    }
};

} // namespace faiss
