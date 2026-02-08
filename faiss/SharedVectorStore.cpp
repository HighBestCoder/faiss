/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/SharedVectorStore.h>

#include <cstring>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

SharedVectorStore::SharedVectorStore(size_t d, size_t code_size)
        : d(d), code_size(code_size) {}

void SharedVectorStore::reclaim_deleted_slots(
        const std::vector<uint64_t>& deleted_bitmap,
        size_t ntotal) {
    free_list.clear();
    for (size_t i = 0; i < ntotal; i++) {
        if ((deleted_bitmap[i >> 6] >> (i & 63)) & 1) {
            free_list.push_back(i);
        }
    }
}

idx_t SharedVectorStore::allocate_slot(const float* vec) {
    idx_t slot;
    if (!free_list.empty()) {
        slot = free_list.back();
        free_list.pop_back();
        memcpy(codes.data() + slot * code_size, vec, code_size);
    } else {
        slot = ntotal_store;
        ntotal_store++;
        codes.resize(ntotal_store * code_size);
        memcpy(codes.data() + slot * code_size, vec, code_size);
    }
    return slot;
}

} // namespace faiss
