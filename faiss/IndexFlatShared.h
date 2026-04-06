/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/SharedVectorStore.h>

namespace faiss {

struct IndexHNSW;

struct IndexFlatShared : IndexFlatCodes {
    std::shared_ptr<SharedVectorStore> store;

    /// local_id [0, ntotal) -> store slot
    std::vector<idx_t> storage_id_map;

    /// per-index deletion bitmap, indexed by store slot
    std::vector<uint64_t> deleted_bitmap;

    /// true when storage_id_map[i] == i for all i (fresh build, no deletions)
    bool is_identity_map = true;

    IndexFlatShared() = default;

    IndexFlatShared(
            std::shared_ptr<SharedVectorStore> store,
            MetricType metric = METRIC_L2);

    IndexFlatShared(
            std::shared_ptr<SharedVectorStore> store,
            const std::vector<idx_t>& old_storage_id_map,
            const std::vector<uint64_t>& old_deleted_bitmap,
            MetricType metric = METRIC_L2);

    idx_t resolve_id(idx_t local_id) const {
        return storage_id_map[local_id];
    }

    bool is_deleted(idx_t store_slot) const {
        return (deleted_bitmap[store_slot >> 6] >> (store_slot & 63)) & 1;
    }

    void mark_deleted(idx_t local_id) {
        idx_t store_slot = storage_id_map[local_id];
        deleted_bitmap[store_slot >> 6] |= (uint64_t(1) << (store_slot & 63));
    }

    size_t count_alive() const;

    void add(idx_t n, const float* x) override;

    void reconstruct(idx_t key, float* recons) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    void permute_entries(const idx_t* perm);

    void reset() override;
};

void build_storage_id_map(
        const SharedVectorStore& store,
        const std::vector<uint64_t>& deleted_bitmap,
        const std::vector<idx_t>& old_storage_id_map,
        std::vector<idx_t>& new_storage_id_map);

IndexHNSW* build_new_index(
        std::shared_ptr<SharedVectorStore> store,
        const IndexHNSW& current_index,
        int M = 32,
        int efConstruction = 40,
        MetricType metric = METRIC_L2);

} // namespace faiss
