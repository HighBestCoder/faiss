/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlatShared.h>

#include <cstring>
#include <numeric>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/prefetch.h>

namespace faiss {

/*************************************************************
 * IndirectFlatL2Dis — L2 distance computer with indirection
 * through storage_id_map. Mirrors FlatL2Dis from IndexFlat.cpp
 * but resolves local IDs to store slots before accessing vectors.
 *************************************************************/

namespace {

struct IndirectFlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    const float* b;
    const idx_t* id_map;
    bool identity_map;
    size_t ndis = 0;
    size_t npartial_dot_products = 0;

    explicit IndirectFlatL2Dis(
            const IndexFlatShared& storage,
            const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.store->codes.data(),
                      storage.store->code_size,
                      q),
              d(storage.d),
              b(reinterpret_cast<const float*>(storage.store->codes.data())),
              id_map(storage.storage_id_map.data()),
              identity_map(storage.is_identity_map) {}

    idx_t resolve(idx_t i) const {
        if (identity_map) {
            return i;
        }
        return id_map[i];
    }

    void set_query(const float* x) override {
        q = x;
    }

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr(q, (const float*)code, d);
    }

    float operator()(idx_t i) override {
        ndis++;
        return fvec_L2sqr(q, b + resolve(i) * d, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + resolve(j) * d, b + resolve(i) * d, d);
    }

    // 16 floats = 64 bytes = 1 cache line
    static constexpr size_t CACHE_LINE_FLOATS = 16;

    void prefetch(idx_t i) override {
        if (!identity_map) {
            prefetch_L2(&id_map[i]);
        }
        idx_t resolved = resolve(i);
        const float* vec = b + resolved * d;
        prefetch_L2(vec);
        if (d > CACHE_LINE_FLOATS) {
            prefetch_L2(vec + CACHE_LINE_FLOATS);
        }
        if (d > 2 * CACHE_LINE_FLOATS) {
            prefetch_L2(vec + 2 * CACHE_LINE_FLOATS);
        }
        if (d > 3 * CACHE_LINE_FLOATS) {
            prefetch_L2(vec + 3 * CACHE_LINE_FLOATS);
        }
    }

    void prefetch_batch_4(
            idx_t i0,
            idx_t i1,
            idx_t i2,
            idx_t i3,
            int level) override {
        const float* p0 = b + resolve(i0) * d;
        const float* p1 = b + resolve(i1) * d;
        const float* p2 = b + resolve(i2) * d;
        const float* p3 = b + resolve(i3) * d;
        if (level == 1) {
            prefetch_L1(p0);
            prefetch_L1(p1);
            prefetch_L1(p2);
            prefetch_L1(p3);
            // Prefetch additional cache lines for large vectors
            if (d > CACHE_LINE_FLOATS) {
                prefetch_L1(p0 + CACHE_LINE_FLOATS);
                prefetch_L1(p1 + CACHE_LINE_FLOATS);
                prefetch_L1(p2 + CACHE_LINE_FLOATS);
                prefetch_L1(p3 + CACHE_LINE_FLOATS);
            }
        } else if (level == 3) {
            prefetch_L3(p0);
            prefetch_L3(p1);
            prefetch_L3(p2);
            prefetch_L3(p3);
        } else {
            prefetch_L2(p0);
            prefetch_L2(p1);
            prefetch_L2(p2);
            prefetch_L2(p3);
            if (d > CACHE_LINE_FLOATS) {
                prefetch_L2(p0 + CACHE_LINE_FLOATS);
                prefetch_L2(p1 + CACHE_LINE_FLOATS);
                prefetch_L2(p2 + CACHE_LINE_FLOATS);
                prefetch_L2(p3 + CACHE_LINE_FLOATS);
            }
        }
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // DPDK-style: prefetch L2 early, then L1 for deeper cache lines
        const float* __restrict y0 = b + resolve(idx0) * d;
        prefetch_L2(y0);
        const float* __restrict y1 = b + resolve(idx1) * d;
        prefetch_L2(y1);
        const float* __restrict y2 = b + resolve(idx2) * d;
        prefetch_L2(y2);
        const float* __restrict y3 = b + resolve(idx3) * d;
        prefetch_L2(y3);

        if (d > CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + CACHE_LINE_FLOATS);
            prefetch_L1(y1 + CACHE_LINE_FLOATS);
            prefetch_L1(y2 + CACHE_LINE_FLOATS);
            prefetch_L1(y3 + CACHE_LINE_FLOATS);
        }
        if (d > 2 * CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y1 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y2 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y3 + 2 * CACHE_LINE_FLOATS);
        }
        if (d > 3 * CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y1 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y2 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y3 + 3 * CACHE_LINE_FLOATS);
        }

        // Phase 3: Compute distances (data should be cache-hot)
        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        fvec_L2sqr_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }

    void distances_batch_8(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            const idx_t idx4,
            const idx_t idx5,
            const idx_t idx6,
            const idx_t idx7,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3,
            float& dis4,
            float& dis5,
            float& dis6,
            float& dis7) final override {
        ndis += 8;

        // DPDK-style software pipeline: resolve + prefetch L2, then L1 deeper lines
        const float* __restrict y0 = b + resolve(idx0) * d;
        prefetch_L2(y0);
        const float* __restrict y1 = b + resolve(idx1) * d;
        prefetch_L2(y1);
        const float* __restrict y2 = b + resolve(idx2) * d;
        prefetch_L2(y2);
        const float* __restrict y3 = b + resolve(idx3) * d;
        prefetch_L2(y3);

        const float* __restrict y4 = b + resolve(idx4) * d;
        prefetch_L2(y4);
        const float* __restrict y5 = b + resolve(idx5) * d;
        prefetch_L2(y5);
        const float* __restrict y6 = b + resolve(idx6) * d;
        prefetch_L2(y6);
        const float* __restrict y7 = b + resolve(idx7) * d;
        prefetch_L2(y7);

        // Phase 2: L1 prefetch for deeper cache lines of all 8 vectors
        if (d > CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + CACHE_LINE_FLOATS);
            prefetch_L1(y1 + CACHE_LINE_FLOATS);
            prefetch_L1(y2 + CACHE_LINE_FLOATS);
            prefetch_L1(y3 + CACHE_LINE_FLOATS);
            prefetch_L1(y4 + CACHE_LINE_FLOATS);
            prefetch_L1(y5 + CACHE_LINE_FLOATS);
            prefetch_L1(y6 + CACHE_LINE_FLOATS);
            prefetch_L1(y7 + CACHE_LINE_FLOATS);
        }
        if (d > 2 * CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y1 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y2 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y3 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y4 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y5 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y6 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y7 + 2 * CACHE_LINE_FLOATS);
        }
        if (d > 3 * CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y1 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y2 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y3 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y4 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y5 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y6 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y7 + 3 * CACHE_LINE_FLOATS);
        }


        fvec_L2sqr_batch_8(
                q,
                y0,
                y1,
                y2,
                y3,
                y4,
                y5,
                y6,
                y7,
                d,
                dis0,
                dis1,
                dis2,
                dis3,
                dis4,
                dis5,
                dis6,
                dis7);
    }

    float partial_dot_product(
            const idx_t i,
            const uint32_t offset,
            const uint32_t num_components) final override {
        npartial_dot_products++;
        return fvec_inner_product(
                q + offset, b + resolve(i) * d + offset, num_components);
    }

    void partial_dot_product_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dp0,
            float& dp1,
            float& dp2,
            float& dp3,
            const uint32_t offset,
            const uint32_t num_components) final override {
        npartial_dot_products += 4;

        const float* __restrict y0 = b + resolve(idx0) * d;
        const float* __restrict y1 = b + resolve(idx1) * d;
        const float* __restrict y2 = b + resolve(idx2) * d;
        const float* __restrict y3 = b + resolve(idx3) * d;

        float dp0_ = 0, dp1_ = 0, dp2_ = 0, dp3_ = 0;
        fvec_inner_product_batch_4(
                q + offset,
                y0 + offset,
                y1 + offset,
                y2 + offset,
                y3 + offset,
                num_components,
                dp0_,
                dp1_,
                dp2_,
                dp3_);
        dp0 = dp0_;
        dp1 = dp1_;
        dp2 = dp2_;
        dp3 = dp3_;
    }
};

/*************************************************************
 * IndirectFlatIPDis — Inner Product distance computer with
 * indirection through storage_id_map.
 *************************************************************/

struct IndirectFlatIPDis : FlatCodesDistanceComputer {
    size_t d;
    const float* b;
    const idx_t* id_map;
    bool identity_map;
    size_t ndis = 0;

    explicit IndirectFlatIPDis(
            const IndexFlatShared& storage,
            const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.store->codes.data(),
                      storage.store->code_size,
                      q),
              d(storage.d),
              b(reinterpret_cast<const float*>(storage.store->codes.data())),
              id_map(storage.storage_id_map.data()),
              identity_map(storage.is_identity_map) {}

    idx_t resolve(idx_t i) const {
        if (identity_map) {
            return i;
        }
        return id_map[i];
    }

    void set_query(const float* x) override {
        q = x;
    }

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_inner_product(q, (const float*)code, d);
    }

    float operator()(idx_t i) override {
        ndis++;
        return fvec_inner_product(q, b + resolve(i) * d, d);
    }

    float symmetric_dis(idx_t i, idx_t j) final override {
        return fvec_inner_product(
                b + resolve(j) * d, b + resolve(i) * d, d);
    }

    static constexpr size_t CACHE_LINE_FLOATS = 16;

    void prefetch(idx_t i) override {
        if (!identity_map) {
            prefetch_L2(&id_map[i]);
        }
        idx_t resolved = resolve(i);
        const float* vec = b + resolved * d;
        prefetch_L2(vec);
        if (d > CACHE_LINE_FLOATS) {
            prefetch_L2(vec + CACHE_LINE_FLOATS);
        }
        if (d > 2 * CACHE_LINE_FLOATS) {
            prefetch_L2(vec + 2 * CACHE_LINE_FLOATS);
        }
        if (d > 3 * CACHE_LINE_FLOATS) {
            prefetch_L2(vec + 3 * CACHE_LINE_FLOATS);
        }
    }

    void prefetch_batch_4(
            idx_t i0,
            idx_t i1,
            idx_t i2,
            idx_t i3,
            int level) override {
        const float* p0 = b + resolve(i0) * d;
        const float* p1 = b + resolve(i1) * d;
        const float* p2 = b + resolve(i2) * d;
        const float* p3 = b + resolve(i3) * d;
        if (level == 1) {
            prefetch_L1(p0);
            prefetch_L1(p1);
            prefetch_L1(p2);
            prefetch_L1(p3);
            if (d > CACHE_LINE_FLOATS) {
                prefetch_L1(p0 + CACHE_LINE_FLOATS);
                prefetch_L1(p1 + CACHE_LINE_FLOATS);
                prefetch_L1(p2 + CACHE_LINE_FLOATS);
                prefetch_L1(p3 + CACHE_LINE_FLOATS);
            }
        } else if (level == 3) {
            prefetch_L3(p0);
            prefetch_L3(p1);
            prefetch_L3(p2);
            prefetch_L3(p3);
        } else {
            prefetch_L2(p0);
            prefetch_L2(p1);
            prefetch_L2(p2);
            prefetch_L2(p3);
            if (d > CACHE_LINE_FLOATS) {
                prefetch_L2(p0 + CACHE_LINE_FLOATS);
                prefetch_L2(p1 + CACHE_LINE_FLOATS);
                prefetch_L2(p2 + CACHE_LINE_FLOATS);
                prefetch_L2(p3 + CACHE_LINE_FLOATS);
            }
        }
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // DPDK-style: prefetch L2 early, then L1 for deeper cache lines
        const float* __restrict y0 = b + resolve(idx0) * d;
        prefetch_L2(y0);
        const float* __restrict y1 = b + resolve(idx1) * d;
        prefetch_L2(y1);
        const float* __restrict y2 = b + resolve(idx2) * d;
        prefetch_L2(y2);
        const float* __restrict y3 = b + resolve(idx3) * d;
        prefetch_L2(y3);

        if (d > CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + CACHE_LINE_FLOATS);
            prefetch_L1(y1 + CACHE_LINE_FLOATS);
            prefetch_L1(y2 + CACHE_LINE_FLOATS);
            prefetch_L1(y3 + CACHE_LINE_FLOATS);
        }
        if (d > 2 * CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y1 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y2 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y3 + 2 * CACHE_LINE_FLOATS);
        }
        if (d > 3 * CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y1 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y2 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y3 + 3 * CACHE_LINE_FLOATS);
        }

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        fvec_inner_product_batch_4(
                q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }

    void distances_batch_8(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            const idx_t idx4,
            const idx_t idx5,
            const idx_t idx6,
            const idx_t idx7,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3,
            float& dis4,
            float& dis5,
            float& dis6,
            float& dis7) final override {
        ndis += 8;

        // DPDK-style software pipeline: resolve + prefetch L2, then L1 deeper lines
        const float* __restrict y0 = b + resolve(idx0) * d;
        prefetch_L2(y0);
        const float* __restrict y1 = b + resolve(idx1) * d;
        prefetch_L2(y1);
        const float* __restrict y2 = b + resolve(idx2) * d;
        prefetch_L2(y2);
        const float* __restrict y3 = b + resolve(idx3) * d;
        prefetch_L2(y3);

        const float* __restrict y4 = b + resolve(idx4) * d;
        prefetch_L2(y4);
        const float* __restrict y5 = b + resolve(idx5) * d;
        prefetch_L2(y5);
        const float* __restrict y6 = b + resolve(idx6) * d;
        prefetch_L2(y6);
        const float* __restrict y7 = b + resolve(idx7) * d;
        prefetch_L2(y7);

        if (d > CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + CACHE_LINE_FLOATS);
            prefetch_L1(y1 + CACHE_LINE_FLOATS);
            prefetch_L1(y2 + CACHE_LINE_FLOATS);
            prefetch_L1(y3 + CACHE_LINE_FLOATS);
            prefetch_L1(y4 + CACHE_LINE_FLOATS);
            prefetch_L1(y5 + CACHE_LINE_FLOATS);
            prefetch_L1(y6 + CACHE_LINE_FLOATS);
            prefetch_L1(y7 + CACHE_LINE_FLOATS);
        }
        if (d > 2 * CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y1 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y2 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y3 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y4 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y5 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y6 + 2 * CACHE_LINE_FLOATS);
            prefetch_L1(y7 + 2 * CACHE_LINE_FLOATS);
        }
        if (d > 3 * CACHE_LINE_FLOATS) {
            prefetch_L1(y0 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y1 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y2 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y3 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y4 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y5 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y6 + 3 * CACHE_LINE_FLOATS);
            prefetch_L1(y7 + 3 * CACHE_LINE_FLOATS);
        }

        fvec_inner_product_batch_8(
                q,
                y0,
                y1,
                y2,
                y3,
                y4,
                y5,
                y6,
                y7,
                d,
                dis0,
                dis1,
                dis2,
                dis3,
                dis4,
                dis5,
                dis6,
                dis7);
    }
};

} // anonymous namespace

/*************************************************************
 * IndexFlatShared implementation
 *************************************************************/

IndexFlatShared::IndexFlatShared(
        std::shared_ptr<SharedVectorStore> store,
        MetricType metric)
        : IndexFlatCodes(store->code_size, store->d, metric),
          store(std::move(store)) {
    codes = MaybeOwnedVector<uint8_t>::create_view(
            this->store->codes.data(),
            this->store->codes.size(),
            this->store);
}

IndexFlatShared::IndexFlatShared(
        std::shared_ptr<SharedVectorStore> store,
        const std::vector<idx_t>& old_storage_id_map,
        const std::vector<uint64_t>& old_deleted_bitmap,
        MetricType metric)
        : IndexFlatCodes(store->code_size, store->d, metric),
          store(std::move(store)) {
    codes = MaybeOwnedVector<uint8_t>::create_view(
            this->store->codes.data(),
            this->store->codes.size(),
            this->store);

    build_storage_id_map(
            *this->store,
            old_deleted_bitmap,
            old_storage_id_map,
            storage_id_map);

    ntotal = storage_id_map.size();
    is_identity_map = false;

    size_t bitmap_words =
            (this->store->ntotal_store + 63) / 64;
    deleted_bitmap.assign(bitmap_words, 0);
}

size_t IndexFlatShared::count_alive() const {
    size_t count = 0;
    for (size_t i = 0; i < (size_t)ntotal; i++) {
        idx_t slot = storage_id_map[i];
        if (!is_deleted(slot)) {
            count++;
        }
    }
    return count;
}

void IndexFlatShared::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(store);
    for (idx_t i = 0; i < n; i++) {
        idx_t slot = store->allocate_slot(x + i * d);
        storage_id_map.push_back(slot);
    }
    ntotal += n;

    codes = MaybeOwnedVector<uint8_t>::create_view(
            store->codes.data(), store->codes.size(), store);

    size_t bitmap_words = (store->ntotal_store + 63) / 64;
    if (deleted_bitmap.size() < bitmap_words) {
        deleted_bitmap.resize(bitmap_words, 0);
    }
}

void IndexFlatShared::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    idx_t slot = storage_id_map[key];
    memcpy(recons, store->get_vector(slot), sizeof(float) * d);
}

FlatCodesDistanceComputer* IndexFlatShared::get_FlatCodesDistanceComputer()
        const {
    if (metric_type == METRIC_L2) {
        return new IndirectFlatL2Dis(*this);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        return new IndirectFlatIPDis(*this);
    } else {
        FAISS_THROW_MSG(
                "IndexFlatShared: only L2 and IP metrics are supported");
    }
}

void IndexFlatShared::permute_entries(const idx_t* perm) {
    std::vector<idx_t> new_map(ntotal);
    for (idx_t i = 0; i < ntotal; i++) {
        new_map[i] = storage_id_map[perm[i]];
    }
    std::swap(storage_id_map, new_map);
}

void IndexFlatShared::reset() {
    storage_id_map.clear();
    deleted_bitmap.clear();
    ntotal = 0;
}

/*************************************************************
 * build_storage_id_map — builds new storage_id_map from old
 * map, skipping entries marked as deleted in old_deleted_bitmap.
 *************************************************************/

void build_storage_id_map(
        const SharedVectorStore& store,
        const std::vector<uint64_t>& deleted_bitmap,
        const std::vector<idx_t>& old_storage_id_map,
        std::vector<idx_t>& new_storage_id_map) {
    new_storage_id_map.clear();
    new_storage_id_map.reserve(old_storage_id_map.size());

    for (size_t i = 0; i < old_storage_id_map.size(); i++) {
        idx_t slot = old_storage_id_map[i];
        bool is_del = false;
        if (!deleted_bitmap.empty()) {
            is_del = (deleted_bitmap[slot >> 6] >> (slot & 63)) & 1;
        }
        if (!is_del) {
            new_storage_id_map.push_back(slot);
        }
    }
}

/*************************************************************
 * build_new_index — build a new IndexHNSW from shared store,
 * skipping deleted vectors (zero-copy rebuild).
 *************************************************************/

IndexHNSW* build_new_index(
        std::shared_ptr<SharedVectorStore> store,
        const IndexHNSW& current_index,
        int M,
        int efConstruction,
        MetricType metric) {
    auto* old_shared =
            dynamic_cast<const IndexFlatShared*>(current_index.storage);
    FAISS_THROW_IF_NOT_MSG(
            old_shared, "current_index.storage must be IndexFlatShared");

    std::vector<idx_t> new_storage_id_map;
    build_storage_id_map(
            *store,
            old_shared->deleted_bitmap,
            old_shared->storage_id_map,
            new_storage_id_map);

    idx_t n_alive = new_storage_id_map.size();

    auto* shared_storage = new IndexFlatShared(store, metric);
    shared_storage->storage_id_map = std::move(new_storage_id_map);
    shared_storage->ntotal = n_alive;
    shared_storage->is_identity_map = false;
    size_t bitmap_words = (store->ntotal_store + 63) / 64;
    shared_storage->deleted_bitmap.assign(bitmap_words, 0);

    auto* new_index = new IndexHNSW(shared_storage, M);
    new_index->own_fields = true;
    new_index->hnsw.efConstruction = efConstruction;
    new_index->is_trained = true;

    // Do NOT set new_index->ntotal before add(): n0 must be 0
    // so hnsw_add_vertices iterates pt_id over [0, n_alive).
    new_index->add(n_alive, nullptr);

    return new_index;
}

} // namespace faiss
