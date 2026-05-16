/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/CodesStorage.h>

#include <cstring>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

void CodesStorage::gather(size_t n, const idx_t* ids, uint8_t* dst) const {
    auto v = try_view();
    FAISS_THROW_IF_NOT_MSG(
            v.has_value(),
            "Form-2 CodesStorage must override gather()");
    const size_t cs = code_size();
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(dst + i * cs, v->data + ids[i] * cs, cs);
    }
}

void CodesStorage::flush(const Index* /*idx*/) {
    FAISS_THROW_MSG("flush() not supported by this CodesStorage");
}

InMemoryCodesStorage::InMemoryCodesStorage(size_t cs) : code_size_(cs) {
    FAISS_THROW_IF_NOT(cs > 0);
}

InMemoryCodesStorage::InMemoryCodesStorage(
        size_t cs,
        std::vector<uint8_t>&& bytes)
        : code_size_(cs), buffer_(std::move(bytes)) {
    FAISS_THROW_IF_NOT(cs > 0);
    FAISS_THROW_IF_NOT(buffer_.size() % cs == 0);
}

void InMemoryCodesStorage::append(size_t n, const uint8_t* src) {
    if (n == 0) {
        return;
    }
    const size_t old = buffer_.size();
    buffer_.resize(old + n * code_size_);
    std::memcpy(buffer_.data() + old, src, n * code_size_);
}

void InMemoryCodesStorage::reset() {
    buffer_.clear();
}

void InMemoryCodesStorage::permute(const idx_t* perm) {
    const size_t n = num_codes();
    std::vector<uint8_t> next(buffer_.size());
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(
                next.data() + i * code_size_,
                buffer_.data() + perm[i] * code_size_,
                code_size_);
    }
    buffer_.swap(next);
}

std::optional<CodesView> InMemoryCodesStorage::try_view() const {
    CodesView v;
    v.data = buffer_.data();
    v.nbytes = buffer_.size();
    return v;
}

} // namespace faiss

#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexNSG.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexPreTransform.h>

namespace faiss {

const IndexFlatCodes* find_codes_storage(const Index* idx) {
    if (idx == nullptr) return nullptr;
    if (auto* fc = dynamic_cast<const IndexFlatCodes*>(idx)) return fc;
    if (auto* h = dynamic_cast<const IndexHNSW*>(idx))
        return find_codes_storage(h->storage);
    if (auto* h = dynamic_cast<const IndexNSG*>(idx))
        return find_codes_storage(h->storage);
    if (auto* h = dynamic_cast<const IndexNNDescent*>(idx))
        return find_codes_storage(h->storage);
    if (auto* h = dynamic_cast<const IndexRefine*>(idx)) {
        if (auto* fc = find_codes_storage(h->refine_index)) return fc;
        return find_codes_storage(h->base_index);
    }
    if (auto* h = dynamic_cast<const IndexIDMap*>(idx))
        return find_codes_storage(h->index);
    if (auto* h = dynamic_cast<const IndexPreTransform*>(idx))
        return find_codes_storage(h->index);
    return nullptr;
}

IndexFlatCodes* find_codes_storage(Index* idx) {
    return const_cast<IndexFlatCodes*>(
            find_codes_storage(static_cast<const Index*>(idx)));
}

} // namespace faiss
