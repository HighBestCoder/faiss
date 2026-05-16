/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/MetadataStore.h>

namespace faiss {

bool InMemoryMetadataStore::load(AppendableMetadata& out) {
    if (!has_) {
        return false;
    }
    out = state_;
    return true;
}

void InMemoryMetadataStore::commit(const AppendableMetadata& m) {
    state_ = m;
    has_ = true;
}

} // namespace faiss
