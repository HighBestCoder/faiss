/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_VISITED_TABLE_H
#define FAISS_VISITED_TABLE_H

#include <stdint.h>

#include <optional>
#include <vector>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/prefetch.h>

namespace faiss {

/// A fast, reusable Visited Set for graph search algorithms.
///
/// At our index sizes (1M..50M) the byte+visno bit-vector is faster
/// than the hash-set path that upstream PR #4735 added in v1.14.1
/// (see arch-2026-05-9/root_cause.md). The hash-set path has been
/// removed; the constructor still accepts a 2nd argument so callers
/// that pass `use_hashset=false` continue to compile, but the hint
/// is ignored — the implementation is always vector-backed.
struct VisitedTable {
    std::vector<uint8_t> visited;
    uint8_t visno;

    explicit VisitedTable(
            size_t size,
            std::optional<bool> /*use_hashset*/ = std::nullopt)
            : visited(size, 0), visno(1) {}

    /// set flag #no to true, return whether this changed it.
    bool set(size_t no) {
        if (visited[no] == visno) {
            return false;
        }
        visited[no] = visno;
        return true;
    }

    /// get flag #no
    bool get(size_t no) const {
        return visited[no] == visno;
    }

    void prefetch(size_t no) const {
        prefetch_L2(&visited[no]);
    }

    /// reset all flags to false
    void advance();
};

} // namespace faiss

#endif
