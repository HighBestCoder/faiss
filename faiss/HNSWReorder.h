/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <faiss/Index.h>

namespace faiss {

struct HNSW; // forward declaration

/// Strategy for reordering HNSW graph nodes to improve cache locality.
enum class ReorderStrategy {
    BFS,      ///< Breadth-first from entry_point
    RCM,      ///< Reverse Cuthill-McKee (minimizes graph bandwidth)
    DFS,      ///< Depth-first from entry_point
    CLUSTER,  ///< Level-descending, neighbors placed consecutively
    WEIGHTED  ///< Score = (1 + level) * degree, high-score nodes first
};

/// Generate a permutation for reordering HNSW graph nodes.
/// All functions return perm where perm[new_id] = old_id.
///
/// @param hnsw      the HNSW graph structure
/// @param strategy  which reorder strategy to use
/// @return permutation vector of size hnsw.levels.size()
std::vector<idx_t> generate_permutation(
        const HNSW& hnsw,
        ReorderStrategy strategy);

/// Individual permutation generators.
/// All return perm where perm[new_id] = old_id.

std::vector<idx_t> generate_bfs_permutation(const HNSW& hnsw);
std::vector<idx_t> generate_rcm_permutation(const HNSW& hnsw);
std::vector<idx_t> generate_dfs_permutation(const HNSW& hnsw);
std::vector<idx_t> generate_cluster_permutation(const HNSW& hnsw);
std::vector<idx_t> generate_weighted_permutation(const HNSW& hnsw);

} // namespace faiss
