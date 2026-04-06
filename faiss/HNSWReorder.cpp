/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/HNSWReorder.h>

#include <algorithm>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

#include <faiss/impl/HNSW.h>

namespace faiss {

namespace {

size_t get_node_degree(const HNSW& hnsw, HNSW::storage_idx_t node) {
    size_t begin, end;
    hnsw.neighbor_range(node, 0, &begin, &end);
    size_t degree = 0;
    for (size_t j = begin; j < end; j++) {
        if (hnsw.neighbors[j] >= 0)
            degree++;
    }
    return degree;
}

std::vector<size_t> precompute_degrees(const HNSW& hnsw) {
    size_t ntotal = hnsw.levels.size();
    std::vector<size_t> degrees(ntotal);
    for (size_t i = 0; i < ntotal; i++) {
        degrees[i] = get_node_degree(hnsw, i);
    }
    return degrees;
}

} // namespace

std::vector<idx_t> generate_bfs_permutation(const HNSW& hnsw) {
    size_t ntotal = hnsw.levels.size();
    std::vector<idx_t> perm;
    perm.reserve(ntotal);

    std::vector<bool> visited(ntotal, false);
    std::deque<HNSW::storage_idx_t> bfs_queue;

    if (hnsw.entry_point >= 0) {
        bfs_queue.push_back(hnsw.entry_point);
        visited[hnsw.entry_point] = true;
    }

    while (!bfs_queue.empty()) {
        HNSW::storage_idx_t current = bfs_queue.front();
        bfs_queue.pop_front();
        perm.push_back(current);

        size_t begin, end;
        hnsw.neighbor_range(current, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            HNSW::storage_idx_t neighbor = hnsw.neighbors[j];
            if (neighbor >= 0 && !visited[neighbor]) {
                visited[neighbor] = true;
                bfs_queue.push_back(neighbor);
            }
        }
    }

    for (size_t i = 0; i < ntotal; i++) {
        if (!visited[i]) {
            perm.push_back(i);
        }
    }

    return perm;
}

std::vector<idx_t> generate_rcm_permutation(const HNSW& hnsw) {
    size_t ntotal = hnsw.levels.size();
    std::vector<idx_t> perm;
    perm.reserve(ntotal);

    std::vector<bool> visited(ntotal, false);

    auto degrees = precompute_degrees(hnsw);

    // Find peripheral node (minimum degree)
    HNSW::storage_idx_t start_node = 0;
    size_t min_degree = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < ntotal; i++) {
        if (degrees[i] > 0 && degrees[i] < min_degree) {
            min_degree = degrees[i];
            start_node = i;
        }
    }

    // BFS to find the farthest node from start_node (pseudo-peripheral)
    {
        std::deque<HNSW::storage_idx_t> queue;
        std::vector<bool> temp_visited(ntotal, false);
        queue.push_back(start_node);
        temp_visited[start_node] = true;
        HNSW::storage_idx_t last_node = start_node;

        while (!queue.empty()) {
            last_node = queue.front();
            queue.pop_front();

            size_t begin, end;
            hnsw.neighbor_range(last_node, 0, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                HNSW::storage_idx_t neighbor = hnsw.neighbors[j];
                if (neighbor >= 0 && !temp_visited[neighbor]) {
                    temp_visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        start_node = last_node;
    }

    // BFS from pseudo-peripheral, sorting neighbors by degree (ascending)
    std::deque<HNSW::storage_idx_t> queue;
    queue.push_back(start_node);
    visited[start_node] = true;

    while (!queue.empty()) {
        HNSW::storage_idx_t current = queue.front();
        queue.pop_front();
        perm.push_back(current);

        std::vector<std::pair<size_t, HNSW::storage_idx_t>> neighbor_degrees;
        size_t begin, end;
        hnsw.neighbor_range(current, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            HNSW::storage_idx_t neighbor = hnsw.neighbors[j];
            if (neighbor >= 0 && !visited[neighbor]) {
                visited[neighbor] = true;
                neighbor_degrees.push_back({degrees[neighbor], neighbor});
            }
        }

        std::sort(neighbor_degrees.begin(), neighbor_degrees.end());

        for (auto& p : neighbor_degrees) {
            queue.push_back(p.second);
        }
    }

    for (size_t i = 0; i < ntotal; i++) {
        if (!visited[i]) {
            perm.push_back(i);
        }
    }

    std::reverse(perm.begin(), perm.end());

    return perm;
}

std::vector<idx_t> generate_dfs_permutation(const HNSW& hnsw) {
    size_t ntotal = hnsw.levels.size();
    std::vector<idx_t> perm;
    perm.reserve(ntotal);

    std::vector<bool> visited(ntotal, false);
    std::vector<HNSW::storage_idx_t> stack;

    if (hnsw.entry_point >= 0) {
        stack.push_back(hnsw.entry_point);
    }

    while (!stack.empty()) {
        HNSW::storage_idx_t current = stack.back();
        stack.pop_back();

        if (visited[current])
            continue;
        visited[current] = true;
        perm.push_back(current);

        std::vector<HNSW::storage_idx_t> neighbors;
        size_t begin, end;
        hnsw.neighbor_range(current, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            HNSW::storage_idx_t neighbor = hnsw.neighbors[j];
            if (neighbor >= 0 && !visited[neighbor]) {
                neighbors.push_back(neighbor);
            }
        }

        for (auto it = neighbors.rbegin(); it != neighbors.rend(); ++it) {
            stack.push_back(*it);
        }
    }

    for (size_t i = 0; i < ntotal; i++) {
        if (!visited[i]) {
            perm.push_back(i);
        }
    }

    return perm;
}

std::vector<idx_t> generate_cluster_permutation(const HNSW& hnsw) {
    size_t ntotal = hnsw.levels.size();
    std::vector<idx_t> perm;
    perm.reserve(ntotal);

    std::vector<bool> visited(ntotal, false);

    std::vector<std::pair<int, HNSW::storage_idx_t>> nodes_by_level;
    for (size_t i = 0; i < ntotal; i++) {
        nodes_by_level.push_back({hnsw.levels[i], i});
    }
    std::sort(nodes_by_level.rbegin(), nodes_by_level.rend());

    for (auto& p : nodes_by_level) {
        HNSW::storage_idx_t node = p.second;

        if (visited[node])
            continue;

        visited[node] = true;
        perm.push_back(node);

        size_t begin, end;
        hnsw.neighbor_range(node, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            HNSW::storage_idx_t neighbor = hnsw.neighbors[j];
            if (neighbor >= 0 && !visited[neighbor]) {
                visited[neighbor] = true;
                perm.push_back(neighbor);
            }
        }
    }

    return perm;
}

std::vector<idx_t> generate_weighted_permutation(const HNSW& hnsw) {
    size_t ntotal = hnsw.levels.size();

    auto degrees = precompute_degrees(hnsw);

    std::vector<std::pair<double, HNSW::storage_idx_t>> scores;
    scores.reserve(ntotal);
    for (size_t i = 0; i < ntotal; i++) {
        double level_weight = 1.0 + hnsw.levels[i];
        scores.push_back({level_weight * degrees[i], i});
    }

    std::sort(scores.rbegin(), scores.rend());

    std::vector<idx_t> perm;
    perm.reserve(ntotal);
    std::vector<bool> visited(ntotal, false);

    for (auto& p : scores) {
        HNSW::storage_idx_t node = p.second;
        if (visited[node])
            continue;

        visited[node] = true;
        perm.push_back(node);

        std::vector<std::pair<double, HNSW::storage_idx_t>> neighbor_scores;
        size_t begin, end;
        hnsw.neighbor_range(node, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            HNSW::storage_idx_t neighbor = hnsw.neighbors[j];
            if (neighbor >= 0 && !visited[neighbor]) {
                double nlevel = 1.0 + hnsw.levels[neighbor];
                neighbor_scores.push_back(
                        {nlevel * degrees[neighbor], neighbor});
            }
        }

        std::sort(neighbor_scores.rbegin(), neighbor_scores.rend());
        for (auto& np : neighbor_scores) {
            if (!visited[np.second]) {
                visited[np.second] = true;
                perm.push_back(np.second);
            }
        }
    }

    return perm;
}

std::vector<idx_t> generate_permutation(
        const HNSW& hnsw,
        ReorderStrategy strategy) {
    switch (strategy) {
        case ReorderStrategy::BFS:
            return generate_bfs_permutation(hnsw);
        case ReorderStrategy::RCM:
            return generate_rcm_permutation(hnsw);
        case ReorderStrategy::DFS:
            return generate_dfs_permutation(hnsw);
        case ReorderStrategy::CLUSTER:
            return generate_cluster_permutation(hnsw);
        case ReorderStrategy::WEIGHTED:
            return generate_weighted_permutation(hnsw);
        default:
            return generate_bfs_permutation(hnsw);
    }
}

} // namespace faiss
