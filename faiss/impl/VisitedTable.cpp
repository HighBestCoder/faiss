/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/VisitedTable.h>

#include <cstring>

namespace faiss {

void VisitedTable::advance() {
    // 254 rather than 255 because some sites use both visno and visno+1.
    if (visno < 254) {
        ++visno;
    } else {
        memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
        visno = 1;
    }
}

} // namespace faiss
