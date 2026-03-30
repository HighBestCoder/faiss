/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace faiss {

inline bool try_enable_hugepages(void* ptr, size_t size) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (ptr && size > 0) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t aligned = (addr + 4095) & ~uintptr_t(4095);
        size_t offset = aligned - addr;
        if (size > offset) {
            return madvise(
                           reinterpret_cast<void*>(aligned),
                           size - offset,
                           MADV_HUGEPAGE) == 0;
        }
    }
#endif
    (void)ptr;
    (void)size;
    return false;
}

} // namespace faiss
