/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark: IndexHNSW with different storage backends
 * 
 * Compares:
 * 1. IndexFlat (direct array storage) - baseline
 * 2. IndexIndirectFlat (pointer mapping storage) - indirect access via vector<float*>
 * 3. IndexIndirectFlatHash (hash map storage) - indirect access via unordered_map
 * 
 * Parameters: 500K vectors, 768 dimensions
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/prefetch.h>

#include <omp.h>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#ifdef __SSE2__
#include <xmmintrin.h>
#endif

//=============================================================================
// DPDK-style optimization macros
//=============================================================================

// Branch prediction hints
#ifndef likely
#if defined(__GNUC__) || defined(__clang__)
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x)   (x)
#define unlikely(x) (x)
#endif
#endif

// Cache line size (typically 64 bytes on x86)
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

namespace {

//=============================================================================
// Utility functions
//=============================================================================

double get_time_sec() {
    return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
}

size_t get_memory_usage_kb() {
#ifdef __linux__
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<size_t>(usage.ru_maxrss);
#else
    return 0;
#endif
}

std::vector<float> generate_random_vectors(size_t n, size_t d, unsigned seed = 42) {
    std::vector<float> data(n * d);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < n * d; i++) {
        data[i] = dist(rng);
    }

    // Normalize vectors
    for (size_t i = 0; i < n; i++) {
        float norm = 0;
        for (size_t j = 0; j < d; j++) {
            norm += data[i * d + j] * data[i * d + j];
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < d; j++) {
            data[i * d + j] /= norm;
        }
    }

    return data;
}

std::vector<faiss::idx_t> compute_ground_truth(
        const float* queries,
        const float* database,
        size_t nq,
        size_t nb,
        size_t d,
        size_t k) {
    std::vector<faiss::idx_t> gt(nq * k);

#pragma omp parallel for
    for (int64_t q = 0; q < (int64_t)nq; q++) {
        const float* query = queries + q * d;
        std::vector<std::pair<float, faiss::idx_t>> distances(nb);

        for (size_t i = 0; i < nb; i++) {
            const float* vec = database + i * d;
            float dist = faiss::fvec_L2sqr(query, vec, d);
            distances[i] = {dist, static_cast<faiss::idx_t>(i)};
        }

        std::partial_sort(
                distances.begin(),
                distances.begin() + k,
                distances.end(),
                [](auto& a, auto& b) { return a.first < b.first; });

        for (size_t i = 0; i < k; i++) {
            gt[q * k + i] = distances[i].second;
        }
    }

    return gt;
}

double compute_recall(
        const faiss::idx_t* results,
        const faiss::idx_t* ground_truth,
        size_t nq,
        size_t k) {
    size_t correct = 0;
    for (size_t q = 0; q < nq; q++) {
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < k; j++) {
                if (results[q * k + i] == ground_truth[q * k + j]) {
                    correct++;
                    break;
                }
            }
        }
    }
    return static_cast<double>(correct) / (nq * k);
}

//=============================================================================
// Custom Storage 1: Indirect Flat (vector of pointers)
//=============================================================================

struct IndexIndirectFlat;

struct IndirectFlatL2Dis : faiss::DistanceComputer {
    const IndexIndirectFlat& storage;
    size_t d;
    const float* q = nullptr;

    explicit IndirectFlatL2Dis(const IndexIndirectFlat& s);

    void set_query(const float* x) override {
        q = x;
    }

    float operator()(faiss::idx_t i) override;

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override;
};

struct IndexIndirectFlat : faiss::Index {
    size_t d;
    
    // Actual data storage (contiguous for fair memory comparison)
    std::vector<float> data_storage;
    
    // Indirect mapping: id -> pointer to vector data
    std::vector<float*> id_to_ptr;

    explicit IndexIndirectFlat(faiss::idx_t dim)
            : faiss::Index(dim, faiss::METRIC_L2), d(dim) {}

    void add(faiss::idx_t n, const float* x) override {
        size_t old_size = data_storage.size();
        data_storage.resize(old_size + n * d);
        std::memcpy(data_storage.data() + old_size, x, n * d * sizeof(float));

        // Build pointer mapping
        for (faiss::idx_t i = 0; i < n; i++) {
            id_to_ptr.push_back(data_storage.data() + old_size + i * d);
        }
        ntotal += n;
    }

    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override {
        // Brute force search (not used by HNSW, but required by interface)
#pragma omp parallel for
        for (faiss::idx_t q = 0; q < n; q++) {
            const float* query = x + q * d;
            std::vector<std::pair<float, faiss::idx_t>> dists(ntotal);
            
            for (faiss::idx_t i = 0; i < ntotal; i++) {
                dists[i] = {faiss::fvec_L2sqr(query, id_to_ptr[i], d), i};
            }
            
            std::partial_sort(
                    dists.begin(), dists.begin() + k, dists.end(),
                    [](auto& a, auto& b) { return a.first < b.first; });
            
            for (faiss::idx_t i = 0; i < k; i++) {
                distances[q * k + i] = dists[i].first;
                labels[q * k + i] = dists[i].second;
            }
        }
    }

    void reset() override {
        data_storage.clear();
        id_to_ptr.clear();
        ntotal = 0;
    }

    void reconstruct(faiss::idx_t key, float* recons) const override {
        std::memcpy(recons, id_to_ptr[key], d * sizeof(float));
    }

    faiss::DistanceComputer* get_distance_computer() const override {
        return new IndirectFlatL2Dis(*this);
    }
};

IndirectFlatL2Dis::IndirectFlatL2Dis(const IndexIndirectFlat& s)
        : storage(s), d(s.d) {}

float IndirectFlatL2Dis::operator()(faiss::idx_t i) {
    return faiss::fvec_L2sqr(q, storage.id_to_ptr[i], d);
}

float IndirectFlatL2Dis::symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
    return faiss::fvec_L2sqr(storage.id_to_ptr[i], storage.id_to_ptr[j], d);
}

//=============================================================================
// Custom Storage 2: Hash Map Storage (unordered_map)
//=============================================================================

struct IndexHashMapFlat;

struct HashMapFlatL2Dis : faiss::DistanceComputer {
    const IndexHashMapFlat& storage;
    size_t d;
    const float* q = nullptr;

    explicit HashMapFlatL2Dis(const IndexHashMapFlat& s);

    void set_query(const float* x) override {
        q = x;
    }

    float operator()(faiss::idx_t i) override;

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override;
};

struct IndexHashMapFlat : faiss::Index {
    size_t d;
    
    // Actual data storage
    std::vector<float> data_storage;
    
    // Hash map: id -> pointer to vector data
    std::unordered_map<faiss::idx_t, float*> id_to_ptr;

    explicit IndexHashMapFlat(faiss::idx_t dim)
            : faiss::Index(dim, faiss::METRIC_L2), d(dim) {}

    void add(faiss::idx_t n, const float* x) override {
        size_t old_size = data_storage.size();
        data_storage.resize(old_size + n * d);
        std::memcpy(data_storage.data() + old_size, x, n * d * sizeof(float));

        // Build hash map
        for (faiss::idx_t i = 0; i < n; i++) {
            faiss::idx_t id = ntotal + i;
            id_to_ptr[id] = data_storage.data() + old_size + i * d;
        }
        ntotal += n;
    }

    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override {
        // Brute force search
#pragma omp parallel for
        for (faiss::idx_t q = 0; q < n; q++) {
            const float* query = x + q * d;
            std::vector<std::pair<float, faiss::idx_t>> dists(ntotal);
            
            faiss::idx_t idx = 0;
            for (const auto& kv : id_to_ptr) {
                dists[idx++] = {faiss::fvec_L2sqr(query, kv.second, d), kv.first};
            }
            
            std::partial_sort(
                    dists.begin(), dists.begin() + k, dists.end(),
                    [](auto& a, auto& b) { return a.first < b.first; });
            
            for (faiss::idx_t i = 0; i < k; i++) {
                distances[q * k + i] = dists[i].first;
                labels[q * k + i] = dists[i].second;
            }
        }
    }

    void reset() override {
        data_storage.clear();
        id_to_ptr.clear();
        ntotal = 0;
    }

    void reconstruct(faiss::idx_t key, float* recons) const override {
        std::memcpy(recons, id_to_ptr.at(key), d * sizeof(float));
    }

    faiss::DistanceComputer* get_distance_computer() const override {
        return new HashMapFlatL2Dis(*this);
    }

    const float* get_vector(faiss::idx_t i) const {
        return id_to_ptr.at(i);
    }
};

HashMapFlatL2Dis::HashMapFlatL2Dis(const IndexHashMapFlat& s)
        : storage(s), d(s.d) {}

float HashMapFlatL2Dis::operator()(faiss::idx_t i) {
    return faiss::fvec_L2sqr(q, storage.get_vector(i), d);
}

float HashMapFlatL2Dis::symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
    return faiss::fvec_L2sqr(storage.get_vector(i), storage.get_vector(j), d);
}

//=============================================================================
// Custom Storage 3: Chunked Vector Storage
// Data is stored in fixed-size chunks. Access requires:
//   chunk_idx = idx >> chunk_shift  (bit shift instead of division)
//   chunk_internal_idx = idx & chunk_mask  (bit mask instead of modulo)
//=============================================================================

struct IndexChunkedFlat;

struct ChunkedFlatL2Dis : faiss::DistanceComputer {
    const IndexChunkedFlat& storage;
    size_t d;
    const float* q = nullptr;
    
    size_t chunk_shift;
    size_t chunk_mask;
    const std::vector<float*>& chunk_ptrs;

    explicit ChunkedFlatL2Dis(const IndexChunkedFlat& s);

    void set_query(const float* x) override {
        q = x;
    }

    inline const float* get_vec(faiss::idx_t i) const {
        return chunk_ptrs[i >> chunk_shift] + (i & chunk_mask) * d;
    }

    float operator()(faiss::idx_t i) override;

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override;
    
    void distances_batch_4(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        // DPDK-style prefetching: prefetch next vector while loading current
        // This hides memory latency by overlapping loads with cache prefetch
        
        // Prefetch all 4 vectors into L2 cache first (larger cache, lower priority)
        prefetch_L2(get_vec(idx0));
        prefetch_L2(get_vec(idx1));
        prefetch_L2(get_vec(idx2));
        prefetch_L2(get_vec(idx3));
        
        // Now load vectors - they should be in cache or on their way
        const float* v0 = get_vec(idx0);
        prefetch_L1(v0 + 64);  // Prefetch rest of vector (next cache line)
        
        const float* v1 = get_vec(idx1);
        prefetch_L1(v1 + 64);
        
        const float* v2 = get_vec(idx2);
        prefetch_L1(v2 + 64);
        
        const float* v3 = get_vec(idx3);
        prefetch_L1(v3 + 64);
        
        faiss::fvec_L2sqr_batch_4(q, v0, v1, v2, v3, d, dis0, dis1, dis2, dis3);
    }
};

struct IndexChunkedFlat : faiss::Index {
    size_t d;
    size_t chunk_element_size;
    size_t chunk_shift;
    size_t chunk_mask;
    
    std::vector<std::vector<float>> chunks;
    std::vector<float*> chunk_ptrs;

    explicit IndexChunkedFlat(faiss::idx_t dim, size_t chunk_size = 1024)
            : faiss::Index(dim, faiss::METRIC_L2), d(dim), chunk_element_size(chunk_size) {
        chunk_shift = 0;
        size_t tmp = chunk_size;
        while (tmp > 1) {
            tmp >>= 1;
            chunk_shift++;
        }
        chunk_mask = chunk_element_size - 1;
    }

    const float* get_vector(faiss::idx_t i) const {
        return chunk_ptrs[i >> chunk_shift] + (i & chunk_mask) * d;
    }

    float* get_vector_mutable(faiss::idx_t i) {
        return chunk_ptrs[i >> chunk_shift] + (i & chunk_mask) * d;
    }

    void add(faiss::idx_t n, const float* x) override {
        for (faiss::idx_t i = 0; i < n; i++) {
            faiss::idx_t global_idx = ntotal + i;
            size_t chunk_idx = global_idx >> chunk_shift;
            size_t internal_idx = global_idx & chunk_mask;
            
            if (chunk_idx >= chunks.size()) {
                chunks.emplace_back(chunk_element_size * d);
                chunk_ptrs.push_back(chunks.back().data());
            }
            
            std::memcpy(
                chunk_ptrs[chunk_idx] + internal_idx * d,
                x + i * d,
                d * sizeof(float));
        }
        ntotal += n;
    }

    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override {
#pragma omp parallel for
        for (faiss::idx_t q = 0; q < n; q++) {
            const float* query = x + q * d;
            std::vector<std::pair<float, faiss::idx_t>> dists(ntotal);
            
            for (faiss::idx_t i = 0; i < ntotal; i++) {
                dists[i] = {faiss::fvec_L2sqr(query, get_vector(i), d), i};
            }
            
            std::partial_sort(
                    dists.begin(), dists.begin() + k, dists.end(),
                    [](auto& a, auto& b) { return a.first < b.first; });
            
            for (faiss::idx_t i = 0; i < k; i++) {
                distances[q * k + i] = dists[i].first;
                labels[q * k + i] = dists[i].second;
            }
        }
    }

    void reset() override {
        chunks.clear();
        chunk_ptrs.clear();
        ntotal = 0;
    }

    void reconstruct(faiss::idx_t key, float* recons) const override {
        std::memcpy(recons, get_vector(key), d * sizeof(float));
    }

    faiss::DistanceComputer* get_distance_computer() const override {
        return new ChunkedFlatL2Dis(*this);
    }
};

ChunkedFlatL2Dis::ChunkedFlatL2Dis(const IndexChunkedFlat& s)
        : storage(s), d(s.d), chunk_shift(s.chunk_shift), 
          chunk_mask(s.chunk_mask), chunk_ptrs(s.chunk_ptrs) {}

float ChunkedFlatL2Dis::operator()(faiss::idx_t i) {
    return faiss::fvec_L2sqr(q, get_vec(i), d);
}

float ChunkedFlatL2Dis::symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
    return faiss::fvec_L2sqr(get_vec(i), get_vec(j), d);
}

//=============================================================================
// Custom Storage 4: Hugepage-backed Chunked Storage (Linux only)
// Uses mmap with MAP_HUGETLB for 2MB hugepages to reduce TLB misses
//=============================================================================

#ifdef __linux__

struct IndexHugepageChunkedFlat;

struct HugepageFlatL2Dis : faiss::DistanceComputer {
    const IndexHugepageChunkedFlat& storage;
    size_t d;
    const float* q = nullptr;
    
    size_t chunk_shift;
    size_t chunk_mask;
    float* const* chunk_ptrs;

    explicit HugepageFlatL2Dis(const IndexHugepageChunkedFlat& s);

    void set_query(const float* x) override {
        q = x;
    }

    FAISS_ALWAYS_INLINE const float* get_vec(faiss::idx_t i) const {
        return chunk_ptrs[i >> chunk_shift] + (i & chunk_mask) * d;
    }

    float operator()(faiss::idx_t i) override;

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override;
    
    void distances_batch_4(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        prefetch_L2(get_vec(idx0));
        prefetch_L2(get_vec(idx1));
        prefetch_L2(get_vec(idx2));
        prefetch_L2(get_vec(idx3));
        
        const float* v0 = get_vec(idx0);
        prefetch_L1(v0 + 64);
        
        const float* v1 = get_vec(idx1);
        prefetch_L1(v1 + 64);
        
        const float* v2 = get_vec(idx2);
        prefetch_L1(v2 + 64);
        
        const float* v3 = get_vec(idx3);
        prefetch_L1(v3 + 64);
        
        faiss::fvec_L2sqr_batch_4(q, v0, v1, v2, v3, d, dis0, dis1, dis2, dis3);
    }
};

struct IndexHugepageChunkedFlat : faiss::Index {
    size_t d;
    size_t chunk_element_size;
    size_t chunk_byte_size;
    size_t chunk_shift;
    size_t chunk_mask;
    bool use_hugepages;
    
    std::vector<float*> chunk_ptrs;

    static constexpr size_t HUGEPAGE_SIZE = 2 * 1024 * 1024;  // 2MB

    explicit IndexHugepageChunkedFlat(faiss::idx_t dim, size_t chunk_size = 16384, bool try_hugepages = true)
            : faiss::Index(dim, faiss::METRIC_L2), d(dim), 
              chunk_element_size(chunk_size), use_hugepages(try_hugepages) {
        chunk_shift = 0;
        size_t tmp = chunk_size;
        while (tmp > 1) {
            tmp >>= 1;
            chunk_shift++;
        }
        chunk_mask = chunk_element_size - 1;
        chunk_byte_size = chunk_element_size * d * sizeof(float);
        
        // Round up to hugepage size if using hugepages
        if (use_hugepages) {
            chunk_byte_size = ((chunk_byte_size + HUGEPAGE_SIZE - 1) / HUGEPAGE_SIZE) * HUGEPAGE_SIZE;
        }
    }
    
    ~IndexHugepageChunkedFlat() override {
        for (float* ptr : chunk_ptrs) {
            if (ptr) {
                munmap(ptr, chunk_byte_size);
            }
        }
    }

    float* allocate_chunk() {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        
        if (use_hugepages) {
            flags |= MAP_HUGETLB;
        }
        
        void* ptr = mmap(nullptr, chunk_byte_size, PROT_READ | PROT_WRITE, flags, -1, 0);
        
        if (ptr == MAP_FAILED) {
            if (use_hugepages) {
                // Fallback to regular pages
                flags = MAP_PRIVATE | MAP_ANONYMOUS;
                ptr = mmap(nullptr, chunk_byte_size, PROT_READ | PROT_WRITE, flags, -1, 0);
                if (ptr == MAP_FAILED) {
                    throw std::runtime_error("mmap failed");
                }
                use_hugepages = false;
            } else {
                throw std::runtime_error("mmap failed");
            }
        }
        
        return static_cast<float*>(ptr);
    }

    const float* get_vector(faiss::idx_t i) const {
        return chunk_ptrs[i >> chunk_shift] + (i & chunk_mask) * d;
    }

    float* get_vector_mutable(faiss::idx_t i) {
        return chunk_ptrs[i >> chunk_shift] + (i & chunk_mask) * d;
    }

    void add(faiss::idx_t n, const float* x) override {
        for (faiss::idx_t i = 0; i < n; i++) {
            faiss::idx_t global_idx = ntotal + i;
            size_t chunk_idx = global_idx >> chunk_shift;
            size_t internal_idx = global_idx & chunk_mask;
            
            if (unlikely(chunk_idx >= chunk_ptrs.size())) {
                chunk_ptrs.push_back(allocate_chunk());
            }
            
            std::memcpy(
                chunk_ptrs[chunk_idx] + internal_idx * d,
                x + i * d,
                d * sizeof(float));
        }
        ntotal += n;
    }

    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override {
#pragma omp parallel for
        for (faiss::idx_t q = 0; q < n; q++) {
            const float* query = x + q * d;
            std::vector<std::pair<float, faiss::idx_t>> dists(ntotal);
            
            for (faiss::idx_t i = 0; i < ntotal; i++) {
                dists[i] = {faiss::fvec_L2sqr(query, get_vector(i), d), i};
            }
            
            std::partial_sort(
                    dists.begin(), dists.begin() + k, dists.end(),
                    [](auto& a, auto& b) { return a.first < b.first; });
            
            for (faiss::idx_t i = 0; i < k; i++) {
                distances[q * k + i] = dists[i].first;
                labels[q * k + i] = dists[i].second;
            }
        }
    }

    void reset() override {
        for (float* ptr : chunk_ptrs) {
            if (ptr) {
                munmap(ptr, chunk_byte_size);
            }
        }
        chunk_ptrs.clear();
        ntotal = 0;
    }

    void reconstruct(faiss::idx_t key, float* recons) const override {
        std::memcpy(recons, get_vector(key), d * sizeof(float));
    }

    faiss::DistanceComputer* get_distance_computer() const override {
        return new HugepageFlatL2Dis(*this);
    }
};

HugepageFlatL2Dis::HugepageFlatL2Dis(const IndexHugepageChunkedFlat& s)
        : storage(s), d(s.d), chunk_shift(s.chunk_shift), 
          chunk_mask(s.chunk_mask), chunk_ptrs(s.chunk_ptrs.data()) {}

float HugepageFlatL2Dis::operator()(faiss::idx_t i) {
    return faiss::fvec_L2sqr(q, get_vec(i), d);
}

float HugepageFlatL2Dis::symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
    return faiss::fvec_L2sqr(get_vec(i), get_vec(j), d);
}

#endif // __linux__

//=============================================================================
// Custom Storage 5: Compressed Pointer Storage
// Uses 32-bit offsets instead of 64-bit pointers for better cache efficiency
//=============================================================================

struct IndexCompressedPtrFlat;

struct CompressedPtrFlatL2Dis : faiss::DistanceComputer {
    const IndexCompressedPtrFlat& storage;
    size_t d;
    const float* q = nullptr;
    
    const float* base_ptr;
    const uint32_t* offsets;

    explicit CompressedPtrFlatL2Dis(const IndexCompressedPtrFlat& s);

    void set_query(const float* x) override {
        q = x;
    }

    FAISS_ALWAYS_INLINE const float* get_vec(faiss::idx_t i) const {
        return base_ptr + offsets[i];
    }

    float operator()(faiss::idx_t i) override;

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override;
    
    void distances_batch_4(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        prefetch_L2(get_vec(idx0));
        prefetch_L2(get_vec(idx1));
        prefetch_L2(get_vec(idx2));
        prefetch_L2(get_vec(idx3));
        
        const float* v0 = get_vec(idx0);
        prefetch_L1(v0 + 64);
        
        const float* v1 = get_vec(idx1);
        prefetch_L1(v1 + 64);
        
        const float* v2 = get_vec(idx2);
        prefetch_L1(v2 + 64);
        
        const float* v3 = get_vec(idx3);
        prefetch_L1(v3 + 64);
        
        faiss::fvec_L2sqr_batch_4(q, v0, v1, v2, v3, d, dis0, dis1, dis2, dis3);
    }
};

struct IndexCompressedPtrFlat : faiss::Index {
    size_t d;
    std::vector<float> data;
    std::vector<uint32_t> offsets;

    explicit IndexCompressedPtrFlat(faiss::idx_t dim)
            : faiss::Index(dim, faiss::METRIC_L2), d(dim) {}

    const float* get_vector(faiss::idx_t i) const {
        return data.data() + offsets[i];
    }

    float* get_vector_mutable(faiss::idx_t i) {
        return data.data() + offsets[i];
    }

    void add(faiss::idx_t n, const float* x) override {
        size_t old_size = data.size();
        data.resize(old_size + n * d);
        std::memcpy(data.data() + old_size, x, n * d * sizeof(float));
        
        offsets.reserve(offsets.size() + n);
        for (faiss::idx_t i = 0; i < n; i++) {
            offsets.push_back(static_cast<uint32_t>(old_size + i * d));
        }
        ntotal += n;
    }

    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override {
#pragma omp parallel for
        for (faiss::idx_t q = 0; q < n; q++) {
            const float* query = x + q * d;
            std::vector<std::pair<float, faiss::idx_t>> dists(ntotal);
            
            for (faiss::idx_t i = 0; i < ntotal; i++) {
                dists[i] = {faiss::fvec_L2sqr(query, get_vector(i), d), i};
            }
            
            std::partial_sort(
                    dists.begin(), dists.begin() + k, dists.end(),
                    [](auto& a, auto& b) { return a.first < b.first; });
            
            for (faiss::idx_t i = 0; i < k; i++) {
                distances[q * k + i] = dists[i].first;
                labels[q * k + i] = dists[i].second;
            }
        }
    }

    void reset() override {
        data.clear();
        offsets.clear();
        ntotal = 0;
    }

    void reconstruct(faiss::idx_t key, float* recons) const override {
        std::memcpy(recons, get_vector(key), d * sizeof(float));
    }

    faiss::DistanceComputer* get_distance_computer() const override {
        return new CompressedPtrFlatL2Dis(*this);
    }
};

CompressedPtrFlatL2Dis::CompressedPtrFlatL2Dis(const IndexCompressedPtrFlat& s)
        : storage(s), d(s.d), base_ptr(s.data.data()), offsets(s.offsets.data()) {}

float CompressedPtrFlatL2Dis::operator()(faiss::idx_t i) {
    return faiss::fvec_L2sqr(q, get_vec(i), d);
}

float CompressedPtrFlatL2Dis::symmetric_dis(faiss::idx_t i, faiss::idx_t j) {
    return faiss::fvec_L2sqr(get_vec(i), get_vec(j), d);
}

//=============================================================================
// Benchmark Results
//=============================================================================

struct BenchmarkResult {
    std::string storage_type;
    double build_time_sec;
    double search_time_sec;
    double search_qps;
    double recall;
    size_t memory_kb;
};

void print_header() {
    std::cout << std::setw(25) << "Storage Type"
              << std::setw(14) << "Build(s)"
              << std::setw(14) << "Search(s)"
              << std::setw(14) << "QPS"
              << std::setw(12) << "Recall@10"
              << std::setw(14) << "Memory(MB)"
              << std::endl;
    std::cout << std::string(93, '-') << std::endl;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::setw(25) << r.storage_type
              << std::setw(14) << std::fixed << std::setprecision(3) << r.build_time_sec
              << std::setw(14) << std::fixed << std::setprecision(3) << r.search_time_sec
              << std::setw(14) << std::fixed << std::setprecision(1) << r.search_qps
              << std::setw(12) << std::fixed << std::setprecision(4) << r.recall
              << std::setw(14) << std::fixed << std::setprecision(2) 
              << (r.memory_kb / 1024.0)
              << std::endl;
}

//=============================================================================
// Benchmark functions
//=============================================================================

BenchmarkResult benchmark_hnsw_flat(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb,
        size_t nq,
        size_t d,
        size_t k,
        int M,
        int efConstruction,
        int efSearch) {
    BenchmarkResult result;
    result.storage_type = "IndexFlat (baseline)";

    size_t mem_before = get_memory_usage_kb();

    double t0 = get_time_sec();
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;

    result.memory_kb = get_memory_usage_kb() - mem_before;

    // Warmup
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.hnsw.efSearch = efSearch;
    index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

    // Benchmark
    double t2 = get_time_sec();
    index.search(nq, queries, k, distances.data(), labels.data());
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;
    result.recall = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

BenchmarkResult benchmark_hnsw_indirect(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb,
        size_t nq,
        size_t d,
        size_t k,
        int M,
        int efConstruction,
        int efSearch) {
    BenchmarkResult result;
    result.storage_type = "IndexIndirectFlat (vector<ptr>)";

    size_t mem_before = get_memory_usage_kb();

    double t0 = get_time_sec();
    
    // Create custom storage
    IndexIndirectFlat* storage = new IndexIndirectFlat(d);
    
    // Create HNSW with custom storage
    faiss::IndexHNSW index(storage, M);
    index.own_fields = true;  // HNSW will delete storage
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;

    result.memory_kb = get_memory_usage_kb() - mem_before;

    // Warmup
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.hnsw.efSearch = efSearch;
    index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

    // Benchmark
    double t2 = get_time_sec();
    index.search(nq, queries, k, distances.data(), labels.data());
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;
    result.recall = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

BenchmarkResult benchmark_hnsw_hashmap(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb,
        size_t nq,
        size_t d,
        size_t k,
        int M,
        int efConstruction,
        int efSearch) {
    BenchmarkResult result;
    result.storage_type = "IndexHashMapFlat (unordered_map)";

    size_t mem_before = get_memory_usage_kb();

    double t0 = get_time_sec();
    
    // Create custom storage
    IndexHashMapFlat* storage = new IndexHashMapFlat(d);
    
    // Create HNSW with custom storage
    faiss::IndexHNSW index(storage, M);
    index.own_fields = true;
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;

    result.memory_kb = get_memory_usage_kb() - mem_before;

    // Warmup
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.hnsw.efSearch = efSearch;
    index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

    // Benchmark
    double t2 = get_time_sec();
    index.search(nq, queries, k, distances.data(), labels.data());
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;
    result.recall = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

BenchmarkResult benchmark_hnsw_chunked(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb,
        size_t nq,
        size_t d,
        size_t k,
        int M,
        int efConstruction,
        int efSearch,
        size_t chunk_size) {
    BenchmarkResult result;
    result.storage_type = "IndexChunkedFlat (chunk=" + std::to_string(chunk_size) + ")";

    size_t mem_before = get_memory_usage_kb();

    double t0 = get_time_sec();
    
    IndexChunkedFlat* storage = new IndexChunkedFlat(d, chunk_size);
    
    faiss::IndexHNSW index(storage, M);
    index.own_fields = true;
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;

    result.memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.hnsw.efSearch = efSearch;
    index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

    double t2 = get_time_sec();
    index.search(nq, queries, k, distances.data(), labels.data());
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;
    result.recall = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

#ifdef __linux__
BenchmarkResult benchmark_hnsw_hugepage(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb,
        size_t nq,
        size_t d,
        size_t k,
        int M,
        int efConstruction,
        int efSearch,
        size_t chunk_size) {
    BenchmarkResult result;
    result.storage_type = "IndexHugepageChunked";

    size_t mem_before = get_memory_usage_kb();

    double t0 = get_time_sec();
    
    IndexHugepageChunkedFlat* storage = new IndexHugepageChunkedFlat(d, chunk_size, true);
    
    faiss::IndexHNSW index(storage, M);
    index.own_fields = true;
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;

    result.memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.hnsw.efSearch = efSearch;
    index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

    double t2 = get_time_sec();
    index.search(nq, queries, k, distances.data(), labels.data());
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;
    result.recall = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}
#endif

BenchmarkResult benchmark_hnsw_compressed_ptr(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb,
        size_t nq,
        size_t d,
        size_t k,
        int M,
        int efConstruction,
        int efSearch) {
    BenchmarkResult result;
    result.storage_type = "IndexCompressedPtr (32-bit)";

    size_t mem_before = get_memory_usage_kb();

    double t0 = get_time_sec();
    
    IndexCompressedPtrFlat* storage = new IndexCompressedPtrFlat(d);
    
    faiss::IndexHNSW index(storage, M);
    index.own_fields = true;
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;

    result.memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.hnsw.efSearch = efSearch;
    index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

    double t2 = get_time_sec();
    index.search(nq, queries, k, distances.data(), labels.data());
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;
    result.recall = compute_recall(labels.data(), ground_truth, nq, k);

    return result;
}

//=============================================================================
// Graph-reordered storage: BFS traversal for memory locality
//=============================================================================

std::vector<faiss::idx_t> generate_bfs_permutation(const faiss::HNSW& hnsw) {
    size_t ntotal = hnsw.levels.size();
    std::vector<faiss::idx_t> perm;
    perm.reserve(ntotal);
    
    std::vector<bool> visited(ntotal, false);
    std::deque<faiss::HNSW::storage_idx_t> bfs_queue;
    
    if (hnsw.entry_point >= 0) {
        bfs_queue.push_back(hnsw.entry_point);
        visited[hnsw.entry_point] = true;
    }
    
    while (!bfs_queue.empty()) {
        faiss::HNSW::storage_idx_t current = bfs_queue.front();
        bfs_queue.pop_front();
        perm.push_back(current);
        
        size_t begin, end;
        hnsw.neighbor_range(current, 0, &begin, &end);
        
        for (size_t j = begin; j < end; j++) {
            faiss::HNSW::storage_idx_t neighbor = hnsw.neighbors[j];
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

BenchmarkResult benchmark_hnsw_graph_reordered(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb,
        size_t nq,
        size_t d,
        size_t k,
        int M,
        int efConstruction,
        int efSearch) {
    BenchmarkResult result;
    result.storage_type = "IndexHNSW+GraphReorder";

    size_t mem_before = get_memory_usage_kb();

    double t0 = get_time_sec();
    
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    auto perm = generate_bfs_permutation(index.hnsw);
    index.permute_entries(perm.data());
    
    double t1 = get_time_sec();
    result.build_time_sec = t1 - t0;

    result.memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.hnsw.efSearch = efSearch;
    index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

    double t2 = get_time_sec();
    index.search(nq, queries, k, distances.data(), labels.data());
    double t3 = get_time_sec();

    result.search_time_sec = t3 - t2;
    result.search_qps = nq / result.search_time_sec;
    
    std::vector<faiss::idx_t> inverse_perm(nb);
    for (size_t i = 0; i < nb; i++) {
        inverse_perm[perm[i]] = i;
    }
    
    std::vector<faiss::idx_t> remapped_gt(nq * k);
    for (size_t i = 0; i < nq * k; i++) {
        remapped_gt[i] = inverse_perm[ground_truth[i]];
    }
    
    result.recall = compute_recall(labels.data(), remapped_gt.data(), nq, k);

    return result;
}

} // anonymous namespace

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[]) {
    // Default parameters
    size_t nb = 500000;   // 500K vectors
    size_t d = 768;       // 768 dimensions
    size_t nq = 1000;     // 1000 queries
    size_t k = 10;        // top-10
    int M = 32;           // HNSW M parameter
    int efConstruction = 40;
    int efSearch = 64;

    // Parse arguments
    if (argc > 1) nb = std::atoi(argv[1]);
    if (argc > 2) d = std::atoi(argv[2]);
    if (argc > 3) nq = std::atoi(argv[3]);
    if (argc > 4) efSearch = std::atoi(argv[4]);
    
    std::string mode = "all";
    if (argc > 5) mode = argv[5];

    std::cout << "================================================================" << std::endl;
    std::cout << "  HNSW Storage Backend Benchmark" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Database size: " << nb << " vectors" << std::endl;
    std::cout << "  Dimension: " << d << std::endl;
    std::cout << "  Queries: " << nq << std::endl;
    std::cout << "  k: " << k << std::endl;
    std::cout << "  HNSW M: " << M << std::endl;
    std::cout << "  HNSW efConstruction: " << efConstruction << std::endl;
    std::cout << "  HNSW efSearch: " << efSearch << std::endl;
    std::cout << "  Threads: " << omp_get_max_threads() << std::endl;
    std::cout << std::endl;

    // Generate data
    std::cout << "Generating " << nb << " random vectors of dimension " << d << "..." << std::endl;
    auto database = generate_random_vectors(nb, d, 42);
    
    std::cout << "Generating " << nq << " query vectors..." << std::endl;
    auto queries = generate_random_vectors(nq, d, 123);

    std::cout << "Computing ground truth (this may take a while)..." << std::endl;
    auto ground_truth = compute_ground_truth(
            queries.data(), database.data(), nq, nb, d, k);
    
    std::cout << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "  Results" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    print_header();

    BenchmarkResult result_flat, result_chunked;

    if (mode == "all" || mode == "flat") {
        std::cout << "Running benchmark: IndexFlat..." << std::endl;
        result_flat = benchmark_hnsw_flat(
                database.data(), queries.data(), ground_truth.data(),
                nb, nq, d, k, M, efConstruction, efSearch);
        print_result(result_flat);
    }

    if (mode == "all") {
        std::cout << "Running benchmark: IndexIndirectFlat..." << std::endl;
        auto result_indirect = benchmark_hnsw_indirect(
                database.data(), queries.data(), ground_truth.data(),
                nb, nq, d, k, M, efConstruction, efSearch);
        print_result(result_indirect);

        std::cout << "Running benchmark: IndexHashMapFlat..." << std::endl;
        auto result_hashmap = benchmark_hnsw_hashmap(
                database.data(), queries.data(), ground_truth.data(),
                nb, nq, d, k, M, efConstruction, efSearch);
        print_result(result_hashmap);
    }

    if (mode == "all" || mode == "chunked") {
        std::vector<size_t> chunk_sizes = {16384};
        for (size_t chunk_size : chunk_sizes) {
            std::cout << "Running benchmark: IndexChunkedFlat (chunk=" << chunk_size << ")..." << std::endl;
            result_chunked = benchmark_hnsw_chunked(
                    database.data(), queries.data(), ground_truth.data(),
                    nb, nq, d, k, M, efConstruction, efSearch, chunk_size);
            print_result(result_chunked);
        }
    }

#ifdef __linux__
    if (mode == "all" || mode == "hugepage") {
        std::cout << "Running benchmark: IndexHugepageChunked..." << std::endl;
        auto result_hugepage = benchmark_hnsw_hugepage(
                database.data(), queries.data(), ground_truth.data(),
                nb, nq, d, k, M, efConstruction, efSearch, 16384);
        print_result(result_hugepage);
    }
#endif

    if (mode == "all" || mode == "compressed") {
        std::cout << "Running benchmark: IndexCompressedPtr..." << std::endl;
        auto result_compressed = benchmark_hnsw_compressed_ptr(
                database.data(), queries.data(), ground_truth.data(),
                nb, nq, d, k, M, efConstruction, efSearch);
        print_result(result_compressed);
    }

    if (mode == "all" || mode == "reorder") {
        std::cout << "Running benchmark: IndexHNSW+GraphReorder..." << std::endl;
        auto result_reorder = benchmark_hnsw_graph_reordered(
                database.data(), queries.data(), ground_truth.data(),
                nb, nq, d, k, M, efConstruction, efSearch);
        print_result(result_reorder);
    }

    if (mode == "flat" || mode == "chunked" || mode == "hugepage" || mode == "compressed" || mode == "reorder") {
        std::cout << "\nBenchmark complete (mode=" << mode << ")." << std::endl;
        return 0;
    }

    // Summary
    std::cout << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "  Performance Comparison (relative to baseline)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Benchmark complete." << std::endl;

    return 0;
}
