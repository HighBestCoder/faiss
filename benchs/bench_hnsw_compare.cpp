/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/prefetch.h>

#include <omp.h>

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

#ifdef __linux__
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#ifdef ENABLE_HDF5
#include <hdf5.h>
#endif

#ifdef ENABLE_VSAG
#include <vsag/vsag.h>
#endif

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

namespace {

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
        std::vector<faiss::idx_t> gt_set(ground_truth + q * k, ground_truth + (q + 1) * k);
        std::sort(gt_set.begin(), gt_set.end());
        for (size_t i = 0; i < k; i++) {
            if (std::binary_search(gt_set.begin(), gt_set.end(), results[q * k + i])) {
                correct++;
            }
        }
    }
    return static_cast<double>(correct) / (nq * k);
}

struct BenchmarkResult {
    std::string name;
    int ef_search;
    double build_time_sec;
    double search_time_sec;
    double search_qps;
    double recall;
    size_t memory_kb;
};

void print_header() {
    std::cout << std::setw(35) << std::left << "Index Type"
              << std::setw(10) << std::right << "efSearch"
              << std::setw(12) << "Build(s)"
              << std::setw(12) << "Search(s)"
              << std::setw(12) << "QPS"
              << std::setw(12) << "Recall@10"
              << std::setw(12) << "Memory(MB)" << std::endl;
    std::cout << std::string(105, '-') << std::endl;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::setw(35) << std::left << r.name
              << std::setw(10) << std::right << r.ef_search
              << std::setw(12) << std::fixed << std::setprecision(2) << r.build_time_sec
              << std::setw(12) << std::setprecision(3) << r.search_time_sec
              << std::setw(12) << std::setprecision(1) << r.search_qps
              << std::setw(12) << std::setprecision(4) << r.recall
              << std::setw(12) << std::setprecision(1) << r.memory_kb / 1024.0
              << std::endl;
}

#ifdef ENABLE_HDF5
struct HDF5Dataset {
    std::vector<float> train;
    std::vector<float> test;
    std::vector<int32_t> neighbors;
    size_t nb;
    size_t nq;
    size_t dim;
    size_t gt_k;
};

bool load_hdf5_dataset(const std::string& filepath, HDF5Dataset& dataset) {
    hid_t file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Failed to open HDF5 file: " << filepath << std::endl;
        return false;
    }

    auto read_dataset = [&](const char* name, auto& vec, hid_t type) -> std::pair<size_t, size_t> {
        hid_t dset = H5Dopen2(file_id, name, H5P_DEFAULT);
        if (dset < 0) return {0, 0};
        
        hid_t space = H5Dget_space(dset);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(space, dims, nullptr);
        
        vec.resize(dims[0] * dims[1]);
        H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
        
        H5Sclose(space);
        H5Dclose(dset);
        return {dims[0], dims[1]};
    };

    auto [nb, dim] = read_dataset("train", dataset.train, H5T_NATIVE_FLOAT);
    auto [nq, dim2] = read_dataset("test", dataset.test, H5T_NATIVE_FLOAT);
    auto [nq2, gt_k] = read_dataset("neighbors", dataset.neighbors, H5T_NATIVE_INT32);

    dataset.nb = nb;
    dataset.nq = nq;
    dataset.dim = dim;
    dataset.gt_k = gt_k;

    H5Fclose(file_id);
    
    std::cout << "Loaded HDF5 dataset: " << filepath << std::endl;
    std::cout << "  Train: " << nb << " x " << dim << std::endl;
    std::cout << "  Test: " << nq << " x " << dim2 << std::endl;
    std::cout << "  Ground truth: " << nq2 << " x " << gt_k << std::endl;
    
    return true;
}
#endif

class IndexCacheAlignedFlat : public faiss::Index {
public:
    size_t d;
    size_t vector_size_bytes;
    size_t aligned_vector_size;
    std::vector<float*> vectors;
    
    explicit IndexCacheAlignedFlat(faiss::idx_t dim)
            : faiss::Index(dim, faiss::METRIC_L2), d(dim) {
        vector_size_bytes = d * sizeof(float);
        aligned_vector_size = ((vector_size_bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
    }
    
    ~IndexCacheAlignedFlat() override {
        for (float* ptr : vectors) {
            if (ptr) {
                free(ptr);
            }
        }
    }
    
    const float* get_vector(faiss::idx_t i) const {
        return vectors[i];
    }
    
    void add(faiss::idx_t n, const float* x) override {
        for (faiss::idx_t i = 0; i < n; i++) {
            float* aligned_ptr = static_cast<float*>(aligned_alloc(CACHE_LINE_SIZE, aligned_vector_size));
            std::memcpy(aligned_ptr, x + i * d, d * sizeof(float));
            vectors.push_back(aligned_ptr);
        }
        ntotal += n;
    }
    
    void search(
            faiss::idx_t,
            const float*,
            faiss::idx_t,
            float*,
            faiss::idx_t*,
            const faiss::SearchParameters* = nullptr) const override {
    }
    
    void reset() override {
        for (float* ptr : vectors) {
            free(ptr);
        }
        vectors.clear();
        ntotal = 0;
    }
    
    faiss::DistanceComputer* get_distance_computer() const override;
};

struct CacheAlignedFlatL2Dis : faiss::DistanceComputer {
    const IndexCacheAlignedFlat& storage;
    size_t d;
    std::vector<float> q_copy;

    explicit CacheAlignedFlatL2Dis(const IndexCacheAlignedFlat& s)
            : storage(s), d(s.d), q_copy(s.d) {}

    void set_query(const float* x) override {
        std::memcpy(q_copy.data(), x, d * sizeof(float));
    }

    float operator()(faiss::idx_t i) override {
        return faiss::fvec_L2sqr(q_copy.data(), storage.get_vector(i), d);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        return faiss::fvec_L2sqr(storage.get_vector(i), storage.get_vector(j), d);
    }
    
    void distances_batch_4(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        faiss::fvec_L2sqr_batch_4(
                q_copy.data(),
                storage.get_vector(idx0),
                storage.get_vector(idx1),
                storage.get_vector(idx2),
                storage.get_vector(idx3),
                d,
                dis0, dis1, dis2, dis3);
    }

    void distances_batch_8(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            const faiss::idx_t idx4,
            const faiss::idx_t idx5,
            const faiss::idx_t idx6,
            const faiss::idx_t idx7,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3,
            float& dis4,
            float& dis5,
            float& dis6,
            float& dis7) override {
        faiss::fvec_L2sqr_batch_8(
                q_copy.data(),
                storage.get_vector(idx0),
                storage.get_vector(idx1),
                storage.get_vector(idx2),
                storage.get_vector(idx3),
                storage.get_vector(idx4),
                storage.get_vector(idx5),
                storage.get_vector(idx6),
                storage.get_vector(idx7),
                d,
                dis0, dis1, dis2, dis3, dis4, dis5, dis6, dis7);
    }
};

faiss::DistanceComputer* IndexCacheAlignedFlat::get_distance_computer() const {
    return new CacheAlignedFlatL2Dis(*this);
}

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

    float operator()(faiss::idx_t i) override {
        return faiss::fvec_L2sqr(q, get_vec(i), d);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        return faiss::fvec_L2sqr(get_vec(i), get_vec(j), d);
    }
    
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

    void distances_batch_8(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            const faiss::idx_t idx4,
            const faiss::idx_t idx5,
            const faiss::idx_t idx6,
            const faiss::idx_t idx7,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3,
            float& dis4,
            float& dis5,
            float& dis6,
            float& dis7) override {
        prefetch_L2(get_vec(idx0));
        prefetch_L2(get_vec(idx1));
        prefetch_L2(get_vec(idx2));
        prefetch_L2(get_vec(idx3));
        prefetch_L2(get_vec(idx4));
        prefetch_L2(get_vec(idx5));
        prefetch_L2(get_vec(idx6));
        prefetch_L2(get_vec(idx7));
        
        const float* v0 = get_vec(idx0);
        prefetch_L1(v0 + 64);
        const float* v1 = get_vec(idx1);
        prefetch_L1(v1 + 64);
        const float* v2 = get_vec(idx2);
        prefetch_L1(v2 + 64);
        const float* v3 = get_vec(idx3);
        prefetch_L1(v3 + 64);
        const float* v4 = get_vec(idx4);
        prefetch_L1(v4 + 64);
        const float* v5 = get_vec(idx5);
        prefetch_L1(v5 + 64);
        const float* v6 = get_vec(idx6);
        prefetch_L1(v6 + 64);
        const float* v7 = get_vec(idx7);
        prefetch_L1(v7 + 64);
        
        faiss::fvec_L2sqr_batch_8(
                q, v0, v1, v2, v3, v4, v5, v6, v7, d,
                dis0, dis1, dis2, dis3, dis4, dis5, dis6, dis7);
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
            faiss::idx_t,
            const float*,
            faiss::idx_t,
            float*,
            faiss::idx_t*,
            const faiss::SearchParameters* = nullptr) const override {
    }

    void reset() override {
        chunks.clear();
        chunk_ptrs.clear();
        ntotal = 0;
    }

    faiss::DistanceComputer* get_distance_computer() const override {
        return new ChunkedFlatL2Dis(*this);
    }
};

ChunkedFlatL2Dis::ChunkedFlatL2Dis(const IndexChunkedFlat& s)
        : storage(s), d(s.d), chunk_shift(s.chunk_shift), 
          chunk_mask(s.chunk_mask), chunk_ptrs(s.chunk_ptrs) {}

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

    float operator()(faiss::idx_t i) override {
        return faiss::fvec_L2sqr(q, get_vec(i), d);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        return faiss::fvec_L2sqr(get_vec(i), get_vec(j), d);
    }
    
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

    void distances_batch_8(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            const faiss::idx_t idx4,
            const faiss::idx_t idx5,
            const faiss::idx_t idx6,
            const faiss::idx_t idx7,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3,
            float& dis4,
            float& dis5,
            float& dis6,
            float& dis7) override {
        prefetch_L2(get_vec(idx0));
        prefetch_L2(get_vec(idx1));
        prefetch_L2(get_vec(idx2));
        prefetch_L2(get_vec(idx3));
        prefetch_L2(get_vec(idx4));
        prefetch_L2(get_vec(idx5));
        prefetch_L2(get_vec(idx6));
        prefetch_L2(get_vec(idx7));
        
        const float* v0 = get_vec(idx0);
        prefetch_L1(v0 + 64);
        const float* v1 = get_vec(idx1);
        prefetch_L1(v1 + 64);
        const float* v2 = get_vec(idx2);
        prefetch_L1(v2 + 64);
        const float* v3 = get_vec(idx3);
        prefetch_L1(v3 + 64);
        const float* v4 = get_vec(idx4);
        prefetch_L1(v4 + 64);
        const float* v5 = get_vec(idx5);
        prefetch_L1(v5 + 64);
        const float* v6 = get_vec(idx6);
        prefetch_L1(v6 + 64);
        const float* v7 = get_vec(idx7);
        prefetch_L1(v7 + 64);
        
        faiss::fvec_L2sqr_batch_8(
                q, v0, v1, v2, v3, v4, v5, v6, v7, d,
                dis0, dis1, dis2, dis3, dis4, dis5, dis6, dis7);
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

    static constexpr size_t HUGEPAGE_SIZE = 2 * 1024 * 1024;

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
            faiss::idx_t,
            const float*,
            faiss::idx_t,
            float*,
            faiss::idx_t*,
            const faiss::SearchParameters* = nullptr) const override {
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

    faiss::DistanceComputer* get_distance_computer() const override {
        return new HugepageFlatL2Dis(*this);
    }
};

HugepageFlatL2Dis::HugepageFlatL2Dis(const IndexHugepageChunkedFlat& s)
        : storage(s), d(s.d), chunk_shift(s.chunk_shift), 
          chunk_mask(s.chunk_mask), chunk_ptrs(s.chunk_ptrs.data()) {}

#endif // __linux__

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

std::vector<BenchmarkResult> benchmark_faiss_hnsw(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb, size_t nq, size_t d, size_t k,
        int M, int efConstruction, const std::vector<int>& efSearchValues) {
    std::vector<BenchmarkResult> results;

    size_t mem_before = get_memory_usage_kb();
    double t0 = get_time_sec();
    
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    double build_time = t1 - t0;
    size_t memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    for (int efSearch : efSearchValues) {
        BenchmarkResult result;
        result.name = "FAISS IndexHNSWFlat";
        result.ef_search = efSearch;
        result.build_time_sec = build_time;
        result.memory_kb = memory_kb;

        index.hnsw.efSearch = efSearch;
        index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

        double t2 = get_time_sec();
        index.search(nq, queries, k, distances.data(), labels.data());
        double t3 = get_time_sec();

        result.search_time_sec = t3 - t2;
        result.search_qps = nq / result.search_time_sec;
        result.recall = compute_recall(labels.data(), ground_truth, nq, k);
        results.push_back(result);
    }

    return results;
}

std::vector<BenchmarkResult> benchmark_faiss_cache_aligned(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb, size_t nq, size_t d, size_t k,
        int M, int efConstruction, const std::vector<int>& efSearchValues) {
    std::vector<BenchmarkResult> results;

    size_t mem_before = get_memory_usage_kb();
    double t0 = get_time_sec();
    
    IndexCacheAlignedFlat* storage = new IndexCacheAlignedFlat(d);
    faiss::IndexHNSW index(storage, M);
    index.own_fields = true;
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    double build_time = t1 - t0;
    size_t memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    for (int efSearch : efSearchValues) {
        BenchmarkResult result;
        result.name = "FAISS CacheAligned";
        result.ef_search = efSearch;
        result.build_time_sec = build_time;
        result.memory_kb = memory_kb;

        index.hnsw.efSearch = efSearch;
        index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

        double t2 = get_time_sec();
        index.search(nq, queries, k, distances.data(), labels.data());
        double t3 = get_time_sec();

        result.search_time_sec = t3 - t2;
        result.search_qps = nq / result.search_time_sec;
        result.recall = compute_recall(labels.data(), ground_truth, nq, k);
        results.push_back(result);
    }

    return results;
}

std::vector<BenchmarkResult> benchmark_faiss_graph_reorder(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb, size_t nq, size_t d, size_t k,
        int M, int efConstruction, const std::vector<int>& efSearchValues) {
    std::vector<BenchmarkResult> results;

    size_t mem_before = get_memory_usage_kb();
    double t0 = get_time_sec();
    
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    auto perm = generate_bfs_permutation(index.hnsw);
    index.permute_entries(perm.data());
    
    double t1 = get_time_sec();
    double build_time = t1 - t0;
    size_t memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<faiss::idx_t> inverse_perm(nb);
    for (size_t i = 0; i < nb; i++) {
        inverse_perm[perm[i]] = i;
    }
    std::vector<faiss::idx_t> remapped_gt(nq * k);
    for (size_t i = 0; i < nq * k; i++) {
        remapped_gt[i] = inverse_perm[ground_truth[i]];
    }

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    for (int efSearch : efSearchValues) {
        BenchmarkResult result;
        result.name = "FAISS GraphReorder";
        result.ef_search = efSearch;
        result.build_time_sec = build_time;
        result.memory_kb = memory_kb;

        index.hnsw.efSearch = efSearch;
        index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

        double t2 = get_time_sec();
        index.search(nq, queries, k, distances.data(), labels.data());
        double t3 = get_time_sec();

        result.search_time_sec = t3 - t2;
        result.search_qps = nq / result.search_time_sec;
        result.recall = compute_recall(labels.data(), remapped_gt.data(), nq, k);
        results.push_back(result);
    }

    return results;
}

std::vector<BenchmarkResult> benchmark_faiss_chunked(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb, size_t nq, size_t d, size_t k,
        int M, int efConstruction, const std::vector<int>& efSearchValues) {
    std::vector<BenchmarkResult> results;

    size_t mem_before = get_memory_usage_kb();
    double t0 = get_time_sec();
    
    IndexChunkedFlat* storage = new IndexChunkedFlat(d, 1024);
    faiss::IndexHNSW index(storage, M);
    index.own_fields = true;
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    double build_time = t1 - t0;
    size_t memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    for (int efSearch : efSearchValues) {
        BenchmarkResult result;
        result.name = "FAISS Chunked";
        result.ef_search = efSearch;
        result.build_time_sec = build_time;
        result.memory_kb = memory_kb;

        index.hnsw.efSearch = efSearch;
        index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

        double t2 = get_time_sec();
        index.search(nq, queries, k, distances.data(), labels.data());
        double t3 = get_time_sec();

        result.search_time_sec = t3 - t2;
        result.search_qps = nq / result.search_time_sec;
        result.recall = compute_recall(labels.data(), ground_truth, nq, k);
        results.push_back(result);
    }

    return results;
}

#ifdef __linux__
std::vector<BenchmarkResult> benchmark_faiss_hugepage(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb, size_t nq, size_t d, size_t k,
        int M, int efConstruction, const std::vector<int>& efSearchValues) {
    std::vector<BenchmarkResult> results;

    size_t mem_before = get_memory_usage_kb();
    double t0 = get_time_sec();
    
    IndexHugepageChunkedFlat* storage = new IndexHugepageChunkedFlat(d, 16384, true);
    faiss::IndexHNSW index(storage, M);
    index.own_fields = true;
    index.hnsw.efConstruction = efConstruction;
    index.add(nb, data);
    
    double t1 = get_time_sec();
    double build_time = t1 - t0;
    size_t memory_kb = get_memory_usage_kb() - mem_before;

    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);

    for (int efSearch : efSearchValues) {
        BenchmarkResult result;
        result.name = "FAISS Hugepage";
        result.ef_search = efSearch;
        result.build_time_sec = build_time;
        result.memory_kb = memory_kb;

        index.hnsw.efSearch = efSearch;
        index.search(std::min(nq, (size_t)100), queries, k, distances.data(), labels.data());

        double t2 = get_time_sec();
        index.search(nq, queries, k, distances.data(), labels.data());
        double t3 = get_time_sec();

        result.search_time_sec = t3 - t2;
        result.search_qps = nq / result.search_time_sec;
        result.recall = compute_recall(labels.data(), ground_truth, nq, k);
        results.push_back(result);
    }

    return results;
}
#endif

#ifdef ENABLE_VSAG
std::vector<BenchmarkResult> benchmark_vsag_hgraph(
        const float* data,
        const float* queries,
        const faiss::idx_t* ground_truth,
        size_t nb, size_t nq, size_t d, size_t k,
        int M, int efConstruction, const std::vector<int>& efSearchValues) {
    std::vector<BenchmarkResult> results;

    size_t mem_before = get_memory_usage_kb();
    double t0 = get_time_sec();

    std::string build_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": )" + std::to_string(d) + R"(,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": )" + std::to_string(M) + R"(,
            "ef_construction": )" + std::to_string(efConstruction) + R"(,
            "alpha": 1.2
        }
    })";
    
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    
    auto index_result = engine.CreateIndex("hgraph", build_params);
    if (!index_result.has_value()) {
        std::cerr << "VSAG HGraph: Failed to create index: " << index_result.error().message << std::endl;
        BenchmarkResult fail_result;
        fail_result.name = "VSAG HGraph (FAILED)";
        results.push_back(fail_result);
        return results;
    }
    auto index = index_result.value();

    auto ids = new int64_t[nb];
    for (size_t i = 0; i < nb; i++) {
        ids[i] = i;
    }
    
    auto base = vsag::Dataset::Make();
    base->NumElements(nb)->Dim(d)->Ids(ids)->Float32Vectors(const_cast<float*>(data))->Owner(false);
    
    auto build_result = index->Build(base);
    if (!build_result.has_value()) {
        std::cerr << "VSAG HGraph: Failed to build index: " << build_result.error().message << std::endl;
        BenchmarkResult fail_result;
        fail_result.name = "VSAG HGraph (FAILED)";
        results.push_back(fail_result);
        delete[] ids;
        engine.Shutdown();
        return results;
    }
    
    double t1 = get_time_sec();
    double build_time = t1 - t0;
    size_t memory_kb = get_memory_usage_kb() - mem_before;

    for (int efSearch : efSearchValues) {
        BenchmarkResult result;
        result.name = "VSAG HGraph";
        result.ef_search = efSearch;
        result.build_time_sec = build_time;
        result.memory_kb = memory_kb;

        std::string search_params = R"({
            "hgraph": {
                "ef_search": )" + std::to_string(efSearch) + R"(
            }
        })";

        std::vector<int64_t> all_results(nq * k);
        
        auto warmup_query = vsag::Dataset::Make();
        warmup_query->NumElements(1)->Dim(d)->Float32Vectors(const_cast<float*>(queries))->Owner(false);
        index->KnnSearch(warmup_query, k, search_params);

        double t2 = get_time_sec();
        
        for (size_t q = 0; q < nq; q++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(d)->Float32Vectors(const_cast<float*>(queries + q * d))->Owner(false);
            auto knn_result = index->KnnSearch(query, k, search_params);
            if (knn_result.has_value()) {
                auto res = knn_result.value();
                for (size_t i = 0; i < k && i < res->GetDim(); i++) {
                    all_results[q * k + i] = res->GetIds()[i];
                }
            }
        }
        
        double t3 = get_time_sec();

        result.search_time_sec = t3 - t2;
        result.search_qps = nq / result.search_time_sec;

        size_t correct = 0;
        for (size_t q = 0; q < nq; q++) {
            std::vector<faiss::idx_t> gt_set(ground_truth + q * k, ground_truth + (q + 1) * k);
            std::sort(gt_set.begin(), gt_set.end());
            for (size_t i = 0; i < k; i++) {
                if (std::binary_search(gt_set.begin(), gt_set.end(), all_results[q * k + i])) {
                    correct++;
                }
            }
        }
        result.recall = static_cast<double>(correct) / (nq * k);
        results.push_back(result);
    }

    delete[] ids;
    engine.Shutdown();
    return results;
}
#endif

}

int main(int argc, char* argv[]) {
    size_t nb = 100000;
    size_t d = 128;
    size_t nq = 2000;
    size_t k = 10;
    int M = 32;
    int efConstruction = 200;
    std::vector<int> efSearchValues = {50, 100, 200, 400};
    std::string hdf5_path;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find(".hdf5") != std::string::npos || arg.find(".h5") != std::string::npos) {
            hdf5_path = arg;
        } else if (arg == "-nb" && i + 1 < argc) {
            nb = std::atoi(argv[++i]);
        } else if (arg == "-d" && i + 1 < argc) {
            d = std::atoi(argv[++i]);
        } else if (arg == "-nq" && i + 1 < argc) {
            nq = std::atoi(argv[++i]);
        } else if (arg == "-M" && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (arg == "-efConstruction" && i + 1 < argc) {
            efConstruction = std::atoi(argv[++i]);
        } else if (arg == "-efSearch" && i + 1 < argc) {
            // Parse comma-separated efSearch values: e.g., "50,100,200,400"
            efSearchValues.clear();
            std::string vals = argv[++i];
            size_t pos = 0;
            while ((pos = vals.find(',')) != std::string::npos) {
                efSearchValues.push_back(std::atoi(vals.substr(0, pos).c_str()));
                vals.erase(0, pos + 1);
            }
            efSearchValues.push_back(std::atoi(vals.c_str()));
        }
    }

    std::vector<float> database;
    std::vector<float> queries;
    std::vector<faiss::idx_t> ground_truth;

#ifdef ENABLE_HDF5
    if (!hdf5_path.empty()) {
        HDF5Dataset dataset;
        if (!load_hdf5_dataset(hdf5_path, dataset)) {
            std::cerr << "Failed to load HDF5 dataset" << std::endl;
            return 1;
        }
        nb = dataset.nb;
        nq = dataset.nq;
        d = dataset.dim;
        database = std::move(dataset.train);
        queries = std::move(dataset.test);
        
        // Convert int32 neighbors to idx_t (HDF5 has gt_k neighbors per query, we need k)
        ground_truth.resize(nq * k);
        for (size_t q = 0; q < nq; q++) {
            for (size_t i = 0; i < k; i++) {
                ground_truth[q * k + i] = static_cast<faiss::idx_t>(dataset.neighbors[q * dataset.gt_k + i]);
            }
        }
    } else
#endif
    {
        std::cout << "Generating " << nb << " random vectors of dimension " << d << "..." << std::endl;
        database = generate_random_vectors(nb, d, 42);
        
        std::cout << "Generating " << nq << " query vectors..." << std::endl;
        queries = generate_random_vectors(nq, d, 123);
        
        std::cout << "Computing ground truth..." << std::endl;
        ground_truth = compute_ground_truth(queries.data(), database.data(), nq, nb, d, k);
    }

    std::cout << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "  FAISS vs VSAG HGraph Comparison Benchmark" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Database size: " << nb << " vectors" << std::endl;
    std::cout << "  Dimension: " << d << std::endl;
    std::cout << "  Queries: " << nq << std::endl;
    std::cout << "  k: " << k << std::endl;
    std::cout << "  HNSW M: " << M << std::endl;
    std::cout << "  HNSW efConstruction: " << efConstruction << std::endl;
    std::cout << "  efSearch values: ";
    for (size_t i = 0; i < efSearchValues.size(); i++) {
        std::cout << efSearchValues[i];
        if (i < efSearchValues.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "  Threads: " << omp_get_max_threads() << std::endl;
#ifdef ENABLE_VSAG
    std::cout << "  VSAG: ENABLED" << std::endl;
#else
    std::cout << "  VSAG: DISABLED (compile with -DENABLE_VSAG)" << std::endl;
#endif
#ifdef ENABLE_HDF5
    std::cout << "  HDF5: ENABLED" << std::endl;
#else
    std::cout << "  HDF5: DISABLED (compile with -DENABLE_HDF5)" << std::endl;
#endif
    std::cout << std::endl;

    std::cout << "================================================================" << std::endl;
    std::cout << "  Results" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    print_header();

    std::vector<BenchmarkResult> all_results;

    std::cout << "Building and benchmarking FAISS IndexHNSWFlat..." << std::endl;
    auto results_faiss = benchmark_faiss_hnsw(
            database.data(), queries.data(), ground_truth.data(),
            nb, nq, d, k, M, efConstruction, efSearchValues);
    for (const auto& r : results_faiss) {
        print_result(r);
        all_results.push_back(r);
    }
    std::cout << std::string(105, '-') << std::endl;

    std::cout << "Building and benchmarking FAISS CacheAligned..." << std::endl;
    auto results_aligned = benchmark_faiss_cache_aligned(
            database.data(), queries.data(), ground_truth.data(),
            nb, nq, d, k, M, efConstruction, efSearchValues);
    for (const auto& r : results_aligned) {
        print_result(r);
        all_results.push_back(r);
    }
    std::cout << std::string(105, '-') << std::endl;

    std::cout << "Building and benchmarking FAISS GraphReorder..." << std::endl;
    auto results_reorder = benchmark_faiss_graph_reorder(
            database.data(), queries.data(), ground_truth.data(),
            nb, nq, d, k, M, efConstruction, efSearchValues);
    for (const auto& r : results_reorder) {
        print_result(r);
        all_results.push_back(r);
    }
    std::cout << std::string(105, '-') << std::endl;

    std::cout << "Building and benchmarking FAISS Chunked..." << std::endl;
    auto results_chunked = benchmark_faiss_chunked(
            database.data(), queries.data(), ground_truth.data(),
            nb, nq, d, k, M, efConstruction, efSearchValues);
    for (const auto& r : results_chunked) {
        print_result(r);
        all_results.push_back(r);
    }
    std::cout << std::string(105, '-') << std::endl;

#ifdef __linux__
    std::cout << "Building and benchmarking FAISS Hugepage..." << std::endl;
    auto results_hugepage = benchmark_faiss_hugepage(
            database.data(), queries.data(), ground_truth.data(),
            nb, nq, d, k, M, efConstruction, efSearchValues);
    for (const auto& r : results_hugepage) {
        print_result(r);
        all_results.push_back(r);
    }
    std::cout << std::string(105, '-') << std::endl;
#endif

#ifdef ENABLE_VSAG
    std::cout << "Building and benchmarking VSAG HGraph..." << std::endl;
    auto results_vsag = benchmark_vsag_hgraph(
            database.data(), queries.data(), ground_truth.data(),
            nb, nq, d, k, M, efConstruction, efSearchValues);
    for (const auto& r : results_vsag) {
        print_result(r);
        all_results.push_back(r);
    }
    std::cout << std::string(105, '-') << std::endl;
#endif

    std::cout << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "  Summary (Recall-QPS Table)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    
    // Print summary table grouped by index type
    std::cout << std::setw(35) << std::left << "Index Type"
              << std::setw(10) << std::right << "efSearch"
              << std::setw(12) << "Recall@10"
              << std::setw(12) << "QPS" << std::endl;
    std::cout << std::string(69, '-') << std::endl;
    
    for (const auto& r : all_results) {
        std::cout << std::setw(35) << std::left << r.name
                  << std::setw(10) << std::right << r.ef_search
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.recall
                  << std::setw(12) << std::setprecision(1) << r.search_qps
                  << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Benchmark complete." << std::endl;

    return 0;
}
