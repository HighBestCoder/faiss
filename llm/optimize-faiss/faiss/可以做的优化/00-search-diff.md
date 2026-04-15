# Dense Vector Search 路径对比分析：cortex.core vs beta

日期：2026-04-01
场景：vectordbbench 10M 向量 dense search
已知条件：M=32, efConstruction=512, efSearch=512，两者参数完全一致，都走 Search RPC

---

## 1. 调用图对比

### 1.1 beta: Search 调用图

```
gRPC VDSSServiceImpl::Search()          [comp/vdss/grpc/server/service_impl.cpp:428]
  |-- proto repeated float -> std::vector<float> 拷贝              // ALLOC #1
  |
  v
vde_cpp::vde_search()                   [comp/vde/src/api/vde_cpp_api.cpp:196]
  |-- 传 float* 指针（零拷贝）
  |
  v
VDECollection::Search()                 [comp/vde/src/api/vde_collection.cpp:620]
  |-- shared_lock(rw_mutex_)                                       // 读锁
  |-- SearchWithoutPredicateFilter(query, dim, top_k)
  |     |
  |     v
  |   VDECollection::SearchWithoutPredicateFilter()  [vde_collection.cpp:577]
  |     |-- std::vector<float> normalized_query(query, query+dim)  // ALLOC #2（始终拷贝）
  |     |-- NormalizeVector() 原地归一化
  |     |
  |     v
  |   FaissDriver::Search()             [faiss_driver.cpp:174]
  |     |-- alloc distances(top_k), labels(top_k)                  // ALLOC #3
  |     |-- index_->hnsw.efSearch = ef_search_                     // 直接写字段
  |     |-- index_->search(1, query, top_k, ...)                   // FAISS 调用（裸 IndexHNSWFlat）
  |     |    （不传 SearchParameters，params=nullptr）
  |     |-- id_map_[label] 重映射                                   // vector<uint64_t>
  |     |-- deleted_ids_.find() 过滤                                // std::set O(log n)
  |     |-- return vector<SearchResult>（按值返回）
  |
  |-- CollectionVectorId -> VectorIdentifier 映射                   // unordered_map
  |-- return vector<SearchResult>
```

**gRPC 到 FAISS 总层数：4**

### 1.2 cortex.core: Search 调用图

```
gRPC PointsImpl::Search()               [vde_grpc_service.cpp:2725]
  |-- proto repeated float -> VDEVector.data.assign()              // ALLOC #1
  |-- 构造 VDESearchParams { hnsw_ef = 512 }
  |
  v
vde_cpp::vde_search_with_params()        [vde_cpp_api.cpp:867]
  |-- 传 float* 指针（零拷贝）
  |-- 提取 hnsw_ef = 512
  |
  v
VDECollection::Search()                  [collection.cpp:1683]
  |-- shared_lock(rw_mutex_)                                       // 读锁
  |
  v
LocalShard::Search()                     [local_shard.cpp:684]
  |-- 纯转发（无拷贝、无锁）
  |
  v
Segment::Search()                        [segment.cpp:1402]
  |-- vector_data_map_.find(vector_name)                           // std::map 查找
  |-- （无 filter 时跳过 payload_store_->Filter()）
  |
  v
VectorData::Search()                     [vector_data.cpp:384]
  |-- （cosine 时）normalized.assign(query, query+dim)             // ALLOC #2（条件拷贝）
  |-- normalize_driver->Normalize() 原地归一化
  |
  v
FaissVectorIndexDriver::Search()         [faiss_vector_index_driver.cpp:375]
  |-- GetDeleteRatio() + overfetch 计算                             // 浮点计算
  |-- （无 delete 无 filter 时 search_k == top_k）
  |-- alloc distances(search_k), labels(search_k)                  // ALLOC #3
  |-- inner = idmap_->index                                        // 提取内部索引指针
  |-- 构造 SearchParametersHNSW { efSearch=512 }                    // 栈上对象
  |-- inner->search(1, query, search_k, ..., &hnsw_params)         // FAISS 调用（传 params）
  |    （传 SearchParameters → faiss 内部 3 次 dynamic_cast）
  |-- idmap_->id_map[label] 重映射循环                              // vector<idx_t>
  |-- deleted_ids_.contains() 过滤                                  // HybridIdSet O(1)
  |-- results 写入输出参数
  |
  <-- 回溯
Segment: id_tracker_->GetExternalId(iid) 重映射                     // unordered_map
```

**gRPC 到 FAISS 总层数：7**

---

## 2. Search 函数接口差异

| 方面 | beta FaissDriver::Search | cortex.core FaissVectorIndexDriver::Search |
|------|-------------------------|-------------------------------------------|
| **签名** | `Search(float*, dim, top_k) -> vector<SearchResult>` | `Search(float*, dim, top_k, optional<IdFilter>&, vector<ScoredInternalId>&, hnsw_ef) -> Result<Empty>` |
| **返回方式** | 按值返回 `vector<SearchResult>` | 写入输出参数 `results`，返回状态码 |
| **Filter 参数** | 无（仅内部 deleted_ids_ 检查） | 有 `optional<IdFilter>` 函数对象 |
| **efSearch 传递** | 直接写 `index_->hnsw.efSearch`，不传 params | 构造 `SearchParametersHNSW` 传入 |
| **Overfetch** | 无（精确请求 top_k 个结果） | 根据 delete_ratio 和 filter 动态 overfetch |
| **FAISS 调用目标** | `index_->search()`（裸 IndexHNSWFlat） | `idmap_->index->search()`（从 IndexIDMap2 提取内部索引） |
| **ID 映射** | 自定义 `vector<uint64_t> id_map_` | IndexIDMap2 内置的 `id_map`（`vector<idx_t>`） |
| **删除 ID 结构** | `std::set<uint64_t>` — O(log n) | `HybridIdSet` — O(1) bitmap |

---

## 3. FAISS 使用方式差异

### 3.1 索引包装

| 方面 | beta | cortex.core |
|------|------|-------------|
| 索引类型 | 裸 `faiss::IndexHNSWFlat`（无包装） | `faiss::IndexHNSWFlat` 包在 `faiss::IndexIDMap2` 里 |
| 搜索目标 | `index_->search()` 直接调用 | `idmap_->index->search()` 绕过 IndexIDMap2 |
| add 方式 | `index_->add()` 顺序 ID + 自定义 id_map_ | `idmap_->add_with_ids()` 触发 rev_map 插入 |
| 额外内存 | 仅 `vector<uint64_t> id_map_`（80 MB @ 10M） | IndexIDMap2 的 `id_map` + `rev_map`（见下文分析） |

### 3.2 efSearch 传递方式（关键差异）

**beta:**
```cpp
index_->hnsw.efSearch = ef_search_;      // 直接写字段
index_->search(1, query, top_k, distances, labels);  // 不传 params
```
→ faiss 内部：`params == nullptr`，不走 `dynamic_cast` 分支

**cortex.core:**
```cpp
faiss::SearchParametersHNSW hnsw_params;
hnsw_params.efSearch = 512;
inner->search(1, query, search_k, distances, labels, &hnsw_params);
```
→ faiss 内部：`params != nullptr`，触发 **3 次 `dynamic_cast<const SearchParametersHNSW*>`**：
  1. `hnsw_search()` [IndexHNSW.cpp:253]
  2. `HNSW::search()` [HNSW.cpp:951]
  3. `search_from_candidates()` [HNSW.cpp:610]

### 3.3 IndexIDMap2 的 rev_map 内存开销

`faiss::IndexIDMap2` 在每次 `add_with_ids` 时向 `rev_map` (unordered_map<idx_t, idx_t>) 插入条目。

10M vectors 时 rev_map 内存估算：
```
std::unordered_map 每个桶(bucket)：
  - key:    8 bytes (idx_t = int64_t)
  - value:  8 bytes (idx_t = int64_t)
  - hash:   隐含在桶数组中
  - node*:  8 bytes (链表指针)
  - 每个 node 额外 malloc header: ~16 bytes

每个条目实际开销 ≈ 40-56 bytes
10,000,000 × 48 bytes ≈ 480 MB
加上桶数组（~2倍条目数 × 8 bytes）≈ 160 MB
总计 ≈ 640 MB
```

这 640 MB 在搜索期间**完全不被访问**（搜索绕过了 IndexIDMap2），但占据物理内存，
导致：
- 增加 RSS，加重内存压力
- 增大页表，增加 TLB miss
- 与 HNSW 图数据竞争 LLC（最后一级缓存）

**beta 完全没有此开销** — 用的是裸索引 + `vector<uint64_t>` 前向映射。

---

## 4. 所有可能导致性能差异的点（完整列表）

### P0：IndexIDMap2 rev_map 内存压力（影响：高）

**问题：** cortex.core 用 `IndexIDMap2` 包装所有 FAISS 索引。10M 时 rev_map 占 ~640 MB 无用内存。
**影响机制：** 虽然搜索已绕过 IndexIDMap2，但 rev_map 的物理内存占用会：
- 压缩可用于 HNSW 图 + 向量存储的 LLC 空间
- 增加系统页表项数，引发更多 TLB miss
- 在内存紧张时可能引发 swap 或 OOM 压力

**beta 对比：** 无此问题，只有 `vector<uint64_t> id_map_`（~80 MB @ 10M）。

### P1：faiss 内部 3 次 dynamic_cast（影响：低-中）

**问题：** cortex.core 传递 `SearchParametersHNSW` 给 `inner->search()`，导致 faiss 内部在每次搜索中执行 3 次 `dynamic_cast<const SearchParametersHNSW*>`。
**beta：** 不传 params（`nullptr`），3 处都走 `if(params)` 的 false 分支，零 dynamic_cast。
**预计开销：** 每次 dynamic_cast ~10-30ns，3 次 = 30-90ns/query。在高 QPS 下可量测但不是主因。

### P2：额外的抽象层数（影响：低-中）

**cortex.core 比 beta 多 3 层调用：**
- `LocalShard::Search()` — 纯转发，~几 ns
- `Segment::Search()` — `std::map::find(vector_name)` 查找 + `id_tracker_->GetExternalId()` 映射
- `VectorData::Search()` — 条件归一化分支

**具体额外开销：**
1. `std::map<string, unique_ptr<VectorData>>::find(vector_name)` — 字符串比较，~50-100ns
2. `id_tracker_->GetExternalId(iid)` per result — `unordered_map` 查找，top_k 次，~10-20ns each
3. 更深的调用栈 → 可能降低指令缓存局部性

**总计：** ~200-500ns/query（beta 无此开销）

### P3：Overfetch 计算逻辑（影响：低）

**cortex.core：** 每次搜索计算 `GetDeleteRatio()` + 浮点乘法 + 条件分支。
**beta：** 无 overfetch — 在非 filter 搜索中精确请求 top_k。

当无 delete 且无 filter 时，cortex.core 的 `search_k == top_k`（不多不少），但计算本身仍需 ~10-20ns。

如果存在 soft delete（即使 1% = 100K），cortex.core 会 overfetch：
- `search_k = top_k * (1 + 0.01 * 3.0) = top_k * 1.03` — 影响极小
- 但 beta 在此场景下可能返回不足 top_k 个结果（无 overfetch 补偿）

### P4：id_map remap 循环开销差异（影响：低）

**cortex.core：**
```cpp
for (uint32_t j = 0; j < search_k; ++j) {    // 遍历所有 search_k 个结果
    if (labels[j] >= 0 && labels[j] < id_map_size) {
        labels[j] = id_map[labels[j]];        // 原地替换
    }
}
```
→ 遍历 **search_k** 个结果做 remap，然后再遍历一次做 post-filter。

**beta：**
```cpp
for (uint32_t i = 0; i < top_k; ++i) {        // 遍历 top_k 个结果
    if (labels[i] >= 0 && labels[i] < id_map_.size()) {
        uint64_t external_id = id_map_[labels[i]];
        if (deleted_ids_.find(external_id) == deleted_ids_.end()) {
            results.push_back(...);
        }
    }
}
```
→ 单次遍历 **top_k** 个结果，同时做 remap + filter。

cortex.core 的两遍遍历（remap 遍 + filter 遍）vs beta 的单遍遍历 — 差异在 top_k 较小（如 10-100）时几乎可忽略。

### P5：SearchResult 返回方式（影响：低）

**beta：** `vector<SearchResult>` 按值返回，触发 RVO（返回值优化）。
**cortex.core：** 写入输出参数 `vector<ScoredInternalId>& results`，然后在 Segment 层再构造 `vector<ScoredPoint>`。

cortex.core 多了一次 vector 分配（`ScoredInternalId -> ScoredPoint` 转换），但 top_k 通常很小（10-100），影响可忽略。

### P6：VDESearchResult -> proto 转换路径差异（影响：低）

**beta gRPC 结果构造 [service_impl.cpp:468-489]：**
- `std::string(vde_results[i].uuid)` — 字符串拷贝
- `proto_vector->add_data()` — 逐元素添加（if with_vector）

**cortex.core gRPC 结果构造 [vde_grpc_service.cpp:2820-2838]：**
- `result->mutable_id()->set_uuid(std::string(r.uuid))` — 同样字符串拷贝
- 同样的 proto 填充

两者基本等价，差异可忽略。

### P7：条件归一化 vs 无条件归一化（影响：cortex.core 有优势）

**beta：** `SearchWithoutPredicateFilter` 始终分配 `normalized_query` 并调用 `NormalizeVector()`，即使没有 normalizer（NormalizeVector 返回 0 但内存已分配）。
**cortex.core：** `VectorData::Search` 仅在 `normalize_driver != nullptr` 时才分配和拷贝。

这是 cortex.core 的优势——对于 EUCLIDEAN 距离不需要额外分配。

### P8：deleted_ids 查找效率（影响：cortex.core 有优势）

**beta：** `std::set<uint64_t>::find()` — O(log n)，红黑树遍历，缓存不友好。
**cortex.core：** `HybridIdSet::contains()` — O(1) bitmap 查找（ID < 100M），缓存友好。

10M 场景下这是 cortex.core 的明显优势。

### P9：hnsw_ef 参数传递的线程安全性（影响：无性能差异，但 beta 有隐患）

**beta：** `index_->hnsw.efSearch = ef_search_` 在每次搜索前直接写入，多线程并发写同一字段（data race）。由于写入的值始终相同，实际无害，但理论上是 UB。
**cortex.core：** 通过 `SearchParametersHNSW` 栈对象传递，线程安全，但引入 dynamic_cast 开销（见 P1）。

---

## 5. 综合评估：影响排序

| 排名 | 原因 | 影响等级 | 方向 | 是否可修复 |
|------|------|---------|------|-----------|
| **1** | **IndexIDMap2 rev_map ~640 MB 内存浪费**（10M 时）→ LLC/TLB 压力 | **高** | cortex.core 劣势 | 是 — 去掉 IndexIDMap2，改用裸索引 + vector id_map |
| **2** | **3 次 dynamic_cast / query**（SearchParametersHNSW） | **低-中** | cortex.core 劣势 | 是 — 改用直接写 efSearch 字段（像 beta） |
| **3** | **额外 3 层抽象** + map 查找 + id_tracker 映射 | **低-中** | cortex.core 劣势 | 架构性，不易改 |
| **4** | overfetch 计算逻辑 | **低** | cortex.core 劣势 | 可简化 |
| **5** | 两遍遍历 vs 单遍遍历 | **极低** | cortex.core 劣势 | 可合并 |
| **6** | deleted_ids O(1) vs O(log n) | **中** | cortex.core **优势** | — |
| **7** | 条件归一化（仅 cosine 时拷贝） | **低** | cortex.core **优势** | — |

---

## 6. 关键发现：VisitedTable 10MB 反复 malloc/free 问题

### 6.1 问题发现

**两者使用完全相同的 libfaiss.so**（MD5: `ee3ba67212249dfeeed6d26f62c7d63d`），
faiss 内部代码完全一致。但 cortex.core 使用 jemalloc（LD_PRELOAD），beta 使用 glibc malloc。

### 6.2 根因：faiss 的批量查询设计 vs 我们的单条查询调用

faiss 的 `hnsw_search` 函数是为**批量查询（nq >> 1）设计**的：

```cpp
// IndexHNSW.cpp:266-286 — faiss 的设计意图
#pragma omp parallel if (i1 - i0 > 1)    // 多个查询时开 OMP 并行
{
    VisitedTable vt(index->ntotal);       // 每个 OMP 线程分配一个（10M = 10MB）
    DistanceComputer* dis = ...;          // 每个 OMP 线程一个

    #pragma omp for
    for (idx_t i = i0; i < i1; i++) {     // 多个查询在线程内串行执行
        hnsw.search(*dis, res, vt, ...);  // 复用同一个 vt
    }
}
// ← vt 在 OMP 块结束时析构
```

**faiss 预期的高效用法：**
- 调用 `index->search(nq=1000, ...)` 一次传入 1000 个查询
- OMP 开 N 个线程，每个线程分配一个 VisitedTable
- 每个线程处理 `1000/N` 个查询，**复用同一个 VisitedTable**
- VisitedTable 有 `advance()` 优化：通过递增 `visno`（1→2→...→249）标记已访问节点，**免去 memset**
- 只有每 249 次查询才做一次 `memset(10MB, 0)`
- 结果：1000 个查询只需 N 次 malloc + ~4 次 memset

```cpp
// VisitedTable 的 advance() 优化 [AuxIndexStructures.h:190-197]
void advance() {
    visno++;                   // 递增版本号，旧标记自动失效
    if (visno == 250) {        // 每 249 次才真正清零
        memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
        visno = 1;
    }
}
```

**我们的实际调用方式（两个代码库都是）：**
- 每个 gRPC 请求调用 `inner->search(nq=1, ...)`
- `#pragma omp parallel if (1-0 > 1)` → **false**，不开并行
- VisitedTable 在主线程构造：**malloc(10MB) + memset(0, 10MB)**
- 单次搜索完毕，VisitedTable 析构：**free(10MB)**
- 下一个请求到来，**重新 malloc(10MB) + memset(0, 10MB) + free(10MB)**
- `advance()` 的免 memset 优化**完全失效**（每个 vt 只用一次就销毁）

### 6.3 每次搜索的 VisitedTable 开销估算

```
每次搜索：
  malloc(10MB)       → ~10-50μs（系统调用或 allocator 管理）
  memset(0, 10MB)    → ~500μs-1ms（受内存带宽限制，~10GB/s → 10MB/10GB/s = 1ms）
  [HNSW 图遍历]     → ~2-4ms（实际搜索计算）
  free(10MB)         → ~10-50μs

  VisitedTable 占总搜索时间：~15-30%
```

如果单次 HNSW search（efSearch=512, 10M vectors）总耗时约 3-5ms，
那 VisitedTable 的 malloc+memset+free 就占了 **0.5-1ms，即 10-30%**。

### 6.4 faiss 内部完整的每次搜索分配

| 分配 | 大小 | 来源 |
|------|------|------|
| `VisitedTable vt(ntotal)` | **10 MB** | `IndexHNSW.cpp:268` |
| `std::unique_ptr<DistanceComputer>` | ~几十字节 | `IndexHNSW.cpp:271` |
| `MinimaxHeap(ef=512)` | ~6 KB（ids+dis 各 512 × 4/8 bytes） | `HNSW.cpp:970` |

VisitedTable 的 10MB 是绝对主导。

### 6.5 为什么 jemalloc 会加重这个问题

虽然 VisitedTable 问题在两个代码库中都存在（同一个 libfaiss.so），
但 **allocator 不同导致 10MB alloc/free 的实际代价不同**：

**glibc malloc (beta)：**
- 10MB > M_MMAP_THRESHOLD（默认 128KB）→ 使用 `mmap()`
- 内核用 COW 零页映射（lazy allocation），实际物理页在 memset 触碰时通过 page fault 分配
- `free()` → `munmap()` 立即释放虚拟地址空间
- 但 glibc 有**动态 mmap 阈值调整**：频繁大分配会提高阈值，让更多分配走 arena
- 如果走 arena（sbrk），free 后内存保留在进程内，下次 malloc 可快速复用
- 关键点：**glibc 在高频大分配场景下可能自适应切换到 arena 模式**，避免频繁 mmap/munmap 系统调用

**jemalloc (cortex.core)：**
- 10MB 走 jemalloc 的 **large allocation** 路径（大于 large_maxclass 时走 extent/mmap）
- free 后进入 **dirty extent cache**，不立即 munmap
- 下次分配时可从 dirty cache 复用 extent（避免 mmap 系统调用）
- 但 jemalloc 有后台线程周期性 `madvise(MADV_DONTNEED)` 清理 dirty pages
- 如果清理后再分配，需要重新 page fault
- **jemalloc 的 extent 管理有锁开销**（全局 extent mutex），高并发搜索时可能成为瓶颈
- jemalloc 的 `dirty_decay_ms` 和 `muzzy_decay_ms` 配置影响 dirty page 保留时间

### 6.6 jemalloc 可能导致搜索性能下降的具体机制

**glibc malloc (beta)：**
- 10MB > M_MMAP_THRESHOLD（默认 128KB）→ 使用 `mmap()`
- 内核用 COW 零页映射（lazy allocation），实际物理页在 memset 触碰时通过 page fault 分配
- `free()` → `munmap()` 立即释放虚拟地址空间
- 但 glibc 有**动态 mmap 阈值调整**：频繁大分配会提高阈值，让更多分配走 arena
- 如果走 arena（sbrk），free 后内存保留在进程内，下次 malloc 可快速复用
- 关键点：**glibc 在高频大分配场景下可能自适应切换到 arena 模式**，避免频繁 mmap/munmap 系统调用

**jemalloc (cortex.core)：**
- 10MB 走 jemalloc 的 **large allocation** 路径（大于 large_maxclass 时走 extent/mmap）
- free 后进入 **dirty extent cache**，不立即 munmap
- 下次分配时可从 dirty cache 复用 extent（避免 mmap 系统调用）
- 但 jemalloc 有后台线程周期性 `madvise(MADV_DONTNEED)` 清理 dirty pages
- 如果清理后再分配，需要重新 page fault
- **jemalloc 的 extent 管理有锁开销**（全局 extent mutex），高并发搜索时可能成为瓶颈
- jemalloc 的 `dirty_decay_ms` 和 `muzzy_decay_ms` 配置影响 dirty page 保留时间

### jemalloc 可能导致搜索性能下降的具体机制

1. **10MB VisitedTable 的 extent 锁竞争**
   - 多个 gRPC 线程并发搜索，每个都要 malloc/free 10MB
   - jemalloc 的 large allocation 路径需要获取 extent 锁
   - 并发越高，锁竞争越严重
   - glibc 的 mmap/munmap 虽然也有内核锁，但在 arena 模式下可能更高效

2. **dirty page decay 造成的不确定性**
   - jemalloc 后台线程可能在搜索间隙 madvise 掉 VisitedTable 的页面
   - 下次 memset 时触发 page fault（~10MB / 4KB = 2500 次 page fault）
   - 每次 page fault ~1-5us → 2500 × 3us = ~7.5ms，这是灾难性的

3. **jemalloc 的 metadata 内存**
   - jemalloc 为每个 extent 维护 metadata
   - 10M 个 rev_map node（每个单独 malloc）产生大量 extent metadata
   - 这些 metadata 消耗额外内存并增加管理开销

4. **IndexIDMap2 rev_map 的 10M 次独立 malloc**
   - `unordered_map` 的 separate chaining 为每个 node 单独 `new`
   - 10M 个小分配（每个 ~32 bytes）在 jemalloc 中产生大量 tcache/bin 活动
   - 虽然这些分配发生在 build 阶段不影响 search，但它们占据的内存碎片
     会影响 jemalloc 的整体分配效率（bin fragmentation）

---

## 7. 核心结论

### 最关键的发现：nq=1 调用模式导致 VisitedTable 反复 malloc/free 10MB

**这是两个代码库共有的问题**（同一个 libfaiss.so），但 cortex.core 因为使用 jemalloc 而受到更大影响。

faiss 的 HNSW search 为批量查询设计（`nq >> 1`，一次传多个查询向量），
VisitedTable（`ntotal` 字节 = 10MB @ 10M vectors）本应在 OMP 线程内被多个查询**复用**，
`advance()` 通过递增 `visno` 实现免 memset 重置（每 249 次才真正 memset 一次）。

但我们以 `nq=1` 方式调用（每个 gRPC 请求一次 `index->search(1, ...)`），导致：
- OMP 并行不触发（`if(1 > 1)` = false）
- VisitedTable **每次搜索都新建、用一次、销毁**
- `advance()` 的免 memset 优化完全失效
- **每次搜索白白浪费 ~0.5-1ms 在 malloc(10MB) + memset(0, 10MB) + free(10MB) 上**
- 如果单次 HNSW search 总耗时 3-5ms，这就是 **10-30% 的无谓开销**

### 30% 回归的原因分解

| 原因 | 估计影响 | 说明 |
|------|---------|------|
| **jemalloc 对 10MB VisitedTable 的管理开销** | **10-20%** | extent 锁竞争 + dirty page decay 导致 page fault 波动，比 glibc arena 模式更慢 |
| **IndexIDMap2 rev_map ~614 MB 内存** | **5-10%** | 10M 独立 node 碎片化 + TLB 压力 + jemalloc 下碎片更严重 |
| **上层代码路径开销叠加** | **3-5%** | 3 次 dynamic_cast + 3 层额外抽象 + map 查找 + id_tracker 映射 |
| **jemalloc 对小分配的间接影响** | **2-5%** | distances/labels/results 等中小 vector 分配 |

---

## 8. gRPC 线程模型调研

### 8.1 gRPC C++ Sync Server 的线程模型

经代码调研，**gRPC C++ sync server 使用的是动态线程池（`SyncRequestThreadManager`），而非每次 RPC 创建短命线程。**

具体机制：
- 每个 `ServerCompletionQueue`（CQ）对应一个 `SyncRequestThreadManager`
- `ThreadManager` 维护一组**轮询线程**（polling threads），它们在 CQ 上调用 `AsyncNext()` 等待 RPC 到来
- 当一个轮询线程拿到 RPC 后，**在同一线程上**执行用户的同步 handler
- handler 执行完毕后，线程**回到轮询状态**，继续处理下一个 RPC —— **线程被复用**
- 如果活跃轮询线程数低于 `min_pollers`，`ThreadManager` 会动态创建新线程（上限 `max_pollers`）
- 空闲线程超过 CQ timeout 后可能被回收

### 8.2 默认参数（问题所在）

gRPC sync server 的默认线程参数来自 `ServerBuilder::SyncServerSettings`
（`third-party/grpc/linux-x64/include/grpcpp/server_builder.h`）：

```cpp
struct SyncServerSettings {
    SyncServerSettings()
        : num_cqs(1), min_pollers(1), max_pollers(2), cq_timeout_msec(10000) {}

    int num_cqs;         // CQ 数量
    int min_pollers;     // 每个 CQ 最小轮询线程数
    int max_pollers;     // 每个 CQ 最大轮询线程数
    int cq_timeout_msec; // CQ AsyncNext 超时
};
```

**默认值：1 个 CQ，每个 CQ 1-2 个轮询线程。**

这意味着默认配置下，**最多只有 2 个线程同时处理 RPC**，其余请求会排队。

### 8.3 两个代码库的实际配置

| 配置项 | cortex.core (`vde_grpc_server.cpp`) | beta (`vdss_grpc_server.cpp`) |
|-------|--------------------------------------|-------------------------------|
| `SetSyncServerOption(NUM_CQS)` | 未调用（默认 1） | 未调用（默认 1） |
| `SetSyncServerOption(MIN_POLLERS)` | 未调用（默认 1） | 未调用（默认 1） |
| `SetSyncServerOption(MAX_POLLERS)` | 未调用（默认 2） | 未调用（默认 2） |
| `SetResourceQuota` | 未调用 | 未调用 |
| `AddChannelArgument` | 未调用 | 未调用 |
| **实际并发搜索线程数** | **最多 2** | **最多 2** |

**两个代码库都没有配置 gRPC 线程参数**，全部使用默认值。

### 8.4 对 thread_local VisitedTable 的影响

**好消息：`thread_local` 方案在当前 gRPC 线程模型下是可行的。**

由于 gRPC sync server 的线程是**长生命周期的轮询线程**（不是每次 RPC 新建），
`thread_local VisitedTable` 只会在每个轮询线程首次搜索时分配一次，
之后该线程处理的所有后续 RPC 搜索都会复用同一个 VisitedTable，
`advance()` 的免 memset 优化也能正常生效。

默认 `max_pollers=2` 意味着：
- 最多只有 2 个 `thread_local VisitedTable` 实例（2 × 10MB = 20MB）
- 内存开销极小
- 但并发能力被限制在 2 个同时搜索

### 8.5 潜在风险：gRPC 线程动态创建/回收

`ThreadManager` 的线程并非完全固定 —— 在负载波动时会动态扩缩：

1. **高负载时**：如果 2 个轮询线程都在执行 handler，且有新 RPC 到来，
   `ThreadManager` 不会额外创建线程（已达 `max_pollers=2`），新 RPC 排队等待。
2. **负载降低时**：空闲线程超过 `cq_timeout_msec`（默认 10s）后可能被回收，
   其上的 `thread_local VisitedTable` 随线程析构。
3. **负载再次上升时**：新创建的线程需要重新分配 VisitedTable。

但在持续高 QPS 的 benchmark 场景下（vectordbbench），线程基本保持稳定，
`thread_local` 方案是有效的。

### 8.6 gRPC 线程模型优化建议

#### 建议 A：增大 max_pollers 和 num_cqs（推荐，必须做）

当前默认 `max_pollers=2` 严重限制了搜索并发度。在 10M benchmark 场景下，
每次搜索耗时 3-5ms，2 个线程的理论 QPS 上限仅 ~400-600。

```cpp
// vde_grpc_server.cpp — 添加线程配置
grpc::ServerBuilder builder;

// 方法 1：增大每个 CQ 的轮询线程数
builder.SetSyncServerOption(
    grpc::ServerBuilder::SyncServerOption::NUM_CQS, 4);        // 4 个 CQ
builder.SetSyncServerOption(
    grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, 4);     // 每个 CQ 至少 4 个线程
builder.SetSyncServerOption(
    grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, 8);     // 每个 CQ 最多 8 个线程
// 总并发线程：4 CQ × 8 pollers = 最多 32 个同时搜索

// 方法 2：通过 ResourceQuota 限制总线程数上限
grpc::ResourceQuota quota;
quota.SetMaxThreads(64);  // 整个 server 最多 64 个线程
builder.SetResourceQuota(quota);
```

**注意**：增大线程数会增加 `thread_local VisitedTable` 的总内存开销
（32 线程 × 10MB = 320MB），但相比 IndexIDMap2 rev_map 的 640MB 仍然可接受。

#### 建议 B：考虑 gRPC Callback API（长期优化）

gRPC 还有两种异步模型：

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| **Sync Server**（当前） | 简单，handler 阻塞线程 | handler 快速返回（<1ms） |
| **Async (CQ)**  | 手动状态机，最大控制 | 需要精细控制线程/连接 |
| **Callback API** | 框架管理线程，事件驱动回调 | handler 可以异步执行 |

Callback API 适合搜索场景的原因：
- 框架内部维护固定大小的线程池执行回调
- 回调线程是长生命周期的，`thread_local` 可完美配合
- 避免 sync server 的 "线程阻塞在 handler 中" 问题

但迁移到 Callback API 需要重构所有 gRPC handler 为 Reactor 模式，工程量较大，
建议作为长期优化方向。

---

## 9. 建议修复方案（按优先级排序，更新版）

#### 方案 1：复用 VisitedTable（效果最大，改动最小）

在 `FaissVectorIndexDriver` 中维护 `thread_local VisitedTable`，避免每次搜索重新分配：

```cpp
// 在 FaissVectorIndexDriver 或调用层
thread_local std::unique_ptr<faiss::VisitedTable> tl_vt;

// 搜索前：
if (!tl_vt || tl_vt->visited.size() != ntotal) {
    tl_vt = std::make_unique<faiss::VisitedTable>(ntotal);
}
tl_vt->advance();  // 免 memset 重置（每 249 次才真正清零）
```

**关于 gRPC 线程模型**：经调研确认 gRPC sync server 使用长生命周期的轮询线程池（非短命线程），
`thread_local` 方案可行。默认 `max_pollers=2` 时只有 2 个 VisitedTable 实例（20MB），
增大到 32 线程后为 320MB，仍可接受。

但这需要修改 faiss 的 `hnsw_search` 函数让其接受外部 VisitedTable，
或者在我们的 faiss fork 中做这个改动。

**替代方案 — 对象池**：如果不想依赖 `thread_local`（例如未来切换到 Callback API
后回调线程可能不固定），可以用显式对象池：

```cpp
class VisitedTablePool {
    std::mutex mu_;
    std::vector<std::unique_ptr<faiss::VisitedTable>> pool_;
    size_t ntotal_;
public:
    explicit VisitedTablePool(size_t ntotal) : ntotal_(ntotal) {}

    std::unique_ptr<faiss::VisitedTable> acquire() {
        std::lock_guard<std::mutex> lock(mu_);
        if (!pool_.empty()) {
            auto vt = std::move(pool_.back());
            pool_.pop_back();
            vt->advance();
            return vt;
        }
        return std::make_unique<faiss::VisitedTable>(ntotal_);
    }

    void release(std::unique_ptr<faiss::VisitedTable> vt) {
        std::lock_guard<std::mutex> lock(mu_);
        pool_.push_back(std::move(vt));
    }
};
```

对象池的 mutex 开销极小（获取/归还各一次，无竞争时 ~20ns），
远低于 malloc(10MB)+memset(10MB)+free(10MB) 的开销。

#### 方案 1.5：增大 gRPC 线程池（与方案 1 配合使用）

```cpp
// vde_grpc_server.cpp
builder.SetSyncServerOption(
    grpc::ServerBuilder::SyncServerOption::NUM_CQS, 4);
builder.SetSyncServerOption(
    grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, 4);
builder.SetSyncServerOption(
    grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, 8);
```

当前默认 `max_pollers=2` 可能成为 benchmark 并发瓶颈。
但需注意：beta 也是默认 2，所以这不是 cortex.core 相对 beta 回归的原因。
增大线程池是独立的性能优化。

#### 方案 2：去掉 jemalloc（验证最快）

不用 LD_PRELOAD jemalloc，直接用 glibc malloc 运行 cortex.core。
如果性能恢复到接近 beta，则 jemalloc 是主因。

#### 方案 3：调整 jemalloc 配置

```bash
MALLOC_CONF="dirty_decay_ms:-1,muzzy_decay_ms:-1,oversize_threshold:16777216"
```
- `dirty_decay_ms:-1`：禁止 dirty page 衰减（保留 free 后的页面，下次 malloc 秒复用）
- `oversize_threshold:16MB`：10MB 分配走 jemalloc 管理而非直接 mmap

#### 方案 4：去除 IndexIDMap2

改用裸 `IndexHNSWFlat` + 自定义 `vector<uint64_t> id_map_`（beta 的方式），
消除 614 MB 内存浪费和 10M 个小分配的碎片化。

### 建议验证步骤

1. **去掉 jemalloc 跑 benchmark** — 最快验证 jemalloc 是否是主因
2. **perf record** — `perf record -g` 看 malloc/free/memset 在搜索时间中的占比
3. **对比 RSS** — `VmRSS` 预期 cortex.core 比 beta 多 ~614 MB
4. **尝试方案 1** — 修改 faiss fork，支持外部传入 VisitedTable
5. **验证 gRPC 线程数** — 在 benchmark 运行时 `ps -eLf | grep vde` 确认实际线程数，
   验证是否被 `max_pollers=2` 限制了并发

---

## 10. 实测数据：glibc malloc vs jemalloc 对 10MB VisitedTable 的影响

### 10.1 Benchmark 数据

测试场景：10M 向量 HNSW 搜索，不同并发数下的 QPS 对比：

| 并发数 | QPS (glibc malloc) | QPS (jemalloc) | glibc 优势 |
|--------|-------------------|----------------|-----------|
| 1      | 98.81             | 59.14          | **+67%**  |
| 5      | 478.78            | 281.17         | **+70%**  |
| 10     | 771.88            | 441.69         | **+75%**  |
| 20     | 794.14            | 454.79         | **+75%**  |
| 30     | 723.38            | 453.58         | **+59%**  |
| 40     | 488.36            | 413.39         | **+18%**  |
| 60     | 254.16            | 199.66         | **+27%**  |
| 80     | 311.03            | 351.58         | **-12%**  |

**关键发现：在低中并发（1-30）下 glibc 比 jemalloc 快 59-75%，只有高并发（80）时 jemalloc 才微弱反超。**

### 10.2 根因分析：为什么 glibc malloc 在此场景下远快于 jemalloc

VisitedTable 的分配模式很特殊：**每个线程反复 malloc(10MB) → 用完 → free(10MB) → 立刻又 malloc(10MB)**。这是一个"短生命周期大块分配、高频循环"的模式。

#### glibc malloc 的优势

1. **mmap 阈值自适应（M_MMAP_THRESHOLD 动态调整）**

   glibc 对 >=128KB 的分配默认用 `mmap`/`munmap`，但它有一个动态调整机制 —
   当检测到反复分配释放相同大小时，会提升阈值，改用 `brk`/`sbrk` arena 管理。
   这意味着 free 后的 10MB 块留在进程地址空间内，下次 malloc 直接从 free list 复用，
   **接近零开销**。

2. **per-thread arena 无锁复用**

   glibc 会为每个线程分配独立 arena（最多 `8 × CPU cores` 个）。低并发时每个
   gRPC 线程用自己的 arena，**无锁竞争**。free 的 10MB 留在本线程 arena 的
   free list 中，下次 malloc 直接命中 —— 不需要任何全局锁。

3. **懒回收机制**

   即使走 mmap 路径，glibc 的 `munmap` 后 `mmap` 新页面时，内核可能复用之前的
   物理页（尤其在内存不紧张时），通过 `madvise(MADV_DONTNEED)` 让内核懒回收，
   物理页可能还在。

#### jemalloc 的劣势

1. **大分配走 extent/large allocation 路径**

   jemalloc 对 >=4MB 的分配使用独立的 extent 管理。每次 free 10MB 会进入
   extent dirty/muzzy 回收流程，涉及：
   - radix tree 查找（定位 extent metadata）
   - extent 合并/拆分尝试
   - decay ticker 检查
   
   这比 glibc 的 "放回 arena free list" 重得多。

2. **Decay 机制的持续开销**

   jemalloc 有后台 dirty page decay（默认 10s）和 muzzy page decay（默认 10s）。
   **每次 `free` 时都会检查 decay ticker**，可能触发 `madvise(MADV_DONTNEED)` 调用。
   对于每秒几百次的 10MB free，这个检查开销很显著。

   更糟糕的是：如果 decay 触发了 `madvise(MADV_DONTNEED)` 释放了物理页，
   下次 malloc 复用这个 extent 后做 `memset` 时会触发大量 page fault：
   ```
   10MB / 4KB = 2,560 次 page fault
   每次 page fault ~1-5μs
   总计 ~2.5-12.8ms —— 比搜索本身还慢
   ```

3. **tcache 不覆盖大分配**

   jemalloc 的 thread cache (tcache) 默认只缓存 <=32KB 的 size class。
   10MB 远超此范围，**每次 free 都要回到 arena 层**，涉及 extent mutex。
   不像 glibc 的 per-thread arena 可以无锁缓存大块。

### 10.3 为什么 80 并发时 jemalloc 反超

```
并发 40: glibc 488 vs jemalloc 413  (glibc 赢 18%)
并发 60: glibc 254 vs jemalloc 200  (glibc 赢 27%)
并发 80: glibc 311 vs jemalloc 352  (jemalloc 赢 12%)
```

注意 glibc 在 40→60 并发时 QPS **暴跌**（488→254，下降 48%），说明
**glibc 的 per-thread arena 数量耗尽，锁竞争爆发**。

glibc 默认最多 `8 × CPU cores` 个 arena。假设 8-core 机器，最多 64 个 arena，
但实际上 glibc 不会预分配这么多 —— 它按需创建，且大分配更容易集中到少数 arena。
当 60 个并发线程同时 malloc/free 10MB 时，多个线程争抢同一个 arena 的 mutex，
导致性能骤降。

而 jemalloc 虽然单线程慢，但其**锁粒度更细**（per-extent 而非 per-arena），
高并发时衰减更平缓。jemalloc 的 arena 数量默认是 `4 × CPU cores`，
且 extent 操作的锁竞争比 glibc 的 arena mutex 轻。

80 并发时 glibc QPS 从 254 回升到 311，可能是操作系统线程调度在极高并发下趋于稳定
（线程多到排队，反而减少了锁竞争的瞬时冲突）。

### 10.4 总结对比

| 特性 | glibc malloc | jemalloc |
|------|-------------|----------|
| 10MB free→malloc 复用速度 | 极快（arena free list 直接命中） | 慢（extent 管理 + decay 检查） |
| 低并发锁竞争 | 无（per-thread arena） | 低但有 extent mutex |
| 高并发锁竞争 | arena 数量耗尽后严重 | 更好的锁粒度，衰减平缓 |
| page fault 风险 | 低（arena 内复用不 munmap） | 高（decay 可能释放物理页） |
| **最适合场景** | **短生命周期、重复大小、高频循环** | 长生命周期、碎片治理、多样化分配 |

### 10.5 结论

**这也正好印证了 thread_local VisitedTable 复用方案（Section 9 方案 1）的必要性** ——
它彻底消除了 10MB 的 malloc/free 循环，无论使用哪个 allocator 都不再是瓶颈。
实施后：
- jemalloc 的 extent 管理开销消失
- glibc 的 arena 锁竞争（高并发时）消失
- memset 开销被 `advance()` 的免 memset 优化取代（每 249 次搜索才清零一次）
- 两个 allocator 的性能差异将大幅缩小，因为 10MB 热路径分配被完全消除
