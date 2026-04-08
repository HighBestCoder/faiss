# FAISS HNSW 搜索性能优化 — 完整技术报告

> 项目：FAISS v1.14.1 HNSW 索引搜索性能优化
> 分支：`bench/optimize-all`（基于 v1.14.1 tag）
> 日期：2026-04-08
> 硬件环境：Azure VM, 62GB RAM, 16 cores, Intel Xeon (Ice Lake, AVX-512)
> 数据集：Cohere 768D 余弦相似度，1M / 10M 向量

---

## 一、项目背景

### 1.1 问题定义

FAISS 的 HNSW（Hierarchical Navigable Small World）索引是工业界最常用的近似最近邻搜索算法之一。在大规模向量检索场景（千万级以上），原生 FAISS v1.14.1 的 HNSW 搜索存在以下性能瓶颈：

1. **距离计算效率低**：标量循环依赖编译器自动向量化，无法充分利用 AVX-512 的 512-bit 寄存器宽度
2. **内存访问模式差**：10M×768D 的向量数据约 30 GB，远超 CPU 缓存，HNSW 的随机访问导致大量 cache miss 和 TLB miss
3. **线程管理开销大**：单条查询仍触发 OpenMP 线程池调度，微秒级开销在高 QPS 场景下不可忽略
4. **重复内存分配**：每次搜索分配/释放 10 MB 级的 VisitedTable 数组

### 1.2 优化目标

- 在不改变搜索结果正确性的前提下，提升搜索 QPS
- 尽量不增加内存开销
- 所有优化可独立开关，便于评估贡献

### 1.3 最终成果

对比基准：客户实际使用的**原生 FAISS v1.14.1 + HNSW32,Flat (FP32，无量化)**。
优化版本：**V1-16 全部优化 + SQfp16 量化**。

#### 1M 数据集 (efSearch=64, efConstruction=40)

| 指标 | V0 原生 Flat (FP32) | V1-16 + SQfp16 | 提升 |
|------|---------------------|----------------|------|
| QPS 单线程 | 1,110 | 1,538 | **+38.6%** |
| QPS 16线程 | 10,912 | 13,404 | **+22.8%** |
| Recall@10 | 96.15% | 95.98% | -0.17pp (可忽略) |
| 内存 (RSS) | ~3,200 MB | 1,946 MB | **-39.1%** |

#### 10M 数据集 — 高精度搜索 (M=32, efConstruction=512, efSearch=512)

| 指标 | V0 原生 Flat (FP32) | V1-16 + SQfp16 | 提升 |
|------|---------------------|----------------|------|
| QPS 单线程 | 141 | 177 | **+25.5%** |
| QPS 16线程 | 1,325 | 1,581 | **+19.3%** |
| Recall@10 | 99.60% | 99.02% | -0.58pp (可接受) |
| 内存 (RSS) | 33,798 MB | 19,157 MB | **-43.3%** |
| 构建时间 | 8,031s (2.2h) | 7,041s (2.0h) | **-12.3%** |

> **核心价值**：在 QPS 提升 19-39% 的同时，内存减少 39-43%，Recall 损失 < 0.6pp。

---

## 二、HNSW 搜索过程分析

要理解各项优化，首先需要了解 HNSW 搜索的核心流程。

### 2.1 搜索分两个阶段

```
阶段 1: 上层贪心搜索 (greedy_update_nearest)
  从最高层的入口点开始，每层只找 1 个最近邻
  层数 = log(N) / log(M)，计算量很小

阶段 2: Layer 0 的候选搜索 (search_from_candidates)
  这是搜索的主要开销（>95% 时间）
  维护大小为 efSearch 的候选堆
  每次取最近候选节点，遍历其所有邻居（最多 2×M = 64 个）
  对未访问的邻居计算距离，更新候选堆
  最多进行 efSearch 步
```

### 2.2 搜索热点的 CPU 时间分布

对于 768D 向量、efSearch=64 的典型查询：

| 操作 | 占比 | 说明 |
|------|------|------|
| 距离计算 (fvec_inner_product) | ~60% | 每次 768 维浮点内积 |
| 内存访问等待 (cache miss stall) | ~20% | 随机访问 30 GB 向量数据 |
| 候选堆操作 (MinimaxHeap) | ~8% | pop_min, count_below |
| VisitedTable 分配/查询 | ~7% | malloc + memset 10 MB |
| OpenMP 调度 | ~3% | 线程池唤醒/同步 |
| 其他 | ~2% | 图结构遍历等 |

各项优化精准地针对以上每个热点。

---

## 三、优化详解

### O1: OpenMP 条件守卫 — 消除单查询线程调度开销

**解决的问题**

当 `n=1`（单条查询）时，OpenMP 的 `#pragma omp parallel` 仍然会触发线程池管理，包括：线程唤醒、任务分发、barrier 同步。这些操作的固定开销约 10-130 微秒。在 QPS > 1000 的场景下（每查询 < 1 ms），这个开销占比可达 10-13%。

**实现方式**

在所有搜索相关的 `#pragma omp parallel` 指令上添加 `if(n > 1)` 条件：

```cpp
// 修改前
#pragma omp parallel for
for (idx_t i = 0; i < n; i++) { ... }

// 修改后
#pragma omp parallel for if(n > 1)
for (idx_t i = 0; i < n; i++) { ... }
```

当 `n=1` 时，循环直接在调用线程执行，跳过所有 OpenMP 开销。

**修改文件**

- `faiss/IndexHNSW.cpp`
- `faiss/IndexBinaryHNSW.cpp`
- `faiss/IndexFlatCodes.cpp`
- `faiss/IndexIDMap.cpp`
- `faiss/IndexNNDescent.cpp`
- `faiss/IndexNSG.cpp`
- `faiss/IndexScalarQuantizer.cpp`

这个主要是避免掉坑里。因为在faiss代码里面，有些地方是无脑开omp。导致的后果是，

---

### O2: 动态 OMP 调度 — 解决构建阶段负载不均

**解决的问题**

HNSW 构建阶段 `hnsw_add_vertices()` 使用 `schedule(static)` 将向量均匀分配给各线程。但 HNSW 插入的特性是：后插入的向量需要在更大的图中搜索邻居，计算量远大于先插入的。静态调度导致分配到后半段的线程工作量远超前半段，前半段线程空转等待。

**实现方式**

```cpp
// 修改前
#pragma omp for schedule(static)

// 修改后
#pragma omp for schedule(dynamic, 64)
```

改为动态调度后，每个线程完成当前 64 个向量的插入后，立即领取下一批任务，自动实现负载均衡。

**修改文件**

- `faiss/IndexHNSW.cpp` — `hnsw_add_vertices()` 函数

**效果**：构建时间减少约 **-4%**，搜索性能不直接受影响

---

### O3: AVX-512 batch_8 距离计算 — 主要性能贡献者

**解决的问题**

距离计算占搜索时间的 ~60%，是最核心的热点。原生 FAISS 中：
1. `fvec_inner_product` / `fvec_L2sqr` 是标量循环，依赖编译器自动向量化
2. HNSW 搜索路径中邻居距离是"缓冲 4 个、批量计算"（`batch_4`）

两个问题：
- 编译器自动向量化不稳定，无法保证使用 AVX-512
- AVX-512 有 32 个 512-bit 寄存器（每个存 16 个 float），batch_4 只用了 4 个累加器，寄存器利用率仅 12.5%

**实现方式**

**Step 1**: 新增手写 SIMD 的 `batch_8` 距离函数

```cpp
// AVX-512 实现 (distances_avx512.cpp)
void fvec_inner_product_batch_8(
    const float* x,           // 查询向量 (768D)
    const float* y0, ..., const float* y7,  // 8 个数据库向量
    size_t d,
    float& dp0, ..., float& dp7)
{
    __m512 sum0=_mm512_setzero_ps(), ..., sum7=_mm512_setzero_ps();
    for (size_t i = 0; i < d; i += 16) {
        __m512 xi = _mm512_loadu_ps(x + i);  // 加载查询向量（只加载一次）
        sum0 = _mm512_fmadd_ps(xi, _mm512_loadu_ps(y0+i), sum0);  // FMA
        sum1 = _mm512_fmadd_ps(xi, _mm512_loadu_ps(y1+i), sum1);
        ...
        sum7 = _mm512_fmadd_ps(xi, _mm512_loadu_ps(y7+i), sum7);
    }
    dp0 = _mm512_reduce_add_ps(sum0);
    ...
}
```

关键优化点：
- 查询向量 `x[i]` 每个 16-float 块只从内存加载 **1 次**，在 8 个 FMA 中复用
- 8 个累加器同时进行 FMA 运算，充分利用 CPU 的乱序执行能力
- 每次循环迭代执行 8 个 FMA 指令，利用 AVX-512 的 FMA 吞吐（每周期 2 个 FMA）

**Step 2**: 在 `DistanceComputer` 基类添加 `distances_batch_8()` 虚方法

```cpp
struct DistanceComputer {
    // 原有
    virtual void distances_batch_4(const idx_t idx0, ..., float& dis0, ...);
    // 新增
    virtual void distances_batch_8(const idx_t idx0, ..., float& dis0, ...);
};
```

**Step 3**: 修改 HNSW 搜索热路径，将缓冲区从 4 扩大到 8

```cpp
// 修改前 (HNSW.cpp search_from_candidates)
storage_idx_t saved_j[4];
if (counter == 4) { qdis.distances_batch_4(...); counter = 0; }

// 修改后
storage_idx_t saved_j[8];
if (counter == 8) { qdis.distances_batch_8(...); counter = 0; }
// 剩余 4-7 个用 batch_4 处理
```

4 个 HNSW 热路径全部升级：`search_from_candidates`、`search_from_candidate_unbounded`、`greedy_update_nearest`、`search_neighbors_to_add`

**修改文件**

- `faiss/impl/DistanceComputer.h` — batch_8 虚方法
- `faiss/impl/HNSW.cpp` — 4 个搜索热路径
- `faiss/IndexFlat.cpp` — FlatL2Dis/FlatIPDis batch_8 实现
- `faiss/IndexFlatCodes.cpp` — GenericFlatCodesDistanceComputer batch_8
- `faiss/utils/distances.cpp` — 调度函数
- `faiss/utils/distances.h` — 声明
- `faiss/utils/distances_dispatch.h` — SIMD 层级调度模板
- `faiss/utils/simd_impl/distances_avx512.cpp` — AVX-512 实现
- `faiss/utils/simd_impl/distances_avx2.cpp` — AVX2 实现
- `faiss/utils/simd_impl/distances_autovec-inl.h` — 标量 fallback

**效果**：单线程 QPS 提升约 **+20%**（最大单项贡献）

---

### O4: 维度感知软件预取 — 减少 Cache Miss 等待

**解决的问题**

10M × 768D 的向量数据约 30 GB，远超 CPU 的 L3 缓存（通常 25-50 MB）。HNSW 搜索是随机访问模式——从候选节点的邻居中取出节点 ID 后，需要加载该节点的 3072 字节向量数据来计算距离。这几乎必定是一次 L3 cache miss（延迟 100-200 个时钟周期）。

CPU 的乱序执行窗口无法完全隐藏这么长的延迟。如果能在实际需要数据之前提前发起预取，让数据在计算前就到达缓存，就能把延迟"隐藏"在其他计算背后。

**实现方式**

在 `DistanceComputer` 基类添加 `prefetch()` 虚方法：

```cpp
virtual void prefetch(idx_t id, int lines = 3) {
    // 仅在向量数据 >= 1200 字节 (约 dim >= 300) 时启用
    if (code_size >= 1200) {
        const char* ptr = codes + id * code_size;
        for (int i = 0; i < lines; i++) {
            _mm_prefetch(ptr + i * 64, _MM_HINT_T0);  // prefetch 到 L1
        }
    }
}
```

在 HNSW 搜索热路径中，遍历邻居节点时提前预取：

```cpp
// 在访问邻居 v1 的距离之前
qdis.prefetch(v1);   // 提前加载向量数据到缓存
vt.prefetch(v1);     // 提前加载 VisitedTable 条目（原有）
```

**维度阈值的设计考量**

- dim=128（SIFT）：向量仅 512 字节，大概率已在缓存中，prefetch 指令的开销（占用取指/发射槽位）反而拖慢性能（实测 -20%）
- dim=768（Cohere）：向量 3072 字节，必定 cache miss，prefetch 收益显著
- 阈值设为 1200 字节（dim ≈ 300），是经过多个数据集验证的经验值

**修改文件**

- `faiss/impl/DistanceComputer.h` — prefetch 虚方法（+22 行）
- `faiss/impl/HNSW.cpp` — 4 个热路径中插入 prefetch 调用（+4 行）

**效果**：与 O3/O14 协同，减少 cache miss 约 **+3-5%**

---

### O5: FP16 SIMD 距离计算库

**解决的问题**

FP32 向量每维 4 字节，FP16 每维 2 字节。使用 SQfp16 存储可以将内存减半，但需要高效的 FP16 距离计算支持。FAISS 的 ScalarQuantizer 虽然支持 FP16，但内部距离计算路径在 v1.14.1 中需要正确的 SIMD dispatch 配置才能启用 AVX-512。

**实现方式**

独立的 FP16 距离计算库，三层 SIMD 实现：

```
AVX-512F + F16C: 每次处理 32 个 float16 → 转换为 2 组 __m512 → FMA
AVX2 + F16C:     每次处理 8 个 float16 → _mm256_cvtph_ps → FMA
Scalar:          逐元素 fp16_ieee_to_fp32_value() → 标量乘加
```

API 包括：`fp16vec_L2sqr`、`fp16vec_inner_product`、`batch_4`、`batch_8` 变体，以及 FP32↔FP16 转换工具。

**修改文件**

- `faiss/utils/distances_fp16.h` — 声明（140 行）
- `faiss/utils/distances_fp16_simd.cpp` — 实现（1249 行）
- `tests/test_distances_fp16.cpp` — 117 个单元测试

**SIMD Dispatch 修复**（关键发现）

移植完所有优化后，SQfp16 的 QPS 只有 431（期望 ~1500）。排查发现：

1. 构建脚本缺少 `-DFAISS_OPT_LEVEL=avx512`，导致 v1.14.1 的 SIMD dispatch 机制 fallback 到标量路径
2. 启用后出现 ODR（One Definition Rule）冲突——batch_8 模板特化和显式特化同时存在

修复方案：添加 `FAISS_SKIP_AUTOVEC_BATCH_8` 预处理守卫，更新 `llm/build.sh` 加入 `-DFAISS_OPT_LEVEL=avx512`。

**效果**：SQfp16 QPS 从 431 → 1,304（**+202%**）。这不是 O5 库本身的贡献，而是修复了 dispatch 配置使 SQ 的 AVX-512 路径正确启用。

---

### O10: 透明大页 (THP) — 减少 TLB Miss

**解决的问题**

10M 向量 × 3072 字节 = 30 GB 向量数据，使用标准 4 KB 页面需要约 **750 万个页表项**。CPU 的 TLB（Translation Lookaside Buffer）通常只能缓存 1000-2000 个页表项。HNSW 的随机访问模式导致几乎每次向量访问都触发 TLB miss → 多级页表遍历（100+ 个时钟周期）。

**实现方式**

在索引构建完成后，对向量存储区域调用 `madvise(MADV_HUGEPAGE)`，请求内核使用 2 MB 透明大页：

```cpp
// numa_helpers.h (新文件)
inline void try_enable_hugepages(void* ptr, size_t size) {
    // 对齐到 2MB 边界
    uintptr_t aligned = (uintptr_t(ptr) + 0x1FFFFF) & ~0x1FFFFF;
    size_t usable = size - (aligned - uintptr_t(ptr));
    madvise((void*)aligned, usable, MADV_HUGEPAGE);
}

// IndexHNSW.cpp 的 add() 末尾
auto* flat = dynamic_cast<IndexFlatCodes*>(storage);
if (flat) {
    try_enable_hugepages(flat->codes.data(), flat->codes.size());
}
```

2 MB 大页将页表项从 750 万降至约 **1.5 万**，TLB miss 率大幅下降。

**适用条件**

| 数据集 | 维度 | 大页效果 | 原因 |
|--------|------|---------|------|
| 10M × 768D | 3072 B/vec | **+13-25%** | 数据量大、向量长 |
| 1M × 960D | 3840 B/vec | **+25.3%** | 向量很长 |
| 1M × 128D | 512 B/vec | +12.5% | 数据量中等 |
| 1.2M × 100D | 400 B/vec | **-40%** | 向量短，大页反而增加浪费 |
| 290K × 256D | 1024 B/vec | **-44%** | 数据量太小 |

**结论**：向量 > 500 字节 **且** 数据量 > 50 万时启用大页才有收益。

**修改文件**

- `faiss/utils/numa_helpers.h` — 新增（38 行）
- `faiss/IndexHNSW.cpp` — add() 末尾（+8 行）
- `faiss/CMakeLists.txt` — 头文件注册

**效果**：高维大数据集 **+13-25%**，与 O14 BFS 重排协同效果更佳

---

### O11: 跨节点邻居批处理 — 提升 batch_8 命中率

**解决的问题**

O3 引入了 batch_8 距离计算，但 HNSW 原始代码的缓冲区是**每个候选节点独立**的。每次弹出一个候选节点后，遍历其邻居并缓冲。问题是：HNSW M=32 时每个节点有最多 64 个邻居，但大部分已被访问过。平均每个候选节点只有 2-3 个未访问邻居进入缓冲区，很少能凑满 8 个。batch_8 的实际命中率只有 ~30%。

**实现方式**

将缓冲区移到候选节点循环**外部**，让多个候选节点的未访问邻居在同一个缓冲区中累积：

```cpp
// 修改前：缓冲区每轮重置
while (candidates.size() > 0) {
    int counter = 0;               // ← 每个候选节点重置
    storage_idx_t saved_j[8];
    for (j in neighbors) { ... }
}

// 修改后：缓冲区跨节点累积
int batch_counter = 0;             // ← 移到循环外
storage_idx_t batch_ids[8];
while (candidates.size() > 0) {
    for (ni in neighbor_ids) {
        batch_ids[batch_counter++] = v1;
        if (batch_counter == 8) {   // 凑满 8 个就触发 batch_8
            qdis.distances_batch_8(...);
            batch_counter = 0;
        }
    }
}
flush_batch();  // 处理剩余（先试 batch_4，再标量）
```

同时引入**滑动窗口预取**：收集当前节点的所有邻居 ID 后，以 8 为窗口提前预取后续邻居的向量数据和 VisitedTable 条目。

**修改文件**

- `faiss/impl/HNSW.cpp` — `search_from_candidates` 完全重写（+114/-78 行）

**效果**：batch_8 命中率从 ~30% 提升至 ~80%，QPS 提升约 **+10-17%**

---

### O12: SIMD 加速 MinimaxHeap::count_below — 加速早停判断

**解决的问题**

`search_from_candidates` 中的早停条件需要调用 `MinimaxHeap::count_below(thresh)`，统计堆中距离小于阈值的元素个数。原始实现是逐元素标量比较循环。efSearch=256 时需要遍历 256 个 float，每个候选节点都要调用一次。

**实现方式**

```cpp
// AVX-512 实现
int MinimaxHeap::count_below(float thresh) {
    int count = 0;
    __m512 vt = _mm512_set1_ps(thresh);
    size_t i = 0;
    for (; i + 16 <= k; i += 16) {
        __m512 vd = _mm512_loadu_ps(dis.data() + i);
        __mmask16 mask = _mm512_cmp_ps_mask(vd, vt, _CMP_LT_OS);
        count += _mm_popcnt_u32(mask);
    }
    // 处理剩余元素
    for (; i < k; i++) {
        if (dis[i] < thresh) count++;
    }
    return count;
}
```

每次迭代比较 16 个 float（AVX-512）或 8 个 float（AVX2），配合 `popcnt` 指令统计符合条件的个数。

**修改文件**

- `faiss/impl/HNSW.cpp` — count_below 函数（+43 行）

**效果**：单项加速约 **+3-5%**（count_below 本身在总时间中占比不大）

---

### O13: SharedVectorStore — 零拷贝重建

**解决的问题**

生产环境中 HNSW 索引需要定期重建（删除向量后图质量退化）。传统重建流程会**复制所有向量**到新索引，内存峰值翻倍。对于 10M × 768D 的索引（~30 GB 向量数据），重建需要 ~60 GB 内存，往往超出机器可用内存。

**实现方式**

引入共享存储架构：

```
┌──────────────────────────────────────┐
│         SharedVectorStore            │
│  (存储实际向量数据, shared_ptr 管理)    │
│  + free_list (已删除 slot 的回收列表)  │
└──────────┬───────────────────────────┘
           │ shared_ptr
    ┌──────┴──────┐
    │             │
┌───┴────┐  ┌────┴───┐
│ 旧索引  │  │ 新索引  │   ← 两个索引共享同一份向量数据
│(删除后) │  │(重建中) │
│storage_ │  │storage_ │
│id_map[] │  │id_map[] │   ← 各自维护 HNSW节点ID → Store位置 映射
└────────┘  └────────┘
```

- `SharedVectorStore`：持有向量数据，多个索引通过 `shared_ptr` 引用
- `IndexFlatShared`：使用 `storage_id_map` 做间接寻址，支持 `deleted_bitmap`
- 重建时 `IndexHNSW::add(n, nullptr)` — 传入 `nullptr` 表示不复制向量，直接从 SharedVectorStore 读取
- `compact_store()`：重建完成后，用原地 cycle-following 算法重排存储，消除间接寻址开销

**修改文件**

- `faiss/SharedVectorStore.h/cpp` — 新增
- `faiss/IndexFlatShared.h/cpp` — 新增（744 行）
- `faiss/IndexHNSW.cpp` — 零拷贝 add 路径

**效果**：重建期间内存节省 **~78%**（仅复制图结构 + 映射表，不复制向量数据），搜索性能影响 < 2%

---

### O14: BFS 图重排 — 提升缓存局部性

**解决的问题**

HNSW 的节点 ID 按插入顺序分配，与图拓扑完全无关。搜索时访问的邻居节点在内存中随机分布，导致 CPU 缓存几乎无法利用空间局部性——即使两个节点是图中的近邻，它们的向量数据在内存中可能相距数 GB。

**实现方式**

提供 5 种图重排策略，根据图拓扑重新分配节点 ID，使图中相邻的节点在内存中也相邻：

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| **BFS** | 从入口点出发 BFS 遍历，按访问顺序编号 | 通用，效果最稳定 |
| RCM | Reverse Cuthill-McKee，最小化图带宽 | 稀疏图 |
| DFS | 深度优先遍历编号 | 特定图结构 |
| Cluster | 按层级+度数排序 | 多层图 |
| Weighted | 按 (1+level)×degree 排序 | 高连接度节点 |

```cpp
// 使用方式
auto perm = generate_permutation(hnsw, ReorderStrategy::BFS);
index->permute_entries(perm.data());
// perm[new_id] = old_id，之后搜索结果需要反向映射
```

`permute_entries` 会同时重排：
- 图结构（邻居列表中的 ID 引用）
- 向量数据（`IndexFlatCodes::codes` 数组）

重排后，BFS 搜索路径上访问的节点在内存中是连续的，CPU 的 L1/L2 prefetcher 和硬件预取可以有效工作。

**修改文件**

- `faiss/HNSWReorder.h/cpp` — 新增（367 行）
- `faiss/IndexFlatShared.cpp` — dual-mode permute_entries
- `faiss/CMakeLists.txt`

**效果**：单独 BFS 重排 **+18%** QPS，与大页 (O10) 配合使用效果 **+10-15%**

---

### O16: VisitedTable 复用 — 消除逐查询内存分配

**解决的问题**

每次 HNSW 搜索调用都会新分配一个 `VisitedTable`，大小为 `ntotal` 字节。10M 索引意味着每次搜索要 `malloc(10 MB)` + `memset(10 MB, 0)` + `free(10 MB)`。在 QPS > 1000 的场景下，这些操作每秒执行上千次，成为显著开销。

`VisitedTable` 有一个精巧的设计：内部维护 `visited_generation` 计数器，调用 `advance()` 时只需将计数器 +1 就能"逻辑清空"整个表（不需要 memset）。但如果每次都重新分配，这个设计就浪费了。

**实现方式**

在 `SearchParametersHNSW` 中添加 `visited_table` 字段：

```cpp
struct SearchParametersHNSW : SearchParameters {
    int efSearch = 0;
    VisitedTable* visited_table = nullptr;  // ← 新增
};
```

搜索函数中，如果调用者提供了预分配的 VisitedTable，就复用它：

```cpp
VisitedTable* vt_ptr;
std::unique_ptr<VisitedTable> local_vt;
if (external_vt && is_single_query) {
    vt_ptr = external_vt;
    vt_ptr->advance();      // O(1) 逻辑清空
} else {
    local_vt = std::make_unique<VisitedTable>(ntotal);  // O(ntotal) 分配+清零
    vt_ptr = local_vt.get();
}
```

调用方只需创建一次 VisitedTable，后续所有查询复用同一份：

```cpp
// 客户端代码
VisitedTable vt(index->ntotal);
SearchParametersHNSW params;
params.efSearch = 64;
params.visited_table = &vt;
for (int i = 0; i < nq; i++) {
    index->search(1, query_i, k, distances, labels, &params);
}
```

**修改文件**

- `faiss/impl/HNSW.h` — SearchParametersHNSW 添加字段（+1 行）
- `faiss/IndexHNSW.cpp` — hnsw_search 逻辑（+16 行）

**效果**：单线程 QPS 提升约 **+5%**（10M 规模下更明显）

---

## 四、优化贡献分解

### 4.1 各优化的独立贡献估算

基于 10M 数据集、HNSW32,SQfp16、efSearch=64 的测试结果：

```
总提升: 单线程 +30.4% (954 → 1,244 QPS)

O3  SIMD batch_8 距离计算    ████████████████████  ~20%   ← 最大贡献
O14 BFS 图重排              ██████████████        ~12%
O16 VisitedTable 复用       █████                 ~5%
O4  软件预取                ████                  ~4%
O11 跨节点邻居批处理        ███                   ~3%    (与 O3 协同)
O10 透明大页                ██                    ~2%    (与 O14 协同)
O1  OMP 条件守卫            ██                    ~2%
O12 SIMD count_below        █                     ~1%
```

> 注：各项优化之间存在协同效应和递减效应，独立贡献之和不等于总提升。O3+O11 协同使 batch_8 命中率从 30% 提升到 80%；O14+O10 协同使重排后的连续数据被大页保护更有效。

### 4.2 搜索 vs 构建优化

| 类别 | 优化 | 影响阶段 |
|------|------|---------|
| 搜索性能 | O1, O3, O4, O11, O12, O14, O16 | search() |
| 构建性能 | O2 | add() |
| 内存节省 | O5 (SQfp16), O13 (零拷贝重建) | 部署 |
| 搜索+构建 | O10 (大页) | 两者都影响 |

---

## 五、完整性能对比

> **对比口径**：客户实际使用的**原生 FAISS v1.14.1 + HNSW32,Flat (FP32)**作为基线，与**V1-16 全部优化 + SQfp16**对比。
> 同时列出同量化方式的纯搜索优化对比（V0 SQfp16 vs V1-16 SQfp16）作为参考。

### 5.1 1M 数据集 (efSearch=64, efConstruction=40)

| 指标 | V0 Flat (FP32) | V1-16 SQfp16 | 变化 |
|------|---------------|-------------|------|
| QPS 单线程 | 1,110 | 1,538 | **+38.6%** |
| QPS 16线程 | 10,912 | 13,404 | **+22.8%** |
| Recall@10 | 96.15% | 95.98% | -0.17pp |
| 内存 (RSS) | ~3,200 MB | 1,946 MB | **-39.1%** |

纯搜索优化参考（同为 SQfp16）：V0 SQfp16 QPS_1T=1,165 → V1-16 SQfp16 QPS_1T=1,538（+32.0%）

### 5.2 10M 数据集 — 高精度搜索 (M=32, efC=512, efS=512)

| 指标 | V0 Flat (FP32) | V1-16 SQfp16 | 变化 |
|------|---------------|-------------|------|
| QPS 单线程 | 141 | 177 | **+25.5%** |
| QPS 16线程 | 1,325 | 1,581 | **+19.3%** |
| Recall@10 | 99.60% | 99.02% | -0.58pp |
| 内存 (RSS) | 33,798 MB | **19,157 MB** | **-43.3%** |
| 构建时间 | 8,031s (2.2h) | 7,041s (2.0h) | **-12.3%** |

### 5.3 10M 数据集 — 纯搜索优化对比 (V0 SQfp16 vs V1-16 SQfp16)

以下对比在相同量化条件下（均为 SQfp16），仅体现搜索算法优化的贡献：

| efSearch | V0 SQfp16 QPS_1T | V1-16 SQfp16 QPS_1T | 变化 |
|----------|------------------|---------------------|------|
| 64 | 954 | 1,244 | +30.4% |
| 128 | 525 | 684 | +30.3% |
| 256 | 273 | 366 | +34.1% |

### 5.4 与 RaBitQ 量化的对比

RaBitQ 是 FAISS v1.14.1 内置的新量化方案。在 10M 数据集、Recall@10 ≈ 95% 的条件下：

| 方案 | QPS 单线程 | QPS 16线程 | 内存 | 构建时间 |
|------|-----------|-----------|------|---------|
| V0 原生 Flat (efC=512,efS=512) | 141 | 1,325 | 33,798 MB | 8,031s |
| V1-16 HNSW+SQfp16 (efC=512,efS=512) | **177** | **1,581** | 19,157 MB | 7,041s |
| IVF4096,RaBitQfs4 (nprobe=256) | 29 | 451 | **5,718 MB** | **488s** |

HNSW+SQfp16 相比原生 Flat：QPS +25%，内存 -43%。
RaBitQ 相比原生 Flat：QPS -79%，内存 -83%。各有适用场景。

---

## 六、版本定义与复现

### 6.1 版本定义

| 版本 | 说明 | 库文件 |
|------|------|--------|
| **V0** | 原生 FAISS v1.14.1，零优化 | `llm/faiss-1.14.1-origin/lib/libfaiss_avx512.so` |
| **V1-12** | O1/O2/O3/O4/O10/O11/O12（库内自动生效） | `install/lib/libfaiss_avx512.so` |
| **V1-16** | V1-12 + O14(BFS图重排) + O16(VT复用) | 同上 + bench_task.cpp 显式调用 |

### 6.2 编译参数

```bash
# FAISS 库编译
cmake -DFAISS_OPT_LEVEL=avx512 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -ffast-math -funroll-loops" \
      -DBLA_VENDOR=Intel10_64_dyn ..

# 客户端程序编译
g++ -O3 -march=native -mtune=native -std=c++17 \
    -I install/include \
    -o bench_task bench_task.cpp \
    -L install/lib -lfaiss_avx512 \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 \
    -lpthread -lm -ldl \
    -Wl,-rpath,install/lib
```

### 6.3 关键注意事项

1. **必须使用 `-lfaiss_avx512`** 而非 `-lfaiss`。后者加载通用库，SQfp16 的距离计算走标量路径，QPS 从 1538 暴降到 450
2. **必须设置 `-DFAISS_OPT_LEVEL=avx512`**。缺失此配置会导致 ScalarQuantizer 的 SIMD dispatch 回退到标量路径
3. **O14 图重排** 需要临时内存（约等于 codes 数组大小）。10M Flat 索引需要 ~64 GB，在 62 GB 机器上 OOM。SQfp16 只需 ~15 GB 临时内存，可正常完成

---

## 七、关键结论与建议

### 7.1 对搜索 QPS 贡献最大的优化

1. **O3 (SIMD batch_8)** — 最大单项贡献（~20%），必须启用
2. **O14 (BFS 图重排)** — 第二大贡献（~12%），缓存友好性改善显著
3. **O11 (跨节点批处理)** — 与 O3 协同，使 batch_8 充分发挥

### 7.2 相比客户现有方案的整体提升

客户现有方案：**原生 FAISS v1.14.1 + HNSW32,Flat (FP32，无量化)**

我们的方案：**V1-16 全部搜索优化 + SQfp16 量化**

综合收益：
- **QPS 提升 19-39%**：搜索算法优化（O1-O16）贡献 +25-34%，SQfp16 的更小数据量进一步改善缓存命中
- **内存减少 39-43%**：SQfp16 将每向量存储从 3072 字节降至 1536 字节
- **Recall 损失 < 0.6pp**：99.60% → 99.02%，对业务影响可忽略
- **构建时间减少 12%**：动态 OMP 调度 + SQfp16 更小的数据搬运量

### 7.3 适用场景

| 场景 | 推荐方案 |
|------|---------|
| 高 QPS + 高 Recall | HNSW32,SQfp16 (V1-16) |
| 内存受限 (< 8 GB) | IVF4096,RaBitQfs4 |
| 超大规模 (100M+) | IVF + RaBitQ (可能是唯一能装进内存的方案) |
| 需要频繁重建 | O13 SharedVectorStore 零拷贝架构 |

---

## 附录 A: 优化一览表

| ID | 优化名称 | 目标 | 独立效果 |
|----|---------|------|---------|
| O1 | OMP 条件守卫 | 消除 n=1 时的线程调度开销 | +2% |
| O2 | 动态 OMP 调度 | 构建阶段负载均衡 | 构建 -4% |
| O3 | AVX-512 batch_8 | 距离计算指令级并行 | **+20%** |
| O4 | 维度感知预取 | 减少向量数据 cache miss | +3-5% |
| O5 | FP16 SIMD 库 | 支持 FP16 距离计算 | (基础设施) |
| O10 | 透明大页 | 减少 TLB miss | +2-25% |
| O11 | 跨节点批处理 | 提升 batch_8 命中率 | +10-17% |
| O12 | SIMD count_below | 加速早停判断 | +3-5% |
| O13 | SharedVectorStore | 零拷贝重建 | 内存 -78% |
| O14 | BFS 图重排 | 缓存局部性 | +10-18% |
| O16 | VisitedTable 复用 | 消除 malloc/memset | +5% |

## 附录 B: 代码仓库信息

- 分支：`bench/optimize-all`
- 基线：FAISS v1.14.1 tag (`471ddad72`)
- 最新 commit：`436c609f8` (RaBitQ benchmark)
- 结果报告：`llm/arch-2026-04-07/benchmark-report.md`（HNSW 对比）
- 结果报告：`llm/arch-2026-04-07/rabitq-benchmark-report.md`（RaBitQ 对比）
