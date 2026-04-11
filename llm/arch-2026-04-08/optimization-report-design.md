# FAISS HNSW 搜索优化 — 技术设计文档

> 项目：FAISS v1.14.1 HNSW 索引搜索优化
> 分支：`bench/optimize-all`（基于 v1.14.1 tag）
> 日期：2026-04-08
> 目标平台：Intel Xeon (Ice Lake, AVX-512)

---

## 一、HNSW 搜索的瓶颈在哪里

在讨论任何一项优化之前，先要搞清楚 HNSW 搜索到底把时间花在了哪里。

### 1.1 搜索过程

HNSW 搜索分两个阶段：

1. **上层贪心下行** (`greedy_update_nearest`)：从最高层入口点开始，每层只找 1 个最近邻，逐层往下走。层数 = log(N)/log(M)，计算量很小，不是瓶颈。

2. **Layer 0 候选扩展** (`search_from_candidates`)：这里是真正的主循环，占搜索时间的 95% 以上。它维护一个大小为 efSearch 的候选堆，每次弹出最近的候选节点，遍历该节点的所有邻居（最多 2×M 个），对未访问过的邻居计算距离，然后更新候选堆。

所以 HNSW 搜索的本质是：**在一张大图上做局部 BFS，每一跳的代价主要取决于"距离计算"和"向量数据的内存访问"。**

### 1.2 时间花在哪里

对 768 维向量、efSearch=64 的典型查询做 profiling，时间分布大致如下：

| 热点 | 占比 | 本质 |
|------|------|------|
| 距离计算 (`fvec_inner_product`) | ~60% | 高维浮点内积 |
| 内存访问等待 (cache miss stall) | ~20% | 随机访问大块向量数据 |
| 候选堆操作 (`MinimaxHeap`) | ~8% | pop_min、count_below |
| VisitedTable 分配和查询 | ~7% | 每次搜索 malloc + memset |
| OpenMP 调度 | ~3% | 线程池唤醒和同步 |
| 其他 | ~2% | 图结构遍历等 |

下面所有的优化，都精确地对应到这张表里的某一行或某几行。

### 1.3 为什么不能只靠编译器

原始 FAISS 的距离计算路径是标量循环，依赖编译器自动向量化。这在 HNSW 场景下通常不够，有三个原因：

1. 编译器无法保证把关键循环稳定地优化成 AVX-512 的理想形式。同一段代码换个编译器版本或优化级别，生成的指令可能完全不同。
2. 编译器不会帮你做"跨邻居批量计算"这种源码层面的重组。它能向量化一个循环，但不会把四个独立的循环合并成一个。
3. 编译器完全不管 cache miss、TLB miss、OpenMP 调度开销、VisitedTable 重复分配这些问题。

所以这轮优化的思路不是"开更多编译参数"，而是**针对 CPU 执行模型和 HNSW 的具体访问模式做定向设计**。

---

## 二、各项优化的原因、理由和实现

---

### O1: OpenMP 条件守卫

#### 原因

FAISS 代码里很多搜索路径无条件使用 `#pragma omp parallel for`，即使只有 1 条查询也会触发 OpenMP 线程池。OpenMP 并行区的固定开销包括：线程唤醒、任务分发、barrier 同步，合计约 10-130 微秒。

问题在于：当 QPS > 1000 时，每条查询只有不到 1 毫秒的执行时间。10-130 微秒的固定开销意味着 **1-13% 的时间被花在了"准备并行"上，而不是"实际计算"上**。对单条查询来说，根本不需要并行，直接在调用线程上执行反而更快。

#### 理由

这不是 OpenMP 的 bug，而是粒度不匹配。OpenMP 对长任务很有效，但对"只处理一条向量"这种微任务，它的调度成本大于收益。FAISS 很多地方是无脑加 `omp parallel`，没有区分 batch 大小。

#### 实现

在所有搜索相关的并行循环上加 `if(n > 1)` 条件：

```cpp
// 修改前
#pragma omp parallel for
for (idx_t i = 0; i < n; i++) { ... }

// 修改后
#pragma omp parallel for if(n > 1)
for (idx_t i = 0; i < n; i++) { ... }
```

`n=1` 时不进入并行区，循环直接在调用线程执行。`n > 1` 时行为与原来完全一致。

涉及的文件：`IndexHNSW.cpp`、`IndexBinaryHNSW.cpp`、`IndexFlatCodes.cpp`、`IndexIDMap.cpp`、`IndexNNDescent.cpp`、`IndexNSG.cpp`、`IndexScalarQuantizer.cpp`。

#### 边界条件

- 只对 `n=1` 的单查询场景有价值，batch 查询不受影响。
- 不改变任何搜索结果。

---

### O2: 动态 OpenMP 调度

#### 原因

HNSW 构建阶段 `hnsw_add_vertices()` 原始使用 `schedule(static)`，把向量均匀切分给各线程。但 HNSW 插入有一个天然特性：**越晚插入的向量，需要在越大的图里搜索邻居，计算量越大。**

假设 10M 个向量分配给 16 个线程，每个线程处理 625K 个。前面的线程处理的是"图还很小"时的插入，很快就完成了；后面的线程处理的是"图已经很大"时的插入，要花数倍时间。结果就是：前半段线程早早做完、空转等待，后半段线程成为拖尾瓶颈。

#### 理由

这是经典的负载不均衡问题。静态调度假设每个 iteration 成本相同，但 HNSW 插入不满足这个假设。动态调度让每个线程"做完一批就领下一批"，自动实现负载均衡。

#### 实现

```cpp
// 修改前
#pragma omp for schedule(static)

// 修改后
#pragma omp for schedule(dynamic, 64)
```

chunk 大小选 64 的理由：
- 太小（如 1）：每插一个向量就调度一次，OpenMP runtime 开销过大。
- 太大（如 10000）：尾部仍然会失衡。
- 64 是一个工程折中，足够大以摊薄调度成本，足够小以保证均衡。

涉及的文件：`faiss/IndexHNSW.cpp` — `hnsw_add_vertices()` 函数。

#### 边界条件

- 仅影响构建阶段，不影响搜索路径。
- 动态调度本身有一点 runtime 开销，但远远被负载均衡收益覆盖。

---

### O3: AVX-512 `batch_8` 距离计算

#### 原因

距离计算占搜索时间的 ~60%，是最大的单一热点。原始 FAISS 的问题有两层：

**第一层：编译器自动向量化不可靠。** `fvec_inner_product` 和 `fvec_L2sqr` 是标量循环，依赖编译器优化成 SIMD 指令。但编译器的行为不稳定——换编译器版本、换优化级别、甚至换代码上下文，生成的指令都可能不同。你没法保证它一定会用 AVX-512，也没法保证它用了 AVX-512 就是最优的展开方式。

**第二层：`batch_4` 浪费了 AVX-512 的寄存器资源。** HNSW 搜索路径中，原始代码是"缓冲 4 个邻居，然后批量计算 4 个距离"（`batch_4`）。AVX-512 有 32 个 512-bit 寄存器，每个能存 16 个 float。`batch_4` 只用了 4 个累加器 —— 寄存器利用率仅 **12.5%**。CPU 的 FMA 单元吞吐是每周期 2 个 FMA，但 `batch_4` 每次循环只发 4 个 FMA，远没有把流水线喂满。

#### 理由

要从根本上解决距离计算效率，需要同时做两件事：

1. **手写 SIMD 内核**，不依赖编译器。明确使用 `_mm512_fmadd_ps` 等 intrinsic，保证指令选择确定。
2. **把 batch 大小从 4 扩大到 8**，让 8 个累加器并行做 FMA。每次循环迭代发 8 个 FMA 指令，配合 AVX-512 每周期 2 FMA 的吞吐，流水线能在 4 个周期内完成 —— 而这 4 个周期里 load 单元也刚好能加载下一轮数据。

#### 实现

整个实现分三层，缺一不可：

**第一层：SIMD 内核**

在 `distances_avx512.cpp` 中实现 `fvec_inner_product_batch_8` 和 `fvec_L2sqr_batch_8`：

```cpp
void fvec_inner_product_batch_8(
    const float* x,                          // 查询向量
    const float* y0, ..., const float* y7,   // 8 个数据库向量
    size_t d,
    float& dp0, ..., float& dp7)
{
    __m512 sum0=_mm512_setzero_ps(), ..., sum7=_mm512_setzero_ps();
    for (size_t i = 0; i < d; i += 16) {
        __m512 xi = _mm512_loadu_ps(x + i);  // 查询向量只加载一次
        sum0 = _mm512_fmadd_ps(xi, _mm512_loadu_ps(y0+i), sum0);
        sum1 = _mm512_fmadd_ps(xi, _mm512_loadu_ps(y1+i), sum1);
        ...
        sum7 = _mm512_fmadd_ps(xi, _mm512_loadu_ps(y7+i), sum7);
    }
    dp0 = _mm512_reduce_add_ps(sum0);
    ...
}
```

关键点：查询向量 `x[i:i+16]` 每轮只从内存加载 **1 次**，在 8 个 FMA 中复用。这意味着对 query 的内存带宽消耗降低了 8 倍。

同时提供 AVX2 版本（每次处理 8 个 float，用 `__m256` 和 `_mm256_fmadd_ps`）和标量 fallback，保证在非 AVX-512 平台上仍然正确运行。

**第二层：`DistanceComputer` 接口扩展**

在 `DistanceComputer` 基类添加 `distances_batch_8()` 虚方法：

```cpp
struct DistanceComputer {
    virtual void distances_batch_4(...);  // 原有
    virtual void distances_batch_8(...);  // 新增
};
```

各存储后端（FlatL2Dis、FlatIPDis、GenericFlatCodesDistanceComputer 等）分别实现各自的 `batch_8`，内部调用对应的 SIMD 内核。

**第三层：HNSW 热路径重组**

把 HNSW 搜索中 4 个核心函数的缓冲区从 4 扩大到 8：

```cpp
// 修改前
storage_idx_t saved_j[4];
if (counter == 4) { qdis.distances_batch_4(...); counter = 0; }

// 修改后
storage_idx_t saved_j[8];
if (counter == 8) { qdis.distances_batch_8(...); counter = 0; }
// 剩余 4-7 个用 batch_4 处理，不足 4 个用标量
```

四个热路径全部升级：`search_from_candidates`、`search_from_candidate_unbounded`、`greedy_update_nearest`、`search_neighbors_to_add`。

涉及的文件：`DistanceComputer.h`、`HNSW.cpp`、`IndexFlat.cpp`、`IndexFlatCodes.cpp`、`distances.cpp`、`distances.h`、`distances_dispatch.h`、`distances_avx512.cpp`、`distances_avx2.cpp`、`distances_autovec-inl.h`。

#### 为什么三层都必须改

- 只有 SIMD 内核，没有上层接口 → 热路径调不到。
- 只有接口，没有热路径重组 → 实际调用频率上不去。
- 只有热路径改动，没有 SIMD 内核 → 换个名字走旧实现，没有性能收益。

#### 边界条件

- `batch_8` 只有在邻居缓冲区确实能凑到 8 个时才会触发。如果大部分邻居都已访问过，凑不满 8 个，那么实际命中率会很低 —— 这正是 O11 要解决的问题。

---

### O4: 维度感知软件预取

#### 原因

10M × 768D 的向量数据约 30 GB，远超 CPU 的 L3 缓存（通常 25-50 MB）。HNSW 搜索是随机访问模式 —— 从邻接表取出一个邻居节点 ID 后，需要去内存里加载他的向量数据（3072 字节）来算距离。这个加载几乎一定是 L3 cache miss，延迟 100-200 个时钟周期。

CPU 的乱序执行窗口通常只有 200-300 条指令，无法完全隐藏这么长的内存延迟。如果能在"知道要访问哪个邻居"的时候就提前发出预取指令，让数据在真正需要时已经到达 L1 缓存，那么内存延迟就能被"隐藏"在其他有用计算背后。

#### 理由

关键问题是：预取不是对所有情况都有益。

- **dim=768**（我们的目标场景）：每个向量 3072 字节，跨 48 个 cache line。这么大的数据几乎不可能已经在缓存里，预取收益明确。
- **dim=128**（SIFT 数据集）：每个向量仅 512 字节，8 个 cache line。数据量相对小，有一定概率已经在缓存中。实测表明，对这种场景强制预取反而 **降低** 了性能约 20%，因为 prefetch 指令本身占用了取指/发射槽位。

所以预取必须是**维度感知**的：大向量才值得预取，小向量不要画蛇添足。

#### 实现

在 `DistanceComputer` 基类添加 `prefetch()` 虚方法：

```cpp
virtual void prefetch(idx_t id, int lines = 3) {
    if (code_size >= 1200) {  // 约 dim >= 300 才启用
        const char* ptr = codes + id * code_size;
        for (int i = 0; i < lines; i++) {
            _mm_prefetch(ptr + i * 64, _MM_HINT_T0);  // 预取到 L1
        }
    }
}
```

在 HNSW 热路径中，遍历邻居时提前预取：

```cpp
qdis.prefetch(v1);   // 提前加载向量数据到缓存
vt.prefetch(v1);     // 提前加载 VisitedTable 条目（这个是 FAISS 原有的）
```

**阈值 1200 字节**（约 dim=300）是经过 SIFT-128D、Cohere-768D、GloVe-960D 等多个数据集验证的经验值。

涉及的文件：`DistanceComputer.h`（+22 行）、`HNSW.cpp`（+4 行）。

#### 边界条件

- 对低维数据（dim < 300）不启用，避免负面影响。
- 预取的效果和 O14（图重排）有协同关系：重排后访问模式更有规律，预取的命中率更高。

---

### O5: FP16 SIMD 距离库与 SIMD Dispatch 修正

#### 原因

使用 SQfp16（标量量化 FP16）存储向量，每维从 4 字节降到 2 字节，内存减半。对 HNSW 这种严重受内存访问限制的算法来说，减少数据量不仅节省内存，还能直接提升搜索效率 —— 因为每次从内存加载的向量变小了，cache miss 的代价也变小了。

但这有一个前提：**FP16 距离计算路径必须走到 SIMD 实现，不能退回标量路径。**

#### 理由

FAISS v1.14.1 的 ScalarQuantizer 虽然支持 FP16，但内部距离计算依赖一套 SIMD dispatch 机制来选择使用 AVX-512、AVX2 还是标量。这个 dispatch 机制需要编译时设置 `-DFAISS_OPT_LEVEL=avx512` 才能正确工作。

实际移植过程中发现了两个关键问题：

1. **编译参数缺失**：构建脚本没有设置 `-DFAISS_OPT_LEVEL=avx512`，导致 SIMD dispatch 回退到标量路径。SQfp16 的距离计算走标量，QPS 只有 431 —— 比 FP32 还慢。
2. **ODR（One Definition Rule）冲突**：启用 avx512 后，`batch_8` 的模板特化和显式特化同时存在，链接器在不同翻译单元中看到了同一个符号的两个定义。

这两个问题告诉我们一个重要事实：**"有 SIMD 实现" ≠ "SIMD 实现被使用"。** 编译、链接、dispatch 三个环节必须全部正确，FP16 路径才能真正生效。

#### 实现

**FP16 距离库**

独立建设一套 FP16 距离计算库，三层 SIMD 实现：

```
AVX-512F + F16C: 每次处理 32 个 float16，转换为 2 组 __m512，做 FMA
AVX2 + F16C:     每次处理 8 个 float16，_mm256_cvtph_ps 转换后做 FMA
Scalar:          逐元素 fp16_ieee_to_fp32_value() → 标量乘加
```

API 包括 `fp16vec_L2sqr`、`fp16vec_inner_product`、`batch_4`、`batch_8` 变体。

**Dispatch 修正**

1. 在 `llm/build.sh` 中加入 `-DFAISS_OPT_LEVEL=avx512`。
2. 添加 `FAISS_SKIP_AUTOVEC_BATCH_8` 宏守卫，解决 `batch_8` ODR 冲突。

涉及的文件：`distances_fp16.h`（140 行）、`distances_fp16_simd.cpp`（1249 行）、`test_distances_fp16.cpp`（117 个测试）。

#### 边界条件

- FP16 存储有精度损失。对 768D 余弦相似度场景，Recall 损失约 0.17-0.58 pp，可以接受。
- **必须链接 `libfaiss_avx512.so`**（而不是 `libfaiss.so`），否则 SQ 距离计算仍会退回标量路径。

---

### O10: 透明大页（THP）

#### 原因

10M × 768D 的向量数据 = 30 GB。使用标准 4 KB 页面，需要约 **750 万个页表项**。CPU 的 TLB（Translation Lookaside Buffer）通常只能缓存 1000-2000 个页表项。

HNSW 搜索是随机跳转访问，每次访问一个向量就可能跳到一个完全不同的地址区域。这意味着几乎每次向量访问都会 TLB miss → 触发多级页表遍历（代价 100+ 个时钟周期）。TLB miss 的代价和 cache miss 是叠加的 —— 即使数据在 L3 cache 里，TLB miss 也会增加额外延迟。

#### 理由

使用 2 MB 大页可以把页表项从 750 万降到 **1.5 万**，大幅减少 TLB miss。

但大页并不是对所有场景都有益。核心判断标准是两个条件必须同时满足：

1. **向量要足够大**（> 500 字节）：小向量在 4 KB 页面内就能装好几个，大页带来的收益不明显，反而可能因为页面内部碎片增加浪费。
2. **数据量要足够大**（> 50 万向量）：小数据集的整体内存可能只有几百 MB，TLB 压力本来就不大。

反面案例：
- 1.2M × 100D（400 B/vec）：强开 THP 后性能**下降 40%**。
- 290K × 256D（1024 B/vec）：性能**下降 44%**。

#### 实现

在索引构建完成后，对向量存储区域调用 `madvise(MADV_HUGEPAGE)`：

```cpp
// numa_helpers.h（新文件）
inline void try_enable_hugepages(void* ptr, size_t size) {
    uintptr_t aligned = (uintptr_t(ptr) + 0x1FFFFF) & ~0x1FFFFF;  // 对齐到 2MB 边界
    size_t usable = size - (aligned - uintptr_t(ptr));
    madvise((void*)aligned, usable, MADV_HUGEPAGE);
}
```

在 `IndexHNSW::add()` 末尾调用：

```cpp
auto* flat = dynamic_cast<IndexFlatCodes*>(storage);
if (flat) {
    try_enable_hugepages(flat->codes.data(), flat->codes.size());
}
```

涉及的文件：`numa_helpers.h`（新增 38 行）、`IndexHNSW.cpp`（+8 行）、`CMakeLists.txt`。

#### 边界条件

- 需要 Linux 内核开启 THP 支持（通常默认开启 `madvise` 模式）。
- 对向量 < 500 字节或数据集 < 50 万的场景，**不应启用**。
- 和 O14（BFS 图重排）有协同效应：重排后访问更具局部性，大页覆盖的连续地址区间更容易被有效利用。

---

### O11: 跨节点邻居批处理

#### 原因

O3 引入了 `batch_8` 距离计算内核，理论吞吐很高。但问题是：**在原始 HNSW 热路径中，batch_8 的实际命中率很低。**

原始代码的缓冲逻辑是**每个候选节点独立**的。弹出一个候选节点后，遍历其邻居，把未访问的邻居放入缓冲区，凑满 4 个（或 8 个）就批量计算。但是 HNSW 的特性是：M=32 时每个节点最多 64 个邻居，其中**大部分已经被访问过**。平均每个候选节点只有 2-3 个未访问邻居，很少能凑满 8 个。

实测 `batch_8` 命中率只有 ~30%，大部分时候走的还是 `batch_4` 甚至标量路径。O3 的内核能力被白白浪费了。

#### 理由

根本原因是缓冲区作用域太小。一个候选节点的 2-3 个未访问邻居凑不满 8 个，但连续处理 3-4 个候选节点，就能累积 8-12 个未访问邻居。只要把缓冲区从"每候选节点"提升为"跨候选节点"，batch_8 的命中率就能大幅提升。

这不改变搜索语义 —— 距离计算是顺序无关的，先算哪个邻居不影响最终结果。唯一的区别是"几个邻居被攒在一起算"。

#### 实现

将缓冲区移到候选节点循环**外部**：

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
        if (batch_counter == 8) {
            qdis.distances_batch_8(...);
            batch_counter = 0;
        }
    }
}
flush_batch();  // 剩余的先试 batch_4，再标量
```

同时引入**滑动窗口预取**：收集当前节点的所有邻居 ID 后，以 8 为窗口提前预取后续邻居的向量数据和 VisitedTable 条目。

涉及的文件：`HNSW.cpp` — `search_from_candidates` 完全重写（+114/-78 行）。

#### 为什么 O3 和 O11 必须一起理解

- 没有 O3 → O11 只是把邻居攒在一起，没有高效内核来计算，价值有限。
- 没有 O11 → O3 的内核大部分时候命中不了，80% 的吞吐能力被浪费。

这两项是一个配套设计：**O3 负责"算得快"，O11 负责"有足够多的活可以高效地算"。**

#### 边界条件

- 改动集中在 `search_from_candidates`，是最热的路径，需要极其谨慎地保证缓冲 flush 时机和访问状态一致性。
- 在 efSearch 很小（如 16）的极端情况下，候选节点本来就少，跨节点累积的空间有限，收益会减弱。

---

### O12: SIMD 加速 `MinimaxHeap::count_below`

#### 原因

`search_from_candidates` 中有一个早停判断：调用 `MinimaxHeap::count_below(thresh)`，统计堆中距离小于阈值的元素个数。原始实现是逐元素标量比较循环。当 efSearch=256 时，每次调用需要遍历 256 个 float，而且每弹出一个候选节点就调用一次。

虽然 `count_below` 在总时间中只占 ~8% 的一部分，但它的计算模式极其简单规整 —— 一段连续 float 和同一个阈值逐个比较 —— 是教科书级的 SIMD 应用场景。

#### 理由

对这种"结构简单、频率高、天然适合向量化"的函数，用 SIMD 加速属于低风险高确定性的优化。不做它不会出什么大问题，但做了之后就把这个小热点彻底清理掉了。

#### 实现

```cpp
int MinimaxHeap::count_below(float thresh) {
    int count = 0;
    __m512 vt = _mm512_set1_ps(thresh);
    size_t i = 0;
    for (; i + 16 <= k; i += 16) {
        __m512 vd = _mm512_loadu_ps(dis.data() + i);
        __mmask16 mask = _mm512_cmp_ps_mask(vd, vt, _CMP_LT_OS);
        count += _mm_popcnt_u32(mask);
    }
    for (; i < k; i++) {
        if (dis[i] < thresh) count++;
    }
    return count;
}
```

AVX-512 每次迭代比较 16 个 float，AVX2 每次 8 个，尾部用标量处理。

涉及的文件：`HNSW.cpp`（+43 行）。

#### 边界条件

- 不改变任何搜索语义或结果。
- 对小 efSearch（如 16-32）效果不明显，因为待比较的元素很少。

---

### O13: `SharedVectorStore` 零拷贝重建

#### 原因

HNSW 索引在生产中需要定期重建：删除向量后图的连通性和质量会退化。传统重建流程是"创建新索引 → 把所有向量**复制**到新索引 → 重建图结构"。

问题在于：10M × 768D 的索引约 30 GB 向量数据。重建时需要同时持有旧索引和新索引，峰值内存约 60 GB —— 很可能超出机器可用内存。结果就是"理论上需要重建，实际上机器内存不够装不下。"

#### 理由

根本问题是：**重建的目的是重建图结构，不是重建向量数据。** 向量数据在重建前后完全相同（只是少了被删除的那些）。把所有向量复制一遍完全是浪费。

解决思路是把"向量存储"和"图结构"这两部分解耦开来。旧索引和新索引可以共享同一份向量数据，各自只维护自己的图结构和映射关系。

#### 实现

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

核心组件：
- `SharedVectorStore`：持有底层向量数据，使用 `shared_ptr` 管理生命周期。
- `IndexFlatShared`：替代 `IndexFlat` 作为存储后端，使用 `storage_id_map` 做间接寻址，支持 `deleted_bitmap`。
- `IndexHNSW::add(n, nullptr)`：传入 `nullptr` 表示"不复制向量，直接从 SharedVectorStore 读取"。
- `compact_store()`：重建完成后，用原地 cycle-following 算法重排存储，消除间接寻址开销。

涉及的文件：`SharedVectorStore.h/cpp`（新增）、`IndexFlatShared.h/cpp`（新增 744 行）、`IndexHNSW.cpp`。

#### 边界条件

- 重建期间搜索走间接寻址路径，有约 1-2% 的性能开销。`compact_store()` 执行后恢复直接寻址。
- 需要维护 shared_ptr 所有权、删除槽位和映射关系的一致性。
- 如果不需要重建（比如只读场景），这项优化不是首要需求。

---

### O14: BFS 图重排

#### 原因

HNSW 的节点 ID 是按插入顺序分配的，和图拓扑完全无关。两个节点在图上是邻居，但它们的向量数据在内存中可能相距数 GB。

这导致搜索路径上的向量访问呈现高度离散的随机跳转，CPU 的缓存系统和硬件预取器几乎无法利用空间局部性。更具体地说：
- L1/L2 hardware prefetcher 依赖访问的连续性来预测下一次访问位置。如果连续访问的地址间距太大（数 GB），预取器无法学习到有效模式。
- CPU 的 L1/L2 cacheline 替换也无法利用"邻居可能很快被访问"这个事实，因为邻居的数据在完全不同的地址。

#### 理由

如果不改变图语义，只通过重新编号和重排存储，让"图上相邻的节点在内存中也相邻"，就能显著改善 cache 局部性。

选择 **BFS** 作为主要重排策略的理由：
1. HNSW 搜索从入口点开始，向外逐层扩展 —— 这本身就是 BFS 的访问模式。
2. BFS 编号把入口点附近的节点排在前面，搜索路径上连续访问的节点在内存中也是连续的。
3. 相比 RCM、DFS 等其他图重排策略，BFS 在 HNSW 上效果最稳定。

#### 实现

提供 5 种重排策略（BFS、RCM、DFS、Cluster、Weighted），BFS 为默认推荐：

```cpp
auto perm = generate_permutation(hnsw, ReorderStrategy::BFS);
index->permute_entries(perm.data());
```

`permute_entries` 同时重排两部分：
1. **图结构**：邻接表中所有 ID 引用按新编号更新。
2. **向量数据**：`IndexFlatCodes::codes` 数组按新顺序重排。

涉及的文件：`HNSWReorder.h/cpp`（新增 367 行）、`IndexFlatShared.cpp`、`CMakeLists.txt`。

#### 为什么 O14 和 O10、O4 有协同效应

- **和 O10 的协同**：重排后连续访问的节点在内存中也连续，大页覆盖的 2 MB 区间更可能包含即将访问的数据。
- **和 O4 的协同**：重排后邻居节点在地址上更近，软件预取请求的目标地址更可能在预取完成时确实被使用。

所以 O14 不仅自身有收益，还放大了 O10 和 O4 的效果。

#### 边界条件

- 需要额外临时内存（约等于 codes 数组大小）。10M FP32 索引需要约 64 GB 临时内存，在 62 GB 机器上会 OOM。SQfp16 只需约 15 GB，可以完成。
- 重排后搜索结果中的内部 ID 需要做反向映射。
- 适合索引建好后做一次离线重排的场景。如果数据持续快速变动，重排的维护成本更高。

---

### O16: VisitedTable 复用

#### 原因

每次 HNSW 搜索都会新分配一个 `VisitedTable`，大小等于 `ntotal` 字节。10M 索引意味着每次搜索：`malloc(10 MB)` + `memset(10 MB, 0)` + `free(10 MB)`。在 QPS > 1000 的场景下，这每秒执行上千次。

讽刺的是，`VisitedTable` 本身有一个精巧的设计：内部维护一个 `visited_generation` 计数器，调用 `advance()` 时只需把计数器 +1 就能"逻辑清空"整个表（$O(1)$ 而非 $O(n)$）。但如果每次搜索都重新分配一个新的 VisitedTable，这个设计就完全浪费了 —— 你都没有第二次 `advance()` 的机会，因为对象在搜索结束就被释放了。

#### 理由

VisitedTable 是一个典型的"查询级临时状态"对象。它的特征是：
1. 生命周期可以跨查询。
2. 逻辑清空成本远低于物理清空。
3. 内容不需要长期保留。

满足这三个条件的对象，天然适合复用而不是每次新建。

#### 实现

在 `SearchParametersHNSW` 中添加可选字段：

```cpp
struct SearchParametersHNSW : SearchParameters {
    int efSearch = 0;
    VisitedTable* visited_table = nullptr;  // 新增
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

调用方的使用方式：

```cpp
VisitedTable vt(index->ntotal);  // 只分配一次
SearchParametersHNSW params;
params.efSearch = 64;
params.visited_table = &vt;
for (int i = 0; i < nq; i++) {
    index->search(1, query_i, k, distances, labels, &params);
    // vt 被自动复用，仅 advance() 一次
}
```

涉及的文件：`HNSW.h`（+1 行）、`IndexHNSW.cpp`（+16 行）。

#### 为什么放在 SearchParameters 里而不是用 thread-local 缓存

1. `SearchParameters` 已经是查询级配置的标准扩展点。
2. 不强迫全局状态或隐式缓存。
3. 调用方可以显式管理生命周期，不会引入不可控的内存占用。

如果用 TLS 或隐藏缓存，虽然使用上更"自动"，但会引入更难理解的生命周期问题，也更难在多租户场景下控制内存。

#### 边界条件

- 调用方不传 `visited_table` 时，行为和原来完全一致。
- 每个并发查询上下文必须有自己的 VisitedTable，不能多线程共享一个可写的 VT。

---

## 三、优化之间的协同关系

这些优化不是孤立的，有几组重要的协同关系：

### 3.1 计算路径：O3 + O11 + O5

- **O3** 提供高吞吐 `batch_8` 内核。
- **O11** 让 `batch_8` 的实际命中率从 ~30% 提升到 ~80%。
- **O5** 让 FP16 路径也走到正确的 SIMD 实现。

缺 O11，O3 的能力发挥不出来；缺 O5 的 dispatch 修正，FP16 路径悄悄退回标量。

### 3.2 访存路径：O14 + O10 + O4

- **O14（BFS 重排）** 改善空间局部性，让连续访问的节点在内存中也连续。
- **O10（大页）** 减少 TLB miss，大页覆盖的连续地址范围和重排后的局部性匹配。
- **O4（预取）** 提前拉取数据到缓存，重排后预取目标更准确。

O14 是这组的基础 —— 有了更好的空间布局，O10 和 O4 才能更有效地工作。

### 3.3 控制开销：O1 + O12 + O16

- **O1** 减少单查询的线程调度固定成本。
- **O12** 减少候选堆早停判断的小热点。
- **O16** 消除查询级的内存分配和清零。

这三项都在减少"每次查询不得不付出的固定税费"。单独看每项都不大，但累积起来不可忽视。

---

## 四、编译和链接的关键约束

有三件事如果做错了，上面很多优化会静默失效：

1. **必须使用 `-lfaiss_avx512` 链接**，而不是 `-lfaiss`。后者加载通用库，SQfp16 距离计算走标量路径。
2. **必须在 cmake 中设置 `-DFAISS_OPT_LEVEL=avx512`**。否则 ScalarQuantizer 的 SIMD dispatch 回退到标量。
3. **`batch_8` 的 ODR 冲突必须处理**。使用 `FAISS_SKIP_AUTOVEC_BATCH_8` 宏守卫防止模板特化和显式特化冲突。

这些不是可选的"最佳实践"，而是必要条件。

```bash
# FAISS 库编译
cmake -DFAISS_OPT_LEVEL=avx512 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -ffast-math -funroll-loops" \
      -DBLA_VENDOR=Intel10_64_dyn ..

# 客户端程序编译
g++ -O3 -march=native -mtune=native -std=c++17 \
    -I install/include -o bench_task bench_task.cpp \
    -L install/lib -lfaiss_avx512 \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 \
    -lpthread -lm -ldl -Wl,-rpath,install/lib
```

---

## 附录：优化一览表

| ID | 名称 | 解决的问题 | 影响阶段 |
|----|------|-----------|---------|
| O1 | OMP 条件守卫 | 单查询走 OpenMP 的无谓调度开销 | search |
| O2 | 动态 OMP 调度 | 构建阶段静态调度导致的负载不均 | build |
| O3 | AVX-512 `batch_8` | 距离计算的寄存器利用率和指令级并行不足 | search |
| O4 | 维度感知预取 | 高维向量 cache miss 的内存延迟 | search |
| O5 | FP16 SIMD 库 + dispatch 修正 | FP16 路径静默退回标量 | search |
| O10 | 透明大页 | 大规模随机访问的 TLB miss | search + build |
| O11 | 跨节点邻居批处理 | `batch_8` 命中率太低 | search |
| O12 | SIMD `count_below` | 早停判断的标量循环 | search |
| O13 | `SharedVectorStore` | 重建期间内存峰值翻倍 | rebuild |
| O14 | BFS 图重排 | 节点内存布局与图拓扑脱耦导致的 cache 局部性差 | search |
| O16 | VisitedTable 复用 | 每查询 malloc+memset+free 大数组 | search |
