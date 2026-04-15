# FAISS HNSW 构建优化方案

> 场景：10M × 768d 向量，IndexHNSWFlat，M=32，efConstruction=512

## 一、软件/指令工程层面

### 1. 手写 AVX2 `fvec_L2sqr_batch_4`（优先级最高）

**现状**：`fvec_L2sqr_batch_4` 和 `fvec_inner_product_batch_4`（`distances_simd.cpp:235-301`）都是纯标量循环，依赖编译器 `FAISS_PRAGMA_IMPRECISE_LOOP` 自动向量化。而单向量的 `fvec_L2sqr` 已有完整的 SSE/AVX2/AVX-512 手写版本。

**优化方案**：

```cpp
// AVX2 手写 batch_4，768 维时每次处理 8 个 float
#ifdef __AVX2__
void fvec_L2sqr_batch_4_avx2(
        const float* x,
        const float* y0, const float* y1,
        const float* y2, const float* y3,
        const size_t d,
        float& dis0, float& dis1, float& dis2, float& dis3) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (size_t i = 0; i < d; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 d0 = _mm256_sub_ps(xv, _mm256_loadu_ps(y0 + i));
        __m256 d1 = _mm256_sub_ps(xv, _mm256_loadu_ps(y1 + i));
        __m256 d2 = _mm256_sub_ps(xv, _mm256_loadu_ps(y2 + i));
        __m256 d3 = _mm256_sub_ps(xv, _mm256_loadu_ps(y3 + i));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    dis0 = horizontal_sum(acc0);
    dis1 = horizontal_sum(acc1);
    dis2 = horizontal_sum(acc2);
    dis3 = horizontal_sum(acc3);
}
#endif
```

**收益**：
- x[i] 只从内存加载一次，被 4 次距离计算复用
- 每次循环 4 条 FMA 指令，充分利用 CPU 流水线（ILP）
- 768 / 8 = 96 次迭代，768 是 8 的倍数无需处理尾部
- 预期距离计算提速 2-4x

### 2. Batch 大小从 4 扩展到 8/16

**现状**：`search_neighbors_to_add`（`HNSW.cpp:413-445`）和搜索路径中固定 batch=4。AVX-512 寄存器有 16 个 `__m512`，一次能处理更多邻居。

**优化方案**：

```cpp
// batch_8: 更好利用 AVX-512 寄存器和内存带宽
void fvec_L2sqr_batch_8(const float* x,
    const float* y0, ..., const float* y7,
    size_t d, float& dis0, ..., float& dis7);
```

对应修改 HNSW.cpp 中的 buffered_ids 从 4 扩大到 8。

**收益**：AVX-512 环境下额外提速 30-50%。

### 3. 向量数据预取（Prefetch）

**现状**：搜索路径（`HNSW.cpp:662`）只预取了 visited 表，**没有预取向量数据本身**。10M × 768d × 4B ≈ 29 GB 远超 L3 cache，每次距离计算都会 cache miss。

**优化方案**：

```cpp
// 在遍历邻居列表时，提前预取下一批向量数据
for (size_t j = begin; j < end; j++) {
    int v1 = hnsw.neighbors[j];
    prefetch_L2(vt.visited.data() + v1);
    // 新增：预取向量数据到 L2 cache（3072 bytes = 48 个 cacheline）
    const char* vec_ptr = (const char*)(codes + (size_t)v1 * code_size);
    prefetch_L2(vec_ptr);
    prefetch_L2(vec_ptr + 64);
    prefetch_L2(vec_ptr + 128);  // 按需预取前几个 cacheline
}
```

同样应用于构建路径 `search_neighbors_to_add` 和 `greedy_update_nearest`。

**收益**：整体 10-30%，隐藏内存延迟。

### 4. OpenMP 调度策略优化

**现状**：`IndexHNSW.cpp:147` 使用 `schedule(static)`，注释称 dynamic 在某些 LLVM 版本上 segfault。

```cpp
#pragma omp for schedule(static)
```

HNSW 插入耗时极度不均匀——早期插入快（图小），后期越来越慢。static 分配导致线程负载不均。

**优化方案**：

```cpp
#pragma omp for schedule(dynamic, 64)
```

在新版编译器上测试是否仍有 segfault 问题。如果不行，可用 `schedule(guided, 32)` 作为折中。

**收益**：多线程效率提升 20-40%。

### 5. FP16/BF16 距离计算

**现状**：全部使用 FP32 存储和计算。

**优化方案**：

- 向量数据用 FP16 存储，距离计算时转换为 FP32（AVX-512 有原生 `_mm512_cvtph_ps`）
- 或使用 BF16（更适合机器学习场景）
- 内存占用从 29 GB 降到 14.5 GB，更多数据留在 cache

```cpp
// FP16 距离计算
__m256 xv = _mm256_loadu_ps(x_fp32 + i);
__m256 yv = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y_fp16 + i)));
__m256 diff = _mm256_sub_ps(xv, yv);
acc = _mm256_fmadd_ps(diff, diff, acc);
```

**收益**：
- 内存减半，cache 命中率提升约 2x
- 对 HNSW 图质量影响极小（距离只用于排序，不需要高精度）

---

## 二、算法层面

### 6. 插入顺序空间局部性排序

**现状**：按层级分批，每批内随机打乱：

```cpp
for (int j = i0; j < i1; j++) {
    std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);
}
```

**优化方案**：

用 Random Projection 或 k-means 聚类对向量按空间位置排序后再构建。相邻插入的向量更可能是彼此的邻居 → 距离计算时向量数据更可能已在 cache 中。

```
1. 对 10M 向量做随机投影到 1D → 排序
2. 或者做 k-means(k=1024) → 按 cluster 分组排序
3. 按排序后的顺序插入 HNSW
```

**收益**：构建提速 10-20%，cache 友好。

### 7. 更激进的 Early Termination

**现状**：`search_neighbors_to_add` 的停止条件：

```cpp
if (currEv.d > results.top().d) {
    break;  // 候选集最近的比结果集最远的还要远
}
```

**优化方案**：

在 efConstruction 很大时（如 512），增加额外的收敛检测：若已收集到足够多的候选且 top-M 的距离已稳定（近几步无更新），提前终止。

```cpp
int no_improvement_count = 0;
float prev_top_m_dist = results.top().d;
// ... 在循环中
if (results.size() >= M && results.top().d >= prev_top_m_dist) {
    no_improvement_count++;
    if (no_improvement_count > threshold) break;
} else {
    no_improvement_count = 0;
    prev_top_m_dist = results.top().d;
}
```

**收益**：大 efConstruction 场景下减少 30-50% 无效距离计算。

### 8. Neighbor Selection 距离缓存

**现状**：`add_link` 中 shrink 邻居列表时，`symmetric_dis` 被重复计算。`shrink_neighbor_list` 中对 output 中每对节点调用 `symmetric_dis`，这些距离没有被缓存。

**优化方案**：

在 shrink 过程中维护一个距离缓存 map，避免对同一对节点重复计算距离。

**收益**：构建提速 5-10%。

### 9. 双向边对称距离复用

**现状**：插入节点 A 后与邻居 B 建立双向边时：

```cpp
add_link(*this, ptdis, pt_id, other_id, ...);  // A→B
add_link(*this, ptdis, other_id, pt_id, ...);   // B→A，重新算 d(B,A)
```

d(A,B) 在第一步已经知道，但 B→A 的 add_link 内部会重算 `symmetric_dis(B, A)`。

**优化方案**：

将已知的 d(A,B) 作为参数传递给 B→A 的 add_link，避免重复计算。

**收益**：构建提速 5-10%。

---

## 三、系统层面

### 10. NUMA 感知内存分配

多路服务器（如双路 Xeon）上，跨 NUMA 节点访问内存带宽减半、延迟翻倍。29 GB 的向量数据不会自动按 NUMA 优化分配。

**方案**：
```bash
# 方案 A：交错分配（简单）
numactl --interleave=all ./build_index

# 方案 B：手动绑定（更优）
# 向量数据 mbind() 到本地 NUMA 节点
```

**收益**：双路机器上 20-30%。

### 11. 大页（Huge Pages）

29 GB 数据用 4KB 页面 → 约 750 万个页表项 → 大量 TLB miss。

**方案**：
```bash
# 预留 2MB 大页
echo 15000 > /proc/sys/vm/nr_hugepages

# 或用 transparent huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

用 `mmap(MAP_HUGETLB)` 或 `aligned_alloc` + `madvise(MADV_HUGEPAGE)` 分配向量存储。

**收益**：TLB miss 减少 → 整体 5-15%。

---

## 四、优先级总结

| 编号 | 优化项 | 难度 | 预期收益 | 影响范围 |
|------|--------|------|----------|----------|
| 1 | 手写 AVX2 batch_4 | 中 | 距离计算 2-4x | 构建 + 搜索 |
| 3 | 向量数据预取 | 低 | 整体 10-30% | 构建 + 搜索 |
| 5 | FP16 存储 + 计算 | 中 | 内存 2x, cache 2x | 构建 + 搜索 |
| 4 | dynamic 调度 | 低 | 多线程效率 20-40% | 构建 |
| 6 | 空间局部性排序 | 中 | 构建 10-20% | 构建 |
| 9 | 对称距离复用 | 低 | 构建 5-10% | 构建 |
| 2 | batch_8/16 | 中 | AVX-512 额外 30-50% | 构建 + 搜索 |
| 7 | 激进 early termination | 中 | 大 efC 下 30-50% | 构建 |
| 8 | 距离缓存 | 低 | 构建 5-10% | 构建 |
| 10 | NUMA 感知 | 低 | 双路机器 20-30% | 构建 + 搜索 |
| 11 | 大页 | 低 | 5-15% | 构建 + 搜索 |

**最优先实施**：#1（手写 SIMD batch）、#3（向量预取）、#4（dynamic 调度），改动量小但效果显著。
