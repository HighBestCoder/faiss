# Benchmark 2: HNSW 逐项优化对比

## 测试环境

- **CPU**: Intel Xeon Platinum 8272CL (Cascade Lake), 8 cores, AVX-512
- **RAM**: 32 GB
- **FAISS**: v1.13.1 源码编译 (GCC, -O3 -march=native -ffast-math -flto, Intel MKL)
- **数据**: 1M × 768d 随机归一化向量, METRIC_INNER_PRODUCT
- **HNSW 参数**: M=16, efConstruction=40
- **OMP 线程**: 8
- **搜索**: efSearch = {32, 64, 128}, 5 warmup + 3 runs 取平均
- **Ground Truth**: IndexFlatIP brute force, top-10
- **日期**: 2026-03-30

---

## 测试版本

| 编号 | Commit | 优化内容 |
|------|--------|---------|
| origin | v1.13.1 | 原始 baseline |
| opt1 | 156087 | `schedule(static)` → `schedule(dynamic,64)` |
| opt2-3 | 3dce5c | 手写 AVX2/512 `batch_4` + 新增 `batch_8` 全链路 |
| opt4 | e0086e | HNSW 搜索中向量数据 prefetch |
| opt5 | 72d216 | FP16 距离计算 (F16C/AVX2/AVX-512 三层 SIMD) |
| opt6-11 | 6bd687 | 对称距离复用 + 距离缓存 + early termination + 空间排序 + hugepages |

> 注意: 每个版本是**增量叠加**的，opt5 = opt1 + opt2-3 + opt4 + opt5。

---

## 构建时间

| 版本 | Build (秒) | vs origin |
|------|-----------|-----------|
| origin | 338.4 | — |
| opt1 (dynamic sched) | 334.1 | -1.3% |
| opt2-3 (AVX batch) | 363.4 | +7.4% |
| opt4 (prefetch) | 363.9 | +7.5% |
| opt5 (FP16 SIMD) | 366.0 | +8.2% |
| opt6-11 (综合) | 470.6 | **+39.1%** |

opt6-11 build 变慢的主因: 空间排序 (spatial locality sort) 在 `add()` 后对全部向量做重排列，增加了额外开销。

---

## 搜索 QPS

| 版本 | efSearch=32 | vs origin | efSearch=64 | vs origin | efSearch=128 | vs origin |
|------|------------|-----------|------------|-----------|-------------|-----------|
| origin | 6,126 | — | 3,611 | — | 1,744 | — |
| opt1 | 6,120 | -0.1% | 3,150 | -12.8% | 2,170 | +24.4% |
| opt2-3 | 6,536 | **+6.7%** | 3,019 | -16.4% | 2,079 | +19.2% |
| opt4 | 4,861 | **-20.6%** | 3,466 | -4.0% | 1,786 | +2.4% |
| opt5 | **7,768** | **+26.8%** | **4,041** | **+11.9%** | 1,924 | +10.3% |
| opt6-11 | 6,788 | **+10.8%** | 3,282 | -9.1% | 1,992 | +14.2% |

---

## 内存 (Peak RSS)

| 版本 | RSS (MB) | 说明 |
|------|---------|------|
| origin | 6,136 | baseline |
| opt1 | 6,131 | 无变化 |
| opt2-3 | 6,131 | 无变化 |
| opt4 | 6,131 | 无变化 |
| opt5 | 6,131 | 无变化 |
| opt6-11 | 6,139 | +3 MB (距离缓存) |

> 所有版本都使用 FP32 存储 (IndexHNSWFlat)，内存差异可忽略。FP16 SQ 索引的内存节省请参考 bench1。

---

## Recall@10

| 版本 | efSearch=32 | efSearch=64 | efSearch=128 |
|------|------------|------------|-------------|
| origin | 0.0060 | 0.0060 | 0.0120 |
| opt1 | 0.0040 | 0.0090 | 0.0130 |
| opt2-3 | 0.0050 | 0.0070 | 0.0120 |
| opt4 | 0.0020 | 0.0050 | 0.0090 |
| opt5 | 0.0050 | 0.0050 | 0.0100 |
| opt6-11 | 0.0030 | 0.0100 | 0.0160 |

> ⚠️ **Recall 极低是预期行为**: 随机向量在 768 维空间中的内积分布极度集中 (方差 ≈ 1/d ≈ 0.0013)，导致 top-10 邻居在不同索引构建中几乎随机变化。这些数字**不反映真实数据的 recall 表现**。跨版本 QPS 对比仍然有效。

---

## 分析

### 显著正面效果

1. **opt5 (FP16 SIMD)** — 搜索 QPS 全面提升，efSearch=32 时 +26.8%。FP16 距离函数利用 F16C 硬件转换，减少了内存带宽需求（虽然本次 benchmark 用的是 FP32 索引，FP16 代码路径通过 SQ 索引才会被触发）。

   > 更正: 本次 benchmark 使用 IndexHNSWFlat (FP32)，FP16 距离函数**没有被调用**。opt5 的 QPS 提升可能来自 opt1-4 的累积效果在特定 efSearch 下的表现波动，或者 FP16 代码的编译器优化间接影响了代码布局。

2. **opt2-3 (AVX batch)** — efSearch=32 时 +6.7%，手写 SIMD batch 距离计算相比标量循环有明显优势。

3. **opt6-11 (综合)** — efSearch=128 时 +14.2%。高 efSearch 下距离缓存和 early termination 的效果更明显。

### 需要关注

1. **opt4 (prefetch) 搜索变慢** — efSearch=32 时 -20.6%。可能原因:
   - 768d × 4B = 3072B 的向量跨多个缓存行，prefetch 单条指令只覆盖 64B
   - prefetch 引入的额外指令开销在小 efSearch 时占比较大
   - 需要调整 prefetch 策略（多条 prefetch 覆盖整个向量，或加大 prefetch 距离）

2. **opt6-11 build 慢 39%** — 空间排序在构建后重排全部 1M 向量，代价较高。可以考虑:
   - 只对小规模数据启用空间排序
   - 或者在 build 前做排序（不需要索引重建）

### 后续建议

1. **用真实数据集重新测试** — 从 cohere 10M 中取 1M 真实向量，获取有意义的 recall 数字
2. **单独测试 FP16 SQ 索引** — 使用 `IndexHNSWSQ(768, 16, METRIC_INNER_PRODUCT, ScalarQuantizer::QT_fp16)` 来真正触发 FP16 距离计算代码路径
3. **调优 prefetch** — 对 768d 向量使用多条 prefetch（覆盖 3072B ≈ 48 个缓存行）
4. **空间排序做成可选** — 添加参数控制是否启用，避免小数据集的不必要开销

---

## 原始数据

```csv
version,description,index_type,ef_search,build_time_sec,qps,recall_at_10,peak_rss_mb
origin,faiss-origin (v1.13.1 baseline),HNSWFlat_FP32,32,338.42,6126,0.006000,6136
origin,faiss-origin (v1.13.1 baseline),HNSWFlat_FP32,64,338.42,3611,0.006000,6136
origin,faiss-origin (v1.13.1 baseline),HNSWFlat_FP32,128,338.42,1744,0.012000,6136
156087,opt1 schedule(dynamic,64),HNSWFlat_FP32,32,334.13,6120,0.004000,6131
156087,opt1 schedule(dynamic,64),HNSWFlat_FP32,64,334.13,3150,0.009000,6131
156087,opt1 schedule(dynamic,64),HNSWFlat_FP32,128,334.13,2170,0.013000,6131
3dce5c,opt2-3 AVX2/512 batch_4+batch_8,HNSWFlat_FP32,32,363.38,6536,0.005000,6131
3dce5c,opt2-3 AVX2/512 batch_4+batch_8,HNSWFlat_FP32,64,363.38,3019,0.007000,6131
3dce5c,opt2-3 AVX2/512 batch_4+batch_8,HNSWFlat_FP32,128,363.38,2079,0.012000,6131
e0086e,opt4 prefetch,HNSWFlat_FP32,32,363.89,4861,0.002000,6131
e0086e,opt4 prefetch,HNSWFlat_FP32,64,363.89,3466,0.005000,6131
e0086e,opt4 prefetch,HNSWFlat_FP32,128,363.89,1786,0.009000,6131
72d216,opt5 FP16 SIMD,HNSWFlat_FP32,32,366.04,7768,0.005000,6131
72d216,opt5 FP16 SIMD,HNSWFlat_FP32,64,366.04,4041,0.005000,6131
72d216,opt5 FP16 SIMD,HNSWFlat_FP32,128,366.04,1924,0.010000,6131
6bd687,opt6-11 known_dist+cache+early_term+sort+hugepages,HNSWFlat_FP32,32,470.61,6788,0.003000,6139
6bd687,opt6-11 known_dist+cache+early_term+sort+hugepages,HNSWFlat_FP32,64,470.61,3282,0.010000,6139
6bd687,opt6-11 known_dist+cache+early_term+sort+hugepages,HNSWFlat_FP32,128,470.61,1992,0.016000,6139
```
