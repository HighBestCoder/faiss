# 移植所有搜索路径优化到 v1.14.1

## 背景

v1.13.1+opt 分支上自研的 HNSW 搜索路径优化使 HNSW32,SQfp16 达到 1648 QPS。
但 v1.14.1 baseline 只有 382 QPS，差距 4.3 倍。

目标：将所有搜索路径优化 + SharedVectorStore 架构移植到 v1.14.1，
同时获得新版本特性（RaBitQ 等）和自研优化的双重收益。

## 移植清单（11 个优化 + 1 个架构特性）

| 序号 | ID | 优化 | 源 commit | 移植方式 |
|------|-----|------|-----------|---------|
| 1 | O1 | OMP Guard（单查询跳过线程池） | d4bd2b47 | cherry-pick |
| 2 | O2 | 动态 OMP 调度（HNSW 构建） | 15608779 | cherry-pick |
| 3 | O5 | FP16 SIMD 距离库 | 72d21687 | 手动提取文件 + CMake |
| 4 | O12 | SIMD count_below（AVX512/AVX2） | 29b00333 | cherry-pick |
| 5 | O3 | batch_8 SIMD 距离计算 + HNSW 热路径升级 | 3dce5c2b | 手动移植（6 文件） |
| 6 | O4 | 向量数据预取（维度感知） | e0086e29 | 手动移植 |
| 7 | O11 | 跨节点邻居批处理 | ec03f45e | 手动重写 search_from_candidates |
| 8 | O10 | 透明大页 THP | 6bd687d7 | 手动提取 |
| 9 | O16 | VisitedTable 复用 | patch 文件 | 手动移植 |
| 10 | O13 | SharedVectorStore + IndexFlatShared | 87c59cc8 | 手动提取 + 适配 |
| 11 | O14 | BFS 图重排 | 062724c7 | 手动提取 + 适配 |

## 遇到的问题

### 问题 1：SQfp16 性能远低于预期（431 vs 1648 QPS）

**现象**：移植完所有优化后，HNSW32,SQfp16 只从 382 提升到 431 QPS（+12.8%），
远低于 v1.13.1+opt 的 1648 QPS。

**根因**：v1.14.1 重构了 SIMD 调度架构。

- v1.13.1：单文件架构（ScalarQuantizer.cpp），用编译器宏 `#ifdef __AVX2__` 直接启用 SIMD
- v1.14.1：多文件调度架构（sq-avx2.cpp, sq-avx512.cpp），受 `FAISS_OPT_LEVEL` CMake 变量控制

我们的 build.sh 没有设置 `FAISS_OPT_LEVEL`，默认为 OFF，导致：
1. `COMPILE_SIMD_AVX2` / `COMPILE_SIMD_AVX512` 宏未定义
2. SQ 距离计算走标量路径（逐元素 FP16→FP32 转换 + 标量累加）
3. d=768 需要 768 次标量循环，而 SIMD 路径只需 96 次（AVX2，8 floats/iter）

**解决**：在 build.sh 添加 `-DFAISS_OPT_LEVEL=avx512`

### 问题 2：batch_8 模板特化重复定义（ODR violation）

**现象**：加上 `FAISS_OPT_LEVEL=avx512` 后编译失败：
```
error: redefinition of 'void faiss::fvec_inner_product_batch_8<SIMDLevel::AVX2>'
```

**根因**：O3 移植时在三个地方都定义了 batch_8 函数：
- `distances_autovec-inl.h`（模板特化，AUTOVEC_LEVEL）
- `distances_avx2.cpp`（手写 AVX2 实现）
- `distances_avx512.cpp`（手写 AVX512 实现）

当 `FAISS_OPT_LEVEL=avx512` 时，编译 `faiss_avx2` 目标会 include autovec-inl.h，
此时 `AUTOVEC_LEVEL = SIMDLevel::AVX2`，和 avx2.cpp 里的显式特化冲突。

**解决**：
1. 在 `distances_autovec-inl.h` 的 batch_8 函数外加 `#ifndef FAISS_SKIP_AUTOVEC_BATCH_8` 保护
2. 在 `distances_avx2.cpp` 和 `distances_avx512.cpp` include autovec-inl.h 前定义 `FAISS_SKIP_AUTOVEC_BATCH_8`

### 问题 3：cherry-pick 冲突

**现象**：O3（batch_8）和 O5（FP16 库）的 cherry-pick 因 CMakeLists.txt 结构变化而冲突。

**解决**：放弃 cherry-pick，改为手动提取源文件 + 手动编辑 CMakeLists.txt。

### 问题 4：v1.14.1 API 变化

**现象**：
- v1.14.1 的 `DistanceComputer::prefetch()` 签名变了（多了 `int lines` 参数）
- v1.14.1 的 `FlatCodesDistanceComputer` 新增了 `partial_dot_product_batch_4`
- v1.14.1 的 `VisitedTable` 构造函数多了 `use_visited_hashset` 参数

**解决**：逐一适配 API 变化，保持向后兼容。

### 改进：维度感知 prefetch

根据实际经验，低维度（d < 300）时 prefetch 开销大于收益。
在 `FlatCodesDistanceComputer::prefetch()` 中加了 `code_size >= 1200` 阈值：
- d >= 300 (FP32) 或 d >= 600 (FP16)：启用 prefetch
- 否则跳过

## 测试方法

### 编译验证
每个优化 commit 后都运行 `bash llm/build.sh` 确认编译通过。

### 性能验证
使用 `llm/bench_task.cpp` 在 cohere_medium_1m 数据集（1M 向量，768 维，cosine）上测试：
- 索引类型：HNSW32,Flat 和 HNSW32,SQfp16
- 搜索参数：efSearch = 64, 128, 256（取 Recall@10 >= 95% 的最高 QPS）
- 单线程 QPS（逐条查询）和 16 线程 QPS（批量查询）

编译命令：
```bash
g++ -O3 -march=native -std=c++17 \
    -I install/include -o llm/bench_task llm/bench_task.cpp \
    -L install/lib -lfaiss_avx512 \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
    -liomp5 -lpthread -lm -ldl \
    -Wl,-rpath,/ceph/faiss-dev/install/lib
```

注意：必须链接 `libfaiss_avx512`（不是 `libfaiss`），否则 SQ 走标量路径。

## 结果

### HNSW32,Flat (efSearch=64)

| 阶段 | QPS_1T | QPS_16T | Recall@10 | RSS |
|------|--------|---------|-----------|-----|
| v1.14.1 baseline | 1197 | 8804 | 96.18% | 6340 MB |
| v1.14.1 + all opts | **1281** | **10843** | 96.28% | 6341 MB |
| **提升** | **+7.0%** | **+23.2%** | - | - |

### HNSW32,SQfp16 (efSearch=64)

| 阶段 | QPS_1T | QPS_16T | Recall@10 | RSS |
|------|--------|---------|-----------|-----|
| v1.14.1 baseline | 382 | - | 95.94% | 4875 MB |
| v1.14.1 + opts (无 AVX512 dispatch) | 431 | 5171 | 96.25% | 4875 MB |
| v1.14.1 + opts (AVX512 dispatch) | **1304** | **11799** | 96.36% | 4876 MB |
| v1.13.1+opt 参考 | 1648 | - | - | - |
| **提升（vs baseline）** | **+241%** | - | - | - |

### 关键发现

1. **SIMD dispatch 是 SQfp16 性能的决定性因素**：
   不设 `FAISS_OPT_LEVEL`，SQ 走标量路径，所有搜索优化加一起只提升 12.8%。
   设了 `avx512` 后直接提升 241%。

2. **Flat 的提升主要来自多线程**：
   单线程 +7%（batch_8 + prefetch + cross-node batching），
   16 线程 +23%（OMP guard 消除了单查询时的线程池开销）。

3. **SQfp16 比 Flat 更快**：
   SQfp16 (1304 QPS) > Flat (1281 QPS)，因为 FP16 只需一半内存带宽，
   在 HNSW 的随机访问场景下 cache miss 更少。

4. **与 v1.13.1+opt 仍有 ~20% 差距**：
   v1.14.1+opts (1304) vs v1.13.1+opt (1648)。可能原因：
   - v1.14.1 的 runtime SIMD dispatch 有额外开销
   - SQ 的 DistanceComputer 没有 batch_8 override
   - search_from_candidate_unbounded 未做 cross-node batching

## 分支信息

- 分支：`bench/optimize-all`（基于 v1.14.1 tag）
- 共 12 个 commit（11 个优化 + 1 个 SIMD dispatch 修复）
- CPU：Intel Xeon 8370C (Ice Lake)，支持 AVX-512F/BW/VL/VNNI
