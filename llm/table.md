# FAISS 自研优化 — 分支 / Commit 索引表

> 基于 FAISS v1.14.1 (upstream main aligned 2026-04-05)
> 以下所有分支均 fork 自 upstream 官方代码

---

## 分支概览

| 分支 | Fork 点 | Commits | 主题 |
|------|---------|---------|------|
| `v1.13.1-fix-openmp` | v1.13.1 (338a5fbf) | 1 | OpenMP 条件守卫 |
| `v1.13.1-10m-optimize` | v1.13.1 (338a5fbf) | 6 | HNSW 搜索/构建全套优化 (opt1-11) |
| `yoj/mem_zip` | v1.13.2+ (1ba2dbbd) | 30 | SharedVectorStore + 零拷贝重建 + 图重排 |
| `yoj/opt-1m-vde` | v1.13.2+ (1ba2dbbd) | 31 | = mem_zip + VDE 优化分析文档 |

---

## 分支一: `v1.13.1-fix-openmp` (1 commit)

| Commit | 类型 | 修改文件 | 优化技术 | 效果 |
|--------|------|---------|----------|------|
| `d4bd2b47` | SOURCE | `IndexBinaryHNSW.cpp`, `IndexFlatCodes.cpp`, `IndexIDMap.cpp`, `IndexNNDescent.cpp`, `IndexNSG.cpp`, `IndexScalarQuantizer.cpp` | **OpenMP 条件守卫**: 为 6 个 Index 的 `#pragma omp parallel` 加 `if(n>1)` 守卫, 消除 n=1 单查询时的线程池管理开销 (10-130us/call) | IndexNNDescent +2.79x, IndexIDMap +2.40x, IndexSQ +2.30x, IndexNSG +2.17x |

---

## 分支二: `v1.13.1-10m-optimize` (6 commits)

包含 `v1.13.1-fix-openmp` 的内容, 加上 5 个递增优化。

### 源码修改一览

| Commit | ID | 修改文件 | 优化技术 | 说明 |
|--------|-----|---------|----------|------|
| `d4bd2b47` | opt0 | 6 个 Index*.cpp | **OpenMP 条件守卫** | 同上 |
| `15608779` | opt1 | `IndexHNSW.cpp` | **动态 OMP 调度**: `schedule(static)` → `schedule(dynamic,64)` | 改善 HNSW 构建负载均衡, 高层节点插入耗时不均 |
| `3dce5c2b` | opt2-3 | `IndexFlat.cpp`, `IndexFlatCodes.cpp`, `DistanceComputer.h`, `HNSW.cpp`, `distances.h`, `distances_simd.cpp` | **SIMD batch_4/batch_8 距离计算**: 手写 AVX-512 (16 floats/iter) 和 AVX2 (8 floats/iter) 的 `fvec_L2sqr_batch_8` / `fvec_inner_product_batch_8`; HNSW 4 个热路径从 buffer-4 升级到 buffer-8 | +1140 行, 新增 `DistanceComputer::distances_batch_8()` 虚方法 |
| `e0086e29` | opt4 | `DistanceComputer.h`, `HNSW.cpp` | **软件预取**: 新增 `DistanceComputer::prefetch()` 虚方法, HNSW 4 个热路径在距离计算前预取向量数据到 CPU cache | 重叠内存延迟与计算 |
| `72d21687` | opt5 | `distances_fp16.h` (new), `distances_fp16_simd.cpp` (new), `CMakeLists.txt` | **FP16/BF16 距离计算库**: 3 层 SIMD 分发 (AVX-512F+F16C / AVX2+F16C / scalar); 完整 API: `fp16vec_L2sqr`, `fp16vec_inner_product`, `_batch_4` 等 | +1908 行, 含 117 个单元测试, 未集成到 HNSW 搜索路径 (独立库) |
| `6bd687d7` | opt6-11 | `IndexHNSW.cpp`, `HNSW.cpp`, `numa_helpers.h` (new), `CMakeLists.txt` | **6 项优化打包**: (下方展开) | +136 行 |

### opt6-11 展开

| 子项 | 技术 | 修改位置 | 说明 |
|------|------|---------|------|
| opt6 | **对称距离复用** | `HNSW.cpp: add_link()` | 新增 `known_dist` 参数, 跳过已知 src↔dest 距离的重复计算 |
| opt7 | **距离缓存** | `HNSW.cpp: shrink_neighbor_list()` | `unordered_map<uint64_t, float>` 缓存 symmetric_dis 结果, 避免重复计算 |
| opt8 | **提前终止** | `HNSW.cpp: search_neighbors_to_add()` | 连续 3 次无改善时提前退出, 减少无效搜索 |
| opt9 | **空间局部性排序** | `IndexHNSW.cpp: hnsw_add_vertices()` | 构建时用随机投影排序替代随机 shuffle, 空间邻近的向量连续插入, 改善 cache 局部性 |
| opt10-11 | **透明大页 (THP)** | `IndexHNSW.cpp`, `numa_helpers.h` | 构建完成后对 `IndexFlatCodes::codes` 调用 `madvise(MADV_HUGEPAGE)`, 减少 TLB miss |

---

## 分支三: `yoj/mem_zip` (30 commits)

### 源码修改 (14 commits)

| Commit | 修改文件 | 优化技术 | 说明 |
|--------|---------|----------|------|
| `5d5c1ff2` | `IndexFlatCompressed.cpp/h` (new), `IndexHNSWCompressed.cpp/h` (new) | **压缩向量存储**: LZ4/ZSTD 块级压缩 + 线程级 LRU 解压缓存 | 新增 `IndexFlatCompressed`, `IndexHNSWCompressed` 两个类 |
| `e2665494` | `IndexFlatCompressed.cpp` | Fix: 补缺 `#include <IDSelector.h>` | bug fix |
| `78ff40eb` | `IndexFlatCompressed.cpp/h`, `IndexHNSWCompressed.cpp/h` | **压缩重构**: 块级 → 按向量压缩, 默认 ZSTD, 实例级 cache 隔离 | per-vector 粒度更灵活 |
| `a7b92801` | `DistanceComputer.h`, `HNSW.cpp`, `distances.h`, `distances_simd.cpp` | **基础 API**: 新增 `prefetch()`, `prefetch_batch_4()`, `distances_batch_8()` 虚方法; scalar batch_8 实现 | 基础设施, 为后续 SIMD 优化铺路 |
| `363d3aad` | `HNSW.cpp`, `distances_simd.cpp` | **AVX-512/AVX2 batch_8**: `fvec_L2sqr_batch_8` / `fvec_inner_product_batch_8` SIMD 实现; HNSW search_from_candidates 升级 batch_8 + 预取流水线 | **+21% QPS** |
| `2e47e31e` | `IndexFlat.cpp` | **接线**: `FlatL2Dis`/`FlatIPDis` 添加 `distances_batch_8()` override, 连接 AVX-512 到标准 IndexHNSWFlat | 让标准 Flat 存储也走 batch_8 SIMD |
| `ec03f45e` | `HNSW.cpp` | **跨节点邻居批处理**: `search_from_candidates` 中 batch_ids 缓冲区跨候选节点持久化, 提高 batch_8 命中率 | **+16.8% QPS** |
| `29b00333` | `HNSW.cpp` | **SIMD count_below**: AVX-512/AVX2 优化 `MinimaxHeap::count_below()` (提前终止判断的热路径) | **+3-5% QPS** |
| `87c59cc8` | `SharedVectorStore.cpp/h` (new), `IndexFlatShared.cpp/h` (new), `IndexHNSW.cpp`, `CMakeLists.txt` | **SharedVectorStore 核心架构**: 共享存储 + free_list 回收 + storage_id_map 间接寻址 + deleted_bitmap; 零拷贝 HNSW 重建 (`add(n, nullptr)` 路径); DPDK 风格 prefetch_batch_4 | 核心架构变更, 新增 4 个文件 |
| `062724c7` | `HNSWReorder.cpp/h` (new), `IndexFlatShared.cpp/h` | **重建后管线**: `compact_store()` (cycle-following 原地排列消除间接寻址) + `HNSWReorder` 库 (5 种图重排策略: BFS/RCM/DFS/Cluster/Weighted) | 核心架构变更, 新增 2 个文件 |
| `fbafe3a2` | `IndexHNSW.cpp` | **Fix permute_entries 派发**: `dynamic_cast<IndexFlatShared*>` 优先于 `IndexFlatCodes*`, 修复图重排后 recall 降到 0.007 的 bug | 关键 bug fix |
| `873e76df` | `IndexFlatShared.cpp`, `SharedVectorStore.h` | **THP 支持**: SharedVectorStore 添加 `enable_hugepages()`, 在 add/restore/compact 后调用 `madvise(MADV_HUGEPAGE)` | 减少 TLB miss |

### 构建系统修改 (3 commits)

| Commit | 说明 |
|--------|------|
| `52e5d092` | 新增 `FAISS_ENABLE_COMPRESSED_STORAGE` CMake 选项, 链接 liblz4/libzstd |
| `c2f938ef` | Fix: 交叉编译时添加 LZ4/ZSTD link directories |
| `a7ba57a6` | Fix: LZ4/ZSTD 改为 PUBLIC linkage 解决传递依赖 |

### Benchmark (10 commits)

| Commit | 说明 |
|--------|------|
| `5817d463` | `bench_hnsw_compressed.cpp`: 压缩存储 vs Flat 对比 |
| `c79b2919` | `bench_storage_indirect.cpp`: 间接存储 + DPDK prefetch (+13%) |
| `214fcdc8` | hugepage chunked 存储 (+24%) |
| `8f0bb47a` | `bench_hnsw_compare.cpp`: cache-aligned 存储 + BFS 重排 + VSAG 对比 |
| `a83793f4` | 5 种图重排策略原型 (RCM/Weighted +33-34%) |
| `a2af4e30` | chunked + hugepage 存储 benchmark |
| `c375a16f` | `bench_rebuild_perf.cpp`: 真实数据集 SharedVectorStore 全管线 benchmark |
| `5cd9d026` | Python/Shell: FAISS vs VSAG + THP 测试脚本 |
| `f9b9f284` | 多删除比例对比 (10%-80%) |
| `84994d1c` | Fix: GIST-960 OOM, 3 阶段内存优化 |

### 文档 (3 commits)

| Commit | 说明 |
|--------|------|
| `6b1193b9` | FAISS vs VSAG 性能分析 |
| `da6570d6` | HNSW 优化计划 |
| `a6163fba` | VTune profiling 指引 |

---

## 分支四: `yoj/opt-1m-vde` (31 commits)

= `yoj/mem_zip` 全部 30 个 commit + 1 个额外 commit:

| Commit | 类型 | 说明 |
|--------|------|------|
| `f6b9a54b` | DOC | VDE 优化分析文档: cortex.core vs beta 搜索路径对比 (7 层 vs 4 层), VisitedTable 10MB malloc/free 问题, `dynamic_cast` 开销, gRPC 线程模型分析; 含 `.patch` 文件 (VisitedTable thread_local 复用方案) |

---

## 优化技术汇总 (按 commit 可追溯)

| 编号 | 优化技术 | 分支 | 关键 Commit | 影响文件 | 实测效果 |
|------|----------|------|------------|---------|---------|
| O1 | OpenMP 条件守卫 | fix-openmp, 10m-opt | `d4bd2b47` | 6 个 Index*.cpp | n=1 时 +2.17x ~ +2.79x |
| O2 | 动态 OMP 调度 (构建) | 10m-opt | `15608779` | IndexHNSW.cpp | 构建负载均衡 |
| O3 | SIMD batch_4/batch_8 | 10m-opt, mem_zip | `3dce5c2b`, `363d3aad`, `2e47e31e` | distances_simd.cpp, HNSW.cpp, IndexFlat.cpp | +21% QPS |
| O4 | 软件预取 | 10m-opt, mem_zip | `e0086e29`, `a7b92801` | DistanceComputer.h, HNSW.cpp | 与 batch_8 协同 |
| O5 | FP16 距离计算库 | 10m-opt | `72d21687` | distances_fp16.h/cpp (新文件) | 独立库, 未集成到搜索路径 |
| O6 | 对称距离复用 | 10m-opt | `6bd687d7` | HNSW.cpp | 减少构建时重复距离计算 |
| O7 | 距离缓存 | 10m-opt | `6bd687d7` | HNSW.cpp | shrink_neighbor_list 加速 |
| O8 | 搜索提前终止 | 10m-opt | `6bd687d7` | HNSW.cpp | 连续 3 次无改善则退出 |
| O9 | 空间局部性排序 (构建) | 10m-opt | `6bd687d7` | IndexHNSW.cpp | 随机投影排序替代 shuffle |
| O10 | 透明大页 (THP) | 10m-opt, mem_zip | `6bd687d7`, `873e76df` | numa_helpers.h, IndexHNSW.cpp, SharedVectorStore.h | SIFT-1M +12.5%, GIST-960 +26.4% |
| O11 | 跨节点邻居批处理 | mem_zip | `ec03f45e` | HNSW.cpp | +16.8% QPS |
| O12 | SIMD count_below | mem_zip | `29b00333` | HNSW.cpp | +3-5% QPS |
| O13 | SharedVectorStore 架构 | mem_zip | `87c59cc8` | SharedVectorStore.cpp/h, IndexFlatShared.cpp/h, IndexHNSW.cpp | 零拷贝重建, 内存 -78% |
| O14 | compact_store + 图重排 | mem_zip | `062724c7` | HNSWReorder.cpp/h, IndexFlatShared.cpp/h | BFS +4-21%, RCM+THP +31.3% |
| O15 | 压缩向量存储 | mem_zip | `5d5c1ff2`, `78ff40eb` | IndexFlatCompressed.cpp/h, IndexHNSWCompressed.cpp/h | LZ4/ZSTD 压缩, 实验性 |
| O16 | VisitedTable 复用 | opt-1m-vde | `f6b9a54b` (.patch) | HNSW.h, IndexHNSW.cpp (方案) | 消除 10MB/search 的 malloc+free, 预估 -10-30% 延迟 |
| O17 | gRPC 线程池调优 | opt-1m-vde | `f6b9a54b` (文档) | vde_grpc_server.cpp (方案) | MAX_POLLERS 2→8 |

---

## 注意事项

1. `v1.13.1-*` 分支基于 v1.13.1, 需要 **rebase/cherry-pick 到 v1.14.1** 才能用于当前 main
2. `yoj/mem_zip` 和 `yoj/opt-1m-vde` 基于 v1.13.2+ (1ba2dbbd), 同样需要 rebase 到最新 main
3. 部分优化互相依赖: O3 (batch_8) 依赖 O4 (prefetch) 才能发挥最大效果
4. O5 (FP16 距离库) 是独立库, 需额外集成工作才能用于 HNSW 搜索
5. O15 (压缩存储) 为实验性质, 生产建议使用 O13 (SharedVectorStore) 方案
6. O16/O17 目前仅存在于文档/patch 中, 尚未合入任何分支的源码
