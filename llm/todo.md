# FAISS 性能测试矩阵 — 任务列表

## 测试目标

> **核心目标: 在 Recall@10 >= 95% 的硬约束下, 找到 QPS 最高的 <索引, 编码, 优化> 组合。**

- **约束**: Recall@10 >= 95% (不满足则标记 skip, 不进入后续 Phase)
- **优化目标**: 最大化 QPS (单线程 & 多线程)
- **次要关注**: 内存占用 (同等 QPS 下优先内存更低的方案)

## 测试基准

- **数据集**: cohere_medium_1m (1M vectors, 768D, FP32, cosine)
- **查询集**: 1000 queries, ground truth k=1000
- **测量指标**: QPS_1t, QPS_16t, Recall@10, Recall@100, 内存(RSS), 构建时间
- **FAISS 版本**: v1.14.1 (tag `v1.14.1`, commit `5622e937`)
- **自研优化索引**: 详见 [llm/table.md](table.md)
- **硬件**: 当前测试机 (16 cores, AVX2, Intel MKL + libiomp5)

---

## 结果输出规范

### 目录结构

每个测试用例的结果写入 `llm/results/` 目录:

```
llm/results/
├── phase1/
│   ├── F1_Flat.md
│   ├── F2_SQfp16.md
│   ├── H1_HNSW32_Flat.md
│   ├── H2_HNSW32_SQfp16.md
│   ├── V1_IVF4096_Flat.md
│   └── ...
├── phase2/
│   ├── H8_HNSW16_{best}.md
│   └── ...
├── phase3/
│   ├── H11_omp_guard.md
│   ├── H12_graph_reorder.md
│   └── ...
├── phase4/
│   ├── R1_IVF4096_PQ48x8_RFlat.md
│   └── ...
└── summary.md              ← Phase 5 最终 QPS 排行榜
```

### 单个结果文件格式

```markdown
# {ID}: {index_factory}

## 配置
- index_factory: `{string}`
- 参数: efSearch={}, nprobe={}, ...
- 优化: {baseline / O2+O3+... }
- 分支: `bench/{ID}` (基于 v1.14.1)

## 结果
| 指标 | 值 |
|------|-----|
| 构建时间 | {x}s |
| 内存 (RSS) | {x} MB |
| QPS (1 thread) | {x} |
| QPS (16 threads) | {x} |
| Recall@10 | {x}% |
| Recall@100 | {x}% |

## 状态
{done / skip (Recall@10 < 95%)}

## 备注
{观察、结论、与其他用例的对比}
```

---

## Git 分支工作流

### Phase 1/2 (原版 FAISS, 无源码修改)

```bash
# 每个测试用例
git checkout v1.14.1
git checkout -b bench/{ID}       # 例: bench/H1, bench/V8
# ... 编写测试脚本, 运行 benchmark ...
# ... 将结果写入 llm/results/phase{N}/{ID}_xxx.md ...
git add llm/results/
git commit -m "bench({ID}): {index_factory} — QPS={x}, Recall@10={x}%

测试配置:
- index_factory: {string}
- 参数: {efSearch/nprobe/...}
- 数据集: cohere_medium_1m (1M, 768D, cosine)

结果:
- QPS_1t: {x} | QPS_16t: {x}
- Recall@10: {x}% | Recall@100: {x}%
- 内存: {x}MB | 构建时间: {x}s
- 状态: {done/skip}
"
```

### Phase 3 (软件优化, 需要移植代码)

Phase 3 需要将自研优化从旧分支 cherry-pick 到 v1.14.1。**每个优化的源 commit 记录在 [table.md](table.md) 中。**

```bash
# 步骤 1: 基于 v1.14.1 创建优化分支
git checkout v1.14.1
git checkout -b bench/{ID}       # 例: bench/H11

# 步骤 2: 从 table.md 查找对应优化的源 commit, cherry-pick 或手动移植
# 例如 O2 (omp_guard) 的源 commit 是 d4bd2b47 (见 table.md)
git cherry-pick d4bd2b47         # 如果能直接 cherry-pick
# 或者手动移植 (如果有冲突)

# 步骤 3: 编译 + 运行 benchmark
bash llm/build.sh
# ... 运行测试 ...

# 步骤 4: 提交结果, commit message 中记录优化来源
git add llm/results/
git commit -m "bench({ID}): {优化名} on {index_factory} — QPS={x} (+{y}%)

优化技术: {O2: omp_guard / O3: graph_reorder / ...}
源 commit: {hash} (from branch {branch}, see table.md)
移植方式: {cherry-pick / 手动移植}
代码修改: {列出修改的文件}

测试配置:
- index_factory: {string}
- 基线 QPS: {Phase 2 最优的 QPS}
- 数据集: cohere_medium_1m (1M, 768D, cosine)

结果:
- QPS_1t: {x} (+{y}%) | QPS_16t: {x} (+{y}%)
- Recall@10: {x}% | Recall@100: {x}%
- 内存: {x}MB | 构建时间: {x}s
"
```

### 优化技术 → 源 Commit 速查 (详见 [table.md](table.md))

| 优化 | 技术 | 源 Commit | 源分支 | 修改文件 |
|------|------|----------|--------|---------|
| O2 | OpenMP 条件守卫 | `d4bd2b47` | v1.13.1-fix-openmp | 6 个 Index*.cpp |
| O3 | BFS 图重排 | `062724c7` | yoj/mem_zip | HNSWReorder.cpp/h, IndexFlatShared.cpp/h |
| O4 | 透明大页 (THP) | `6bd687d7` | v1.13.1-10m-optimize | numa_helpers.h, IndexHNSW.cpp |
| O5 | VisitedTable 复用 | `f6b9a54b` (.patch) | yoj/opt-1m-vde | HNSW.h, IndexHNSW.cpp |
| O6 | SIMD batch_8 | `3dce5c2b` + `363d3aad` + `2e47e31e` | v1.13.1-10m-optimize, yoj/mem_zip | distances_simd.cpp, HNSW.cpp, IndexFlat.cpp, DistanceComputer.h |
| O7 | 跨节点邻居批处理 | `ec03f45e` | yoj/mem_zip | HNSW.cpp |
| O8 | SIMD count_below | `29b00333` | yoj/mem_zip | HNSW.cpp |
| O9 | FP16 距离计算库 | `72d21687` | v1.13.1-10m-optimize | distances_fp16.h/cpp (新文件) |
| O10 | SharedVectorStore | `87c59cc8` | yoj/mem_zip | SharedVectorStore.cpp/h, IndexFlatShared.cpp/h, IndexHNSW.cpp |

> 注意: 源 commit 基于 v1.13.x, cherry-pick 到 v1.14.1 时可能存在冲突, 需手动解决。
> 移植后必须 `bash llm/build.sh` 编译通过才能测试。

---

## 三维测试矩阵

### 维度定义

| 维度 | 缩写 | 变量 |
|------|------|------|
| 索引结构 | I | Flat, HNSW, IVF, IVF_HNSW (HNSW做粗量化器) |
| 量化编码 | Q | FP32(Flat), FP16, SQ8, SQ4, PQ, RaBitQ, RaBitQFastScan |
| 软件优化 | O | baseline, omp_guard, graph_reorder, hugepages, visited_table_reuse, simd_batch |

### 优化技术适用范围

| 优化技术 | 适用索引 | 说明 | 源 Commit (见 [table.md](table.md)) |
|----------|----------|------|------|
| O1: baseline | ALL | 无优化，原版 FAISS v1.14.1 | — |
| O2: omp_guard | ALL | OpenMP 条件守卫 (n=1 时跳过并行) | `d4bd2b47` |
| O3: graph_reorder (BFS) | HNSW only | BFS 图重排，改善内存局部性 | `062724c7` |
| O4: hugepages (THP) | HNSW only (dim>=512) | 透明大页，768D 适用 | `6bd687d7` |
| O5: visited_table_reuse | HNSW only | thread_local VisitedTable 复用 | `f6b9a54b` |
| O6: simd_batch | HNSW only | AVX2 batch_4/batch_8 距离计算 | `3dce5c2b` + `363d3aad` + `2e47e31e` |

---

## 第一组：Flat (暴力搜索) — 作为 Recall 基线

| ID | index_factory | 优化 | 分支 | 结果文件 | 预期 |
|----|--------------|------|------|---------|------|
| F1 | `Flat` | baseline | `bench/F1` | `results/phase1/F1_Flat.md` | Recall=100%, QPS 基线 |
| F2 | `SQfp16` | baseline | `bench/F2` | `results/phase1/F2_SQfp16.md` | 内存减半, Recall~99.9% |
| F3 | `SQ8` | baseline | `bench/F3` | `results/phase1/F3_SQ8.md` | 内存 1/4 |
| F4 | `SQ4` | baseline | `bench/F4` | `results/phase1/F4_SQ4.md` | 内存 1/8, Recall 可能低 |

---

## 第二组：HNSW 系列 — 主力索引

### 2A: HNSW × 量化编码 (M=32, efConstruction=40, efSearch=64/128/256)

| ID | index_factory | 优化 | 分支 | 结果文件 | 说明 |
|----|--------------|------|------|---------|------|
| H1 | `HNSW32,Flat` | baseline | `bench/H1` | `results/phase1/H1_HNSW32_Flat.md` | FP32 存储, 当前生产基线 |
| H2 | `HNSW32,SQfp16` | baseline | `bench/H2` | `results/phase1/H2_HNSW32_SQfp16.md` | FP16 存储 |
| H3 | `HNSW32,SQ8` | baseline | `bench/H3` | `results/phase1/H3_HNSW32_SQ8.md` | 8-bit 标量量化 |
| H4 | `HNSW32,SQ4` | baseline | `bench/H4` | `results/phase1/H4_HNSW32_SQ4.md` | 4-bit 标量量化 |
| H5 | `HNSW32,PQ48x8` | baseline | `bench/H5` | `results/phase1/H5_HNSW32_PQ48x8.md` | PQ, M=48 子空间, 8bit |
| H6 | `HNSW32,PQ96x8` | baseline | `bench/H6` | `results/phase1/H6_HNSW32_PQ96x8.md` | PQ, M=96 子空间, 8bit |
| H7 | `HNSW32,PQ48x4` | baseline | `bench/H7` | `results/phase1/H7_HNSW32_PQ48x4.md` | PQ, 4bit, 更激进压缩 |

### 2B: HNSW 参数扫描 (选 2A 中 Recall>=95% 且 QPS 最高的编码)

| ID | index_factory | 参数 | 分支 | 结果文件 | 说明 |
|----|--------------|------|------|---------|------|
| H8 | `HNSW16,{best}` | M=16 | `bench/H8` | `results/phase2/H8_HNSW16.md` | 更少连接, 更小图 |
| H9 | `HNSW48,{best}` | M=48 | `bench/H9` | `results/phase2/H9_HNSW48.md` | 更多连接, 更高 recall |
| H10 | `HNSW64,{best}` | M=64 | `bench/H10` | `results/phase2/H10_HNSW64.md` | 最大连接数 |

### 2C: HNSW × 软件优化 (选 2A/2B 最优配置, 移植代码到 v1.14.1)

| ID | index_factory | 优化 | 源 Commit | 分支 | 结果文件 |
|----|--------------|------|----------|------|---------|
| H11 | `HNSW{best}` | O2: omp_guard | `d4bd2b47` | `bench/H11` | `results/phase3/H11_omp_guard.md` |
| H12 | `HNSW{best}` | O3: graph_reorder | `062724c7` | `bench/H12` | `results/phase3/H12_graph_reorder.md` |
| H13 | `HNSW{best}` | O4: hugepages | `6bd687d7` | `bench/H13` | `results/phase3/H13_hugepages.md` |
| H14 | `HNSW{best}` | O5: visited_table | `f6b9a54b` | `bench/H14` | `results/phase3/H14_visited_table.md` |
| H15 | `HNSW{best}` | O6: simd_batch | `3dce5c2b`+`363d3aad`+`2e47e31e` | `bench/H15` | `results/phase3/H15_simd_batch.md` |
| H16 | `HNSW{best}` | O2+O3+O4+O5+O6 | 全部 | `bench/H16` | `results/phase3/H16_all_combined.md` |

---

## 第三组：IVF 系列

### 3A: IVF × 量化编码 (nlist=4096, nprobe=32/64/128)

| ID | index_factory | 优化 | 分支 | 结果文件 |
|----|--------------|------|------|---------|
| V1 | `IVF4096,Flat` | baseline | `bench/V1` | `results/phase1/V1_IVF4096_Flat.md` |
| V2 | `IVF4096,SQfp16` | baseline | `bench/V2` | `results/phase1/V2_IVF4096_SQfp16.md` |
| V3 | `IVF4096,SQ8` | baseline | `bench/V3` | `results/phase1/V3_IVF4096_SQ8.md` |
| V4 | `IVF4096,PQ48x8` | baseline | `bench/V4` | `results/phase1/V4_IVF4096_PQ48x8.md` |
| V5 | `IVF4096,PQ96x8` | baseline | `bench/V5` | `results/phase1/V5_IVF4096_PQ96x8.md` |
| V6 | `IVF4096,PQ48x4fs` | baseline | `bench/V6` | `results/phase1/V6_IVF4096_PQ48x4fs.md` |
| V7 | `IVF4096,PQ96x4fs` | baseline | `bench/V7` | `results/phase1/V7_IVF4096_PQ96x4fs.md` |
| V8 | `IVF4096,RaBitQ` | baseline | `bench/V8` | `results/phase1/V8_IVF4096_RaBitQ.md` |
| V9 | `IVF4096,RaBitQ4` | baseline | `bench/V9` | `results/phase1/V9_IVF4096_RaBitQ4.md` |
| V10 | `IVF4096,RaBitQfs` | baseline | `bench/V10` | `results/phase1/V10_IVF4096_RaBitQfs.md` |
| V11 | `IVF4096,RaBitQfs4` | baseline | `bench/V11` | `results/phase1/V11_IVF4096_RaBitQfs4.md` |

### 3B: IVF + HNSW 粗量化器 (加速 nprobe 搜索)

| ID | index_factory | 优化 | 分支 | 结果文件 |
|----|--------------|------|------|---------|
| V12 | `IVF4096_HNSW32,Flat` | baseline | `bench/V12` | `results/phase1/V12_IVF4096_HNSW32_Flat.md` |
| V13 | `IVF4096_HNSW32,SQfp16` | baseline | `bench/V13` | `results/phase1/V13_IVF4096_HNSW32_SQfp16.md` |
| V14 | `IVF4096_HNSW32,SQ8` | baseline | `bench/V14` | `results/phase1/V14_IVF4096_HNSW32_SQ8.md` |
| V15 | `IVF4096_HNSW32,PQ48x8` | baseline | `bench/V15` | `results/phase1/V15_IVF4096_HNSW32_PQ48x8.md` |
| V16 | `IVF4096_HNSW32,PQ48x4fs` | baseline | `bench/V16` | `results/phase1/V16_IVF4096_HNSW32_PQ48x4fs.md` |
| V17 | `IVF4096_HNSW32,RaBitQ` | baseline | `bench/V17` | `results/phase1/V17_IVF4096_HNSW32_RaBitQ.md` |
| V18 | `IVF4096_HNSW32,RaBitQfs` | baseline | `bench/V18` | `results/phase1/V18_IVF4096_HNSW32_RaBitQfs.md` |

### 3C: IVF nlist 参数扫描 (选 3A/3B 中 QPS 最高的编码)

| ID | index_factory | 分支 | 结果文件 | 说明 |
|----|--------------|------|---------|------|
| V19 | `IVF1024_HNSW32,{best}` | `bench/V19` | `results/phase2/V19_IVF1024.md` | 更少聚类中心 |
| V20 | `IVF8192_HNSW32,{best}` | `bench/V20` | `results/phase2/V20_IVF8192.md` | 更多聚类中心 |
| V21 | `IVF16384_HNSW32,{best}` | `bench/V21` | `results/phase2/V21_IVF16384.md` | 大量聚类中心 |

### 3D: IVF × 软件优化 (选 3A-3C 最优配置)

| ID | index_factory | 优化 | 源 Commit | 分支 | 结果文件 |
|----|--------------|------|----------|------|---------|
| V22 | `IVF{best}` | O2: omp_guard | `d4bd2b47` | `bench/V22` | `results/phase3/V22_omp_guard.md` |
| V23 | `IVF{best}` | O2+O4 | `d4bd2b47`+`6bd687d7` | `bench/V23` | `results/phase3/V23_omp_hugepages.md` |

---

## 第四组：Refine (二级精排) — 用压缩索引做初筛 + Flat 精排

| ID | index_factory | 优化 | 分支 | 结果文件 |
|----|--------------|------|------|---------|
| R1 | `IVF4096,PQ48x8,RFlat` | baseline | `bench/R1` | `results/phase4/R1_PQ48x8_RFlat.md` |
| R2 | `IVF4096,PQ48x4fs,RFlat` | baseline | `bench/R2` | `results/phase4/R2_PQ48x4fs_RFlat.md` |
| R3 | `IVF4096,RaBitQ,RFlat` | baseline | `bench/R3` | `results/phase4/R3_RaBitQ_RFlat.md` |
| R4 | `IVF4096,RaBitQfs,RFlat` | baseline | `bench/R4` | `results/phase4/R4_RaBitQfs_RFlat.md` |
| R5 | `IVF4096,SQ4,RFlat` | baseline | `bench/R5` | `results/phase4/R5_SQ4_RFlat.md` |

---

## 第五组：Flat 量化独立测试 (无索引结构, 验证量化精度)

| ID | index_factory | 优化 | 分支 | 结果文件 |
|----|--------------|------|------|---------|
| Q1 | `PQ48x8` | baseline | `bench/Q1` | `results/phase1/Q1_PQ48x8.md` |
| Q2 | `PQ96x8` | baseline | `bench/Q2` | `results/phase1/Q2_PQ96x8.md` |
| Q3 | `PQ48x4fs` | baseline | `bench/Q3` | `results/phase1/Q3_PQ48x4fs.md` |
| Q4 | `RaBitQ` | baseline | `bench/Q4` | `results/phase1/Q4_RaBitQ.md` |
| Q5 | `RaBitQfs` | baseline | `bench/Q5` | `results/phase1/Q5_RaBitQfs.md` |

---

## 执行计划

### Phase 1: 基线建立 (F1-F4, H1-H7, V1-V18, Q1-Q5)

1. `git checkout v1.14.1 && git checkout -b bench/{ID}`
2. 编写/运行 benchmark 脚本
3. 将结果写入 `llm/results/phase1/{ID}_xxx.md`
4. git commit, message 中记录完整结果
5. **淘汰 Recall@10 < 95% 的组合 (标记 skip, 不进入后续 Phase)**
6. 按 QPS 排序, 输出 Recall>=95% 的 QPS 排行榜

### Phase 2: 参数调优 (H8-H10, V19-V21)

1. `git checkout v1.14.1 && git checkout -b bench/{ID}`
2. 对 Phase 1 中 QPS 最高的配置做参数扫描
3. 目标: 在保持 Recall>=95% 的前提下, 通过调参进一步提升 QPS
4. 将结果写入 `llm/results/phase2/{ID}_xxx.md`
5. git commit, message 中记录参数变化与 QPS 变化

### Phase 3: 软件优化叠加 (H11-H16, V22-V23)

1. `git checkout v1.14.1 && git checkout -b bench/{ID}`
2. **查阅 [table.md](table.md) 找到对应优化的源 commit**
3. `git cherry-pick {源commit}` 或手动移植代码到 v1.14.1
4. `bash llm/build.sh` 编译通过
5. 运行 benchmark, 与 Phase 2 基线对比
6. 将结果写入 `llm/results/phase3/{ID}_xxx.md`
7. git commit, message 中详细记录:
   - 移植了哪个优化 (编号 + 名称)
   - 源 commit hash + 源分支
   - 修改了哪些文件
   - QPS 提升百分比
   - Recall 是否受影响

### Phase 4: Refine 组合 (R1-R5)

1. `git checkout v1.14.1 && git checkout -b bench/{ID}`
2. 测试二级精排能否在降低内存的同时, 保持 Recall>=95% 且 QPS 可接受
3. 将结果写入 `llm/results/phase4/{ID}_xxx.md`
4. git commit

### Phase 5: 最终对比 — QPS 排行榜

1. 汇总所有 Phase 中 Recall>=95% 的结果
2. 按 QPS 降序排列, 写入 `llm/results/summary.md`
3. 输出产品推荐矩阵:
   - **极速方案**: QPS 最高 (可接受更高内存)
   - **均衡方案**: QPS 与内存兼顾
   - **低内存方案**: 内存最低 (QPS 可接受)

---

## 结果记录模板

每个测试用例记录如下字段:

```
| ID | index_factory | 参数 | 优化 | 构建时间(s) | 内存(MB) | QPS_1t | QPS_16t | Recall@10 | Recall@100 | 状态 |
```

状态: `pending` / `running` / `done` / `skip` (Recall@10 < 95%)

**排序规则**: 所有 `done` 的用例按 QPS_1t 降序排列, QPS 最高的即为当前最优方案。

---

## 有效组合总计

| 组别 | 数量 | 说明 |
|------|------|------|
| Flat 基线 | 4 | F1-F4 |
| HNSW × 编码 | 7 | H1-H7 |
| HNSW × 参数 | 3 | H8-H10 |
| HNSW × 优化 | 6 | H11-H16 |
| IVF × 编码 | 11 | V1-V11 |
| IVF + HNSW粗量化 | 7 | V12-V18 |
| IVF × 参数 | 3 | V19-V21 |
| IVF × 优化 | 2 | V22-V23 |
| Refine | 5 | R1-R5 |
| 纯量化 | 5 | Q1-Q5 |
| **总计** | **53** | |

> 注: Phase 2/3 中标记 `{best}` 的测试用例, 需根据 Phase 1 结果动态确定具体配置。
