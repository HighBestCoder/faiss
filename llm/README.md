# FAISS HNSW Benchmark

## 测试目标

在 Recall@10 >= 95% 的硬约束下，找到 QPS 最高的 <索引, 编码, 优化> 组合。

- 数据集: cohere_10m (10M, 768D, cosine)
- 硬件: 62GB RAM, 16 cores, AVX-512
- FAISS: v1.14.1
- 编译: `g++ -O3 -march=native -mtune=native -std=c++17`, Intel MKL + libiomp5

---

## 两个版本

| 版本 | 索引类型 | 说明 |
|------|---------|------|
| **V0** | HNSW{M},Flat (FP32) | 原始 FAISS v1.14.1，无任何优化 |
| **V1-16** | HNSW{M},SQfp16 (FP16) | 全部自研优化 (O1-O14, O16) |

V1-16 优化内容包括: OpenMP 条件守卫、SIMD batch_8 距离计算、软件预取、跨节点邻居批处理、SIMD count_below、BFS 图重排、VisitedTable 复用等。详见 [table.md](table.md)。

---

## 编译

### 1. 编译 FAISS 库

```bash
# V0: 原始 v1.14.1 (用 git worktree 从 tag 编译，输出到 llm/faiss-1.14.1-origin/)
bash llm/v0_build.sh

# V1-16: 当前分支的优化版 (输出到 install/)
bash llm/build.sh
```

| 版本 | 编译脚本 | 输出库路径 |
|------|---------|-----------|
| V0 | `llm/v0_build.sh` | `llm/faiss-1.14.1-origin/lib/libfaiss_avx512.so` |
| V1-16 | `llm/build.sh` | `install/lib/libfaiss_avx512.so` |

### 2. 编译 benchmark 二进制

```bash
cd /ceph/faiss-dev/llm

make bench-v0    # → llm/faiss-1.14.1-origin/bench
make bench-opt   # → llm/bench_task
make bench-rabitq # → llm/bench_rabitq (RaBitQ 专用)
make clean       # 清理所有二进制
```

---

## 运行测试

### 通用参数格式

```
<binary> <index_factory> [efConstruction=N] [efSearch=N1,N2,...] [nprobe=N1,N2,...] [--data <dir>] [--stream]
```

| 参数 | 说明 |
|------|------|
| `index_factory` | FAISS 索引工厂字符串，如 `"HNSW32,Flat"`, `"HNSW48,SQfp16"` |
| `efConstruction=N` | HNSW 构建参数 (默认使用 FAISS 默认值) |
| `efSearch=N1,N2,...` | 搜索参数扫描列表 (默认 64,128,256) |
| `nprobe=N1,N2,...` | IVF 搜索参数扫描列表 |
| `--data <dir>` | 数据集目录 (默认 `llm/database/cohere_medium_1m`) |
| `--stream` | 批量加载模式，10M+ 数据集必须使用 |

### V0 测试

```bash
# M=32 efC=200 参数扫描 (10M 数据集)
llm/faiss-1.14.1-origin/bench "HNSW32,Flat" \
    efConstruction=200 efSearch=64,128,256,512 \
    --data llm/database/cohere_10m --stream

# M=48 efC=512 参数扫描
llm/faiss-1.14.1-origin/bench "HNSW48,Flat" \
    efConstruction=512 efSearch=64,128,256,512 \
    --data llm/database/cohere_10m --stream
```

### V1-16 测试

```bash
# M=48 efC=200 参数扫描 (10M 数据集, 推荐配置)
llm/bench_task "HNSW48,SQfp16" \
    efConstruction=200 efSearch=64,128,256,512 \
    --data llm/database/cohere_10m --stream

# M=32 efC=512 参数扫描
llm/bench_task "HNSW32,SQfp16" \
    efConstruction=512 efSearch=64,128,256,512 \
    --data llm/database/cohere_10m --stream
```

V1-16 专用开关:

| 开关 | 说明 |
|------|------|
| `--no-reorder` | 跳过 O14 BFS 图重排 |
| `--no-vt-reuse` | 跳过 O16 VisitedTable 复用 |

### 1M 数据集快速测试

```bash
# 1M 数据集不需要 --stream，可直接加载到内存
llm/bench_task "HNSW32,SQfp16" efSearch=64,128,256 \
    --data llm/database/cohere_medium_1m
```

---

## 数据集

| 数据集 | 路径 | 规模 | 维度 | 说明 |
|--------|------|------|------|------|
| cohere_medium_1m | `llm/database/cohere_medium_1m/` | 1M | 768 | 快速测试用 |
| cohere_10m | `llm/database/cohere_10m/` | 10M | 768 | 主力 benchmark |

每个数据集目录包含:
- `base.fvecs` — 底库向量
- `query.fvecs` — 查询向量 (1000 条)
- `groundtruth.ivecs` — ground truth (k=1000)

cosine 相似度通过 L2 normalize + METRIC_INNER_PRODUCT 实现。

---

## 测试结果

### V0 最优配置 (M=32, efC=200, Flat FP32)

| efS | QPS_1T | QPS_16T | R@10 | R@100 |
|-----|--------|---------|------|-------|
| 64 | 1,099 | 10,989 | 96.03% | 83.67% |
| 128 | 630 | 5,908 | 97.54% | 91.70% |
| 256 | 330 | 3,071 | 98.61% | 96.10% |
| 512 | 171 | 1,621 | 99.22% | 98.16% |

内存: 33.8 GB, 构建: 49 分钟

### V1-16 最优配置 (M=48, efC=200, SQfp16)

| efS | QPS_1T | QPS_16T | R@10 | R@100 |
|-----|--------|---------|------|-------|
| 64 | 1,261 | 11,284 | 95.68% | 84.89% |
| 128 | 677 | 6,048 | 97.26% | 92.30% |
| 256 | 349 | 3,030 | 98.05% | 96.11% |
| 512 | 180 | 1,594 | 98.55% | 97.83% |

内存: 20.4 GB, 构建: 48 分钟

### V1-16 vs V0 (同参数对比, M=32, efC=512, efS=64)

| 指标 | V0 (Flat FP32) | V1-16 (SQfp16) | 提升 |
|------|---------------|----------------|------|
| QPS_1T | 978 | 1,232 | +26.0% |
| QPS_16T | 9,583 | 11,293 | +17.8% |
| R@10 | 96.57% | 96.30% | -0.27pp |
| R@100 | 84.90% | 84.85% | -0.05pp |
| RSS | 33,805 MB | 19,157 MB | -43.3% |

---

## 关键规律

- efSearch 翻倍 → QPS 约减半
- 16 线程加速比约 9-10x
- efConstruction 是 Recall 的决定性因素，efC=40 不可用
- M 越大，同 efS 下 Recall 越高但 QPS 越低
- march=native 对 V0 无显著影响 (<2%)，因为 FAISS 距离计算用手写 SIMD

---

## 目录结构

```
llm/
├── README.md              ← 本文件
├── Makefile               ← 编译 benchmark 二进制
├── build.sh               ← V1-16 FAISS 编译脚本
├── v0_build.sh            ← V0 FAISS 编译脚本
├── bench_task.cpp         ← V1-16 benchmark 源码
├── bench_rabitq.cpp       ← RaBitQ benchmark 源码
├── table.md               ← 自研优化 commit 索引表
├── todo.md                ← 完整测试矩阵 (53 个用例)
├── faiss-1.14.1-origin/
│   ├── bench.cpp          ← V0 benchmark 源码
│   ├── bench              ← V0 benchmark 二进制
│   ├── lib/               ← V0 原始 FAISS 库
│   └── include/           ← V0 头文件
├── database/
│   ├── cohere_medium_1m/  ← 1M 数据集
│   └── cohere_10m/        ← 10M 数据集
├── tasks/                 ← 各测试用例定义
├── results/               ← 测试结果
└── arch-2026-04-*/        ← 各阶段分析报告
```

---

## 详细文档

- 完整测试矩阵: [todo.md](todo.md)
- 优化技术索引: [table.md](table.md)
- V0 参数扫描报告: [arch-2026-04-10/V0-parameter-sweep-report.md](arch-2026-04-10/V0-parameter-sweep-report.md)
- V1-16 参数扫描报告: [arch-2026-04-10/parameter-sweep-report.md](arch-2026-04-10/parameter-sweep-report.md)
