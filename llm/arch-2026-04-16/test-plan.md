# 20M HNSW Benchmark Plan (V0 + V1-16, 全量 36 点扫描)

## Context

已完成 10M cohere 数据集上 V0 (Flat FP32) 和 V1-16 (SQfp16 + 全部优化) 的 36 点参数扫描。现在需要在 20M 数据集上重复相同的测试矩阵，观察 scaling 行为。

20M 数据集使用现成的 MS MARCO WebSearch (`llm/database/msmarco_websearch_20m_det_v1/`)，58GB fvecs，9376 queries，exact top-100 ground truth。

## 测试矩阵

每个版本 36 个测试点：

- **M**: 16, 32, 48
- **efConstruction**: 40, 200, 512
- **efSearch**: 64, 128, 256, 512

V0: `HNSW{M},Flat` × 36 点
V1-16: `HNSW{M},SQfp16` × 36 点
**总计: 72 个测试点**

## 约束

- **内存**: 125 GB 总量，122 GB 可用 — 充足（20M FP32 ≈ 57GB，加上 HNSW 图开销）
- **磁盘**: 仅 61 GB 剩余 — 不存储索引文件，benchmark 在内存中完成后丢弃即可
- **时间预估**: 单个 efC=512 M=48 构建在 10M 时约 9400s（~2.6h），20M 预计 ~5h。全部 9 个构建配置 × 2 版本 = 18 次构建，但 efC=40 较快。总计预估 2-3 天

## 执行步骤

### Step 1: 确认二进制就绪

两个 benchmark 二进制均已编译就绪（如需重编译: `cd llm && make bench-v0 bench-opt`）：

| 二进制 | 路径 | 状态 |
|--------|------|------|
| V0 | `llm/faiss-1.14.1-origin/bench` | 已编译 (45KB) |
| V1-16 | `llm/bench_task` | 已编译 (51KB) |

### Step 2: 验证 20M 数据集可用

```bash
# 确认文件完整
ls -lh llm/database/msmarco_websearch_20m_det_v1/
# 预期: base.fvecs (~58GB), query.fvecs (~28MB), groundtruth.ivecs (~3.7MB)
```

### Step 3: 快速冒烟测试

用一个小配置验证数据集和两个 bench 二进制能正常工作：

```bash
# V0 冒烟测试 (efC=40 构建最快)
llm/faiss-1.14.1-origin/bench "HNSW16,Flat" \
    efConstruction=40 efSearch=64 \
    --data llm/database/msmarco_websearch_20m_det_v1 --stream

# V1-16 冒烟测试
llm/bench_task "HNSW16,SQfp16" \
    efConstruction=40 efSearch=64 \
    --data llm/database/msmarco_websearch_20m_det_v1 --stream
```

### Step 4: V0 全量 36 点扫描

每个 (M, efC) 组合构建一次索引，扫描 4 个 efSearch 值。共 9 次构建。

按构建时间从短到长排列（先跑快的，确认流程无误）：

```bash
DATA="--data llm/database/msmarco_websearch_20m_det_v1 --stream"
EFS="efSearch=64,128,256,512"

# --- M=16 ---
llm/faiss-1.14.1-origin/bench "HNSW16,Flat" efConstruction=40  $EFS $DATA
llm/faiss-1.14.1-origin/bench "HNSW16,Flat" efConstruction=200 $EFS $DATA
llm/faiss-1.14.1-origin/bench "HNSW16,Flat" efConstruction=512 $EFS $DATA

# --- M=32 ---
llm/faiss-1.14.1-origin/bench "HNSW32,Flat" efConstruction=40  $EFS $DATA
llm/faiss-1.14.1-origin/bench "HNSW32,Flat" efConstruction=200 $EFS $DATA
llm/faiss-1.14.1-origin/bench "HNSW32,Flat" efConstruction=512 $EFS $DATA

# --- M=48 ---
llm/faiss-1.14.1-origin/bench "HNSW48,Flat" efConstruction=40  $EFS $DATA
llm/faiss-1.14.1-origin/bench "HNSW48,Flat" efConstruction=200 $EFS $DATA
llm/faiss-1.14.1-origin/bench "HNSW48,Flat" efConstruction=512 $EFS $DATA
```

每个命令的输出 redirect 到日志文件并提取结果。

### Step 5: V1-16 全量 36 点扫描

同样 9 次构建，使用 bench_task（默认开启 O14 BFS 重排 + O16 VT 复用）：

```bash
DATA="--data llm/database/msmarco_websearch_20m_det_v1 --stream"
EFS="efSearch=64,128,256,512"

# --- M=16 ---
llm/bench_task "HNSW16,SQfp16" efConstruction=40  $EFS $DATA
llm/bench_task "HNSW16,SQfp16" efConstruction=200 $EFS $DATA
llm/bench_task "HNSW16,SQfp16" efConstruction=512 $EFS $DATA

# --- M=32 ---
llm/bench_task "HNSW32,SQfp16" efConstruction=40  $EFS $DATA
llm/bench_task "HNSW32,SQfp16" efConstruction=200 $EFS $DATA
llm/bench_task "HNSW32,SQfp16" efConstruction=512 $EFS $DATA

# --- M=48 ---
llm/bench_task "HNSW48,SQfp16" efConstruction=40  $EFS $DATA
llm/bench_task "HNSW48,SQfp16" efConstruction=200 $EFS $DATA
llm/bench_task "HNSW48,SQfp16" efConstruction=512 $EFS $DATA
```

### Step 6: 编写 shell runner 脚本

为了自动化 18 次构建 + 结果收集，写一个 `llm/run_20m_sweep.sh` 脚本，功能：

1. 遍历 {V0, V1-16} × {M=16,32,48} × {efC=40,200,512}
2. 每次运行将 stdout/stderr 重定向到 `llm/results/20m/<version>_M<M>_efC<efC>.log`
3. 从日志中提取 QPS_1T, QPS_16T, Recall@10, Recall@100, BUILD_TIME_S, RSS_MB
4. 汇总到 CSV 文件 `llm/results/20m/summary.csv`
5. 每个构建完成后打印进度 (e.g. "=== [3/18] V0 M=16 efC=512 done ===")

### Step 7: 结果报告

测试完成后，生成报告 `llm/arch-2026-04-16/20m-parameter-sweep-report.md`，格式与 10M 报告对齐：

1. 完整 36 点结果表格（V0 和 V1-16 各一份）
2. 达到 R@10 ≥ 95% 的配置
3. 达到 R@100 ≥ 95% 的配置
4. 分析（efC 影响、M 影响、QPS 缩放规律）
5. V0 vs V1-16 对比
6. **10M → 20M scaling 对比**（关键新增内容）：
   - QPS 下降幅度
   - Recall 变化
   - 内存增长
   - 构建时间增长

## 关键文件

| 文件 | 用途 |
|------|------|
| `llm/faiss-1.14.1-origin/bench` | V0 benchmark 二进制 (已编译) |
| `llm/bench_task` | V1-16 benchmark 二进制 (已编译) |
| `llm/faiss-1.14.1-origin/bench.cpp` | V0 benchmark 源码 |
| `llm/bench_task.cpp` | V1-16 benchmark 源码 |
| `llm/Makefile` | 编译规则 |
| `llm/database/msmarco_websearch_20m_det_v1/` | 20M 数据集 |
| `llm/run_20m_sweep.sh` | 新建: 自动化扫描脚本 |
| `llm/results/20m/` | 新建: 结果目录 |

## 验证

1. 冒烟测试通过（Step 3），两个二进制都能正确读取 20M 数据集
2. Recall 数值合理（非零，且随 efC/efS 增大而提高）
3. QPS 与 10M 相比应下降（预计 30-50%，因为图更大，搜索路径更长）
4. 内存应约为 10M 的 2 倍
5. 结果报告格式与 10M 报告一致，可直接对比
