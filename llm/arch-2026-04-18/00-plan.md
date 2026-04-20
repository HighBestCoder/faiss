# 30M HNSW Benchmark Plan (V0 + V1-16)

日期: 2026-04-18

## 目标

在 30M MS MARCO WebSearch 数据集上重复 V0 / V1-16 的 36 点参数扫描，观察 10M → 20M → 30M 的 scaling 行为。

## Step 1: 生成 30M 数据集

从 101M 源数据集中抽取 30M 向量，输出到 `/mnt`（/ceph 仅剩 32GB，不够放 ~92GB 的 30M fvecs）。

```bash
python3 llm/arch-2026-04-18/derive_msmarco_30m.py all \
    --source-root /orion/diskann_bench/msmarco_websearch_simans \
    --output-dir /mnt/msmarco_websearch_30m_det_v1 \
    --target-count 30000000 \
    --seed 20260418 \
    --spec-version msmarco_websearch_30m_det_v1 \
    --temp-dir /mnt/tmp_selection
```

脚本基于 `llm/arch-2026-04-15/derive_msmarco_subset.py`，修改默认参数即可：
- `DEFAULT_TARGET_COUNT = 30_000_000`
- `DEFAULT_OUTPUT_DIR = /mnt/msmarco_websearch_30m_det_v1`
- `DEFAULT_SEED = 20_260_418`
- `DEFAULT_SPEC_VERSION = "msmarco_websearch_30m_det_v1"`

预期产物:
- `/mnt/msmarco_websearch_30m_det_v1/base.fvecs` (~92 GB)
- `/mnt/msmarco_websearch_30m_det_v1/query.fvecs` (~28 MB, 9376 queries)
- `/mnt/msmarco_websearch_30m_det_v1/groundtruth.ivecs` (~3.7 MB, exact top-100)

## Step 2: 验证数据集

```bash
python3 -c "
import struct
with open('/mnt/msmarco_websearch_30m_det_v1/base.fvecs','rb') as f:
    dim = struct.unpack('<i', f.read(4))[0]
    import os; size = os.path.getsize(f.name)
    count = size // (4 + dim*4)
    print(f'base: {count} vectors, {dim} dim')
with open('/mnt/msmarco_websearch_30m_det_v1/query.fvecs','rb') as f:
    dim = struct.unpack('<i', f.read(4))[0]
    size = os.path.getsize(f.name)
    count = size // (4 + dim*4)
    print(f'query: {count} vectors, {dim} dim')
"
```

## Step 3: 内存预估

| M  | V0 FP32 RSS (预估) | V1-16 SQfp16 RSS (预估) |
|----|--------------------|-----------------------|
| 16 | ~97 GB             | ~54 GB                |
| 32 | ~101 GB            | ~57 GB                |
| 48 | ~105 GB            | ~61 GB                |

当前机器 125 GB RAM。V0 M=48 可能接近极限（~105 GB），需要观察。V1-16 全部安全。

## Step 4: V0 全量 36 点扫描

测试矩阵: M={16,32,48} × efC={40,200,512} × efS={64,128,256,512}

```bash
DATA="--data /mnt/msmarco_websearch_30m_det_v1 --stream"
BENCH_V0="llm/faiss-1.14.1-origin/bench"

for M in 16 32 48; do
  for EFC in 40 200 512; do
    $BENCH_V0 "HNSW${M},Flat" efConstruction=${EFC} \
      efSearch=64,128,256,512 $DATA \
      2>&1 | tee llm/results/30m/v0_M${M}_efC${EFC}.log
  done
done
```

日志存放: `llm/results/30m/v0_M{16,32,48}_efC{40,200,512}.log` (9 个文件)

按构建时间排序（先跑快的）:
1. M=16 efC=40   (~30min)
2. M=32 efC=40   (~1.2h)
3. M=48 efC=40   (~1.5h)
4. M=16 efC=200  (~2h)
5. M=32 efC=200  (~3h)
6. M=48 efC=200  (~3.5h)
7. M=16 efC=512  (~4h)
8. M=32 efC=512  (~7h)
9. M=48 efC=512  (~9h, ~105GB RSS, 接近极限)

## Step 5: V1-16 全量 36 点扫描

```bash
DATA="--data /mnt/msmarco_websearch_30m_det_v1 --stream"
BENCH_V1="llm/bench_task"

for M in 16 32 48; do
  for EFC in 40 200 512; do
    $BENCH_V1 "HNSW${M},SQfp16" efConstruction=${EFC} \
      efSearch=64,128,256,512 $DATA \
      2>&1 | tee llm/results/30m/v1-16_M${M}_efC${EFC}.log
  done
done
```

日志存放: `llm/results/30m/v1-16_M{16,32,48}_efC{40,200,512}.log` (9 个文件)

## Step 6: 生成报告和可视化

1. 参数扫描报告: `llm/arch-2026-04-18/30m-parameter-sweep-report.md`
2. HTML 可视化 (16T): `llm/arch-2026-04-18/30m-qps-improvement.html`
3. HTML 可视化 (1T): `llm/arch-2026-04-18/30m-qps-improvement-1t.html`
4. 10M → 20M → 30M scaling 对比分析

## 时间预估

| 阶段 | 预估时间 |
|------|---------|
| 数据集生成 (30M select + materialize) | ~2-3h |
| Ground truth 计算 | ~4-6h |
| V0 9 次构建 + 搜索 | ~30h |
| V1-16 9 次构建 + 搜索 | ~25h |
| 报告 + 可视化 | ~1h |
| **总计** | **~3-4 天** |

## 关键文件

| 文件 | 用途 |
|------|------|
| `llm/arch-2026-04-18/derive_msmarco_30m.py` | 新建: 30M 数据集生成脚本 |
| `llm/arch-2026-04-15/derive_msmarco_subset.py` | 参考: 20M 数据集生成脚本 |
| `/mnt/msmarco_websearch_30m_det_v1/` | 新建: 30M 数据集 (在 /mnt) |
| `llm/faiss-1.14.1-origin/bench` | V0 benchmark 二进制 (已编译) |
| `llm/bench_task` | V1-16 benchmark 二进制 (已编译) |
| `llm/results/30m/` | 新建: 结果目录 |

## 风险

1. **V0 M=48 内存**: RSS 可能达到 ~105 GB，接近 125 GB 上限。如果 OOM，跳过该配置
2. **磁盘**: 30M base.fvecs ~92 GB，/mnt 有 1.1 TB，充裕
3. **Ground truth 时间**: 30M × 9376 queries 的精确 top-100 计算可能需要数小时
