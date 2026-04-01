# 10M HNSW 搜索性能优化 — 2026-04-01

## 背景

vectordbbench 10M 向量场景下，cortex.core 搜索 QPS 比 beta 回归约 30%。
经分析发现三个核心瓶颈：

1. faiss HNSW 每次搜索都 malloc/memset/free 一个 10MB 的 VisitedTable
2. gRPC sync server 默认线程池太小（最多 2 并发）
3. OpenMP 线程池未治理，与 gRPC 多线程冲突

本目录包含完整的分析文档、优化方案、实施 patch 和 benchmark 结果。

---

## 目录文件说明

### 分析文档

| 文件 | 说明 |
|------|------|
| `00-search-diff.md` | **核心分析文档**。详细对比 cortex.core 与 beta 的 search 调用路径差异，包含：调用图对比、FAISS 使用方式差异、所有性能差异点（P0-P9）、VisitedTable 10MB 问题根因分析、jemalloc vs glibc malloc 实测数据与深入分析（Section 10）、gRPC 线程模型调研、修复方案建议。 |
| `01-opt-plan.md` | **实施方案文档**。三个优化任务的详细实施步骤：[A] faiss VisitedTable 复用 patch、[B] gRPC 线程池配置、[C] OMP 线程池治理。包含代码修改细节、Dockerfile 修改、构建步骤、验证方法和预期效果。 |

### Patch

| 文件 | 说明 |
|------|------|
| `0001-perf-eliminate-10MB-VisitedTable-malloc-free-per-sea.patch` | **git format-patch 格式的完整 commit patch**（commit `100be7eb`）。包含本次优化的全部 4 个文件改动，可通过 `git am` 直接应用。 |

#### Patch 中包含的 4 个文件及其作用

**1. `support/third-party-build/faiss-v1.13.1-visitedtable-reuse.patch`**

faiss v1.13.1 源码补丁，修改了 2 个 faiss 文件：

- `faiss/impl/HNSW.h` — 在 `SearchParametersHNSW` 结构体中新增 `VisitedTable* visited_table = nullptr` 字段，允许调用方传入外部 VisitedTable。
- `faiss/IndexHNSW.cpp` — 修改 `hnsw_search()` 函数，当 `params` 中提供了 `visited_table` 且 `nq=1`（单条查询）时，使用外部传入的 VisitedTable 并调用 `advance()` 实现免 memset 重置；当 `nq>1`（批量查询）时仍然各自创建本地 VisitedTable，保证 OMP 多线程安全。

**2. `support/third-party-build/third-party-x64.dockerfile`**

在 Dockerfile 中新增 Step 8B，在 faiss 编译前应用上述 VisitedTable 复用补丁（在已有的 multi-dense patch Step 8A 之后）。确保构建出的 `libfaiss.so` 包含 VisitedTable 外部传入支持。

**3. `comp/vde/src/collection/shard/segment/index/index_drivers/faiss/faiss_vector_index_driver.cpp`**

VDE 搜索层的核心修改：

- 新增 `#include <faiss/impl/AuxIndexStructures.h>` 引入 `VisitedTable` 定义。
- 在 `FaissVectorIndexDriver::Search()` 中使用 `thread_local std::unique_ptr<faiss::VisitedTable>` 实现 VisitedTable 复用。gRPC sync server 的轮询线程是长生命周期的，`thread_local` 只在每个线程首次搜索时分配 10MB，之后该线程所有后续搜索都复用同一个 VisitedTable。当索引大小（`ntotal`）变化时自动重新分配。
- 将 `thread_local` VisitedTable 通过 `hnsw_params.visited_table` 传递给 faiss，faiss 内部使用 `advance()` 实现免 memset 重置（每 249 次搜索才做一次完整 memset）。

**4. `comp/vde/src/server/grpc_api/vde_grpc_server.cpp`**

gRPC server 线程池扩容：

- 新增 `SetSyncServerOption` 配置：`NUM_CQS=2`（2 个 CompletionQueue）、`MIN_POLLERS=2`（每个 CQ 最少 2 个轮询线程）、`MAX_POLLERS=8`（每个 CQ 最多 8 个轮询线程）。
- 将最大并发搜索线程从默认的 2 提升到 16（2 CQ x 8 pollers），适配 8-16 核机器。

### Benchmark 结果

| 文件 | 说明 |
|------|------|
| `100be7eb...report.md` | commit `100be7eb` 的 10M benchmark 报告。峰值 QPS 914.98（并发 20），Recall@100 = 98.82%，串行搜索延迟 10ms，最终 RSS 37.0 GB。 |
| `100be7eb...server.log` | benchmark 期间的 server 日志（含 HNSW 参数 dump、内存采样等）。 |
| `100be7eb...client.log` | benchmark 客户端输出（含各并发级别 QPS/延迟数据）。 |
| `100be7eb...-mem-server-private-vt.log` | benchmark 期间的内存监控日志。 |

---

## 未包含在 commit 中的优化（运行时配置）

**OMP 环境变量治理** — 需在启动脚本中设置，不在编译期生效：

```bash
export OMP_NUM_THREADS=1      # 禁止 OMP 创建额外线程（我们以 nq=1 调用，不需要 OMP 并行）
export KMP_BLOCKTIME=0        # Intel OMP 工作线程立即休眠，不自旋等待
export KMP_AFFINITY=disabled  # 不绑定 CPU core
```

---

## 性能提升预期

| 优化项 | 预计提升 | 机制 |
|--------|---------|------|
| VisitedTable thread_local 复用 | 10-30% latency 降低 | 消除每次搜索的 malloc(10MB) + memset(10MB) + free(10MB) |
| gRPC 线程池扩容 | 并发能力从 2 提升到 16 | NUM_CQS=2, MAX_POLLERS=8 |
| OMP 治理（运行时） | 5-15% latency 降低 | 消除 OMP runtime 开销、spin-wait CPU 浪费、全局锁竞争 |
