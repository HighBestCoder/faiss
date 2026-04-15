# VisitedTable thread_local 复用 + gRPC 线程池 + OMP 治理方案

## 背景

vectordbbench 10M 场景下 cortex.core 搜索性能比 beta 回归 ~30%。根因分析（见 `00-search-diff.md`）发现核心问题：

1. **VisitedTable 10MB 反复 malloc/free**：faiss HNSW search 以 nq=1 调用时，每次搜索都新建/销毁 10MB 的 VisitedTable，浪费 0.5-1ms（占搜索总时间 10-30%）
2. **gRPC 默认线程配置过小**：`max_pollers=2`，限制并发搜索线程数
3. **OpenMP 线程池干扰（大坑）**：faiss 编译时启用了 OMP（`-fopenmp` + Intel `libiomp5`），但没有任何地方调用 `omp_set_num_threads(1)`。多个 gRPC 线程并发搜索时，OMP runtime 创建大量后台线程空转消耗 CPU

---

## 任务 A：faiss VisitedTable 复用 patch

### A1. 准备 faiss 源码

- faiss 版本：`v1.13.1`（commit `338a5fbf8b5aea3008fac68501956cc8a525a468`）
- 本地 clone：`/builds/cortex.core/llm/faiss/`（已 apply multi-dense patch）
- 构建方式：Docker（`support/third-party-build/third-party-x64.dockerfile`）
- 现有 patch：`support/third-party-build/faiss-v1.13.1-multi-dense-support.patch`

### A2. 修改 faiss 源码（2 个文件）

#### 文件 1：`faiss/impl/HNSW.h` — 给 `SearchParametersHNSW` 加 `visited_table` 字段

```cpp
// faiss/impl/HNSW.h, line 47-53
struct SearchParametersHNSW : SearchParameters {
    int efSearch = 16;
    bool check_relative_distance = true;
    bool bounded_queue = true;
    VisitedTable* visited_table = nullptr;  // 新增：外部传入的 VisitedTable

    ~SearchParametersHNSW() {}
};
```

`VisitedTable` 在 `HNSW.h` line 41 已有前向声明（`struct VisitedTable;`），指针类型不需要完整定义，无需额外 include。

#### 文件 2：`faiss/IndexHNSW.cpp` — 在 `hnsw_search` 中使用外部 VisitedTable

修改 `hnsw_search` 函数（line 238-291 的匿名命名空间内模板函数）：

**原代码**（line 266-286）：
```cpp
#pragma omp parallel if (i1 - i0 > 1)
        {
            VisitedTable vt(index->ntotal);  // ← 每次搜索 malloc+memset 10MB
            // ...
            for (idx_t i = i0; i < i1; i++) {
                // ...
                HNSWStats stats = hnsw.search(*dis, res, vt, params);
                // ...
            }
        }
```

**改为**：
```cpp
    // 提取外部 VisitedTable（如果有）
    VisitedTable* external_vt = nullptr;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            efSearch = hnsw_params->efSearch;
            external_vt = hnsw_params->visited_table;
        }
    }

    // ... check_period 计算不变 ...

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            // 关键改动：nq=1 且有外部 VisitedTable 时直接复用，跳过 malloc/memset
            VisitedTable* vt_ptr;
            std::unique_ptr<VisitedTable> local_vt;
            if (external_vt && (i1 - i0 == 1)) {
                vt_ptr = external_vt;
                vt_ptr->advance();  // 免 memset 重置（每 249 次才真正清零）
            } else {
                local_vt = std::make_unique<VisitedTable>(index->ntotal);
                vt_ptr = local_vt.get();
            }

            typename BlockResultHandler::SingleResultHandler res(bres);
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(index->storage));

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                res.begin(i);
                index->set_query_for_search(dis.get(), x, i);

                HNSWStats stats = hnsw.search(*dis, res, *vt_ptr, params);
                n1 += stats.n1;
                n2 += stats.n2;
                ndis += stats.ndis;
                nhops += stats.nhops;
                res.end();
            }
        }
        InterruptCallback::check();
    }
```

**设计要点**：
- `external_vt && (i1 - i0 == 1)` 的双重条件确保：只在单查询（nq=1）且提供了外部 vt 时才复用，批量查询时每个 OMP 线程仍需要独立的 VisitedTable
- `advance()` 在使用前调用，利用 visno 递增机制免去 memset（每 249 次查询才真正 memset 一次 10MB）
- 不提供 `visited_table` 时行为完全不变，保持向后兼容

### A3. 生成 patch 文件

```bash
cd /builds/cortex.core/llm/faiss
# 在 v1.13.1 + multi-dense patch 基础上，只包含 VisitedTable 改动
git diff -- faiss/impl/HNSW.h faiss/IndexHNSW.cpp > \
  /builds/cortex.core/support/third-party-build/faiss-v1.13.1-visitedtable-reuse.patch
```

注意：需要先 revert multi-dense patch 的改动，或者用其他方式确保 patch 只包含 VisitedTable 相关改动。最安全的方式是：

```bash
cd /tmp
git clone https://github.com/facebookresearch/faiss.git faiss-patch
cd faiss-patch
git checkout v1.13.1
# 先 apply multi-dense patch
git apply /builds/cortex.core/support/third-party-build/faiss-v1.13.1-multi-dense-support.patch
git add -A && git commit -m "multi-dense support"
# 再做 VisitedTable 改动
# ... 编辑文件 ...
git diff > /builds/cortex.core/support/third-party-build/faiss-v1.13.1-visitedtable-reuse.patch
```

### A4. 修改 Dockerfile 应用新 patch

**文件：** `support/third-party-build/third-party-x64.dockerfile`

在 Step 8A（multi-dense patch，line 211-225）之后增加 Step 8B：

```dockerfile
# ============================================
# Step 8B: Patch FAISS for VisitedTable Reuse
# ============================================
# Performance optimization: Allow external VisitedTable to be passed via
# SearchParametersHNSW, enabling thread_local reuse and eliminating
# per-search 10MB malloc+memset+free overhead (10-30% of search time at 10M vectors).
COPY faiss-v1.13.1-visitedtable-reuse.patch /tmp/faiss-visitedtable-reuse.patch
RUN echo "========================================" && \
    echo "Applying FAISS VisitedTable reuse patch" && \
    echo "========================================" && \
    cd ${FAISS_SOURCE_DIR} && \
    git apply /tmp/faiss-visitedtable-reuse.patch && \
    echo "Patch applied successfully" && \
    rm -f /tmp/faiss-visitedtable-reuse.patch && \
    echo "========================================"
```

同样需要修改 `third-party-arm64.dockerfile`（在 Step 9A 之后加同样的步骤）。

### A5. 构建 Docker image 并提取 faiss libs

```bash
cd /builds/cortex.core/support/third-party-build

# 构建（~30-60 分钟）
docker build -f third-party-x64.dockerfile -t third-party-gcc15-x64:v6-vt .

# 从 image 提取 faiss libs
docker create --name temp-vt third-party-gcc15-x64:v6-vt
docker cp temp-vt:/builds/cortex.core/third-party/faiss /builds/cortex.core/third-party/faiss
docker rm temp-vt

# 验证
ls -lh /builds/cortex.core/third-party/faiss/linux-x64/lib/libfaiss.so
# 检查 header 包含新字段
grep "visited_table" /builds/cortex.core/third-party/faiss/linux-x64/include/faiss/impl/HNSW.h
```

### A6. 修改 cortex.core 调用层使用 thread_local VisitedTable

**文件：** `comp/vde/src/collection/shard/segment/index/index_drivers/faiss/faiss_vector_index_driver.cpp`

在 `FaissVectorIndexDriver::Search` 函数（line 375 起），修改 line 449-453 的 hnsw_params 构造：

```cpp
#include <faiss/impl/AuxIndexStructures.h>  // for VisitedTable

// ... 在 Search 函数内 ...
if (hnsw_ef > 0 && hnsw_ptr_) {
    // thread_local VisitedTable 复用：gRPC sync server 线程是长生命周期的，
    // thread_local 只在首次搜索时分配 10MB，之后每次搜索复用同一块内存
    thread_local std::unique_ptr<faiss::VisitedTable> tl_vt;
    faiss::idx_t ntotal = inner->ntotal;
    if (!tl_vt || static_cast<faiss::idx_t>(tl_vt->visited.size()) != ntotal) {
        tl_vt = std::make_unique<faiss::VisitedTable>(ntotal);
    }

    faiss::SearchParametersHNSW hnsw_params;
    hnsw_params.efSearch = static_cast<int>(hnsw_ef);
    hnsw_params.visited_table = tl_vt.get();  // 传入外部 VisitedTable

    inner->search(1, query, static_cast<faiss::idx_t>(search_k),
                  distances.data(), labels.data(), &hnsw_params);
} else {
    // 无 hnsw_ef 时走原路径（不传 params）
    inner->search(1, query, static_cast<faiss::idx_t>(search_k),
                  distances.data(), labels.data());
}
```

**线程安全说明**：
- `thread_local` 变量每个线程独立，无竞争
- gRPC sync server 的轮询线程是长生命周期（非短命线程），`thread_local` 有效
- 当 index rebuild 导致 ntotal 变化时，`tl_vt->visited.size() != ntotal` 会触发重新分配

---

## 任务 B：gRPC 线程池配置

### B1. 修改 gRPC server 配置

**文件：** `comp/vde/src/server/grpc_api/vde_grpc_server.cpp`

在 `run()` 方法（line 53）中，`builder.BuildAndStart()` 之前添加：

```cpp
void ActianVectorAIGrpcServer::run() {
  ::grpc::ServerBuilder builder;

  builder.AddListeningPort(server_address_,
                           ::grpc::InsecureServerCredentials());

  // ... RegisterService 不变 ...

  builder.SetMaxReceiveMessageSize(500 * 1024 * 1024);
  builder.SetMaxSendMessageSize(500 * 1024 * 1024);

  // gRPC sync server 线程池配置
  // 默认值 NUM_CQS=1, MIN_POLLERS=1, MAX_POLLERS=2 并发能力严重不足
  // 适配 8-core/16-core 机器，设为 NUM_CQS=2, MAX_POLLERS=8（最大 16 并发）
  builder.SetSyncServerOption(
      ::grpc::ServerBuilder::SyncServerOption::NUM_CQS, 2);
  builder.SetSyncServerOption(
      ::grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, 2);
  builder.SetSyncServerOption(
      ::grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, 8);

  spdlog::info("Building gRPC server...");
  server_ = builder.BuildAndStart();
  // ... 后面不变 ...
}
```

### B2. 配置选择依据

| 机器配置 | NUM_CQS | MIN_POLLERS | MAX_POLLERS | 最大并发线程 | thread_local VT 内存 |
|---------|---------|-------------|-------------|-------------|---------------------|
| 8-core / 32G | 2 | 2 | 6 | 12 | 120 MB |
| 16-core / 64G | 2 | 2 | 8 | 16 | 160 MB |

**推荐使用 NUM_CQS=2, MAX_POLLERS=8**，原因：
- 搜索是 CPU+内存密集型，线程数 > CPU 核心数后收益递减
- 16 并发搜索线程足以支撑高 QPS（3-5ms/query → 理论 3200-5300 QPS）
- thread_local VisitedTable 内存开销可控（16 × 10MB = 160MB）
- 相比默认的 max_pollers=2（理论上限 ~400-600 QPS），提升约 8 倍并发

---

## 任务 C：OMP 线程池治理（关键！与任务 B 配合）

### C1. 问题分析

**这是一个多线程性能的大坑。** faiss 编译时启用了 OpenMP（`-fopenmp`），使用 Intel OpenMP (`libiomp5`)。但代码中没有任何地方调用 `omp_set_num_threads(1)` 或设置 `OMP_NUM_THREADS` 环境变量。

#### 冲突机制

1. **OMP 线程池默认行为**：OpenMP runtime 在进程首次遇到 `#pragma omp parallel` 时初始化全局线程池（默认线程数 = CPU 核心数 = 8 或 16）

2. **Intel OMP spin-wait**：`KMP_BLOCKTIME` 默认 200ms，OMP 工作线程完成任务后**自旋等待 200ms** 才休眠。即使 nq=1 不触发并行，这些线程仍消耗 CPU

3. **OMP fork-join 开销**：每次进入 `#pragma omp parallel` 区域（即使 `if(false)`），OMP runtime 都有同步开销 — 检查条件、访问内部数据结构、`__kmp_fork_barrier` 全局锁

4. **增大 gRPC 线程后的叠加效应**：16 个 gRPC 线程并发搜索时，每个都碰到 `#pragma omp parallel if(false)`，OMP runtime 的全局数据结构成为竞争热点

#### 当前状态确认

```
已检查代码：comp/vde/src/server/, comp/vde/src/api/, comp/vde/src/collection/
已检查配置：Dockerfile, scripts/, faiss_presets.json
结论：OMP 完全没有被治理，全部使用默认行为
```

**备选**：也可以在 grpc service 启动脚本中设环境变量：
```bash
export OMP_NUM_THREADS=1
export KMP_BLOCKTIME=0       # 立即休眠，不自旋
export KMP_AFFINITY=disabled  # 不绑定 CPU core
```

建议**代码内 + 环境变量两者都做**：代码内保底，环境变量可在不重编译的情况下微调。

---

## 实施步骤总结

| 步骤 | 操作 | 文件 |
|------|------|------|
| 1 | 修改 faiss 源码加 `visited_table` 字段 | `llm/faiss/faiss/impl/HNSW.h` |
| 2 | 修改 faiss `hnsw_search` 使用外部 VT | `llm/faiss/faiss/IndexHNSW.cpp` |
| 3 | 生成 patch 文件 | `support/third-party-build/faiss-v1.13.1-visitedtable-reuse.patch` |
| 4 | 修改 Dockerfile 添加新 patch 步骤 | `support/third-party-build/third-party-x64.dockerfile` |
| 5 | 构建 Docker image | `docker build ...` |
| 6 | 提取 faiss libs 到项目目录 | `third-party/faiss/linux-x64/` |
| 7 | 修改调用层：thread_local VT | `comp/vde/.../faiss_vector_index_driver.cpp` |
| 8 | 修改 gRPC 线程池配置 | `comp/vde/src/server/grpc_api/vde_grpc_server.cpp` |

---

## 验证方法

1. **编译验证**：在 vde4 container 的 `/builds/cortex.core` 下运行 `make` 确保编译通过
2. **OMP 验证**：运行时 `ps -eLf | grep vde` 确认不存在大量 OMP 后台线程
3. **线程验证**：确认 gRPC 线程数 ≈ MIN_POLLERS ~ MAX_POLLERS × NUM_CQS
4. **功能验证**：运行现有单元测试和集成测试
5. **性能验证**：vectordbbench 10M benchmark 对比优化前后 QPS
6. **内存验证**：`cat /proc/<pid>/status | grep VmRSS`（预期增加 ~160MB for thread_local VT）
7. **perf 验证**：`perf record -g` 确认 malloc/memset/free 和 OMP runtime 函数在 profile 中占比大幅下降

## 预期效果

| 优化项 | 预计提升 | 机制 |
|-------|---------|------|
| VisitedTable 复用 | 10-30% latency 降低 | 消除每次搜索的 malloc(10MB)+memset(10MB)+free(10MB) |
| OMP 治理 | 5-15% latency 降低 | 消除 OMP runtime 开销、spin-wait CPU 浪费、全局锁竞争 |
| gRPC 线程池增大 | 4-8x 并发提升 | 从 2 并发 → 16 并发 |
| 三者叠加 | 30-50% QPS 提升 | latency 降低 + 并发提升 |
