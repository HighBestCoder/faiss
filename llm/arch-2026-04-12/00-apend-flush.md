# Faiss HNSW 半增量 flush 方案评估

结论先说：

**你的方案"对 HNSW 做半增量 flush"是可行的。**
但要更精确一点：

- **可增量的部分：`storage/code`**
- **基本没法直接增量的部分：HNSW graph 本身**
- **可以抽成通用框架的部分：manifest / block 管理 / 原子提交 / 恢复 / GC**
- **仍然必须 index-specific 的部分：哪些数据 append-only、哪些数据必须 checkpoint、加载后如何重建并校验内存对象**
- 所以现实可落地的方案就是：  
  **common persistence runtime + index-specific adapter**，在 HNSW 上的第一版落地形态是  
  **code append-only + HNSW graph full snapshot（每次全量写）**

这跟你想的方向基本一致。

---

## 1) 为什么这个方向可行

Faiss 当前的标准持久化就是**整块全量序列化**：

- `faiss/index_io.h`
  - `write_index(...)`
  - `read_index(...)`
  - `IO_FLAG_SKIP_STORAGE`

序列化是纯 forward-only byte stream，没有 seek、没有 section table、没有尾部 offset 索引。`IOWriter` 抽象只有 `operator()(const void* ptr, size_t size, size_t nitems)` 一个写接口，完全不支持回写。

源码里 HNSW 的序列化流程（`faiss/impl/index_write.cpp` L827-850）：

```cpp
// IndexHNSW 分支
write_index_header(idxhnsw, f);    // 写公共头：d, ntotal, is_trained, metric_type
write_HNSW(&idxhnsw->hnsw, f);    // 写 graph 结构
if (io_flags & IO_FLAG_SKIP_STORAGE) {
    uint32_t n4 = fourcc("null");
    WRITE1(n4);
} else {
    write_index(idxhnsw->storage, f);  // 写 storage/codes
}
```

`write_index_header` 写的字段（`faiss/impl/index_write.cpp` L83-94）：

```cpp
WRITE1(idx->d);
WRITE1(idx->ntotal);   // ← ntotal 硬写在流头部，读取时必须与 graph 大小完全匹配
WRITE1(dummy);         // 两个历史占位 idx_t
WRITE1(dummy);
WRITE1(idx->is_trained);
WRITE1(idx->metric_type);
```

`write_HNSW()` 顺序写（`faiss/impl/index_write.cpp` L332-348）：

```cpp
WRITEVECTOR(hnsw->assign_probas);
WRITEVECTOR(hnsw->cum_nneighbor_per_level);
WRITEVECTOR(hnsw->levels);      // size = ntotal
WRITEVECTOR(hnsw->offsets);     // size = ntotal + 1
WRITEVECTOR(hnsw->neighbors);   // 所有邻接数据，THE BIG ONE
WRITE1(hnsw->entry_point);
WRITE1(hnsw->max_level);
WRITE1(hnsw->efConstruction);
WRITE1(hnsw->efSearch);
```

而 `storage` 里 `codes` 本质上是一个连续 `uint8_t` 数组，例如：

- `IndexFlat` → `WRITEXBVECTOR(idxf->codes);`（XB 变体，同样是连续字节数组）
- `IndexPQ`   → `WRITEVECTOR(idxp->codes);`

连续数组本身是 append-only 的内存结构（新向量永远追加在末尾，`ntotal * code_size` 之后），**这是整个方案的基础**。

**code 这部分天然像 append-only buffer。**

---

## 2) 为什么 HNSW graph 不适合做"真正增量 flush"

核心数据结构（`faiss/impl/HNSW.h` L110-135）：

```cpp
std::vector<int>                levels;    // 每个点所在的最高层，size = ntotal
std::vector<size_t>             offsets;   // offsets[i]：点 i 在 neighbors 里的起始位置，size = ntotal + 1
MaybeOwnedVector<storage_idx_t> neighbors; // 所有邻接数据，flat buffer
storage_idx_t                   entry_point = -1;
int                             max_level   = -1;
```

`neighbors[offsets[i]:offsets[i+1]]` 存的是点 i **所有层**的邻居列表（多层打平存储）。

问题不是"新点只往后追加"这么简单。

**HNSW 插入新点时，会回写旧点的邻接表。**

源码证据（`faiss/impl/HNSW.cpp` L490-524，`add_links_starting_from`）：

```cpp
// 先为新点 pt_id 添加出边
add_link(*this, ptdis, pt_id, other_id, level, ...);

// 再为每个邻居 other_id 反向加回 pt_id 的入边 —— 这里会修改旧节点
for (storage_idx_t other_id : neighbors_to_add) {
    omp_set_lock(&locks[other_id]);
    add_link(*this, ptdis, other_id, pt_id, level, ...);  // ← 旧节点被 in-place 修改
    omp_unset_lock(&locks[other_id]);
}
```

`add_link` 内部还会执行 `shrink_neighbor_list`：当某个旧节点的邻居数超过 M 时，会裁剪已有邻居列表，写回 `neighbors[offsets[other_id]...]`。这是 in-place 的随机写，不是 append。

另外，`entry_point` 和 `max_level` 这两个标量在插入更高层节点时也会被更新（`add_with_locks` L578-581）：

```cpp
if (pt_level > max_level) {
    max_level = pt_level;
    entry_point = pt_id;  // ← entry_point 可能被覆盖
}
```

读取端对此也有硬校验（`faiss/impl/index_read.cpp` `read_HNSW` 之后）：标准 `read_index` 会将 `hnsw.levels.size()` 与头部 `ntotal` 做一致性检查，不匹配直接抛异常——这意味着增量读取必须完全绕开标准 `read_index` 路径。

因此 graph 里有**三类**需要完整重写的数据：

| 数据 | 原因 |
|------|------|
| `neighbors` | 旧节点的邻居链接会被 in-place 修改和裁剪（`shrink_neighbor_list`） |
| `offsets` / `levels` | 随节点增加线性增长，且偏移量依赖全局位置，不可分块 |
| `entry_point` / `max_level` | 可能被任意一次 add 更新，必须和 graph 一起原子持久化 |

结论：

- **code 增量 flush：可以**
- **HNSW graph 增量 flush：很难，不建议**
- **HNSW graph 每次全量 flush：合理**

---

## 3) 你的方案应该怎么落地

我建议不要硬改现有 `write_index/read_index` 的单文件格式。

直接做一个**自定义增量持久化格式**，最稳。

### 推荐文件布局

```
index_dir/
  index.meta              ← manifest，原子替换，永远是最新一致状态
  graph.snapshot.000001   ← 每次 flush 全量写一份新 snapshot，不覆盖旧的
  graph.snapshot.000002
  codes.block.000001      ← 第一批向量的 codes，append-only，不修改
  codes.block.000002      ← 第二批新增 codes
  ...
```

### `index.meta` 记录

```json
{
  "format_version": 1,
  "index_type": "HNSWFlat",
  "d": 128,
  "metric": "L2",
  "code_size": 512,
  "ntotal": 1000000,
  "last_flushed_code_id": 999999,
  "current_graph_snapshot": "graph.snapshot.000003",
  "code_blocks": [
    { "file": "codes.block.000001", "start_id": 0,      "end_id": 499999, "size_bytes": 256000000, "checksum": "..." },
    { "file": "codes.block.000002", "start_id": 500000, "end_id": 999999, "size_bytes": 256000000, "checksum": "..." }
  ]
}
```

注意：`entry_point` 和 `max_level` 包含在 graph snapshot 文件里（通过现有 `write_index(..., IO_FLAG_SKIP_STORAGE)` 一起写出），不需要单独在 meta 里记录；但如果想做快速健全性检查，可以在 meta 里冗余记录一份。

### flush 流程

#### 第一次 flush（meta 为空或不存在）

1. 用现有 `write_index(..., IO_FLAG_SKIP_STORAGE)` 写 `graph.snapshot.000001`
2. 把 `codes[0:ntotal]` 整块写入 `codes.block.000001`
3. 写 `meta.tmp`，fsync，rename 成 `index.meta`

#### 后续 flush（meta 存在）

1. 读 meta，得到 `last_flushed_code_id = A`，当前 `ntotal = B`
2. 把 `codes[A+1:B]` 写入新的 `codes.block.XXXXXX`
3. 用现有 `write_index(..., IO_FLAG_SKIP_STORAGE)` 写新的 `graph.snapshot.XXXXXX`（全量，不可避免）
4. 生成新 meta（更新 `ntotal`、`last_flushed_code_id`、`current_graph_snapshot`、追加 block 条目）
5. 原子 rename 替换 `index.meta`
6. 旧 graph snapshot 可以在确认 meta 替换成功后异步 GC 删除

---

## 4) crash consistency 怎么做

这个很重要。建议提交顺序严格如下：

```
1. 写 codes.block.XXXXXX.tmp
2. fsync(codes block tmp)
3. rename → codes.block.XXXXXX         ← codes 已持久化，可被引用

4. 写 graph.snapshot.XXXXXX.tmp
5. fsync(graph snapshot tmp)
6. rename → graph.snapshot.XXXXXX      ← graph 已持久化，可被引用

7. 写 meta.tmp（引用新 codes block + 新 graph snapshot）
8. fsync(meta tmp)
9. rename → index.meta                  ← 原子切换，这是唯一的"提交点"
```

关键约束：

- **旧的 graph snapshot 在 step 9 完成之前绝对不能删除**，它是 crash 恢复的最后一道防线
- crash 发生在 step 9 之前：meta 还指向旧 snapshot + 旧 codes，状态完全一致；新写的 tmp 文件在下次启动时清理
- crash 发生在 step 9 之后：meta 已指向新状态，一致；旧 snapshot 可以安全 GC
- codes block 是 write-once 的，永远不修改，不存在 partial-write 导致 corrupt 读的问题

**不要先覆盖旧 meta。rename 是唯一的原子提交点。**

---

## 5) 应该改 Faiss 哪些地方

### 最小改法（推荐）

**不要动通用 `write_index/read_index`。**
新增一套独立 API：

```cpp
// faiss/index_io_incremental.h（新文件）
void write_index_incremental(
    const IndexHNSWFlat* idx,
    const char* meta_path,
    int io_flags = 0);

IndexHNSWFlat* read_index_incremental(
    const char* meta_path,
    int io_flags = 0);
```

这样风险最小，不会破坏现有格式兼容性，也不需要动 `IndexHNSW` 基类。

### 进一步往通用能力抽象：可以，但要分层

这里有一个很重要的边界：

- **可以做成通用的，是持久化编排框架**
- **不太可能做成完全通用的，是每种 index 的增量语义**

原因是不同 index 对"什么可以 append"、"什么必须全量 checkpoint"、"读取后要补哪些派生状态"的定义不一样。

以几类典型 index 为例：

| index | 可能的 append-only 部分 | 必须 checkpoint / 重建的部分 | 难点 |
|------|-------------------------|-------------------------------|------|
| `IndexHNSWFlat` | `storage->codes` 连续尾部 | HNSW graph 全量 snapshot | graph 会回写旧节点 |
| `IndexHNSWPQ` | `codes` 尾部（理论上可分块） | graph + PQ 读后状态 | SDC table / 派生状态处理更复杂 |
| `IndexIVF*` | 倒排 list 追加有机会分段 | list 元数据 / direct map / 训练态 | 不是简单线性 append |
| `IndexFlat*` | 整个 xb/codes 天然连续 | 几乎没有额外 graph | 反而更像纯 block storage |

所以比较合理的工程化抽象是两层：

### Layer 1：通用 persistence runtime

这一层尽量与 index 类型无关，负责：

- `index.meta` / manifest schema 与版本管理
- block / snapshot 文件命名规则
- `tmp -> fsync -> rename` 的提交协议
- crash recovery
- checksum / size 校验
- old snapshot / orphan tmp 文件 GC

### Layer 2：index-specific adapter

这一层保留 index 语义，负责回答 4 个问题：

1. **当前 index 的 snapshot 是什么？**
2. **当前 index 的 append-only delta 是什么？**
3. **如何从 snapshot + delta blocks 重建内存对象？**
4. **加载后哪些不变量必须校验？**

如果要抽象接口，大致应该长成这种感觉：

```cpp
struct IncrementalPersistenceAdapter {
    virtual std::string index_type() const = 0;

    // 写一个可独立加载的 checkpoint
    virtual void write_snapshot(const Index* idx, const char* path) = 0;

    // 从上次 flush 之后，导出 append-only 增量
    virtual void write_delta_blocks(
            const Index* idx,
            const FlushState& last_state,
            const char* dir,
            ManifestDelta* out) = 0;

    // 从 manifest 重建 index
    virtual Index* read_from_manifest(
            const Manifest& meta,
            const char* dir) = 0;

    // 校验重建后的关键不变量
    virtual void validate_loaded(
            const Index* idx,
            const Manifest& meta) = 0;
};
```

注意这不是说 Faiss 核心里要立刻引入这样一个基类；它表达的是**职责边界**：

- runtime 尽量 common
- adapter 必须承认 index-specific

### v1 仍然建议只做 `IndexHNSWFlat`

虽然框架层可以从第一天就按通用方式设计，但**第一版实现范围仍应收敛到 `IndexHNSWFlat`**。

原因：

- 它的 append-only 部分最清晰：`IndexFlatCodes::codes` 尾部切片即可
- graph snapshot 可直接复用现有 `write_index(..., IO_FLAG_SKIP_STORAGE)`
- 读回后只需要重建匹配 metric 的 `IndexFlat` storage
- 不会一开始就把 PQ/SQ 的派生状态、读时初始化逻辑和 flag 语义一起卷进来

### 内部实现要点

**写路径：**

1. 用 `IO_FLAG_SKIP_STORAGE` 把 graph 和 storage 分开（Faiss 已有此标志）
2. graph snapshot 直接复用已有公开 API：`write_index(idx, snapshot_path, IO_FLAG_SKIP_STORAGE)`
3. codes 部分直接访问 `idx->storage->codes` 这个 `std::vector<uint8_t>`，用自定义 `FileIOWriter` 写出增量片段

**读路径：**

1. 解析 meta JSON，得到 graph snapshot 文件 + codes block 列表
2. 用已有公开 API：`read_index(graph_snapshot_path)` 读回 graph snapshot，并校验类型为目标 index
3. 手动拼接所有 codes block 到新建 storage 的 `codes`
4. 校验 `manifest.ntotal`、snapshot header 的 `ntotal`、graph 大小与 codes 总大小一致后，设置 `idx->ntotal` 和 `storage->ntotal`

**注意 `ntotal` 的坑：** `write_index_header` 会把 `ntotal` 写进 snapshot 头部，因此 manifest 里的 `ntotal`、snapshot header 的 `ntotal`、`hnsw.levels.size()`、codes block 总长度必须完全一致。更稳的做法不是绕开 `read_index`，而是**先用 `read_index` 读一个合法的 graph snapshot，再补齐 storage，并做一致性校验**。

### 需要看的源码点

| 文件 | 关注点 |
|------|--------|
| `faiss/index_io.h` | `IO_FLAG_SKIP_STORAGE`，IOWriter/IOReader 抽象接口 |
| `faiss/impl/index_write.cpp` | `write_index_header()`（L83），IndexHNSW 分支（L827），`IO_FLAG_SKIP_STORAGE` 的写法 |
| `faiss/impl/index_read.cpp` | HNSW 读取分支（L1184）与 graph/header 一致性校验 |
| `faiss/impl/HNSW.h` | `neighbors` / `offsets` / `levels` 的内存布局 |
| `faiss/IndexHNSW.h` | `storage` 成员的类型，`codes` 的位置 |
| `faiss/impl/io.h` | `FileIOWriter` / `VectorIOWriter` 可以直接复用 |

---

## 6) 这个方案什么时候收益大

### 数据量化分析

对 `n` 个 `d` 维向量，HNSW 参数 `M`（默认 32），存储大小大约：

| 类型 | codes 大小 | graph 大小（约） |
|------|-----------|----------------|
| `IndexHNSWFlat` | `n × d × 4 bytes`（float32） | `n × 2M × 4 bytes`（int32 邻居 ID） |
| `IndexHNSWSQ` | `n × d × 1 byte`（SQ8） | 同上 |
| `IndexHNSWPQ` (M_pq=8, nbits=8) | `n × M_pq × 1 byte` | 同上 |

以 `n=1M, d=128, M=32` 为例：

| 类型 | codes | graph | codes 占比 |
|------|-------|-------|-----------|
| HNSWFlat | ~512 MB | ~256 MB | **67%** |
| HNSWSQ   | ~128 MB | ~256 MB | 33% |
| HNSWPQ (M=8) | ~8 MB | ~256 MB | **3%** |

### 结论

**增量 code flush 收益大的场景：**

- `IndexHNSWFlat`，尤其是维度高（d ≥ 64）的情况
- 每次只新增少量向量（比如新增 10%），这时增量写 code 只写 10%，graph 还是全量但相对 codes 体量更小
- flush 频繁、总数据量大

**收益有限的场景：**

- `IndexHNSWPQ` / 量化压缩率高的配置：codes 本来就很小，graph 才是大头，graph 全量写成为瓶颈
- 每次 flush 都几乎涵盖全量数据（比如批量构建完再 flush 一次）

> "大头肯定是 code" —— 这对 `HNSWFlat` 高维向量成立；对 `HNSWPQ` 几乎相反。

---

## 7) 更靠谱的产品化建议

与其把它叫"incremental flush"，我更建议你把它设计成：

**common persistence runtime + per-index adapter**

在 `IndexHNSWFlat` 上的具体落地形态是：

**manifest + append-only code segments + full graph checkpoints**

这是更工程化、也更好维护的模型。它的本质不是 Faiss 原生格式增强，而是：

- **Faiss 内存索引不变**（add/search 行为完全一致）
- **持久化层变成你自己的 segment/checkpoint 系统**
- **通用 runtime 负责提交、恢复、GC；index adapter 负责如何切 snapshot / delta**
- Faiss 只负责提供现有 index snapshot 序列化能力，不负责整体的持久化编排

类比可以参考 diskann-rs 的 delta-layer 设计：base index mmap + in-memory delta + periodic compact，只是我们的 "base" 是 code blocks，"delta" 是每次新增的 code block，"compact" 是 graph snapshot 更新。

---

## 8) 我对你 5 点设计的判断

### [1] metadata 记录 flush 到 A
**可行，建议做。** 即上文的 `last_flushed_code_id`。

### [2] 下次从 A+1 开始把 codes flush 到 block
**可行，核心就是这个。** `idx->storage->codes` 是连续数组，直接按字节偏移切片写出即可，无需任何 Faiss 内部修改。

### [3] HNSW 其他部分全量 flush
**正确，这是最现实的做法。** graph 全量写的成本在高维 flat 场景下远小于 codes，可以接受。

### [4] create index 时设置 metadata file path
**可做，但不要塞进通用 `Index` 基类。** 最好挂在你自己的 wrapper、runtime context 或 helper API 上，对 Faiss 核心代码零侵入。

### [5] flush 时根据 metadata 决定全量还是增量
**合理。**
- meta 文件不存在或为空 → full flush
- meta 存在 → append code block + rewrite graph snapshot + atomic meta swap

---

## 9) 关于"能不能做成 common 功能"的最终判断

**可以做成 common framework，但不能假设增量语义与 index 无关。**

更准确地说：

- **common 的是 persistence runtime**
- **不 common 的是 per-index adapter**

也就是说，真正可复用的公共能力包括：

- manifest 解析与版本迁移
- block/snapshot 生命周期管理
- 原子提交协议
- 恢复逻辑
- checksum / size / 文件存在性校验
- 垃圾回收

而必须让每种 index 自己定义的是：

- 什么数据能增量导出
- 什么数据必须做 checkpoint
- 如何从文件恢复内存结构
- 恢复后哪些 invariants 要检查

所以比较靠谱的路线不是：

> "做一个与所有 index 都无关的统一增量序列化协议"

而是：

> "做一个通用持久化编排框架，再为 `IndexHNSWFlat`、`IndexFlat*`、`IndexIVF*` 等分别提供 adapter"

对当前这个需求来说，最合理的第一步是：

1. 先把 runtime 按通用层设计好
2. 第一个 adapter 只实现 `IndexHNSWFlat`
3. 等 HNSWFlat 方案稳定后，再评估是否值得给 `HNSWPQ`、`IndexFlat`、`IVF` 增加 adapter

---

## 最终判断

**你的思路是对的，而且可以进一步提升成：通用持久化框架 + HNSW 专用 adapter。**

但要明确边界：

- **不是"Faiss 所有 index 都共享一套完全相同的增量语义"**
- 也不是 **"Faiss HNSW 全量增量化"**
- 而是 **"通用 runtime + per-index adapter"**
- 在 HNSW 上的具体体现是 **"storage(code) 增量化 + graph checkpoint 化"**

最大风险点在于实现的 crash consistency——codes 追加和 graph snapshot 更新是两个独立文件操作，meta 的原子 rename 是唯一的提交点，所有 GC 和恢复逻辑都要围绕这个点设计。

如果你愿意，下一步可以直接给出：

1. **`index.meta` 完整结构定义（JSON Schema 或 Protobuf）**
2. **`write_index_incremental / read_index_incremental` 的 C++ 伪代码**
3. **针对 `IndexHNSWFlat` 的最小侵入 patch**
4. **GC 逻辑设计**

---

## 10) `index.meta` 建议结构（v1）

下面给一版偏工程实现导向的 schema。目标不是追求最抽象，而是：

- 能支撑 `IndexHNSWFlat` v1 落地
- 同时给后续别的 index adapter 留扩展位
- 明确 crash recovery / GC / 兼容性检查需要的字段

### 顶层 JSON 示例

```json
{
  "format_version": 1,
  "runtime_type": "incremental-index-store",
  "adapter_type": "hnsw_flat_v1",
  "index_type": "IndexHNSWFlat",
  "state": "committed",
  "index_params": {
    "d": 128,
    "metric": "METRIC_L2",
    "code_size": 512,
    "hnsw_M": 32
  },
  "committed_state": {
    "epoch": 3,
    "ntotal": 1000000,
    "last_flushed_id": 999999,
    "graph_snapshot": {
      "file": "graph.snapshot.000003",
      "bytes": 268435456,
      "checksum": "sha256:...",
      "created_at": "2026-04-12T01:23:45Z"
    },
    "code_blocks": [
      {
        "file": "codes.block.000001",
        "start_id": 0,
        "count": 500000,
        "code_size": 512,
        "bytes": 256000000,
        "checksum": "sha256:..."
      },
      {
        "file": "codes.block.000002",
        "start_id": 500000,
        "count": 500000,
        "code_size": 512,
        "bytes": 256000000,
        "checksum": "sha256:..."
      }
    ]
  },
  "stale_graph_snapshots": [
    "graph.snapshot.000001",
    "graph.snapshot.000002"
  ]
}
```

### 字段说明

#### 顶层元信息

| 字段 | 含义 | 备注 |
|------|------|------|
| `format_version` | manifest 格式版本 | 用于 schema 迁移 |
| `runtime_type` | 通用 runtime 类型 | 防止误读成别的存储格式 |
| `adapter_type` | 当前 index adapter 版本 | 例如 `hnsw_flat_v1` |
| `index_type` | 逻辑 index 类型 | 例如 `IndexHNSWFlat` |
| `state` | manifest 状态 | v1 固定 `committed` 即可 |

#### `index_params`

这是**加载前的快速兼容性检查**用字段。至少建议包含：

- `d`
- `metric`
- `code_size`
- `hnsw_M`

如果后续扩展到别的 adapter，可以增加：

- PQ/SQ 参数
- IVF 训练参数
- numeric type
- 其他恢复时必须匹配的静态配置

#### `committed_state`

这是 manifest 里真正的**提交状态**：

- `epoch`：单调递增的提交序号，便于调试、GC、对账
- `ntotal`：当前提交状态总向量数
- `last_flushed_id`：当前已写到哪个 vector id
- `graph_snapshot`：当前 graph checkpoint 文件
- `code_blocks`：当前提交状态引用的全部 code blocks

#### `gc`

不是提交语义的一部分，只是 hint。即使这些字段丢了，也不影响恢复正确性。

v1 可以只放：

- `stale_graph_snapshots`

后面如果需要，也可以挂：

- orphan tmp 文件列表
- 待清理 block 列表

### 更严格的 schema 约束

v1 建议把下面这些当作**硬约束**：

1. `code_blocks[i].start_id` 必须连续、无重叠、无空洞
2. `code_blocks[i].bytes == code_blocks[i].count * code_blocks[i].code_size`
3. 所有 block 的 `code_size` 必须等于 `index_params.code_size`
4. `sum(code_blocks[*].count) == committed_state.ntotal`
5. 当 `ntotal > 0` 时，`last_flushed_id + 1 == ntotal`；当 `ntotal == 0` 时，`last_flushed_id == -1`
6. `graph_snapshot.file` 必须存在且 checksum 匹配

如果任一条不满足，读取端应视为：

- 文件损坏
- manifest 与文件不一致
- 或不支持的历史格式

总之都应该**硬失败**，而不是尝试“猜测恢复”。

---

## 11) common runtime 接口草案

这一层不关心 HNSW 细节，只负责提交编排。

### 运行时核心对象

```cpp
struct BlockFileRef {
    std::string file;
    uint64_t start_id;
    uint64_t count;
    uint64_t code_size;
    uint64_t bytes;
    std::string checksum;
};

struct SnapshotFileRef {
    std::string file;
    uint64_t bytes;
    std::string checksum;
    std::string created_at;
};

struct IndexParams {
    int d;
    int metric;
    size_t code_size;
    int hnsw_M;
};

struct CommittedState {
    uint64_t epoch;
    uint64_t ntotal;
    int64_t last_flushed_id;
    SnapshotFileRef graph_snapshot;
    std::vector<BlockFileRef> code_blocks;
};

struct Manifest {
    int format_version;
    std::string runtime_type;
    std::string adapter_type;
    std::string index_type;
    std::string state;
    IndexParams index_params;
    CommittedState committed_state;
    std::vector<std::string> stale_graph_snapshots;
};
```

### adapter 接口草案

```cpp
struct IncrementalIndexAdapter {
    virtual ~IncrementalIndexAdapter() = default;

    virtual std::string adapter_type() const = 0;
    virtual std::string index_type() const = 0;

    virtual IndexParams describe_index(const Index* idx) const = 0;

    virtual void validate_index_before_flush(const Index* idx) const = 0;

    virtual SnapshotFileRef write_snapshot(
            const Index* idx,
            const std::string& tmp_path,
            const std::string& final_path) const = 0;

    virtual std::vector<BlockFileRef> write_delta_blocks(
            const Index* idx,
            const std::optional<CommittedState>& previous,
            const std::string& dir) const = 0;

    virtual std::unique_ptr<Index> load_snapshot(
            const SnapshotFileRef& snapshot,
            const std::string& dir) const = 0;

    virtual void attach_code_blocks(
            Index* idx,
            const std::vector<BlockFileRef>& blocks,
            const std::string& dir) const = 0;

    virtual void validate_loaded_index(
            const Index* idx,
            const Manifest& manifest) const = 0;
};
```

### runtime 接口草案

```cpp
class IncrementalIndexStore {
   public:
    explicit IncrementalIndexStore(std::string dir);

    Manifest read_manifest() const;
    bool manifest_exists() const;
    void recover_orphan_tmp_files() const;

    void flush(const Index* idx, const IncrementalIndexAdapter& adapter);

    std::unique_ptr<Index> load(const IncrementalIndexAdapter& adapter) const;

    void gc(const Manifest& manifest) const;
};
```

### runtime 责任边界

#### runtime 必须负责

- 目录存在性检查
- manifest 读写
- `tmp -> fsync -> rename` 原子提交
- 提交失败后的 orphan tmp 清理
- 提交成功后的延迟 GC

#### runtime 不应负责

- 理解 HNSW graph 内部布局
- 解释 PQ/SQ 的派生状态
- 判断某个 index 的 delta 怎么切
- 推断损坏文件该怎么“容错恢复”

这些都应该留给 adapter 或直接硬失败。

---

## 12) `IndexHNSWFlat` adapter 伪代码

### flush 伪代码

```cpp
void write_index_incremental(
        const IndexHNSWFlat* idx,
        const char* dir,
        int io_flags) {
    FAISS_THROW_IF_NOT(idx != nullptr);

    HNSWFlatAdapter adapter;
    IncrementalIndexStore store(dir);

    store.recover_orphan_tmp_files();
    adapter.validate_index_before_flush(idx);

    std::optional<Manifest> old_meta;
    if (store.manifest_exists()) {
        old_meta = store.read_manifest();
    }

    // 1. 导出新增 code blocks
    auto new_blocks = adapter.write_delta_blocks(
            idx,
            old_meta ? std::optional(old_meta->committed_state) : std::nullopt,
            dir);

    // 2. 导出新的 graph snapshot
    auto next_epoch = old_meta ? old_meta->committed_state.epoch + 1 : 1;
    auto snapshot = adapter.write_snapshot(
            idx,
            fmt("%s/graph.snapshot.%06llu.tmp", dir, next_epoch),
            fmt("%s/graph.snapshot.%06llu", dir, next_epoch));

    // 3. 组装新 manifest
    Manifest new_meta;
    new_meta.format_version = 1;
    new_meta.runtime_type = "incremental-index-store";
    new_meta.adapter_type = adapter.adapter_type();
    new_meta.index_type = adapter.index_type();
    new_meta.state = "committed";
    new_meta.index_params = adapter.describe_index(idx);
    new_meta.committed_state.epoch = next_epoch;
    new_meta.committed_state.ntotal = idx->ntotal;
    new_meta.committed_state.last_flushed_id = idx->ntotal == 0 ? -1 : idx->ntotal - 1;
    new_meta.committed_state.graph_snapshot = snapshot;

    if (old_meta) {
        new_meta.committed_state.code_blocks = old_meta->committed_state.code_blocks;
        if (!old_meta->committed_state.graph_snapshot.file.empty()) {
            new_meta.stale_graph_snapshots.push_back(
                    old_meta->committed_state.graph_snapshot.file);
        }
    }
    new_meta.committed_state.code_blocks.insert(
            new_meta.committed_state.code_blocks.end(),
            new_blocks.begin(),
            new_blocks.end());

    // 4. runtime 做硬校验
    validate_manifest_consistency(new_meta);

    // 5. 原子提交 manifest
    store.atomic_replace_manifest(new_meta);

    // 6. manifest 提交后再做 GC
    store.gc(new_meta);
}
```

### `HNSWFlatAdapter::write_delta_blocks` 伪代码

```cpp
std::vector<BlockFileRef> HNSWFlatAdapter::write_delta_blocks(
        const Index* base,
        const std::optional<CommittedState>& previous,
        const std::string& dir) const {
    auto idx = dynamic_cast<const IndexHNSWFlat*>(base);
    FAISS_THROW_IF_NOT(idx != nullptr);
    FAISS_THROW_IF_NOT(idx->storage != nullptr);

    auto storage = dynamic_cast<const IndexFlat*>(idx->storage);
    FAISS_THROW_IF_NOT(storage != nullptr);

    const size_t code_size = storage->code_size;
    const uint64_t prev_ntotal = previous ? previous->ntotal : 0;
    const uint64_t cur_ntotal = idx->ntotal;

    FAISS_THROW_IF_NOT(cur_ntotal >= prev_ntotal);

    if (cur_ntotal == prev_ntotal) {
        return {};
    }

    const uint64_t delta_count = cur_ntotal - prev_ntotal;
    const uint8_t* begin = storage->codes.data() + prev_ntotal * code_size;
    const uint64_t bytes = delta_count * code_size;

    std::string final_file = make_block_filename(dir, prev_ntotal, delta_count);
    std::string tmp_file = final_file + ".tmp";

    write_all(tmp_file, begin, bytes);
    fsync_file(tmp_file);
    rename_atomic(tmp_file, final_file);

    return {BlockFileRef{
            .file = basename(final_file),
            .start_id = prev_ntotal,
            .count = delta_count,
            .code_size = code_size,
            .bytes = bytes,
            .checksum = sha256_file(final_file)}};
}
```

### `HNSWFlatAdapter::write_snapshot` 伪代码

```cpp
SnapshotFileRef HNSWFlatAdapter::write_snapshot(
        const Index* base,
        const std::string& tmp_path,
        const std::string& final_path) const {
    auto idx = dynamic_cast<const IndexHNSWFlat*>(base);
    FAISS_THROW_IF_NOT(idx != nullptr);

    write_index(idx, tmp_path.c_str(), IO_FLAG_SKIP_STORAGE);
    fsync_file(tmp_path);
    rename_atomic(tmp_path, final_path);

    return SnapshotFileRef{
            .file = basename(final_path),
            .bytes = file_size(final_path),
            .checksum = sha256_file(final_path),
            .created_at = utc_now_rfc3339()};
}
```

### load 伪代码

```cpp
IndexHNSWFlat* read_index_incremental(const char* dir, int io_flags) {
    HNSWFlatAdapter adapter;
    IncrementalIndexStore store(dir);

    store.recover_orphan_tmp_files();
    Manifest meta = store.read_manifest();

    validate_manifest_consistency(meta);
    FAISS_THROW_IF_NOT(meta.adapter_type == adapter.adapter_type());

    std::unique_ptr<Index> loaded = adapter.load_snapshot(
            meta.committed_state.graph_snapshot,
            dir);

    adapter.attach_code_blocks(
            loaded.get(),
            meta.committed_state.code_blocks,
            dir);

    adapter.validate_loaded_index(loaded.get(), meta);

    auto* idx = dynamic_cast<IndexHNSWFlat*>(loaded.release());
    FAISS_THROW_IF_NOT(idx != nullptr);
    return idx;
}
```

### `HNSWFlatAdapter::load_snapshot` 伪代码

```cpp
std::unique_ptr<Index> HNSWFlatAdapter::load_snapshot(
        const SnapshotFileRef& snapshot,
        const std::string& dir) const {
    auto path = join_path(dir, snapshot.file);
    auto idx = faiss::read_index_up(path.c_str());

    auto* hnsw = dynamic_cast<IndexHNSWFlat*>(idx.get());
    FAISS_THROW_IF_NOT(hnsw != nullptr);
    FAISS_THROW_IF_NOT(hnsw->storage == nullptr);

    return idx;
}
```

### `HNSWFlatAdapter::attach_code_blocks` 伪代码

```cpp
void HNSWFlatAdapter::attach_code_blocks(
        Index* base,
        const std::vector<BlockFileRef>& blocks,
        const std::string& dir) const {
    auto idx = dynamic_cast<IndexHNSWFlat*>(base);
    FAISS_THROW_IF_NOT(idx != nullptr);
    FAISS_THROW_IF_NOT(idx->storage == nullptr);

    MetricType metric = static_cast<MetricType>(idx->metric_type);
    auto storage = std::make_unique<IndexFlat>(idx->d, metric);

    uint64_t total_count = 0;
    for (const auto& block : blocks) {
        auto path = join_path(dir, block.file);
        auto bytes = read_entire_file(path);

        FAISS_THROW_IF_NOT(bytes.size() == block.bytes);
        FAISS_THROW_IF_NOT(block.bytes == block.count * block.code_size);

        storage->codes.insert(
                storage->codes.end(),
                bytes.begin(),
                bytes.end());
        total_count += block.count;
    }

    storage->ntotal = total_count;
    idx->storage = storage.release();
    idx->own_fields = true;
    idx->ntotal = total_count;
}
```

### `HNSWFlatAdapter::validate_loaded_index` 伪代码

```cpp
void HNSWFlatAdapter::validate_loaded_index(
        const Index* base,
        const Manifest& meta) const {
    auto idx = dynamic_cast<const IndexHNSWFlat*>(base);
    FAISS_THROW_IF_NOT(idx != nullptr);
    FAISS_THROW_IF_NOT(idx->storage != nullptr);

    auto storage = dynamic_cast<const IndexFlat*>(idx->storage);
    FAISS_THROW_IF_NOT(storage != nullptr);

    const uint64_t ntotal = meta.committed_state.ntotal;
    const size_t code_size = meta.index_params.code_size;

    FAISS_THROW_IF_NOT(idx->ntotal == ntotal);
    FAISS_THROW_IF_NOT(storage->ntotal == ntotal);
    FAISS_THROW_IF_NOT(idx->hnsw.levels.size() == ntotal);
    FAISS_THROW_IF_NOT(idx->hnsw.offsets.size() == ntotal + 1);
    FAISS_THROW_IF_NOT(storage->codes.size() == ntotal * code_size);
}
```

---

## 13) 失败语义与恢复策略

v1 一定要把失败语义定义得保守，不要做“聪明恢复”。

### flush 失败

#### codes block 写失败

- 本次 flush 失败
- 不更新 manifest
- tmp 文件保留或下次清理均可
- 旧 manifest 仍然可用

#### graph snapshot 写失败

- 本次 flush 失败
- 不更新 manifest
- 新写入 block 即使已经落盘，也不能被当前 manifest 引用
- 下次启动可按 orphan 文件清理

#### manifest 写失败

- 本次 flush 失败
- 旧 manifest 仍为唯一 committed 状态
- 新 snapshot / new blocks 都视作未提交垃圾文件

### load 失败

以下情况应直接抛错，不尝试降级：

- manifest 不存在
- manifest schema/version 不支持
- checksum 不匹配
- code block 不连续或有空洞
- snapshot 类型不匹配（不是 `IndexHNSWFlat`）
- `ntotal` / `levels.size()` / `offsets.size()` / `codes.size()` 不一致

### startup recovery

启动恢复只做**保守清理**：

1. 读取 manifest
2. 枚举目录下所有 `*.tmp`
3. 删除所有未被 manifest 引用的 tmp 文件
4. 保留 manifest 当前引用的 snapshot 与 blocks
5. 可选：把旧 snapshot 放入待 GC 队列

注意：

- **不要**在恢复逻辑里尝试“猜哪个 snapshot 更新”
- **不要**根据文件时间戳推断提交先后
- **只能以当前 manifest 为准**

---

## 14) v1 实现边界

为了避免 scope creep，v1 建议写死以下边界：

### 明确支持

- `IndexHNSWFlat`
- CPU index
- 单 writer / 外部串行化 flush
- append-only add 场景
- graph full snapshot + code block append

### 明确不支持

- `IndexHNSWPQ` / `IndexHNSWSQ` / `IndexHNSWCagra` / Panorama
- delete / update / permute / merge / compact
- flush 与 add 并发
- mmap 加载路径
- 多进程同时写同一 index_dir
- 通过修改通用 `write_index/read_index` 支持该格式

### 进入 v2 之前再考虑的事情

- adapter 注册机制
- 非 HNSW index 的 adapter
- block compaction
- mmap / lazy load
- tombstone / delete 语义
- 分布式或多 writer 协调
