# Shared Storage Dual-Index 设计文档

> **项目**: FAISS HNSW 共享存储双索引  
> **版本**: v1.0  
> **日期**: 2026-02-07  
> **代码仓库**: `faiss/` 目录

---

## 目录

1. [背景与问题陈述](#1-背景与问题陈述)
2. [需求分析](#2-需求分析)
3. [现有架构分析](#3-现有架构分析)
4. [设计方案](#4-设计方案)
5. [详细设计](#5-详细设计)
6. [实现计划](#6-实现计划)
7. [测试方案](#7-测试方案)
8. [风险与缓解](#8-风险与缓解)
9. [未来扩展](#9-未来扩展)

---

## 1. 背景与问题陈述

### 1.1 背景

FAISS 的 `IndexHNSW` 索引内部持有一个 `storage` 指针（类型为 `Index*`），用于存放实际的向量数据。以 `IndexHNSWFlat` 为例，其 storage 是一个 `IndexFlatL2`（或 `IndexFlat`），内部以一个连续的 `std::vector<uint8_t> codes` 数组保存所有原始向量。HNSW 图结构中使用 `storage_idx_t`（int32_t）作为节点编号，该编号与 storage 中的存储下标是 **一一对应** 的。

当 `IndexHNSW` 需要计算距离时，通过 `storage->get_distance_computer()` 获取 `DistanceComputer`，后者使用 `codes + i * code_size` 直接定位向量数据。当需要重建向量时，`IndexHNSW::reconstruct(key, recons)` 直接委托到 `storage->reconstruct(key, recons)`。

### 1.2 问题

在实际生产环境中，用户有如下场景：

1. 已有一份 HNSW 索引 (index1) 运行一段时间后，大量数据被删除，HNSW 图质量退化，搜索性能下降。
2. 用户希望用当前有效数据 **重建一份新的 HNSW 索引** (index2) 以恢复搜索性能。
3. 重建期间，index1 继续提供搜索服务 (只读)，index2 构建完成后再切换。

**核心矛盾**：按照现有实现，重建 index2 需要将 index1 中的向量通过 `reconstruct()` 复制到 index2 的新 storage 中。这导致 **向量数据在内存中存在两份**。当原始索引已占用 128GB 内存时，双倍内存需求 (256GB) 是不可接受的。

### 1.3 目标

在不复制向量数据的前提下，让两份 HNSW 索引共享同一份底层向量存储，仅通过各自的 ID 映射表引用实际数据，将额外内存开销降低到一个映射表级别 (通常 < 原始数据的 1%)。

---

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 优先级 |
|------|------|--------|
| FR-1 | 两份索引可以共享同一份底层向量数据，不重复存储 | P0 |
| FR-2 | 新索引的构建过程中，旧索引可以继续正常提供搜索服务 | P0 |
| FR-3 | 新 storage 必须实现完整的 `Index` 接口，可作为 `IndexHNSW` 的 storage 使用 | P0 |
| FR-4 | 新 storage 提供正确的 `DistanceComputer`，使 HNSW 搜索 / 构建流程无需修改 | P0 |
| FR-5 | 支持 `reconstruct()` 从共享存储中恢复指定向量 | P1 |
| FR-6 | 支持标记哪些向量在新索引中有效 (过滤已删除的向量) | P1 |
| FR-7 | 两份索引可以独立销毁，底层存储通过引用计数管理生命周期 | P1 |

### 2.2 非功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| NFR-1 | **内存效率** | 新 storage 的额外内存开销应 ≤ `N × sizeof(idx_t)` (N = 向量数) |
| NFR-2 | **搜索性能** | 引入间接映射后，搜索延迟增加不超过 5% |
| NFR-3 | **构建性能** | 索引构建性能无显著退化 |
| NFR-4 | **线程安全** | 旧索引搜索 (只读) 与新索引构建 (只读共享数据 + 写入自身图结构) 可并行 |
| NFR-5 | **兼容性** | 不修改 `IndexHNSW` 本身的代码，仅新增 storage 类型 |

### 2.3 约束条件

- 重建期间，底层向量数据 **只读**，不允许增删改。
- 新索引构建完成后，可以独立获取一组新的向量数据所有权（可选的深拷贝或持续共享）。
- 需要兼容现有的 FAISS 序列化框架（可在后续迭代支持）。

---

## 3. 现有架构分析

### 3.1 类层次结构

```
Index (base)
 ├── IndexFlatCodes
 │    ├── IndexFlat
 │    │    ├── IndexFlatL2
 │    │    └── IndexFlatIP
 │    └── IndexScalarQuantizer
 └── IndexHNSW
      ├── IndexHNSWFlat    (storage = IndexFlatL2)
      ├── IndexHNSWPQ      (storage = IndexPQ)
      └── IndexHNSWSQ      (storage = IndexScalarQuantizer)
```

### 3.2 IndexHNSW 与 Storage 的交互

`IndexHNSW` 通过以下方式使用 storage：

```cpp
// IndexHNSW 关键成员
struct IndexHNSW : Index {
    HNSW hnsw;                  // 图结构
    bool own_fields = false;    // 是否负责释放 storage
    Index* storage = nullptr;   // 向量存储（多态）
};
```

**关键交互点：**

| 操作 | 调用路径 | 说明 |
|------|----------|------|
| 添加向量 | `IndexHNSW::add()` → `storage->add(n, x)` | 先存向量，再构建 HNSW 图 |
| 距离计算 | `storage->get_distance_computer()` | 返回 `DistanceComputer`，HNSW 搜索/构建时通过 `operator()(idx_t i)` 计算距离 |
| 重建向量 | `storage->reconstruct(key, recons)` | 根据 storage 内部下标恢复原始向量 |
| 重置 | `storage->reset()` | 清空向量数据 |

### 3.3 DistanceComputer 工作原理

`IndexFlat` 返回的 `FlatCodesDistanceComputer` 的核心逻辑：

```cpp
struct FlatCodesDistanceComputer : DistanceComputer {
    const uint8_t* codes;   // 指向连续向量数据
    size_t code_size;       // 每个向量的字节大小

    float operator()(idx_t i) override {
        return distance_to_code(codes + i * code_size);  // 直接下标计算
    }

    void prefetch(idx_t i) override {
        prefetch_L2(codes + i * code_size);              // 直接下标预取
    }
};
```

**关键洞察**：HNSW 搜索时传给 `DistanceComputer::operator()` 的 `idx_t i` 就是 storage 内部的连续下标。我们需要在这个环节插入一层 ID 映射。

### 3.4 当前重建索引的内存问题

```
[Index1]                           [Index2 - 重建]
┌──────────────┐                   ┌──────────────┐
│  HNSW Graph  │                   │  HNSW Graph  │
│  (links)     │                   │  (links)     │
├──────────────┤                   ├──────────────┤
│  Storage     │  reconstruct()    │  Storage     │
│  (IndexFlat) │  ──────────────>  │  (IndexFlat) │
│  128GB 向量  │     复制数据       │  128GB 向量  │ ← 重复！
└──────────────┘                   └──────────────┘
                总内存 = 256GB + 2 × 图结构
```

---

## 4. 设计方案

### 4.1 核心思想

引入一个新的 storage 类型 `IndexIndirectFlat`，它不直接拥有向量数据，而是：

1. 持有一个指向 **共享向量存储** (`SharedVectorStore`) 的引用
2. 维护一个 **本地 ID → 共享存储 ID** 的映射表 (`id_map`)
3. 实现完整的 `Index` 接口，可直接替代 `IndexFlat` 作为 `IndexHNSW` 的 storage

```
[Index1]                           [Index2 - 重建]
┌──────────────┐                   ┌──────────────┐
│  HNSW Graph  │                   │  HNSW Graph  │
│  (links)     │                   │  (links)     │
├──────────────┤                   ├──────────────┤
│ IndexIndirect│                   │ IndexIndirect│
│ Flat         │                   │ Flat         │
│ id_map[0→0]  │                   │ id_map[0→2]  │
│ id_map[1→1]  │                   │ id_map[1→5]  │
│ id_map[2→2]  │                   │ id_map[2→7]  │
│ ...          │                   │ ...          │
└──────┬───────┘                   └──────┬───────┘
       │                                  │
       └──────────┐    ┌──────────────────┘
                  ▼    ▼
          ┌──────────────────┐
          │ SharedVectorStore│
          │ (引用计数管理)    │
          │                  │
          │ 128GB 向量数据   │
          │ codes[]          │
          └──────────────────┘
   总内存 = 128GB + 2 × (图结构 + id_map)
```

### 4.2 方案对比

| 方案 | 额外内存 | 侵入性 | 搜索性能影响 | 实现复杂度 |
|------|----------|--------|------------|-----------|
| A. 完整复制 (现状) | +128GB (100%) | 无 | 无 | 低 |
| B. mmap 共享文件 | ~0 | 需要文件系统支持 | 有 (page fault) | 中 |
| **C. IndexIndirectFlat (本方案)** | **~N×8B (<1%)** | **无 (仅新增类)** | **极小 (<5%)** | **中** |
| D. Copy-on-Write 虚拟内存 | ~0 | 需要 OS 支持 | 不可控 | 高 |

**选择方案 C**：在不修改 `IndexHNSW` 的前提下，仅新增 storage 类型，内存开销极低，实现可控。

---

## 5. 详细设计

### 5.1 SharedVectorStore — 共享向量存储

```cpp
#include <atomic>
#include <cstdint>
#include <vector>
#include <faiss/Index.h>
#include <faiss/impl/maybe_owned_vector.h>

namespace faiss {

/// 共享向量存储，提供对底层连续向量数据的只读访问。
/// 通过引用计数管理生命周期，多个 IndexIndirectFlat 可以共享同一个实例。
struct SharedVectorStore {
    /// 向量维度
    size_t d;

    /// 每个向量的字节大小 (d * sizeof(float) for flat storage)
    size_t code_size;

    /// 向量总数
    size_t ntotal;

    /// 底层向量数据（连续存储）
    /// 可以拥有数据，也可以引用外部数据 (MaybeOwnedVector)
    MaybeOwnedVector<uint8_t> codes;

    /// 引用计数
    std::atomic<int> ref_count{0};

    SharedVectorStore(size_t d, size_t code_size);

    /// 从现有 IndexFlatCodes 中创建，选择是否拥有数据
    /// @param source      源 IndexFlatCodes
    /// @param own_data    true = 深拷贝数据, false = 仅引用（source 需保持存活）
    static SharedVectorStore* from_flat_codes(
        const IndexFlatCodes* source,
        bool own_data = false);

    /// 获取第 i 个向量的 const 指针
    const uint8_t* get_code(idx_t i) const {
        return codes.data() + i * code_size;
    }

    /// 获取第 i 个向量的 float* 指针 (flat 格式)
    const float* get_vec(idx_t i) const {
        return reinterpret_cast<const float*>(get_code(i));
    }

    void add_ref();
    void release();
};

} // namespace faiss
```

**设计要点：**
- `MaybeOwnedVector` 支持两种模式：拥有数据 (复制) 或仅引用外部数据 (零拷贝)
- 引用计数确保最后一个使用者销毁时才释放数据
- `from_flat_codes()` 工厂方法支持从现有 `IndexFlatCodes` 创建，`own_data=false` 时零拷贝

### 5.2 IndexIndirectFlat — 间接映射 Storage

```cpp
#include <faiss/IndexFlatCodes.h>
#include <faiss/SharedVectorStore.h>

namespace faiss {

/// 间接映射 storage：通过 id_map 将本地连续索引映射到共享存储中的实际位置。
/// 可作为 IndexHNSW 的 storage 使用，替代 IndexFlat。
///
/// 内部不存储任何向量数据，仅维护一个映射表和指向共享存储的引用。
struct IndexIndirectFlat : Index {

    /// 本地索引 i → 共享存储中的真实 ID
    /// id_map[local_idx] = real_storage_idx
    std::vector<idx_t> id_map;

    /// 共享的底层向量存储（非拥有，通过引用计数管理）
    SharedVectorStore* shared_store = nullptr;

    /// 向量维度推导的 code_size
    size_t code_size;

    /// 构造函数
    /// @param d              向量维度
    /// @param shared_store   共享向量存储
    /// @param metric         度量类型
    IndexIndirectFlat(
        idx_t d,
        SharedVectorStore* shared_store,
        MetricType metric = METRIC_L2);

    ~IndexIndirectFlat() override;

    // ============================================================
    // Index 接口实现
    // ============================================================

    /// 添加向量：将 real_ids 列表中的 ID 加入映射表
    /// 实际数据不会被复制，仅记录映射关系
    /// @param n         要添加的向量数
    /// @param x         忽略（数据已在 shared_store 中），但需保留接口兼容性
    void add(idx_t n, const float* x) override;

    /// 批量添加：指定 real_ids 列表
    void add_with_real_ids(idx_t n, const idx_t* real_ids);

    /// 根据本地 ID 重建向量
    void reconstruct(idx_t key, float* recons) const override;

    /// 批量重建
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    /// 返回支持间接映射的 DistanceComputer
    DistanceComputer* get_distance_computer() const override;

    /// 重置映射表
    void reset() override;

    /// 当前映射的向量数
    /// (ntotal 继承自 Index，在 add 时更新)

    // ============================================================
    // 搜索接口（可选实现，IndexHNSW 不直接使用）
    // ============================================================
    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params = nullptr) const override;

private:
    /// 将本地 ID 转换为共享存储中的真实 ID
    idx_t get_real_id(idx_t local_id) const {
        return id_map[local_id];
    }
};

} // namespace faiss
```

### 5.3 IndirectFlatDistanceComputer — 间接映射距离计算器

这是性能关键路径，HNSW 搜索和构建时每次距离计算都会调用。

```cpp
namespace faiss {

/// 支持间接映射的 DistanceComputer
/// HNSW 传入的 idx_t i 是 IndexIndirectFlat 的本地 ID，
/// 需要通过 id_map 转换为 shared_store 中的真实位置后再计算距离。
struct IndirectFlatDistanceComputer : DistanceComputer {

    /// 本地 ID → 真实 ID 的映射表（只读引用）
    const idx_t* id_map;

    /// 底层真正执行距离计算的 DistanceComputer（来自 shared_store 对应的 IndexFlat）
    /// 拥有此指针的所有权
    FlatCodesDistanceComputer* base_dis;

    IndirectFlatDistanceComputer(
        const idx_t* id_map,
        FlatCodesDistanceComputer* base_dis);

    ~IndirectFlatDistanceComputer() override;

    void set_query(const float* x) override {
        base_dis->set_query(x);
    }

    /// 核心：将本地 ID 映射到真实 ID 后计算距离
    float operator()(idx_t i) override {
        idx_t real_id = id_map[i];
        return (*base_dis)(real_id);
    }

    /// 批量距离计算（4路）
    void distances_batch_4(
            const idx_t idx0, const idx_t idx1,
            const idx_t idx2, const idx_t idx3,
            float& dis0, float& dis1,
            float& dis2, float& dis3) override {
        base_dis->distances_batch_4(
            id_map[idx0], id_map[idx1],
            id_map[idx2], id_map[idx3],
            dis0, dis1, dis2, dis3);
    }

    /// 批量距离计算（8路）
    void distances_batch_8(
            const idx_t idx0, const idx_t idx1,
            const idx_t idx2, const idx_t idx3,
            const idx_t idx4, const idx_t idx5,
            const idx_t idx6, const idx_t idx7,
            float& dis0, float& dis1,
            float& dis2, float& dis3,
            float& dis4, float& dis5,
            float& dis6, float& dis7) override {
        base_dis->distances_batch_8(
            id_map[idx0], id_map[idx1],
            id_map[idx2], id_map[idx3],
            id_map[idx4], id_map[idx5],
            id_map[idx6], id_map[idx7],
            dis0, dis1, dis2, dis3,
            dis4, dis5, dis6, dis7);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return base_dis->symmetric_dis(id_map[i], id_map[j]);
    }

    void prefetch(idx_t i) override {
        base_dis->prefetch(id_map[i]);
    }

    void prefetch_batch_4(
            idx_t i0, idx_t i1, idx_t i2, idx_t i3,
            int level) override {
        base_dis->prefetch_batch_4(
            id_map[i0], id_map[i1],
            id_map[i2], id_map[i3], level);
    }
};

} // namespace faiss
```

**性能分析**：
- 映射查表 `id_map[i]` 是一次内存读取，id_map 数组通常完全驻留 L2/L3 cache
- 对于 128 维 float32 向量 (512 bytes/vec)，映射开销 (8 bytes 查表) 占比约 1.5%
- batch_4 / batch_8 路径可以让 CPU 利用 ILP 并行化映射查表

### 5.4 IndexHNSWSharedFlat — 便捷的顶层封装

```cpp
namespace faiss {

/// 使用共享存储的 HNSW 索引。
/// 等价于 IndexHNSWFlat，但底层 storage 改为 IndexIndirectFlat。
struct IndexHNSWSharedFlat : IndexHNSW {

    IndexHNSWSharedFlat() = default;

    /// 创建一个使用共享存储的 HNSW 索引
    /// @param d              向量维度
    /// @param M              HNSW 的 M 参数
    /// @param shared_store   共享向量存储
    /// @param metric         度量类型
    IndexHNSWSharedFlat(
        int d,
        int M,
        SharedVectorStore* shared_store,
        MetricType metric = METRIC_L2);
};

} // namespace faiss
```

### 5.5 整体架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         用户应用层                                    │
│                                                                      │
│    ┌──────────────────┐              ┌──────────────────────┐        │
│    │ index1 (旧索引)   │              │ index2 (新建索引)     │        │
│    │ IndexHNSW        │              │ IndexHNSWSharedFlat  │        │
│    │  ├ hnsw (graph)  │              │  ├ hnsw (graph)      │        │
│    │  └ storage ──────┤              │  └ storage ──────────┤        │
│    └──────────────────┘              └──────────────────────┘        │
│              │                                  │                    │
│              ▼                                  ▼                    │
│    ┌──────────────────┐              ┌──────────────────────┐        │
│    │ IndexFlatL2      │              │ IndexIndirectFlat    │        │
│    │ (原始 storage)   │              │  ├ id_map[]          │        │
│    │  └ codes[]  ─────┼──┐           │  └ shared_store ──┐ │        │
│    └──────────────────┘  │           └───────────────────│──┘        │
│                          │                               │           │
│                          ▼                               ▼           │
│              ┌───────────────────────────────────────────┐           │
│              │         SharedVectorStore                 │           │
│              │  ├ codes[] ─── (引用 IndexFlatL2::codes)  │           │
│              │  ├ d, code_size, ntotal                   │           │
│              │  └ ref_count = 2                          │           │
│              └───────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.6 核心流程

#### 5.6.1 创建共享存储并构建新索引

```cpp
// 假设 index1 已经运行一段时间，storage 是 IndexFlatL2
auto* index1 = dynamic_cast<IndexHNSW*>(existing_index);
auto* flat_storage = dynamic_cast<IndexFlatCodes*>(index1->storage);

// 1. 从现有 storage 创建共享存储（零拷贝引用）
auto* shared = SharedVectorStore::from_flat_codes(flat_storage, /*own_data=*/false);

// 2. 收集有效（未被删除的）向量 ID 列表
std::vector<idx_t> valid_ids = get_valid_vector_ids(index1);

// 3. 创建新的 HNSW 索引，使用共享存储
auto* index2 = new IndexHNSWSharedFlat(d, M, shared, METRIC_L2);

// 4. 向新 storage 注册有效向量的映射
auto* indirect = dynamic_cast<IndexIndirectFlat*>(index2->storage);
indirect->add_with_real_ids(valid_ids.size(), valid_ids.data());

// 5. 构建 HNSW 图（不需要传入实际向量数据，因为数据已在 shared_store 中）
//    调用内部构建流程...
```

#### 5.6.2 搜索流程（无修改）

```
用户调用 index2->search(query, k, distances, labels)
  │
  ├── IndexHNSW::search()
  │     │
  │     ├── storage->get_distance_computer()
  │     │     └── 返回 IndirectFlatDistanceComputer
  │     │
  │     └── HNSW::search() 使用 DistanceComputer
  │           │
  │           ├── qdis(candidate_id)    // candidate_id 是本地 ID
  │           │     │
  │           │     └── IndirectFlatDistanceComputer::operator()(candidate_id)
  │           │           │
  │           │           ├── real_id = id_map[candidate_id]
  │           │           └── base_dis->operator()(real_id)  // 从共享存储计算真实距离
  │           │
  │           └── 返回 top-k 结果
  │
  └── 返回给用户
```

#### 5.6.3 构建流程

```
hnsw_add_vertices() 为每个新节点构建 HNSW 链接
  │
  ├── storage_distance_computer(index2->storage)
  │     └── 返回 IndirectFlatDistanceComputer
  │
  ├── 对每个新节点 pt_id (本地 ID):
  │     ├── ptdis.set_query(x + pt_id * d)  // 注意：这里 x 需要是实际向量
  │     │   或者通过 shared_store 获取
  │     │
  │     └── HNSW::add_with_locks() 使用 ptdis 计算与邻居的距离
  │           └── ptdis(neighbor_id)  →  id_map 转换 → 共享存储计算
  │
  └── 图构建完成
```

### 5.7 `add()` 流程的特殊处理

标准 `IndexHNSW::add()` 会调用 `storage->add(n, x)` 把向量数据加入 storage。但在我们的场景中，数据已经在 shared_store 里了。有两种处理方式：

**方案 A: 重写 `IndexHNSWSharedFlat::add()`**

```cpp
void IndexHNSWSharedFlat::add(idx_t n, const float* x) {
    // 不调用 storage->add()（数据已在 shared_store 中）
    // 只需更新 ntotal 并构建 HNSW 图
    int n0 = ntotal;
    ntotal += n;
    hnsw_add_vertices(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);
}
```

**方案 B: 在 `IndexIndirectFlat::add()` 中做兼容处理**

```cpp
void IndexIndirectFlat::add(idx_t n, const float* x) {
    // x 参数在此被忽略，实际数据从 shared_store 获取
    // 调用者需要事先通过 add_with_real_ids() 设置好映射
    // 这里只更新 ntotal
    ntotal += n;
}
```

**推荐方案 A**：更清晰，职责分离明确。

### 5.8 构建时获取向量数据

`hnsw_add_vertices()` 需要原始向量数据 `const float* x`。对于共享存储场景，我们需要从 shared_store 中获取：

```cpp
// 在 IndexHNSWSharedFlat::add() 中
void IndexHNSWSharedFlat::add_from_shared(const idx_t* real_ids, idx_t n) {
    auto* indirect = dynamic_cast<IndexIndirectFlat*>(storage);
    indirect->add_with_real_ids(n, real_ids);

    int n0 = ntotal;
    ntotal += n;

    // 收集实际向量数据用于 HNSW 构建
    std::vector<float> x(n * d);
    for (idx_t i = 0; i < n; i++) {
        memcpy(x.data() + i * d,
               indirect->shared_store->get_vec(real_ids[i]),
               d * sizeof(float));
    }

    hnsw_add_vertices(*this, n0, n, x.data(), verbose,
                      hnsw.levels.size() == ntotal);
}
```

> **优化说明**：`x` 只在构建期间临时使用，可以分批处理避免一次性分配过大内存。

---

## 6. 实现计划

### 6.1 文件组织

```
faiss/faiss/
  ├── SharedVectorStore.h          # 新增：共享向量存储
  ├── SharedVectorStore.cpp        # 新增
  ├── IndexIndirectFlat.h          # 新增：间接映射 storage
  ├── IndexIndirectFlat.cpp        # 新增
  ├── IndexHNSW.h                  # 修改：新增 IndexHNSWSharedFlat 声明
  ├── IndexHNSW.cpp                # 修改：新增 IndexHNSWSharedFlat 实现
  └── impl/
       └── DistanceComputer.h      # 不修改（IndirectFlatDistanceComputer 放在 IndexIndirectFlat.cpp 中）
```

### 6.2 实现阶段

| 阶段 | 内容 | 预计工作量 | 交付物 |
|------|------|-----------|--------|
| **Phase 1** | `SharedVectorStore` 基础实现 + 引用计数 | 1 天 | 可从 IndexFlatCodes 创建共享存储 |
| **Phase 2** | `IndexIndirectFlat` + `IndirectFlatDistanceComputer` | 2 天 | 可作为 IndexHNSW 的 storage 使用 |
| **Phase 3** | `IndexHNSWSharedFlat` + `add_from_shared()` | 1 天 | 端到端构建新索引 |
| **Phase 4** | 单元测试 + 集成测试 + 性能基准测试 | 2 天 | 测试通过，性能达标 |
| **Phase 5** | CMake 集成 + 文档 | 0.5 天 | 可构建、可发布 |

### 6.3 CMake 修改

在 `faiss/faiss/CMakeLists.txt` 中添加新文件：

```cmake
set(FAISS_SRC
    # ... existing sources ...
    SharedVectorStore.cpp
    IndexIndirectFlat.cpp
)

set(FAISS_HEADERS
    # ... existing headers ...
    SharedVectorStore.h
    IndexIndirectFlat.h
)
```

---

## 7. 测试方案

### 7.1 单元测试

| 测试用例 | 验证内容 |
|----------|---------|
| `test_shared_store_from_flat` | 从 IndexFlatL2 创建 SharedVectorStore，验证向量数据正确性 |
| `test_shared_store_ref_count` | 引用计数增减、最后释放时的正确行为 |
| `test_indirect_flat_add_with_real_ids` | 添加映射后 ntotal 正确，reconstruct 正确 |
| `test_indirect_flat_reconstruct` | 多个本地 ID 重建向量与原始向量一致 |
| `test_indirect_distance_computer` | 通过间接映射计算距离与直接计算结果一致 |
| `test_indirect_distance_batch` | batch_4 / batch_8 距离计算结果正确 |
| `test_indirect_flat_as_hnsw_storage` | 将 IndexIndirectFlat 作为 IndexHNSW 的 storage，执行搜索 |

### 7.2 集成测试

| 测试场景 | 验证内容 |
|----------|---------|
| `test_dual_index_shared_storage` | 两份索引共享存储，各自搜索结果正确 |
| `test_rebuild_after_deletion` | 模拟删除场景，用有效 ID 构建新索引，搜索质量恢复 |
| `test_concurrent_search_and_build` | index1 搜索与 index2 构建并行执行，无数据竞争 |
| `test_index_destruction_order` | 先销毁 index1 或 index2，共享存储正确管理 |

### 7.3 性能基准测试

使用 `sift-128-euclidean` 数据集：

| 指标 | 基线 (IndexHNSWFlat) | 目标 (IndexHNSWSharedFlat) |
|------|---------------------|---------------------------|
| 搜索延迟 (QPS) | X | ≥ 0.95X |
| 索引构建时间 | Y | ≤ 1.05Y |
| 内存占用 (双索引) | 2 × Z | ≤ 1.05 × Z |
| Recall@10 | R | = R (精确一致) |

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 间接映射导致 cache miss 增加，搜索性能下降 | 中 | 低 | id_map 连续存储，热路径中保持在 L2 cache；batch prefetch |
| SharedVectorStore 引用计数在多线程下出错 | 低 | 高 | 使用 `std::atomic<int>` + RAII guard |
| 构建期间 shared_store 底层数据被意外修改 | 低 | 高 | 文档约束 + 运行时断言检查 ntotal 不变 |
| `hnsw_add_vertices` 需要 `const float* x` 参数 | 确定 | 中 | 从 shared_store 临时拼接向量数据，分批处理控制峰值内存 |
| FAISS 序列化框架不支持共享引用 | 高 | 中 | Phase 1 不支持序列化，后续迭代增加；保存时做深拷贝 |

---

## 9. 未来扩展

### 9.1 支持 SQ/PQ 量化存储

当前设计以 `IndexFlat` 为基础，未来可扩展支持 `IndexScalarQuantizer` 和 `IndexPQ` 作为共享存储的底层格式，进一步降低内存占用。

### 9.2 支持增量更新

允许在 shared_store 中追加新向量（append-only），同时更新两份索引的映射表。需要引入读写锁保护。

### 9.3 支持序列化

设计自定义的序列化格式：
- 保存时：共享存储只保存一份，每份索引保存各自的 id_map
- 加载时：先加载共享存储，再加载各索引的 id_map 并重新建立引用

### 9.4 多索引共享

支持超过两份索引共享同一个 SharedVectorStore，满足 A/B 测试或多策略索引场景。

### 9.5 Copy-on-Write 语义

当某个索引需要修改共享数据时，自动创建副本，其他索引不受影响。

---

## 附录 A: 关键 FAISS 源码引用

| 文件 | 关键内容 |
|------|---------|
| `faiss/IndexHNSW.h:37-38` | `Index* storage` 定义 |
| `faiss/IndexHNSW.cpp:344-355` | `IndexHNSW::add()` — storage->add 后构建图 |
| `faiss/IndexHNSW.cpp:362-364` | `IndexHNSW::reconstruct()` — 委托给 storage |
| `faiss/IndexHNSW.cpp:639-641` | `get_distance_computer()` — 委托给 storage |
| `faiss/impl/DistanceComputer.h:26-101` | `DistanceComputer` 接口定义 |
| `faiss/impl/DistanceComputer.h:210-280` | `FlatCodesDistanceComputer` — 基于连续存储的距离计算 |
| `faiss/IndexFlatCodes.h:22-55` | `IndexFlatCodes` — codes 存储 + 接口 |

## 附录 B: 内存开销估算

假设 N = 1,000,000 向量，d = 128 维 float32:

| 组件 | 大小 |
|------|------|
| 原始向量数据 | N × d × 4 = 512 MB |
| HNSW 图 (M=32) | N × 2M × 4 ≈ 256 MB |
| id_map (每份索引) | N × 8 = 8 MB |
| SharedVectorStore 开销 | < 1 KB |
| **双索引总计** | 512 + 256×2 + 8×2 ≈ **1040 MB** |
| **对比完整复制** | 512×2 + 256×2 ≈ **1536 MB** |
| **节省** | **496 MB (32%)** |

> 当向量维度更高 (d=960) 时，向量数据占比更大，节省比例可达 **~48%**。


