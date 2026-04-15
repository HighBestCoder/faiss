# HNSW 共享向量存储与双索引重建 — 详细设计 (v1)

## 1. 问题陈述

用户持有一个 IndexHNSWFlat 索引（例如 128GB 向量数据）。经过大量删除操作后，HNSW 图结构退化，搜索性能下降。用户需要 **重建索引**（构建第二份 HNSW 图）以恢复性能。

传统做法：`reconstruct` 所有存活向量 → 写入新 storage → 内存翻倍（128GB → 256GB）。对于大规模场景不可接受。

**约束条件：**
- 重建期间，旧索引继续提供搜索服务（read-only）
- 重建期间不会发生 write/delete/update 操作
- 系统会提供一个 **bitmap 形式的 `deleted_id_list`**，标记哪些向量已被删除
- 内存增量应尽可能小（只有图结构 + 映射表的开销，而非完整向量副本）

## 2. 核心思想

在两个索引之间 **共享同一份向量数据（store）**，每个索引仅维护各自的 HNSW 图结构和一个轻量级的 **ID 映射表（`storage_id_map`）**。

```
                    ┌──────────────────────────┐
                    │    SharedVectorStore      │
                    │                          │
                    │  codes: uint8_t[]        │  ← 唯一一份向量数据
                    │  free_list               │  ← 可复用 slot 队列
                    │  ntotal_store            │  ← 已分配 slot 总数
                    │  code_size, d            │
                    └───────┬──────┬───────────┘
                            │      │
               shared_ptr   │      │  shared_ptr
                    ┌───────┘      └───────┐
                    ▼                      ▼
          ┌─────────────────┐    ┌─────────────────┐
          │  当前索引         │    │  新索引           │
          │  (IndexHNSW)     │    │  (IndexHNSW)     │
          │                  │    │                  │
          │  storage:        │    │  storage:        │
          │  IndexFlatShared │    │  IndexFlatShared │
          │                  │    │                  │
          │  storage_id_map: │    │  storage_id_map: │
          │  [0,2,4,5,7,...] │    │  [0,2,5,7,...]   │
          │  (当前存活向量)    │    │  (跳过新一轮删除)  │
          │                  │    │                  │
          │  deleted_bitmap: │    │  deleted_bitmap: │
          │  [0,0,0,1,0,...] │    │  [全 0]           │
          │  (本索引的删除)    │    │  (新索引无删除)    │
          │                  │    │                  │
          │  HNSW graph      │    │  HNSW graph      │
          │  (当前图)         │    │  (新建图)         │
          └─────────────────┘    └─────────────────┘
```

**关键洞察：**
- 每次 rebuild 时，两个索引各自持有一份 `storage_id_map`。当前索引的映射包含当前所有存活向量（含被标记删除但 HNSW 图仍引用的），新索引的映射跳过当前索引 `deleted_bitmap` 标记的已删除向量
- `deleted_bitmap` 属于 IndexFlatShared（每个索引独立维护），而非 SharedVectorStore。这样不同索引可以有各自独立的删除状态
- 两者读同一份 `codes`，rebuild 期间无并发写冲突（旧索引 read-only）
- rebuild + swap + 旧索引销毁后，通过 `reclaim_deleted_slots(old_bitmap)` 回收空洞，新向量可复用这些 slot

## 3. 数据结构设计

### 3.1 SharedVectorStore

```cpp
namespace faiss {

/// 共享向量数据的持有者。通过 shared_ptr 被多个 IndexFlatShared 引用。
/// 继承 MaybeOwnedVectorOwner 以复用 faiss 已有的 view/ownership 语义。
///
/// SharedVectorStore 的 slot 有两种状态：
///   1. 已分配 (allocated)  — 存有向量数据，可能被某个索引使用，也可能已被索引标记删除
///   2. 可用 (available)    — 在 free_list 中，可被 allocate_slot() 分配给新向量
///
/// 注意：删除状态（deleted）由 IndexFlatShared 通过自身的 deleted_bitmap 管理，
/// SharedVectorStore 不感知哪些 slot 被逻辑删除。只有通过 reclaim_deleted_slots()
/// 将 bitmap 中标记的 slot 回收到 free_list 后，slot 才从 "allocated" 变为 "available"。
///
/// 状态转换:
///   allocated → available   : reclaim_deleted_slots(bitmap) — rebuild + swap 后调用
///   available → allocated   : allocate_slot() — 添加新向量时调用
///   (append)  → allocated   : allocate_slot() 在 free_list 为空时 append 到末尾
struct SharedVectorStore : MaybeOwnedVectorOwner {
    /// 向量数据（owned）。size = capacity * code_size
    /// capacity >= ntotal_store，预留空间以减少 realloc
    std::vector<uint8_t> codes;

    /// 可复用 slot 队列。reclaim_deleted_slots() 从外部 bitmap 扫描填充。
    /// allocate_slot() 优先从此处分配。
    /// 使用栈语义（LIFO），无需排序。
    std::vector<idx_t> free_list;

    size_t ntotal_store;  ///< 已分配的 slot 总数（allocated + available via free_list 内的 slot 仍计入）
    size_t code_size;     ///< 每个向量的字节数 = sizeof(float) * d
    size_t d;             ///< 向量维度

    SharedVectorStore(size_t d, size_t code_size);

    // ---- Slot 生命周期操作 ----

    /// 将外部 bitmap 中标记为已删除的 slot 回收到 free_list。
    /// **前置条件**: 调用时只有一个 IndexFlatShared 引用此 store（旧索引已销毁）。
    /// 回收后：
    ///   - free_list 被这些 slot 填充（旧 free_list 被替换，不是追加）
    ///   - slot 中的向量数据变为无效（会被下次 allocate_slot 覆盖）
    ///   - 调用者应清空自身的 bitmap（bitmap 属于 IndexFlatShared）
    ///
    /// @param deleted_bitmap  旧索引的删除位图（由调用者提供）
    /// @param ntotal          bitmap 覆盖的 slot 范围
    void reclaim_deleted_slots(
            const std::vector<uint64_t>& deleted_bitmap,
            size_t ntotal) {
        free_list.clear();
        for (size_t i = 0; i < ntotal; i++) {
            if ((deleted_bitmap[i >> 6] >> (i & 63)) & 1) {
                free_list.push_back(i);
            }
        }
    }

    /// 分配一个 slot 并写入向量数据。
    /// 优先从 free_list 复用空洞；free_list 为空时 append 到末尾。
    /// @return 分配到的 store 位置
    idx_t allocate_slot(const float* vec) {
        idx_t slot;
        if (!free_list.empty()) {
            slot = free_list.back();
            free_list.pop_back();
            // 写入向量数据到已有位置（覆盖旧数据）
            memcpy(codes.data() + slot * code_size, vec, code_size);
        } else {
            // append 到末尾
            slot = ntotal_store;
            ntotal_store++;
            // 扩展 codes
            codes.resize(ntotal_store * code_size);
            memcpy(codes.data() + slot * code_size, vec, code_size);
        }
        return slot;
    }

    // ---- 数据访问 ----

    /// 获取位置 i 的向量指针
    const uint8_t* get_code(idx_t i) const {
        return codes.data() + i * code_size;
    }

    const float* get_vector(idx_t i) const {
        return reinterpret_cast<const float*>(get_code(i));
    }
};

} // namespace faiss
```

**设计决策：**

| 决策 | 选择 | 理由 |
|------|------|------|
| 继承关系 | `MaybeOwnedVectorOwner` | 复用 faiss 已有的 `MaybeOwnedVector::create_view` + `shared_ptr<Owner>` 生命周期管理 |
| codes 存储 | `std::vector<uint8_t>` | 与 `IndexFlatCodes::codes` 类型一致，可直接创建 view |
| free_list | `vector<idx_t>` LIFO 栈 | 简单高效；无需排序；LIFO 倾向复用低地址 slot，可能略有 cache 优势 |
| deleted_bitmap 不在 store | 放在 IndexFlatShared | 每个索引有自己的删除状态，store 是纯数据容器，不感知删除逻辑 |
| reclaim 接受外部 bitmap | 参数传入 | store 不持有 bitmap，由调用者（swap 后的新索引）传入旧索引的 bitmap |

### 3.2 IndexFlatShared

```cpp
namespace faiss {

/// 基于共享存储的 flat index。作为 IndexHNSW 的 storage 使用。
///
/// 与 IndexFlat 的区别：
///   - 不拥有向量数据，通过 shared_ptr<SharedVectorStore> 共享
///   - 通过 storage_id_map 将本地连续 ID [0, ntotal) 映射到 store 中的实际位置
///   - 支持 add()：通过 store->allocate_slot() 分配 slot（可复用空洞）
///   - 每个索引维护自己的 deleted_bitmap，记录哪些 store slot 被本索引标记删除
struct IndexFlatShared : IndexFlatCodes {
    /// 共享向量存储的引用
    std::shared_ptr<SharedVectorStore> store;

    /// 本地 ID → store 位置的映射。
    /// local_id ∈ [0, ntotal)  →  store_id ∈ [0, store->ntotal_store)
    /// 始终使用显式映射。首次构建时映射为 [0, 1, 2, ..., n-1]。
    /// compact_store() 后映射重新变为恒等，is_identity_map = true。
    std::vector<idx_t> storage_id_map;

    /// is_identity_map 为 true 时，resolve(i) 返回 i，跳过 id_map 访问。
    /// 首次构建和 compact 后为 true。add() 可能使其变为 false（见 §10.6）。
    bool is_identity_map = true;

    /// 删除位图。bit i = 1 表示 store 位置 i 的向量已被本索引标记删除。
    /// 每个索引有自己独立的 bitmap，不与其他索引共享。
    /// 使用 uint64_t 数组实现，支持 SIMD 加速的 popcount。
    /// size = ceil(store->ntotal_store / 64)
    std::vector<uint64_t> deleted_bitmap;

    IndexFlatShared() = default;

    /// 从已有的 SharedVectorStore 创建
    IndexFlatShared(
        std::shared_ptr<SharedVectorStore> store,
        MetricType metric = METRIC_L2);

    /// 从已有的 SharedVectorStore + 旧索引的 storage_id_map 创建（rebuild 时新索引使用）
    /// 遍历旧索引的 storage_id_map，跳过旧索引 bitmap 标记的已删除 slot
    IndexFlatShared(
        std::shared_ptr<SharedVectorStore> store,
        const std::vector<idx_t>& old_storage_id_map,
        const std::vector<uint64_t>& old_deleted_bitmap,
        MetricType metric = METRIC_L2);

    /// 解析本地 ID 到 store 中的实际位置
    idx_t resolve_id(idx_t local_id) const {
        return storage_id_map[local_id];
    }

    // ---- 删除操作 ----

    /// O(1) 判断 store 位置 i 是否被本索引标记为已删除
    bool is_deleted(idx_t store_slot) const {
        return (deleted_bitmap[store_slot >> 6] >> (store_slot & 63)) & 1;
    }

    /// 标记本地 ID 对应的 store slot 为已删除。
    /// 仅设置 bitmap，不影响 store 的 free_list。
    /// HNSW 图中仍有边指向此 slot，搜索时需要读取向量数据计算距离（然后过滤）。
    void mark_deleted(idx_t local_id) {
        idx_t store_slot = storage_id_map[local_id];
        deleted_bitmap[store_slot >> 6] |= (uint64_t(1) << (store_slot & 63));
    }

    /// 获取存活向量数量
    size_t count_alive() const;

    // ---- Index interface overrides ----

    /// 重建单个向量：通过映射找到 store 位置，拷贝数据
    void reconstruct(idx_t key, float* recons) const override;

    /// 添加向量：通过 store->allocate_slot() 分配 slot（优先复用空洞），
    /// 然后将 slot 追加到 storage_id_map。
    ///
    /// 注意：此方法由 IndexHNSW::add() 在写入 storage 时调用。
    /// IndexHNSW::add() 随后会在 HNSW 图中为新节点构建边。
    /// 调用者（IndexHNSW）负责持有写锁。
    void add(idx_t n, const float* x) override {
        for (idx_t i = 0; i < n; i++) {
            idx_t slot = store->allocate_slot(
                reinterpret_cast<const float*>(x + i * code_size));
            storage_id_map.push_back(slot);
            // 确保 bitmap 大小足够覆盖新 slot
            size_t needed_words = (slot / 64) + 1;
            if (deleted_bitmap.size() < needed_words) {
                deleted_bitmap.resize(needed_words, 0);
            }
        }
        ntotal += n;
    }

    /// 返回间接寻址的 DistanceComputer
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /// permute_entries: is_identity_map = true 时物理搬移 store 数据（cycle-following），
    /// is_identity_map = false 时只置换 storage_id_map 间接层。详见 §10.4。
    void permute_entries(const idx_t* perm);

    void reset() override;
};

} // namespace faiss
```

**关键设计点：**

1. **继承 `IndexFlatCodes`**：复用 `code_size`、`ntotal`、`get_distance_computer() → get_FlatCodesDistanceComputer()` 调用链。`codes` 字段通过 `MaybeOwnedVector::create_view` 指向 `store->codes`，不额外分配内存。

2. **始终使用显式 `storage_id_map`**：首次构建时映射为 `[0, 1, 2, ..., n-1]`（所有 slot 都被使用）。rebuild 后映射跳过已删除 slot。设计上统一，无特殊分支。

3. **`deleted_bitmap` 属于索引而非 store**：每个 IndexFlatShared 维护自己的删除位图（按 store slot 索引）。这样不同索引可以有各自独立的删除状态，store 保持为纯数据容器。rebuild 时，新索引的 bitmap 初始为全 0（无删除），旧索引的 bitmap 用于构建新 `storage_id_map`。

4. **`add()` 支持空洞复用**：通过 `store->allocate_slot()` 分配 slot（优先从 free_list 取，否则 append），然后追加到 `storage_id_map`。新向量的 `local_id = ntotal`（递增），`store_slot` 可能是任意位置。HNSW 图只看到连续的 `local_id`，不感知 store 中的空洞。

4. **删除不在 storage 层处理**：HNSW 的删除是在图层面标记节点为 deleted（通过 `IDSelector`）。`storage_id_map[deleted_local_id]` 仍指向有效 store slot，搜索时 HNSW 图遍历可能访问已删除节点来计算距离（然后过滤），所以 slot 数据必须保持有效。只有 rebuild + swap + delete 旧索引后，才能通过 `reclaim_deleted_slots()` 安全回收。

5. **`permute_entries` 双模式**：图重排序（RCM/Weighted 等）调用 `permute_entries` 时，若 `is_identity_map = false`，只置换 `storage_id_map`（8 字节/条目）；若 `is_identity_map = true`（compact 后），则通过 cycle-following 物理搬移 store 中的向量数据，保持恒等映射不被破坏。详见 §10.4。

### 3.3 构建 storage_id_map

```cpp
/// 从旧索引的 storage_id_map + deleted_bitmap 构建新的映射表。
/// 遍历旧映射表，跳过 bitmap 标记为已删除的 slot。
///
/// 注意：free_list 中的 slot 不会出现在任何 storage_id_map 中
/// （因为它们从未被 add() 分配到映射表里），所以天然被跳过。
void build_storage_id_map(
        const SharedVectorStore& store,
        const std::vector<uint64_t>& deleted_bitmap,
        const std::vector<idx_t>& old_storage_id_map,
        std::vector<idx_t>& new_storage_id_map) 
{
    new_storage_id_map.clear();
    new_storage_id_map.reserve(old_storage_id_map.size());

    for (idx_t i = 0; i < (idx_t)old_storage_id_map.size(); i++) {
        idx_t store_slot = old_storage_id_map[i];
        bool deleted = (deleted_bitmap[store_slot >> 6] >> (store_slot & 63)) & 1;
        if (!deleted) {
            new_storage_id_map.push_back(store_slot);
        }
    }
    // new_storage_id_map[new_local_id] = store_position
    // new_local_id ∈ [0, count_alive)
}
```

**时间复杂度：** O(old_ntotal)，遍历旧映射表。比遍历整个 store 更快（跳过 free_list 空洞）。

**空间复杂度：** `count_alive * sizeof(idx_t)` = 存活向量数 × 8 字节。

## 4. DistanceComputer 设计

这是性能最关键的部分。HNSW 搜索热路径中每次距离计算都经过 DistanceComputer。

### 4.1 IndirectFlatL2Dis

```cpp
namespace faiss {

/// 带间接映射的 L2 距离计算器。
/// 通过 storage_id_map 将本地 ID 解析为 store 中的实际位置，
/// 然后复用 fvec_L2sqr_batch_4/8 等 SIMD 优化函数。
struct IndirectFlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    const float* b;          ///< store->codes as float*
    const idx_t* id_map;     ///< storage_id_map.data()
    size_t ndis = 0;

    IndirectFlatL2Dis(
            const IndexFlatShared& storage,
            const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.store->codes.data(),
                      storage.store->code_size,
                      q),
              d(storage.d),
              b(reinterpret_cast<const float*>(storage.store->codes.data())),
              id_map(storage.storage_id_map.data()) {}

    void set_query(const float* x) override {
        q = x;
    }

    /// 核心：解析本地 ID → store 位置
    idx_t resolve(idx_t i) const {
        return id_map[i];
    }

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr(q, (const float*)code, d);
    }

    float operator()(idx_t i) override {
        ndis++;
        return fvec_L2sqr(q, b + resolve(i) * d, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + resolve(j) * d, b + resolve(i) * d, d);
    }

    /// 预取：先解析 id_map（大概率 L1 命中），再预取向量数据
    void prefetch(idx_t i) override {
        idx_t resolved = resolve(i);
        prefetch_L2(b + resolved * d);
    }

    /// 批量预取：id_map 条目通常在同一 cache line（连续 idx_t），
    /// 解析开销可忽略
    void prefetch_batch_4(
            idx_t i0, idx_t i1, idx_t i2, idx_t i3,
            int level) override {
        const float* p0 = b + resolve(i0) * d;
        const float* p1 = b + resolve(i1) * d;
        const float* p2 = b + resolve(i2) * d;
        const float* p3 = b + resolve(i3) * d;
        if (level == 1) {
            prefetch_L1(p0); prefetch_L1(p1); prefetch_L1(p2); prefetch_L1(p3);
        } else if (level == 3) {
            prefetch_L3(p0); prefetch_L3(p1); prefetch_L3(p2); prefetch_L3(p3);
        } else {
            prefetch_L2(p0); prefetch_L2(p1); prefetch_L2(p2); prefetch_L2(p3);
        }
    }

    /// batch_4: 先批量解析，再调用 SIMD 批量距离函数
    void distances_batch_4(
            const idx_t idx0, const idx_t idx1,
            const idx_t idx2, const idx_t idx3,
            float& dis0, float& dis1,
            float& dis2, float& dis3) final override {
        ndis += 4;
        const float* __restrict y0 = b + resolve(idx0) * d;
        const float* __restrict y1 = b + resolve(idx1) * d;
        const float* __restrict y2 = b + resolve(idx2) * d;
        const float* __restrict y3 = b + resolve(idx3) * d;

        float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
        fvec_L2sqr_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0; dis1 = dp1; dis2 = dp2; dis3 = dp3;
    }

    /// batch_8: 同上，8 路解析 + AVX-512 批量距离
    void distances_batch_8(
            const idx_t idx0, const idx_t idx1,
            const idx_t idx2, const idx_t idx3,
            const idx_t idx4, const idx_t idx5,
            const idx_t idx6, const idx_t idx7,
            float& dis0, float& dis1,
            float& dis2, float& dis3,
            float& dis4, float& dis5,
            float& dis6, float& dis7) final override {
        ndis += 8;
        const float* __restrict y0 = b + resolve(idx0) * d;
        const float* __restrict y1 = b + resolve(idx1) * d;
        const float* __restrict y2 = b + resolve(idx2) * d;
        const float* __restrict y3 = b + resolve(idx3) * d;
        const float* __restrict y4 = b + resolve(idx4) * d;
        const float* __restrict y5 = b + resolve(idx5) * d;
        const float* __restrict y6 = b + resolve(idx6) * d;
        const float* __restrict y7 = b + resolve(idx7) * d;

        fvec_L2sqr_batch_8(q, y0, y1, y2, y3, y4, y5, y6, y7, d,
                           dis0, dis1, dis2, dis3, dis4, dis5, dis6, dis7);
    }
};

/// Inner Product 版本结构相同，distance_to_code 调用 fvec_inner_product
struct IndirectFlatIPDis : FlatCodesDistanceComputer {
    // ... 结构与 IndirectFlatL2Dis 对称，省略
};

} // namespace faiss
```

### 4.2 间接寻址的性能影响分析

| 环节 | 无间接层（原始 FlatL2Dis） | 有间接层（IndirectFlatL2Dis） | 差异 |
|------|--------------------------|-------------------------------|------|
| 地址计算 | `codes + i * code_size` | `codes + id_map[i] * code_size` | +1 次内存加载（8 字节） |
| id_map 访问 | — | ~1 cycle (L1 hit, 顺序访问) | 可忽略 |
| 向量数据预取 | 可预取 | 同样可预取（resolve 后） | 无差异 |
| batch_4/8 SIMD | 连续指针 | 离散指针，但 SIMD 函数本身不要求连续 | **无差异** |

**结论：间接层的搜索性能开销 < 2%。** `fvec_L2sqr_batch_4/8` 接受 4/8 个独立指针，不要求内存连续。间接层只增加了地址解析步骤，不影响 SIMD 计算本身。

**`is_identity_map` 快速路径：** 当 `is_identity_map = true` 时（首次构建或 compact 后），`resolve(i)` 直接返回 `i`，跳过 `id_map[i]` 的内存加载。此时 `IndirectFlatL2Dis` / `IndirectFlatIPDis` 的性能与原始 `FlatL2Dis` 完全一致——地址计算退化为 `codes + i * code_size`，无任何额外开销。此外，`prefetch()` 也跳过对 `id_map` 数组的预取，减少不必要的 cache 污染。

### 4.3 与已有性能优化的兼容性

| 优化项 | 兼容性 | 说明 |
|--------|--------|------|
| batch_8 / batch_4 距离计算 (AVX-512) | ✅ 完全兼容 | 间接层在调用 `fvec_L2sqr_batch_8` 之前完成地址解析 |
| DPDK 风格预取 | ✅ 完全兼容 | `prefetch(i)` 内部先 resolve 再 prefetch，流水线不变 |
| Cross-node neighbor batching | ✅ 完全兼容 | 上层 HNSW.cpp 的 batch 逻辑不变，传入的仍是本地 ID |
| 图重排序 (RCM / Weighted / DFS) | ✅ 兼容，双模式 | `permute_entries` 双模式：identity 时物理搬移，否则只置换 `storage_id_map`。详见 §10.4 |
| SIMD count_below (MinimaxHeap) | ✅ 完全兼容 | 与 storage 层无关 |
| Hugepage 存储 | ✅ 可扩展 | `SharedVectorStore::codes` 可改用 hugepage 分配 |
| 压缩存储 (IndexFlatCompressed) | ⚠️ 需评估 | 需验证与间接寻址的兼容性 |

## 5. 索引构建流程

### 5.1 前置条件

索引构建假设 `SharedVectorStore` 已经存在，其中包含所有向量数据。store 由外部创建和管理，索引构建只需要接收一个 `shared_ptr<SharedVectorStore>`。

```
调用者负责:
  1. 创建 SharedVectorStore，填充向量数据
  2. 设置 deleted_bitmap（标记哪些向量已被删除）
  3. 调用 build_new_index() 构建索引
```

### 5.2 构建索引

```cpp
/// 使用共享存储构建新的 HNSW 索引。
/// 遍历当前索引的 storage_id_map，跳过其 deleted_bitmap 标记的已删除向量，
/// 生成新的 storage_id_map，然后构建新的 HNSW 图。
///
/// 向量数据已在 store 中，不需要额外的临时缓冲区。
/// hnsw_add_vertices 通过 store->get_vector(storage_id_map[pt_id]) 直接获取
/// 向量指针传给 set_query()，零拷贝。
///
/// @param store             共享向量存储
/// @param current_index     当前运行的索引（用于获取旧 storage_id_map 和 deleted_bitmap）
/// @param M                 HNSW 参数
/// @param efConstruction    HNSW 参数
/// @return 新的 IndexHNSW
IndexHNSW* build_new_index(
        std::shared_ptr<SharedVectorStore> store,
        const IndexHNSW& current_index,
        int M = 32,
        int efConstruction = 40,
        MetricType metric = METRIC_L2) 
{
    // 1. 获取旧索引的 storage_id_map 和 deleted_bitmap
    auto* old_shared = dynamic_cast<const IndexFlatShared*>(current_index.storage);
    FAISS_THROW_IF_NOT_MSG(old_shared, 
        "current_index.storage must be IndexFlatShared");

    // 2. 构建新的 storage_id_map（遍历旧 storage_id_map，跳过旧索引 bitmap 标记的）
    std::vector<idx_t> new_storage_id_map;
    build_storage_id_map(*store, old_shared->deleted_bitmap, 
        old_shared->storage_id_map, new_storage_id_map);

    idx_t n_alive = new_storage_id_map.size();

    // 3. 创建 IndexFlatShared（带新映射，bitmap 初始全 0）
    auto* shared_storage = new IndexFlatShared(store, metric);
    shared_storage->storage_id_map = std::move(new_storage_id_map);
    shared_storage->ntotal = n_alive;
    // deleted_bitmap 在构造函数中初始化为全 0（新索引无删除）

    // 4. 创建 IndexHNSW
    auto* new_index = new IndexHNSW(shared_storage, M);
    new_index->own_fields = true;
    new_index->hnsw.efConstruction = efConstruction;

    // 5. 构建 HNSW 图（零拷贝，不调用 storage->add()）
    //    数据已在 store 中，storage_id_map 已设好，
    //    跳过 storage->add()，直接构建图结构。
    //    需要修改 hnsw_add_vertices，使其通过
    //    store->get_vector(storage_id_map[pt_id]) 获取向量指针
    //    传给 dis->set_query()，而非从连续数组 x 读取。
    //    详见 §9 对 hnsw_add_vertices 的修改。
    new_index->ntotal = n_alive;
    hnsw_add_vertices(*new_index, 0, n_alive, nullptr, verbose);
    // x = nullptr 表示从 storage 直接读取，不使用外部连续数组

    return new_index;
}
```

**首次构建 vs rebuild 对比：**

| 环节 | 首次构建 | rebuild |
|------|---------|---------|
| SharedVectorStore | 外部创建，填充向量数据 | 已有 store，通过 shared_ptr 共享 |
| 旧 storage_id_map | `[0, 1, 2, ..., n-1]`（无空洞） | 显式 `vector<idx_t>`（可能有空洞） |
| 旧 deleted_bitmap | 全 0（首次构建无删除） | 索引自身的 bitmap，标记了被删除的 slot |
| `build_storage_id_map` | bitmap 全 0，结果不变 | 遍历旧 `storage_id_map`，跳过 bitmap 标记的 |
| store 数据移动 | 无 | 无（共享 ptr） |
| 额外内存 | 无（零拷贝） | 无（零拷贝） |

**零拷贝原理：**

`set_query(const float* x)` 只保存指针，不拷贝数据（faiss 接口契约：*"Pointer x should remain valid while operator() is called"*）。rebuild 期间 store 是只读的，`store->codes` 不会 resize，所以 `store->get_vector(slot)` 返回的指针在整个构图过程中稳定有效。

因此 `hnsw_add_vertices` 可以直接用 `store->get_vector(storage_id_map[pt_id])` 作为 `set_query` 的参数，无需任何临时缓冲区。

## 6. 生命周期管理

### 6.1 所有权模型

```
shared_ptr<SharedVectorStore> 引用计数:

  初始状态:
    old_index->storage (IndexFlatShared) 持有 shared_ptr  → refcount = 1
    用户代码 / 管理器也持有 shared_ptr                     → refcount = 2

  重建期间:
    old_index->storage 持有                               → refcount = 2
    new_index->storage (IndexFlatShared) 持有              → refcount = 3
    用户代码持有                                           → (可选)

  切换后:
    delete old_index                                      → refcount 减 1
    new_index->storage 持有                               → refcount = 1 或 2
    
  最终:
    delete new_index                                      → refcount = 0
    SharedVectorStore 析构，释放 codes 内存
```

**关键保证：** `SharedVectorStore` 的生命周期自动管理，不会出现悬挂指针。只要有任一 IndexFlatShared 存活，store 就不会被释放。

### 6.2 完整循环生命周期

系统支持 **无限次** rebuild 循环。每次循环回收上一轮的空洞，新向量复用空洞位置。

```
                    ┌──────────────────────────────────────────────┐
                    │          完整生命周期状态图                     │
                    │                                              │
                    │   ┌─────────┐  build_new_index  ┌─────────┐ │
                    │   │ Store   │ ─────────────────► │ 单索引   │ │
                    │   │ 已创建   │                    │ 运行中   │ │
                    │   │(外部填充 │                    │(shared   │ │
                    │   │ 向量数据)│                    │  store)  │ │
                    │   └─────────┘                    └────┬────┘ │
                    │                                       │      │
                    │                           add / delete│      │
                    │                           (正常操作)   │      │
                    │                                       ▼      │
                    │                              删除过多？──No──►│
                    │                                 │Yes         │
                    │                                 ▼            │
                    │   ┌─────────────────────────────────────┐    │
                    │   │          Rebuild 流程                │    │
                    │   │                                      │    │
                    │   │  1. build_new_index(store, cur_index)│    │
                    │   │     → 双索引并存，旧索引 read-only    │    │
                    │   │  2. 写锁 + swap(new_index, old_index)│    │
                    │   │  3. delete old_index                 │    │
                    │   │  4. 写锁 + compact_store + reorder   │    │
                    │   │     → is_identity_map = true          │   │
                    │   └──────────────────────┬──────────────┘    │
                    │                          │                    │
                    │                          ▼                    │
                    │                   ┌───────────┐              │
                    │                   │ 单索引     │◄─────────────┘
                    │                   │ 运行中     │  (循环)
                    │                   │(带 free_   │
                    │                   │  list)     │
                    │                   └───────────┘
                    └──────────────────────────────────────────────┘
```

### 6.3 生命周期各阶段详解

#### Phase 0: 首次构建

```cpp
// 外部创建 SharedVectorStore 并填充向量数据
auto store = std::make_shared<SharedVectorStore>(d, code_size);
// ... 填充 store->codes, 设置 store->ntotal_store ...

// 构建首个索引
auto* shared_storage = new IndexFlatShared(store, METRIC_L2);
// storage_id_map = [0, 1, 2, ..., n-1], deleted_bitmap 全 0
shared_storage->storage_id_map.resize(store->ntotal_store);
std::iota(shared_storage->storage_id_map.begin(), 
          shared_storage->storage_id_map.end(), 0);
shared_storage->ntotal = store->ntotal_store;
shared_storage->deleted_bitmap.resize((store->ntotal_store + 63) / 64, 0);

auto* index = new IndexHNSW(shared_storage, M);
index->own_fields = true;
index->hnsw.efConstruction = efConstruction;

// 构建 HNSW 图（零拷贝，同 build_new_index 的步骤 5）
// 数据已在 store 中，直接通过 store->get_vector() 获取指针
index->ntotal = store->ntotal_store;
hnsw_add_vertices(*index, 0, store->ntotal_store, nullptr, verbose);
```

此时 store 状态:
```
slots:    [v0][v1][v2][v3][v4][v5][v6][v7]   (8 个向量)
free_list: []  (空)
index->storage->deleted_bitmap: [0][0][0][0][0][0][0][0]  (全 0)
index->storage->storage_id_map: [0, 1, 2, 3, 4, 5, 6, 7]
```

#### Phase 1: 正常运行 + 标记删除

```cpp
// 删除向量 — 通过索引的 deleted_bitmap 标记
auto* shared = dynamic_cast<IndexFlatShared*>(active_index->storage);
shared->mark_deleted(1);  // local_id 1 → store slot 1
shared->mark_deleted(3);  // local_id 3 → store slot 3
shared->mark_deleted(6);  // local_id 6 → store slot 6
// HNSW 图中 local_id 1, 3, 6 标记为 deleted (IDSelector)
```

此时状态:
```
slots:    [v0][v1][v2][v3][v4][v5][v6][v7]
free_list: []
index->storage->deleted_bitmap: [0][1][0][1][0][0][1][0]  (v1,v3,v6 标记删除)
index->storage->storage_id_map: [0, 1, 2, 3, 4, 5, 6, 7]
```

#### Phase 2: Rebuild

```cpp
// 构建新索引（跳过旧索引 bitmap 标记的 slot）
auto* new_index = build_new_index(store, *active_index, M, efConstruction);
// new_index->storage->storage_id_map = [0, 2, 4, 5, 7]  (跳过 1,3,6)
// new_index->storage->deleted_bitmap = 全 0（新索引无删除）

// old_index 继续服务搜索 ← read-only
// new_index 构建完成，等待切换
```

#### Phase 3: 写锁 + Swap + 清理

```cpp
{
    // 加写锁，阻止新的搜索/add/delete 请求
    std::unique_lock<std::shared_mutex> lock(index_mutex);
    
    // 原子交换：搜索流量切到 new_index
    std::swap(active_index, new_index);
    // 现在 active_index = 新索引, new_index = 旧索引
}
// 释放写锁，新搜索请求使用新索引

// 回收旧索引中被删除的 slot（使用旧索引的 bitmap）
auto* old_shared = dynamic_cast<IndexFlatShared*>(new_index->storage);
store->reclaim_deleted_slots(old_shared->deleted_bitmap, store->ntotal_store);
// free_list = [1, 3, 6]

// 销毁旧索引（此时无人使用它）
delete new_index;  // (这是旧索引) → shared_ptr refcount 减 1
```

此时 store 状态:
```
slots:    [v0][??][v2][??][v4][v5][??][v7]   (slot 1,3,6 数据无效)
free_list: [1, 3, 6]  (三个可复用 slot)
active_index->storage->storage_id_map = [0, 2, 4, 5, 7]
active_index->storage->deleted_bitmap = 全 0
```

#### Phase 4: 正常运行（add / delete / search）

```cpp
// 添加新向量 — 优先复用空洞
active_index->add(3, new_vectors);
// 内部调用 store->allocate_slot() 三次：
//   slot 6 ← free_list.pop_back()
//   slot 3 ← free_list.pop_back()
//   slot 1 ← free_list.pop_back()
// storage_id_map 变为 [0, 2, 4, 5, 7, 6, 3, 1]
// free_list = []  (用完了)

// 继续添加更多向量 — append 到末尾
active_index->add(2, more_vectors);
// store->allocate_slot() append: slot 8, 9
// storage_id_map = [0, 2, 4, 5, 7, 6, 3, 1, 8, 9]
// store->ntotal_store = 10

// 删除向量 — 通过索引的 deleted_bitmap 标记
auto* shared = dynamic_cast<IndexFlatShared*>(active_index->storage);
shared->mark_deleted(5);  // local_id 5 → store slot 7
shared->mark_deleted(8);  // local_id 8 → store slot 8
// HNSW 图中 local_id 5 和 8 标记为 deleted (IDSelector)
```

此时 store 状态:
```
slots:    [v0][v_new1][v2][v_new2][v4][v5][v_new0][v7][v_new3][v_new4]
free_list: []
active_index->storage->deleted_bitmap: 
  [0][0][0][0][0][0][0][1][1][0]  (store slot 7,8 标记删除)
active_index->storage->storage_id_map = [0, 2, 4, 5, 7, 6, 3, 1, 8, 9]
  其中 local_id 5 (→store 7) 和 local_id 8 (→store 8) 已被删除
```

#### Phase 5: 再次 Rebuild（删除过多时）

```cpp
// 与 Phase 2 相同的流程
auto* rebuild_index = build_new_index(store, *active_index, M, efConstruction);
// rebuild_index->storage->storage_id_map = [0, 2, 4, 5, 6, 3, 1, 9]
//   (跳过旧索引 bitmap 标记的 store slot 7 和 8)
// rebuild_index->storage->deleted_bitmap = 全 0

// swap + reclaim + delete 旧索引
{
    std::unique_lock<std::shared_mutex> lock(index_mutex);
    std::swap(active_index, rebuild_index);
}
auto* old_shared = dynamic_cast<IndexFlatShared*>(rebuild_index->storage);
store->reclaim_deleted_slots(old_shared->deleted_bitmap, store->ntotal_store);
// free_list = [7, 8]
delete rebuild_index;

// 循环继续...
```

### 6.4 不变量（Invariants）

在整个生命周期中，以下不变量始终成立：

| 不变量 | 说明 |
|--------|------|
| **每个 occupied slot 恰好被一个索引的 storage_id_map 引用** | 不存在两个索引同时 "拥有" 同一个 slot |
| **deleted_bitmap 属于 IndexFlatShared，不属于 store** | 每个索引独立管理自己的删除状态 |
| **新索引的 deleted_bitmap 初始全 0** | rebuild 产出的索引没有任何删除标记 |
| **free_list 中的 slot 不在任何索引的 storage_id_map 中** | available slot 不会被任何索引引用 |
| **reclaim_deleted_slots() 仅在单索引状态下调用** | 前置条件：旧索引已销毁，只有一个 IndexFlatShared 引用 store |
| **rebuild 期间无 add/delete** | 用户保证 rebuild 期间旧索引 read-only |
| **store->ntotal_store 单调不减** | allocate_slot append 时递增，reclaim 不改变 ntotal_store |
| **HNSW 图中被删除的节点仍可被访问** | 搜索时通过距离计算过滤，所以 deleted slot 的数据必须有效，直到 rebuild + swap 后才可回收 |

## 7. 并发安全性分析

系统有三种运行阶段，各自的并发模式不同：

### 7.1 正常运行阶段（单索引：search + add + delete）

```
Search Threads (shared_lock):
  active_index->search()
    → storage->get_distance_computer()
      → IndirectFlatL2Dis (读 store->codes, 读 storage_id_map)  [READ-ONLY]
    → HNSW graph traversal (读 hnsw.neighbors)                    [READ-ONLY]

Add/Delete Thread (exclusive_lock):
  active_index->add(n, x)
    → storage->add()
      → store->allocate_slot() → 修改 free_list, 可能 resize codes  [WRITE store]
      → storage_id_map.push_back()                                  [WRITE id_map]
    → hnsw_add_vertices → 写 hnsw.neighbors, levels, offsets       [WRITE graph]
  
  shared->mark_deleted(local_id)
    → 修改 index 的 deleted_bitmap                                  [WRITE bitmap]
  // HNSW 图中对应节点标记为 deleted (IDSelector)
```

| 资源 | Search (shared_lock) | Add/Delete (exclusive_lock) | 冲突？ |
|------|---------------------|----------------------------|--------|
| `store->codes` | READ | WRITE (allocate_slot) | ⚠️ **需要锁保护** |
| `store->free_list` | 不访问 | WRITE | ❌ 无冲突 |
| `index->deleted_bitmap` | 不访问 | WRITE (mark_deleted) | ❌ 无冲突 |
| `storage_id_map` | READ | WRITE (push_back) | ⚠️ **需要锁保护** |
| `hnsw.neighbors` | READ | WRITE | ⚠️ **需要锁保护** |

**锁策略：`std::shared_mutex`（读写锁）**
- Search 线程持 `shared_lock` → 多读并发
- Add/Delete 操作持 `unique_lock` → 独占写入
- 这与用户描述的 "客户端加写锁" 一致

> **注意：** `allocate_slot()` 在 free_list 为空时会 `codes.resize()`，这可能导致内存重分配使所有指向 `codes.data()` 的指针失效。因此 add 操作 **必须** 持有写锁，阻止所有 search 线程。如果性能上不可接受，可考虑预分配 `codes.reserve()`。

### 7.2 Rebuild 阶段（双索引并存：old search + new build）

```
Search Threads (shared_lock on old_index):
  old_index->search()
    → IndirectFlatL2Dis (读 store->codes, 读 old storage_id_map)  [READ-ONLY]
    → HNSW graph traversal (读 old hnsw.neighbors)                  [READ-ONLY]

Build Thread (独立，无需锁):
  build_new_index()
    → build_storage_id_map (读 old_shared->deleted_bitmap)            [READ-ONLY on index]
    → reconstruct → 读 store->codes via old storage_id_map         [READ-ONLY on store]
    → hnsw_add_vertices → 写 new hnsw.*                           [WRITE, 独立数据]
```

| 资源 | Search 线程 | Build 线程 | 冲突？ |
|------|------------|-----------|--------|
| `store->codes` | READ | READ | ❌ 无冲突 |
| `old_shared->deleted_bitmap` | 不访问 | READ（构建 id_map 时） | ❌ 无冲突 |
| `old_index->hnsw.*` | READ | 不访问 | ❌ 无冲突 |
| `new_index->hnsw.*` | 不访问 | WRITE | ❌ 无冲突 |
| `old storage_id_map` | READ | READ（reconstruct 时） | ❌ 无冲突 |
| `new storage_id_map` | 不访问 | 构建前一次性生成，构建时 READ | ❌ 无冲突 |
| `store->free_list` | 不访问 | 不访问 | ❌ 无冲突 |

**结论：rebuild 期间完全无锁。** 用户保证此阶段无 add/delete 操作，所有共享数据均为只读。两个索引的图结构完全独立。不需要 mutex、atomic 或任何同步原语。

### 7.3 Swap 阶段（原子切换）

```cpp
// 1. 加写锁，阻止所有搜索和写操作
{
    std::unique_lock<std::shared_mutex> lock(index_mutex);
    std::swap(active_index, old_index);
}
// 2. 释放锁，新搜索走新索引

// 3. 等待所有持有旧 shared_lock 的搜索线程完成（自然完成，无需特殊处理）
// 4. delete old_index（此时无人引用）
// 5. store->reclaim_deleted_slots(old_bitmap, ntotal)（单索引状态，无并发访问 store 的写操作）
```

`reclaim_deleted_slots()` 的前置条件：
- 旧索引已销毁（无人读 deleted slot 的向量数据）
- 只有一个 IndexFlatShared 引用 store
- 旧索引的 bitmap 作为参数传入（在 delete 旧索引之前保存）
- 此时可安全填充 free_list

### 7.4 注意事项

- `shared_ptr` 的引用计数操作是 atomic 的，但仅在索引创建/销毁时发生，不在搜索热路径上
- `DistanceComputer` 是 per-thread 创建的（在 `#pragma omp parallel` 块内），不存在跨线程共享
- 构建 `storage_id_map` 应在 `new_index->add()` 之前完成（单线程，一次性）
- `allocate_slot()` 写入 `codes` 时使用 `memcpy`，修改的是已分配 slot 的内容（free_list 复用）或新末尾空间（append），不影响其他 slot 的已有数据

## 8. 内存开销分析

以 **1 亿向量、128 维、float32、M=32** 为例。

### 8.1 首次 Rebuild（20% 删除率）

| 组件 | 大小 | 说明 |
|------|------|------|
| 向量数据 (store->codes) | 100M × 128 × 4B = **51.2 GB** | 共享，不增加 |
| 删除位图 | 100M / 8 = **12.5 MB** | 每个索引各一份，可忽略 |
| free_list | **0** | 首次 rebuild 前 free_list 为空 |
| 旧索引 storage_id_map | 100M × 8B = **800 MB** | 显式映射 [0,1,...,n-1] |
| 旧索引 HNSW 图 | 100M × 64 × 4B ≈ **25.6 GB** | 已存在 |
| 新索引 storage_id_map | 80M × 8B = **640 MB** | 存活数 × sizeof(idx_t) |
| 新索引 HNSW 图 (M=32) | 80M × 64 × 4B ≈ **20.5 GB** | neighbors + offsets + levels |
| **峰值（build 期间）** | **~99 GB** | store + 两图 + 两 id_map + bitmap |
| **稳态（swap 后，仅新索引）** | **~72.9 GB** | store + 新图 + id_map + bitmap |

### 8.2 稳态运行（slot 复用后）

假设首次 rebuild 后：回收 20M 个 slot → free_list，然后添加 15M 新向量（复用 15M 个 slot），再删除 10M 个向量。

| 组件 | 大小 | 说明 |
|------|------|------|
| 向量数据 (store->codes) | 100M × 512B = **51.2 GB** | ntotal_store 不变（15M 新向量复用空洞） |
| 删除位图 | **12.5 MB** | 100M bits |
| free_list | 5M × 8B = **40 MB** | 20M 回收 - 15M 复用 = 5M 剩余 |
| storage_id_map | 85M × 8B = **680 MB** | 80M 存活 + 15M 新增 - 10M 删除 = 85M |
| HNSW 图 (M=32) | 95M × 64 × 4B ≈ **24.3 GB** | 80M + 15M = 95M 节点（含 10M deleted） |
| **稳态总计** | **~76.2 GB** | |

**关键观察：** 因为新向量复用了空洞，`store->codes` 大小 **没有增长**（仍然是 100M slots × 512B）。只有当 free_list 耗尽后新增向量才会 append，导致 codes 增长。

### 8.3 再次 Rebuild（稳态中 10M 删除后）

| 组件 | 大小 | 说明 |
|------|------|------|
| store->codes | **51.2 GB** | 不变 |
| 旧索引 HNSW 图 | **24.3 GB** | 95M 节点 |
| 新索引 storage_id_map | 85M × 8B = **680 MB** | 跳过 10M deleted |
| 新索引 HNSW 图 | 85M × 64 × 4B ≈ **21.8 GB** | |
| **峰值** | **~98 GB** | store + 两图 + id_map（零拷贝，无临时缓冲区） |
| **swap 后稳态** | **~73.7 GB** | store + 新图 + id_map |

### 8.4 辅助数据结构开销

| 数据结构 | 每元素大小 | 1 亿向量总计 | 说明 |
|----------|-----------|-------------|------|
| `deleted_bitmap` | 1 bit | 12.5 MB | 固定开销 |
| `free_list` | 8 bytes | 最多 800 MB | 极端情况：所有 slot 都被回收 |
| `storage_id_map` | 8 bytes | 800 MB | 与存活向量数成正比 |

> **free_list 峰值不会超过 ntotal_store × 8B。** 实际中 free_list 会被 add 操作快速消耗，稳态下通常远小于峰值。

### 8.5 对比传统方案

| 方案 | 首次 Rebuild 峰值 | 稳态内存 | 再次 Rebuild 峰值 |
|------|-------------------|---------|-------------------|
| 传统（复制向量） | 51.2 + 51.2 + 20.5 = **122.9 GB** | 51.2 + 20.5 = **71.7 GB** | 同首次 |
| 共享存储（本方案，零拷贝） | 51.2 + 20.5 + 0.8 + 0.64 ≈ **73.1 GB** | **~72.9 GB** | **~98 GB** |

> **关键洞察：** 
> 1. 零拷贝构建使得 rebuild 峰值 ≈ 稳态，首次 Rebuild 比传统方案节省 **41%** 峰值内存。
> 2. 稳态内存中 `storage_id_map`（~640 MB）是共享方案相比传统方案的额外开销，占总内存 < 1%。
> 3. slot 复用使得 `store->codes` 大小趋于稳定，不会因反复 rebuild 而无限增长。
> 4. 再次 Rebuild 峰值略高（~98 GB），因为旧索引图（24.3 GB）和新索引图（21.8 GB）短暂共存。

## 9. 零拷贝构建：`hnsw_add_vertices` 与 `IndexHNSW::add` 修改

本方案的核心修改之一。使 HNSW 图构建过程直接从 `SharedVectorStore` 读取向量指针，不需要任何临时缓冲区。

> **设计目标：** `SharedVectorStore` 和 `IndexFlatShared` 保持索引类型无关（index-type agnostic）。仅 `hnsw_add_vertices` 和 `IndexHNSW::add` 需要修改，修改量极小。

### 9.1 背景：原始 faiss 代码

`hnsw_add_vertices`（`faiss/IndexHNSW.cpp`, line 159）在构建图时，为每个节点设置查询向量：

```cpp
dis->set_query(x + (pt_id - n0) * d);
```

这要求向量在**连续内存**中（`x` 指向 `n` 个向量的平坦数组）。但在共享存储方案中，向量在 store 里可能有空洞（被删除的 slot），不是连续的。

`set_query` 的 faiss 合约（`faiss/impl/DistanceComputer.h`, line 27-29）：

```cpp
/// called before computing distances. Pointer x should remain valid
/// while operator() is called.
virtual void set_query(const float* x) = 0;
```

**关键洞察：** `set_query` 仅保存指针，不拷贝数据。只要指针在 `operator()` 调用期间保持有效即可。

### 9.2 `hnsw_add_vertices` 修改

修改仅在 `set_query` 一行：当 `x == nullptr` 时，从 `IndexFlatShared` 的共享 store 直接获取向量指针。

```cpp
// faiss/IndexHNSW.cpp, hnsw_add_vertices 函数内，约 line 155-165

// 原始代码:
// dis->set_query(x + (pt_id - n0) * d);

// 修改后:
if (x) {
    // 传统路径：向量在连续内存中
    dis->set_query(x + (pt_id - n0) * d);
} else {
    // 零拷贝路径：向量已在 SharedVectorStore 中
    // storage 实际类型为 IndexFlatShared，通过 storage_id_map 获取 store 中的 slot
    auto* flat_shared = dynamic_cast<const IndexFlatShared*>(index_hnsw.storage);
    FAISS_THROW_IF_NOT_MSG(flat_shared,
        "x == nullptr requires storage to be IndexFlatShared");
    idx_t storage_id = flat_shared->storage_id_map[pt_id];
    dis->set_query(flat_shared->store->get_vector(storage_id));
}
```

**为什么指针安全：**
- Rebuild 期间旧索引只有读操作，不会有 `add` 触发 `codes.resize()`
- 新索引的 `storage_id_map` 在 `build_new_index` 开始时已经构建完成
- `get_vector` 返回指向 `codes` 内部的指针，在当前 `add_with_locks` 调用期间不会失效
- 每个 `set_query` 设置的指针仅在下一次 `set_query` 之前被使用

### 9.3 `IndexHNSW::add` 修改

当 `x == nullptr` 时，跳过 `storage->add()`（数据已在 store 中），直接构建图。

```cpp
void IndexHNSW::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(googblocks_check(), "check failed");
    FAISS_THROW_IF_NOT(googblocks_check2());
    
    int n0 = ntotal;
    
    if (x) {
        // 传统路径：向量需要添加到 storage
        storage->add(n, x);
    }
    // x == nullptr: 数据已在 SharedVectorStore 中，
    // IndexFlatShared::storage_id_map 已构建完成，
    // storage->ntotal 已设置正确（= n_alive）
    
    ntotal = storage->ntotal;
    hnsw_add_vertices(*this, n0, n, x, verbose,
                      hnsw.levels.size() == ntotal);
}
```

### 9.4 调用方式

在 `build_new_index` 中：

```cpp
IndexHNSW* build_new_index(SharedVectorStore* store,
                           const std::vector<uint64_t>& deleted_bitmap,
                           size_t ntotal_old_index,
                           int M, int efConstruction) {
    // 1. 构建 storage_id_map（仅包含存活向量）
    auto id_map = build_storage_id_map(deleted_bitmap, ntotal_old_index);
    size_t n_alive = id_map.size();
    
    // 2. 创建 IndexFlatShared 作为 storage
    auto* flat_shared = new IndexFlatShared(store, std::move(id_map));
    flat_shared->ntotal = n_alive;
    // deleted_bitmap 初始为全 0（新索引无删除）
    
    // 3. 创建 IndexHNSW，挂载 flat_shared 为 storage
    auto* new_index = new IndexHNSW(flat_shared, M);
    new_index->hnsw.efConstruction = efConstruction;
    
    // 4. 零拷贝构建：x = nullptr 触发从 store 直接读取
    new_index->add(n_alive, nullptr);
    
    return new_index;
}
```

### 9.5 修改范围总结

| 文件 | 修改 | 行数 |
|------|------|------|
| `faiss/IndexHNSW.cpp` — `hnsw_add_vertices` | `set_query` 分支：`x ? 原始路径 : store 直接读取` | ~8 行 |
| `faiss/IndexHNSW.cpp` — `IndexHNSW::add` | `x ? storage->add : 跳过`，其余不变 | ~5 行 |
| `faiss/IndexFlatShared.h/cpp` | 新增文件（index-type agnostic） | 不修改 faiss 核心 |
| `faiss/SharedVectorStore.h/cpp` | 新增文件（index-type agnostic） | 不修改 faiss 核心 |

> **对 faiss 核心的侵入性极小：** 仅修改 `IndexHNSW.cpp` 两处（共 ~13 行），且通过 `if (x)` 保持完全向后兼容——当 `x != nullptr` 时行为与原始 faiss 完全一致。

## 10. Store Compaction 与 Reorder 优化

Rebuild 之后，`storage_id_map` 为非恒等映射（例如 `[0, 2, 4, 5, 7]`），每次距离计算需要间接寻址。本节描述一个可选的后处理流程，通过物理重排 store 中的向量数据，消除间接寻址开销，并进一步应用图重排序提升搜索性能。

### 10.1 设计动机

Rebuild 后 `is_identity_map = false`，搜索热路径的额外开销：
1. 每次距离计算多一次 `id_map[i]` 内存加载（8 字节，通常 L1 命中）
2. 向量在 store 中分布离散，cache 局部性不如连续存储
3. 无法直接应用图重排序（BFS/RCM 等），因为图重排序后 `storage_id_map` 不再恒等，性能优势被间接层抵消

### 10.2 Compact 算法

**目标：** 通过原地交换（cycle-following permutation），使 `store[i]` 存放 `local_id == i` 对应的向量数据，从而恢复 `is_identity_map = true`。

**算法（原地 cycle-following）：**

```cpp
/// 原地压缩 store，使 store[i] == local_id i 的向量。
/// 临时内存开销: O(code_size) = 一个向量大小，满足零拷贝约束。
///
/// @param store      共享向量存储
/// @param id_map     当前 storage_id_map（compact 后变为恒等）
/// @param n_alive    存活向量数量
void compact_store(
        SharedVectorStore& store,
        std::vector<idx_t>& id_map,
        size_t n_alive) {
    std::vector<bool> visited(n_alive, false);
    std::vector<uint8_t> tmp(store.code_size);  // O(d) 临时空间

    for (size_t i = 0; i < n_alive; i++) {
        if (visited[i] || id_map[i] == (idx_t)i) {
            visited[i] = true;
            continue;
        }
        // 跟踪置换循环
        memcpy(tmp.data(),
               store.codes.data() + i * store.code_size,
               store.code_size);
        size_t j = i;
        while (true) {
            size_t src = id_map[j];
            visited[j] = true;
            if (src == (idx_t)i) {
                // 循环闭合：将保存的数据放入 slot j
                memcpy(store.codes.data() + j * store.code_size,
                       tmp.data(), store.code_size);
                break;
            }
            // 将 src 位置的数据搬到 j 位置
            memcpy(store.codes.data() + j * store.code_size,
                   store.codes.data() + src * store.code_size,
                   store.code_size);
            j = src;
        }
    }

    // compact 后状态更新
    store.ntotal_store = n_alive;
    store.codes.resize(n_alive * store.code_size);
    store.free_list.clear();

    // 将 id_map 设为恒等
    for (size_t i = 0; i < n_alive; i++) {
        id_map[i] = i;
    }
}
```

**算法正确性：**
- cycle-following 是经典的原地置换算法。对于置换 `P`，每个元素属于且仅属于一个循环
- 对于每个循环 `(i → P[i] → P[P[i]] → ... → i)`，只需一个临时变量即可完成所有交换
- `visited` 数组确保每个元素只被处理一次，时间复杂度 O(n_alive)
- `storage_id_map` 中的映射 `[i] → store_slot` 天然构成一个置换（每个 store_slot 唯一），因此 cycle-following 可以正确应用

**开销分析：**

| 项目 | 开销 |
|------|------|
| 临时内存 | O(code_size) = 1 个向量 ≈ 512B（128 维） |
| 时间复杂度 | O(n_alive)，每个向量最多被 memcpy 一次 |
| 数据搬移量 | n_alive × code_size（例：100 万 × 512B ≈ 512MB） |
| 耗时估算 | ~100-200ms（受内存带宽限制，~10-20 GB/s） |

### 10.3 执行时序

Compact 修改了 store 中向量的物理位置，因此必须在**旧索引完全停用后**、且**新索引暂停服务期间**执行。

```
完整流程（含 compact + reorder）:

1. build_new_index(store, cur_index)
   → 新索引建好，storage_id_map 非恒等，is_identity_map = false
   → 此时新索引尚未投入使用

2. write-lock → swap(active_index, new_index)
   → 新索引开始服务搜索（is_identity_map = false，通过间接寻址）
   → 释放 write-lock

3. 等待旧索引上的搜索线程全部完成 → 销毁旧索引
   → store 的 shared_ptr 引用计数减 1

4. write-lock → 暂停搜索
   4a. compact_store(store, storage_id_map, n_alive)
       → store[i] == local_id i 的向量
       → is_identity_map = true
   4b.（可选）permute_entries(bfs_perm)
       → 图重排序 + store 内向量物理搬移（cycle-following）
       → is_identity_map 仍为 true
   → 释放 write-lock，恢复搜索

时间窗口: 步骤 4 的 write-lock 持续时间 ≈ compact 耗时 + reorder 耗时
  100 万向量 × 128 维: ~200-400ms
  1000 万向量 × 128 维: ~2-4s
```

**为什么不在 `build_new_index()` 内部做 compact：**
`build_new_index()` 期间旧索引仍在服务搜索，旧索引通过 `shared_ptr` 引用同一个 store。如果在此时 compact store，旧索引的搜索线程会读到被搬移的向量数据，导致**数据损坏**。因此 compact 必须在旧索引被完全销毁后执行。

### 10.4 `permute_entries` 的双模式实现

Compact 之后 `is_identity_map = true`，此时调用图重排序（BFS/RCM/Cluster-Weighted）需要**物理搬移 store 中的向量数据**，以保持恒等映射不被破坏。

```cpp
void IndexFlatShared::permute_entries(const idx_t* perm) {
    if (is_identity_map) {
        // 模式 A：物理搬移向量数据（cycle-following），保持恒等映射
        // perm[new] = old，需要把 store[old] 搬到 store[new]
        std::vector<bool> visited(ntotal, false);
        std::vector<uint8_t> tmp(store->code_size);

        for (idx_t i = 0; i < ntotal; i++) {
            if (visited[i] || perm[i] == i) {
                visited[i] = true;
                continue;
            }
            memcpy(tmp.data(),
                   store->codes.data() + i * store->code_size,
                   store->code_size);
            idx_t j = i;
            while (true) {
                idx_t src = perm[j];
                visited[j] = true;
                if (src == i) {
                    memcpy(store->codes.data() + j * store->code_size,
                           tmp.data(), store->code_size);
                    break;
                }
                memcpy(store->codes.data() + j * store->code_size,
                       store->codes.data() + src * store->code_size,
                       store->code_size);
                j = src;
            }
        }
        // is_identity_map 仍为 true
    } else {
        // 模式 B：只重排间接层（当前实现），不动 store 数据
        std::vector<idx_t> new_map(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            new_map[i] = storage_id_map[perm[i]];
        }
        std::swap(storage_id_map, new_map);
    }
}
```

**模式 A（compact 后）的开销：**
- 临时内存：O(code_size) = 1 个向量
- 时间：O(n_alive × code_size) — 与 compact_store 相同
- 保持 `is_identity_map = true`：搜索性能最优

**模式 B（未 compact）的开销：**
- 临时内存：O(n × sizeof(idx_t)) = n × 8 字节
- 时间：O(n) — 极快
- `is_identity_map` 变为 false（如果之前为 true 需要设为 false）

### 10.5 可用的图重排序策略

当 `is_identity_map = true`（compact 后），以下图重排序策略可以显著提升搜索性能（通过改善 HNSW 图遍历的内存访问局部性）：

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| **BFS** | 从 entry_point 做 BFS，按访问顺序排列节点 | 通用，效果稳定 |
| **RCM (Reverse Cuthill-McKee)** | 选择 peripheral 起点，BFS 时按度数升序排列邻居，最后反转 | 减少矩阵带宽，cache 友好 |
| **DFS** | 深度优先遍历排列 | 对某些拓扑有优势 |
| **Cluster** | 按 HNSW level 降序排列，每个节点紧跟其邻居 | 高层节点优先 |
| **Weighted** | 按 `(1 + level) × degree` 评分降序排列 | 高重要性节点优先 |

**permutation 生成函数**（均在 benchmark 代码中实现，可提取为库函数）：
- `generate_bfs_permutation(const HNSW& hnsw)`
- `generate_rcm_permutation(const HNSW& hnsw)`
- `generate_dfs_permutation(const HNSW& hnsw)`
- `generate_cluster_permutation(const HNSW& hnsw)`
- `generate_weighted_permutation(const HNSW& hnsw)`

所有函数返回 `vector<idx_t> perm`，语义为 `perm[new_id] = old_id`。

**调用方式：**

```cpp
// compact 后，is_identity_map = true
auto perm = generate_bfs_permutation(new_index->hnsw);
new_index->permute_entries(perm.data());
// → IndexFlatShared::permute_entries 走模式 A（物理搬移）
// → HNSW::permute_entries 重排图结构
// → is_identity_map 仍为 true
```

### 10.6 新增向量对 `is_identity_map` 的影响

Compact + reorder 后 `is_identity_map = true`。后续 `add()` 操作的行为：

1. **`free_list` 为空（正常情况）：** `allocate_slot()` 走 append 路径，返回 `slot = ntotal_store`。此时 `storage_id_map.push_back(ntotal_store)`，新条目仍满足 `storage_id_map[i] == i`。**`is_identity_map` 保持 true。**

2. **`free_list` 不为空（先 delete 再 reclaim 后 add）：** `allocate_slot()` 复用 free_list 中的旧 slot，返回的 `slot` 不等于当前 `ntotal`。此时 `storage_id_map[ntotal] = slot`，不满足恒等。**`is_identity_map` 必须设为 false。**

**实现：** 在 `IndexFlatShared::add()` 中检测：

```cpp
void IndexFlatShared::add(idx_t n, const float* x) {
    for (idx_t i = 0; i < n; i++) {
        idx_t slot = store->allocate_slot(x + i * d);
        if (is_identity_map && slot != (idx_t)(ntotal + i)) {
            is_identity_map = false;
        }
        storage_id_map.push_back(slot);
        // ... bitmap resize ...
    }
    ntotal += n;
}
```

### 10.7 完整生命周期（含 compact + reorder）

```
Phase 0: 首次构建
  → storage_id_map = [0,1,...,n-1], is_identity_map = true
  → 搜索走 identity fast path

Phase 1: 正常运行 + 标记删除 + 添加新向量
  → is_identity_map 可能变为 false（如果有 free_list 复用）

Phase 2: Rebuild
  → build_new_index() → 新 storage_id_map 非恒等, is_identity_map = false
  → 新索引开始服务（间接寻址模式）

Phase 3: Compact + Reorder（write-lock 内）
  → compact_store() → is_identity_map = true
  → permute_entries(bfs_perm) → 图 + store 重排, is_identity_map 仍 true
  → 搜索恢复 identity fast path + 图局部性优化

Phase 4: 下一轮正常运行
  → add/delete → 可能 is_identity_map 变 false
  → 再次 rebuild → 回到 Phase 2
```

### 10.8 不变量补充

| 不变量 | 说明 |
|--------|------|
| **compact 后 `store[i]` == `local_id i` 的向量** | cycle-following 保证 |
| **compact 后 `store.ntotal_store == n_alive`** | store 被截断 |
| **compact 后 `free_list` 为空** | 所有 slot 都被占用 |
| **`is_identity_map = true` 时 `permute_entries` 物理搬移数据** | 保持恒等性 |
| **`is_identity_map = true` 时 `resolve()` 返回 `i` 本身** | 跳过间接寻址 |
| **compact 必须在旧索引销毁后执行** | 避免并发读损坏 |

## 12. 序列化 / 反序列化

### 12.1 设计原则

共享存储的序列化需要处理两个问题：
1. **store 数据只写一份**（不要两个索引各存一份）
2. **加载时恢复共享关系**

### 12.2 序列化方案

```
文件布局:

  [SharedVectorStore]
    magic: "SVST"
    d, ntotal_store, code_size
    codes: ntotal_store * code_size bytes
    free_list_size, free_list: free_list_size * 8 bytes

  [IndexHNSW #1 (old)]
    magic: "IHSW"
    store_ref: SHARED (标记引用共享存储)
    storage_id_map: ntotal * 8 bytes
    deleted_bitmap: ceil(ntotal_store/64) * 8 bytes
    hnsw graph data...

  [IndexHNSW #2 (new)]
    magic: "IHSW"
    store_ref: SHARED
    storage_id_map: ntotal * 8 bytes
    deleted_bitmap: ceil(ntotal_store/64) * 8 bytes  (全 0)
    hnsw graph data...
```

**加载流程：**
1. 先加载 `SharedVectorStore`
2. 加载每个 `IndexHNSW`，其 storage 创建为 `IndexFlatShared`，指向同一个 store

> **简化选项：** 如果不需要同时序列化两个索引，可以只序列化新索引。新索引的 `IndexFlatShared` 序列化时将 store 数据内联写出（退化为普通 IndexFlat 的序列化），加载时退化为独立索引。这样不需要修改 faiss 的序列化框架。

## 13. 文件清单与实现计划

### 13.1 新增文件

| 文件 | 内容 |
|------|------|
| `faiss/SharedVectorStore.h` | `SharedVectorStore` 定义 |
| `faiss/SharedVectorStore.cpp` | `SharedVectorStore` 实现、`build_storage_id_map`、`reclaim_deleted_slots` |
| `faiss/IndexFlatShared.h` | `IndexFlatShared` 定义 |
| `faiss/IndexFlatShared.cpp` | `IndexFlatShared` 实现、`IndirectFlatL2Dis`、`IndirectFlatIPDis` |
| `tests/test_shared_storage.cpp` | 单元测试 |
| `benchs/bench_shared_storage.cpp` | 性能基准测试 |

### 13.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `faiss/IndexHNSW.cpp` | `hnsw_add_vertices`: `x==nullptr` 时从 store 直接读取向量指针（~8 行）；`IndexHNSW::add`: `x==nullptr` 时跳过 `storage->add()`（~5 行）。详见 §9 |
| `faiss/CMakeLists.txt` | 添加新文件编译 |

### 13.3 实现顺序

```
Phase 1: 核心数据结构 (1-2 天)
  ├── SharedVectorStore (含 free_list、reclaim_deleted_slots)
  ├── IndexFlatShared (含 storage_id_map、deleted_bitmap)
  ├── IndirectFlatL2Dis / IndirectFlatIPDis
  └── 单元测试: 映射正确性、距离计算一致性

Phase 2: 零拷贝构建 + 索引构建流程 (1-2 天)
  ├── 修改 hnsw_add_vertices: x==nullptr 零拷贝路径（§9.2）
  ├── 修改 IndexHNSW::add: x==nullptr 跳过 storage->add（§9.3）
  ├── build_storage_id_map()
  ├── build_new_index()
  └── 端到端测试: 首次构建 → 删除 → rebuild → swap → reclaim → 再次使用

Phase 3: 性能验证 (1 天)
  ├── 基准测试: 间接层开销测量
  ├── 对比: 共享存储 vs 传统复制
  ├── 峰值内存验证（应 ≈ 稳态，无临时缓冲区）
  └── batch_4/8 / prefetch / 图重排序兼容性验证
```

## 14. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 间接寻址 cache miss 率上升 | 低 | 搜索 QPS 下降 2-5% | id_map 是顺序数组，几乎必然 L1 命中；向量访问本身就是随机的 |
| `hnsw_add_vertices` 未来 faiss 版本接口变更 | 低 | 需要适配 | 封装为独立函数，降低耦合；修改量仅 ~13 行，适配成本低 |
| `permute_entries` 在图重排序时行为不符预期 | 低 | 重排序后搜索结果错误 | 单元测试覆盖：重排序前后搜索结果一致 |
| 序列化/反序列化破坏共享关系 | 中 | 加载后变成两份独立数据 | v1 先不支持双索引联合序列化，仅支持单索引独立序列化 |
| `codes.resize()` 导致搜索线程指针失效 | 低 | 搜索结果异常或 crash | `codes.reserve()` 预分配；或通过读写锁保证 resize 时无搜索线程活跃（§7.1） |

## 15. 开放问题

1. **是否需要支持 >2 个索引共享同一个 store？** 当前设计天然支持（`shared_ptr` 可以被任意数量的 `IndexFlatShared` 持有），但未做专门测试。

2. ~~**Compaction 策略**~~ → **已解决。** 通过 §10 的 `compact_store()` 实现 rebuild 后的 store 压缩 + 恒等化。compact 使用原地 cycle-following 算法，临时内存仅 O(code_size)（一个向量），满足零拷贝约束。compact 后 `is_identity_map = true`，消除间接寻址开销，并可进一步应用图重排序（BFS/RCM 等）提升搜索性能。

3. ~~**增量构建**~~ → **已解决。** `IndexFlatShared::add()` 通过 `store->allocate_slot()` 支持新向量添加（优先复用 free_list 空洞，否则 append）。`storage_id_map` 自动追加新映射。HNSW 图通过 `IndexHNSW::add()` 正常扩展。详见 §3.2 和 §6.3 Phase 3。

4. **与 IndexIDMap 的交互：** 如果用户使用了 `IndexIDMap` 包装，外部 ID → 内部 ID 的映射如何与 `storage_id_map` 协调？当前设计中 `storage_id_map` 是 HNSW local_id → store slot 的映射，与 IndexIDMap 的 external_id → internal_id 映射是正交的两层。IndexIDMap 包装在 IndexHNSW 外面，不感知 storage 层。

5. **`codes.resize()` 导致的指针失效问题：** `allocate_slot()` 在 free_list 为空时 append 会触发 `codes.resize()`，使所有指向 `codes.data()` 的指针/引用失效。这包括搜索线程的 `DistanceComputer::b` 指针。**注意：此问题仅出现在运行时 `add()` 路径，不会出现在 rebuild 期间**——rebuild 期间旧索引只有读操作，新索引通过零拷贝构建不会触发 `codes.resize()`（所有 slot 已预先存在于 store 中）。运行时解决方案：(a) `codes.reserve()` 预分配足够空间，或 (b) 通过读写锁保证 resize 时无搜索线程活跃（见 §7.1）。

6. **store 碎片率监控：** 是否需要提供 `fragmentation_ratio() = free_list.size() / ntotal_store` 接口，帮助用户决定何时触发 rebuild 或 compaction？
