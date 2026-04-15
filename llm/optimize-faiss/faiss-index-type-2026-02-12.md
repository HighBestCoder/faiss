# FAISS Index Type & Storage Type — Vector Database API 设计文档

> **项目**: 基于 FAISS 的向量数据库  
> **版本**: v1.0  
> **日期**: 2026-02-12  
> **范围**: 如何通过 API 将 FAISS 的 Index Type 和 Storage Type 暴露给用户

---

## 目录

1. [设计目标](#1-设计目标)
2. [核心概念拆解](#2-核心概念拆解)
3. [API 设计：双层模型](#3-api-设计双层模型)
4. [Simple Mode API](#4-simple-mode-api)
5. [Advanced Mode API](#5-advanced-mode-api)
6. [合法性约束矩阵](#6-合法性约束矩阵)
7. [参数自动推导规则](#7-参数自动推导规则)
8. [GPU 支持](#8-gpu-支持)
9. [API 到 FAISS index_factory 的映射](#9-api-到-faiss-index_factory-的映射)
10. [内存与性能估算](#10-内存与性能估算)
11. [推荐的 12 个核心组合](#11-推荐的-12-个核心组合)
12. [不暴露给用户的索引类型](#12-不暴露给用户的索引类型)
13. [API Schema 定义](#13-api-schema-定义)
14. [使用示例](#14-使用示例)
15. [附录：FAISS 索引技术参考](#15-附录faiss-索引技术参考)

---

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| **简洁性** | 普通用户无需理解 FAISS 内部概念，通过策略卡一键选择 |
| **灵活性** | 高级用户可精细控制搜索结构、编码方式、预处理和精炼 |
| **安全性** | API 层拦截非法组合，不让用户构造无效索引 |
| **可预测性** | 提供内存估算和性能特征，帮助用户决策 |
| **全规模** | 覆盖从几千到数十亿的数据规模 |
| **CPU + GPU** | 支持 CPU 和 GPU 两种硬件后端 |

---

## 2. 核心概念拆解

FAISS 的索引可以拆解为 **4 个正交维度**，这是 API 设计的基础：

```
┌──────────────────────────────────────────────────────────────────┐
│                    FAISS 索引 = 4 个维度的组合                     │
│                                                                  │
│  ① 搜索结构 (Search Structure)                                   │
│     决定"怎么找"——搜索算法                                        │
│     Flat（暴力搜索）/ IVF（分区搜索）/ HNSW（图搜索）               │
│                                                                  │
│  ② 存储编码 (Storage Encoding)                                   │
│     决定"怎么存"——向量的压缩方式                                   │
│     Flat（原始float）/ SQ（标量量化）/ PQ（乘积量化）/ RaBitQ       │
│                                                                  │
│  ③ 预处理 (Preprocessing) [可选]                                  │
│     在存入索引前对向量做变换                                       │
│     无 / OPQ（优化旋转）/ PCA（降维）/ ITQ / L2Norm               │
│                                                                  │
│  ④ 精炼 (Refinement) [可选]                                      │
│     用原始向量重排初步搜索结果                                     │
│     无 / RFlat（用原始向量精确重排）                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 搜索结构详解

| 搜索结构 | 算法 | 复杂度 | 是否精确 | 需训练 | GPU |
|---------|------|--------|---------|--------|-----|
| **Flat** | 暴力线性扫描，逐一比较所有向量 | O(N × d) | 精确 (100%) | 否 | 支持 |
| **IVF** | k-means 分区 → 只搜索 nprobe 个分区 | O(nprobe × N/nlist × d) | 近似 | 是 | 支持 |
| **HNSW** | 多层导航图，贪心遍历 + 束搜索 | O(efSearch × M × logN) | 近似 | 否 | 不支持 |

### 2.2 存储编码详解

| 编码方式 | 每向量内存 (d=128) | 精度损失 | 需训练 | 说明 |
|---------|-------------------|---------|--------|------|
| **Flat** | 512 B (4×d) | 无 | 否 | 原始 float32，零损失 |
| **SQ8** | 128 B (1×d) | 极小 | 是 | 每维量化到 8-bit，4x 压缩 |
| **SQ4** | 64 B (0.5×d) | 小 | 是 | 每维量化到 4-bit，8x 压缩 |
| **PQ** | M×8 bits | 中等 | 是 | 乘积量化，高压缩比 |
| **RaBitQ** | d/8 B = 16 B | 小 | 是 | 随机二值量化，32x 压缩 |

### 2.3 "Flat" 在不同语境中的含义

> **关键澄清：** "Flat" 在 FAISS 名字里有两种完全不同的含义。

| 索引名 | "Flat" 含义 | 搜索方式 |
|--------|------------|---------|
| Index**Flat** | 整个索引是平铺暴力搜索 | **暴力搜索** |
| IndexHNSW**Flat** | 向量以原始 float 存储（不压缩） | **图遍历**（近似） |
| IndexIVF**Flat** | 倒排表内向量以原始 float 存储 | **分区搜索**（近似） |

---

## 3. API 设计：双层模型

```
┌─────────────────────────────────────────────────────────┐
│                     用户 API 层                          │
│                                                         │
│  ┌──────────────┐          ┌──────────────────────┐     │
│  │ Simple Mode  │          │   Advanced Mode      │     │
│  │ (策略卡选择)  │          │  (4维度精细配置)      │     │
│  │              │          │                      │     │
│  │ 🎯 精确      │          │  search_type: IVF    │     │
│  │ ⚡ 均衡      │          │  encoding: PQ16x8    │     │
│  │ 🚀 高速      │          │  preprocess: OPQ16   │     │
│  │ 💾 省内存    │          │  refine: RFlat       │     │
│  └──────┬───────┘          └──────────┬───────────┘     │
│         │                             │                  │
│         └──────────┬──────────────────┘                  │
│                    ▼                                     │
│         ┌──────────────────┐                            │
│         │ 参数推导 + 校验   │                            │
│         │ (合法性矩阵检查)  │                            │
│         └────────┬─────────┘                            │
│                  ▼                                       │
│         ┌──────────────────┐                            │
│         │ index_factory()  │                            │
│         │ 字符串生成        │                            │
│         └────────┬─────────┘                            │
│                  ▼                                       │
│         ┌──────────────────┐                            │
│         │ FAISS Index 实例  │                            │
│         └──────────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Simple Mode API

### 4.1 用户界面

4 张策略卡，用户选一张即可：

| 策略 | 图标 | 一句话描述 | 特点 |
|------|------|-----------|------|
| **精确搜索** | 🎯 | 100% 召回，绝对精确 | 小数据集首选 |
| **均衡搜索** | ⚡ | 高速 + 高召回（>95%） | 中等规模首选 |
| **极速搜索** | 🚀 | 最快响应，轻微精度损失 | 大规模 + 低延迟 |
| **省内存搜索** | 💾 | 最小内存占用 | 大规模 + 内存受限 |

### 4.2 API 请求

```json
POST /api/v1/collections
{
    "name": "my_collection",
    "dimension": 768,
    "metric": "L2",
    "strategy": "balanced",
    "estimated_vectors": 5000000,
    "hardware": "cpu"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `strategy` | enum | 是 | `"precise"` / `"balanced"` / `"fast"` / `"memory_efficient"` |
| `dimension` | int | 是 | 向量维度 |
| `metric` | enum | 是 | `"L2"` / `"IP"` (内积) / `"cosine"` |
| `estimated_vectors` | int | 否 | 预估向量数量，影响参数自动推导 |
| `hardware` | enum | 否 | `"cpu"` (默认) / `"gpu"` |

### 4.3 策略到 FAISS 配置的自动映射

系统后端根据 `strategy` + `estimated_vectors` + `hardware` 自动选择最优配置：

#### 精确搜索 (`precise`)

| 数据规模 | index_factory 字符串 | 说明 |
|---------|---------------------|------|
| 任意 | `"Flat"` | 暴力搜索，100% 召回 |

#### 均衡搜索 (`balanced`)

| 数据规模 | index_factory 字符串 | 说明 |
|---------|---------------------|------|
| < 5 万 | `"Flat"` | 数据少时暴力搜索已足够快 |
| 5万 ~ 100万 | `"HNSW32,Flat"` | 图搜索 + 无压缩 |
| 100万 ~ 1亿 | `"IVF{nlist},Flat"` | 分区搜索 + 无压缩 |
| > 1亿 | `"IVF{nlist}_HNSW32,Flat"` | HNSW 作为粗量化器的 IVF |

#### 极速搜索 (`fast`)

| 数据规模 | index_factory 字符串 | 说明 |
|---------|---------------------|------|
| < 5 万 | `"Flat"` | 数据少时暴力搜索已足够快 |
| 5万 ~ 100万 | `"HNSW32,SQ8"` | 图搜索 + 标量量化 |
| 100万 ~ 1亿 | `"IVF{nlist},PQ{M}x8"` | 分区 + 乘积量化 |
| > 1亿 | `"OPQ{M},IVF{nlist}_HNSW32,PQ{M}x8"` | 预处理 + HNSW粗量化 + PQ |

#### 省内存搜索 (`memory_efficient`)

| 数据规模 | index_factory 字符串 | 说明 |
|---------|---------------------|------|
| < 5 万 | `"Flat"` | 数据少时无需压缩 |
| 5万 ~ 100万 | `"IVF{nlist},SQ4"` | 分区 + 4-bit 量化 |
| 100万 ~ 1亿 | `"OPQ{M},IVF{nlist},PQ{M}x4"` | 预处理 + 分区 + PQ4 |
| > 1亿 | `"OPQ{M},IVF{nlist}_HNSW32,PQ{M}x4"` | 全套压缩 |

### 4.4 API 响应

```json
{
    "collection_id": "col_abc123",
    "name": "my_collection",
    "strategy": "balanced",
    "resolved_config": {
        "search_type": "HNSW",
        "encoding": "Flat",
        "preprocessing": null,
        "refinement": null,
        "index_factory_string": "HNSW32,Flat",
        "requires_training": false
    },
    "estimated_memory": {
        "per_vector_bytes": 768,
        "total_gb": 3.58,
        "breakdown": {
            "vectors": "2.86 GB",
            "graph_structure": "0.72 GB"
        }
    },
    "capabilities": {
        "exact_results": false,
        "supports_deletion": false,
        "supports_gpu": false,
        "supports_incremental_add": true,
        "recall_estimate": "95-99%"
    }
}
```

---

## 5. Advanced Mode API

### 5.1 四维度配置

用户显式指定 4 个维度的配置：

```json
POST /api/v1/collections
{
    "name": "my_collection",
    "dimension": 768,
    "metric": "L2",
    "index_config": {
        "search_type": "IVF",
        "encoding": "PQ48x8",
        "preprocessing": "OPQ48",
        "refinement": "RFlat",
        "hardware": "cpu",
        "params": {
            "nlist": 4096,
            "nprobe": 64,
            "M": 32,
            "efSearch": 128,
            "efConstruction": 200
        }
    }
}
```

### 5.2 维度一：搜索结构 (`search_type`)

| 值 | 对应 FAISS | 需训练 | GPU | 删除 | 适用规模 |
|----|-----------|--------|-----|------|---------|
| `"Flat"` | IndexFlat / IndexFlatL2 / IndexFlatIP | 否 | 是 | 是 | < 10万 |
| `"IVF"` | IndexIVF 系列 | 是 | 是 | 是 | 10万 ~ 数十亿 |
| `"HNSW"` | IndexHNSW 系列 | 否 | 否 | 否 | 10万 ~ 数千万 |

**IVF 粗量化器选项**（当 `search_type = "IVF"` 时）：

| `ivf_quantizer` | 说明 | 适用场景 |
|-----------------|------|---------|
| `"Flat"` (默认) | 暴力搜索粗量化 | nlist < 16384 |
| `"HNSW32"` | HNSW 作为粗量化器 | nlist >= 16384，大规模 |

### 5.3 维度二：存储编码 (`encoding`)

| 值 | 格式 | 每向量内存 (d=128) | 精度 | 需训练 |
|----|------|-------------------|------|--------|
| `"Flat"` | 原始 float32 | 512 B | 无损 | 否 |
| `"SQ8"` | 标量量化 8-bit | 128 B | 极高 | 是 |
| `"SQ4"` | 标量量化 4-bit | 64 B | 高 | 是 |
| `"SQfp16"` | 半精度浮点 | 256 B | 极高 | 是 |
| `"PQ{M}x{nbits}"` | 乘积量化 | M×nbits/8 B | 中等 | 是 |
| `"RaBitQ"` | 随机二值量化 | d/8 B | 高 | 是 |

**PQ 参数说明**：
- `M`: 子空间数量，必须能整除 `d`。常用值：`d/4`、`d/8`。
- `nbits`: 每子空间量化比特数，常用 `4` 或 `8`。

### 5.4 维度三：预处理 (`preprocessing`)

| 值 | 说明 | 最佳搭配 | 需训练 |
|----|------|---------|--------|
| `null` | 无预处理 | — | — |
| `"OPQ{M}"` | 优化乘积量化旋转 | PQ 编码 | 是 |
| `"PCA{d_out}"` | PCA 降维 | 高维向量 | 是 |
| `"PCAR{d_out}"` | PCA 降维 + 随机旋转 | PQ 编码 | 是 |
| `"ITQ{d_out}"` | 迭代量化旋转 | 二值编码 | 是 |
| `"L2Norm"` | L2 归一化 | cosine 度量 | 否 |

### 5.5 维度四：精炼 (`refinement`)

| 值 | 说明 | 内存代价 | 精度提升 |
|----|------|---------|---------|
| `null` | 无精炼 | 0 | — |
| `"RFlat"` | 用原始向量重排 top-k' | +4×d B/vec | 显著 |

精炼机制：先用压缩编码快速搜出 k' > k 个候选，再用原始向量精确计算距离，返回最终 top-k。额外内存开销 = 需存储一份原始向量。

---

## 6. 合法性约束矩阵

### 6.1 搜索结构 × 存储编码

| | Flat | SQ8 | SQ4 | SQfp16 | PQ | RaBitQ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Flat** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **IVF** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **HNSW** | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |

**说明**：
- **Flat 搜索** 只能搭配 Flat 编码（暴力搜索的意义就是精确）
- **HNSW** 不支持 RaBitQ（FAISS 没有 IndexHNSWRaBitQ 实现）
- **HNSW** 不支持 SQ4 和 SQfp16（仅支持 SQ8）
- **IVF** 支持所有编码类型

### 6.2 预处理约束

| 预处理 | 适用编码 | 约束 |
|--------|---------|------|
| `OPQ{M}` | PQ | M 值必须与 PQ 的 M 一致 |
| `PCA{d_out}` | 任意 | d_out < d |
| `PCAR{d_out}` | PQ | d_out < d，通常 d_out = M × 子空间维度 |
| `L2Norm` | 任意 | 仅用于 cosine 度量转 IP |

### 6.3 精炼约束

| 精炼 | 适用编码 | 说明 |
|------|---------|------|
| `RFlat` | SQ / PQ / RaBitQ | 仅在有损编码时有意义 |
| `RFlat` | Flat | ❌ 无意义（已经是精确距离） |

### 6.4 GPU 约束

| 搜索结构 | GPU 支持 |
|---------|---------|
| Flat | ✅ |
| IVF + Flat/SQ/PQ | ✅ |
| IVF + RaBitQ | ✅ (FastScan 变体) |
| HNSW | ❌ |

---

## 7. 参数自动推导规则

当用户未显式指定参数时，系统按以下规则自动推导：

### 7.1 IVF 参数

| 参数 | 推导规则 | 说明 |
|------|---------|------|
| `nlist` | `4 × sqrt(N)` ~ `16 × sqrt(N)` | N = 预估向量数 |
| `nprobe` | `max(1, nlist / 16)` | 默认搜索 6.25% 的分区 |
| `ivf_quantizer` | nlist < 16384 → `"Flat"`, 否则 `"HNSW32"` | 粗量化器类型 |

**nlist 推导表**：

| N (预估向量数) | 推荐 nlist | 推荐 nprobe |
|---------------|-----------|------------|
| 10万 | 256 ~ 1024 | 16 ~ 64 |
| 100万 | 1024 ~ 4096 | 64 ~ 256 |
| 1000万 | 4096 ~ 16384 | 256 ~ 1024 |
| 1亿 | 16384 ~ 65536 | 1024 ~ 4096 |
| 10亿 | 65536 ~ 262144 | 4096 ~ 16384 |

### 7.2 HNSW 参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `M` | 32 | 4 ~ 64 | 邻居数。越大越准、越耗内存 |
| `efConstruction` | 40 ~ 200 | 16 ~ 500 | 建图搜索宽度 |
| `efSearch` | 64 | 16 ~ 512 | 搜索时束宽度 |

### 7.3 PQ 参数

| 参数 | 推导规则 | 说明 |
|------|---------|------|
| `M` | `d / 4`（均衡）或 `d / 8`（省内存） | 子空间数量，须整除 d |
| `nbits` | 默认 8 | 4 或 8 |

### 7.4 SQ 参数

自动推导，无用户可调参数（量化范围从训练数据学习）。

---

## 8. GPU 支持

### 8.1 GPU 兼容索引

| index_factory 字符串 | GPU 支持 | 说明 |
|---------------------|---------|------|
| `"Flat"` | ✅ | GpuIndexFlat |
| `"IVF{K},Flat"` | ✅ | GpuIndexIVFFlat |
| `"IVF{K},SQ8"` | ✅ | GpuIndexIVFScalarQuantizer |
| `"IVF{K},PQ{M}x8"` | ✅ | GpuIndexIVFPQ |
| `"HNSW*"` | ❌ | 无 GPU 实现 |

### 8.2 API 行为

当 `hardware = "gpu"` 时：
1. 验证所选索引类型是否支持 GPU
2. 若不支持，返回错误并建议替代方案
3. 若支持，使用 `index_cpu_to_gpu()` 或直接创建 GpuIndex

```json
{
    "error": "INCOMPATIBLE_HARDWARE",
    "message": "HNSW indexes do not support GPU. Alternatives: IVF+Flat, IVF+PQ",
    "suggestions": [
        {"search_type": "IVF", "encoding": "Flat", "index_factory": "IVF4096,Flat"},
        {"search_type": "IVF", "encoding": "PQ48x8", "index_factory": "IVF4096,PQ48x8"}
    ]
}
```

---

## 9. API 到 FAISS index_factory 的映射

### 9.1 映射规则

```
index_factory_string = [preprocessing + ","] + search_structure + "," + encoding [+ ",RFlat"]
```

具体拼接逻辑：

```python
def build_index_factory_string(config):
    parts = []
    
    # 1. 预处理
    if config.preprocessing:
        parts.append(config.preprocessing)  # e.g. "OPQ48"
    
    # 2. 搜索结构
    if config.search_type == "Flat":
        # Flat 搜索不需要显式写 "Flat,"，直接用编码
        pass  
    elif config.search_type == "IVF":
        ivf_str = f"IVF{config.params.nlist}"
        if config.params.ivf_quantizer == "HNSW32":
            ivf_str += "_HNSW32"
        parts.append(ivf_str)  # e.g. "IVF4096" or "IVF65536_HNSW32"
    elif config.search_type == "HNSW":
        parts.append(f"HNSW{config.params.M}")  # e.g. "HNSW32"
    
    # 3. 编码
    parts.append(config.encoding)  # e.g. "Flat", "SQ8", "PQ48x8"
    
    # 4. 精炼
    if config.refinement == "RFlat":
        parts.append("RFlat")
    
    return ",".join(parts)
```

### 9.2 映射示例

| 用户配置 | index_factory 字符串 |
|---------|---------------------|
| search=Flat, encoding=Flat | `"Flat"` |
| search=HNSW(M=32), encoding=Flat | `"HNSW32,Flat"` |
| search=HNSW(M=32), encoding=SQ8 | `"HNSW32,SQ8"` |
| search=HNSW(M=32), encoding=PQ48x8 | `"HNSW32,PQ48x8"` |
| search=IVF(4096), encoding=Flat | `"IVF4096,Flat"` |
| search=IVF(4096), encoding=SQ8 | `"IVF4096,SQ8"` |
| search=IVF(4096), encoding=PQ48x8 | `"IVF4096,PQ48x8"` |
| search=IVF(65536, HNSW32), encoding=PQ48x8 | `"IVF65536_HNSW32,PQ48x8"` |
| preprocess=OPQ48, search=IVF(4096), encoding=PQ48x8 | `"OPQ48,IVF4096,PQ48x8"` |
| preprocess=OPQ48, search=IVF(4096), encoding=PQ48x8, refine=RFlat | `"OPQ48,IVF4096,PQ48x8,RFlat"` |
| search=IVF(1024), encoding=RaBitQ | `"IVF1024,RaBitQ"` |

---

## 10. 内存与性能估算

### 10.1 内存估算公式

API 返回每种配置的内存估算，帮助用户决策：

| 组件 | 内存公式 |
|------|---------|
| Flat 编码 | `N × d × 4` 字节 |
| SQ8 编码 | `N × d × 1` 字节 |
| SQ4 编码 | `N × d × 0.5` 字节 |
| PQ{M}x{b} 编码 | `N × M × b / 8` 字节 |
| RaBitQ 编码 | `N × d / 8` 字节 |
| IVF 开销 | `nlist × d × 4 + N × 8` 字节 (质心 + ID) |
| HNSW 图 | `N × M × 2 × 4` 字节 |
| RFlat 精炼 | `+ N × d × 4` 字节 (额外存储原始向量) |

### 10.2 性能特征参考 (SIFT1M, d=128)

| 配置 | 查询延迟 | Recall@1 | 内存/向量 |
|------|---------|----------|----------|
| Flat | 9.1 s / 10K queries | 100% | 512 B |
| HNSW32,Flat (efSearch=64) | 0.033 ms | 97.8% | ~768 B |
| IVF4096,Flat (nprobe=64) | 0.141 ms | 94.7% | ~520 B |
| IVF4096,PQ32x8 (nprobe=64) | ~0.08 ms | ~85% | ~40 B |
| IVF4096,SQ8 (nprobe=64) | ~0.10 ms | ~93% | ~136 B |

---

## 11. 推荐的 12 个核心组合

以下为建议暴露给用户的核心索引配置，覆盖绝大多数场景：

| # | 组合名称 | index_factory | 适用规模 | 训练 | GPU | 特点 |
|---|---------|---------------|---------|------|-----|------|
| 1 | 精确搜索 | `Flat` | < 10万 | 否 | 是 | 100% 召回 |
| 2 | 图搜索 | `HNSW32,Flat` | < 500万 | 否 | 否 | 高速高召回 |
| 3 | 图搜索(压缩) | `HNSW32,SQ8` | < 1000万 | 是 | 否 | 省内存图搜索 |
| 4 | 分区精确 | `IVF{K},Flat` | < 1亿 | 是 | 是 | GPU首选 |
| 5 | 分区量化 | `IVF{K},SQ8` | < 1亿 | 是 | 是 | 均衡 |
| 6 | 分区高压缩 | `IVF{K},PQ{M}x8` | < 10亿 | 是 | 是 | 大规模 |
| 7 | 旋转+分区高压缩 | `OPQ{M},IVF{K},PQ{M}x8` | < 10亿 | 是 | 是 | PQ 最佳实践 |
| 8 | 分区极致压缩 | `OPQ{M},IVF{K},PQ{M}x4` | > 1亿 | 是 | 是 | 极致省内存 |
| 9 | 分区+精炼 | `IVF{K},PQ{M}x8,RFlat` | < 10亿 | 是 | 是 | 压缩+精确重排 |
| 10 | 分区二值量化 | `IVF{K},RaBitQ` | < 1亿 | 是 | 是 | 超高压缩比 |
| 11 | 大规模分区 | `IVF{K}_HNSW32,Flat` | > 1亿 | 是 | 是 | HNSW粗量化 |
| 12 | 大规模压缩 | `OPQ{M},IVF{K}_HNSW32,PQ{M}x8` | > 1亿 | 是 | 是 | 全套大规模方案 |

---

## 12. 不暴露给用户的索引类型

以下 FAISS 索引类型不建议通过 API 暴露，原因如下：

| 索引类型 | 不暴露原因 |
|---------|-----------|
| **IndexLSH** | 效果差，已被 RaBitQ 替代 |
| **IndexLattice** | 实验性质，不成熟 |
| **Index2Layer** | 内部组件，不适合直接使用 |
| **IndexNSG** | HNSW 在多数场景更优 |
| **IndexNNDescent** | 图构建辅助工具，非搜索索引 |
| **NeuralNetCodec** | 实验性质 |
| **SVS/Vamana** | 需要特殊编译 (`FAISS_ENABLE_SVS`) |
| **IndexFlatCompressed** | 边缘用例，增加用户理解成本 |
| **IndexIVFSpectralHash** | 效果不稳定 |
| **PRQ / PLSQ** | 渐进式残差量化，太小众 |
| **IndexIVFFlatPanorama** | 实验性优化变体 |
| **IndexRefine** (非 RFlat) | 复杂度高，RFlat 已足够 |
| **IndexRowwiseMinMax** | 预处理包装器，可内部使用 |
| **IndexSplitVectors** | 底层实现细节 |
| **IndexBinary 系列** | 二值索引是独立场景，需要单独 API |

---

## 13. API Schema 定义

### 13.1 创建 Collection (Simple Mode)

```yaml
CreateCollectionSimple:
  type: object
  required: [name, dimension, metric, strategy]
  properties:
    name:
      type: string
      description: Collection 名称
    dimension:
      type: integer
      minimum: 1
      maximum: 65536
      description: 向量维度
    metric:
      type: string
      enum: [L2, IP, cosine]
      description: 距离度量类型
    strategy:
      type: string
      enum: [precise, balanced, fast, memory_efficient]
      description: 搜索策略
    estimated_vectors:
      type: integer
      minimum: 0
      description: 预估向量数量（用于参数自动推导）
    hardware:
      type: string
      enum: [cpu, gpu]
      default: cpu
      description: 硬件后端
```

### 13.2 创建 Collection (Advanced Mode)

```yaml
CreateCollectionAdvanced:
  type: object
  required: [name, dimension, metric, index_config]
  properties:
    name:
      type: string
    dimension:
      type: integer
      minimum: 1
      maximum: 65536
    metric:
      type: string
      enum: [L2, IP, cosine]
    index_config:
      type: object
      required: [search_type, encoding]
      properties:
        search_type:
          type: string
          enum: [Flat, IVF, HNSW]
          description: 搜索结构
        encoding:
          type: string
          description: |
            存储编码方式：
            - "Flat": 原始 float32
            - "SQ8": 标量量化 8-bit
            - "SQ4": 标量量化 4-bit
            - "SQfp16": 半精度浮点
            - "PQ{M}x{nbits}": 乘积量化 (e.g. "PQ48x8")
            - "RaBitQ": 随机二值量化
        preprocessing:
          type: string
          nullable: true
          description: |
            预处理方式（可选）：
            - null: 无预处理
            - "OPQ{M}": 优化乘积量化旋转
            - "PCA{d_out}": PCA 降维
            - "PCAR{d_out}": PCA + 随机旋转
            - "L2Norm": L2 归一化
        refinement:
          type: string
          nullable: true
          enum: [null, RFlat]
          description: 精炼方式（可选）
        hardware:
          type: string
          enum: [cpu, gpu]
          default: cpu
        params:
          type: object
          description: 搜索参数（未指定则自动推导）
          properties:
            # IVF 参数
            nlist:
              type: integer
              minimum: 1
              description: IVF 聚类数
            nprobe:
              type: integer
              minimum: 1
              description: IVF 搜索时探测的聚类数
            ivf_quantizer:
              type: string
              enum: [Flat, HNSW32]
              default: Flat
              description: IVF 粗量化器类型
            # HNSW 参数
            M:
              type: integer
              minimum: 4
              maximum: 64
              description: HNSW 邻居数
            efConstruction:
              type: integer
              minimum: 16
              maximum: 500
              description: HNSW 建图搜索宽度
            efSearch:
              type: integer
              minimum: 16
              maximum: 512
              description: HNSW 搜索束宽度
```

### 13.3 响应 Schema

```yaml
CollectionResponse:
  type: object
  properties:
    collection_id:
      type: string
    name:
      type: string
    strategy:
      type: string
      nullable: true
      description: Simple Mode 时返回
    resolved_config:
      type: object
      properties:
        search_type:
          type: string
        encoding:
          type: string
        preprocessing:
          type: string
          nullable: true
        refinement:
          type: string
          nullable: true
        index_factory_string:
          type: string
          description: 生成的 FAISS index_factory 字符串
        requires_training:
          type: boolean
          description: 是否需要训练数据
        training_vectors_recommended:
          type: integer
          description: 推荐的训练向量数量
    estimated_memory:
      type: object
      properties:
        per_vector_bytes:
          type: integer
        total_gb:
          type: number
        breakdown:
          type: object
          additionalProperties:
            type: string
    capabilities:
      type: object
      properties:
        exact_results:
          type: boolean
        supports_deletion:
          type: boolean
        supports_gpu:
          type: boolean
        supports_incremental_add:
          type: boolean
        recall_estimate:
          type: string
```

### 13.4 搜索参数调整 API

用户可在运行时动态调整搜索精度/速度平衡：

```yaml
# PATCH /api/v1/collections/{id}/search_params
UpdateSearchParams:
  type: object
  properties:
    nprobe:
      type: integer
      minimum: 1
      description: IVF 搜索探测数（运行时可调）
    efSearch:
      type: integer
      minimum: 1
      description: HNSW 搜索束宽度（运行时可调）
```

**注意**：以下参数在创建后不可修改：
- `search_type`、`encoding`、`preprocessing`、`refinement`（需重建索引）
- `nlist`、`M`、`efConstruction`（需重建索引）

---

## 14. 使用示例

### 14.1 示例 1：小型精确搜索

```bash
# 创建一个精确搜索的 collection（1万条文档嵌入，768维）
curl -X POST /api/v1/collections -d '{
    "name": "faq_embeddings",
    "dimension": 768,
    "metric": "cosine",
    "strategy": "precise"
}'

# 响应
{
    "resolved_config": {
        "search_type": "Flat",
        "encoding": "Flat",
        "index_factory_string": "Flat",
        "requires_training": false
    },
    "estimated_memory": {
        "per_vector_bytes": 3072,
        "total_gb": 0.029
    },
    "capabilities": {
        "exact_results": true,
        "supports_deletion": true,
        "supports_gpu": true
    }
}
```

### 14.2 示例 2：中等规模均衡搜索

```bash
# 500万条产品向量，追求速度和精度平衡
curl -X POST /api/v1/collections -d '{
    "name": "product_vectors",
    "dimension": 256,
    "metric": "L2",
    "strategy": "balanced",
    "estimated_vectors": 5000000
}'

# 响应 → 自动选择 HNSW32,Flat
{
    "resolved_config": {
        "search_type": "HNSW",
        "encoding": "Flat",
        "index_factory_string": "HNSW32,Flat",
        "requires_training": false
    },
    "estimated_memory": {
        "per_vector_bytes": 1280,
        "total_gb": 5.96,
        "breakdown": {
            "vectors": "4.77 GB",
            "graph_structure": "1.19 GB"
        }
    },
    "capabilities": {
        "exact_results": false,
        "supports_deletion": false,
        "supports_gpu": false,
        "recall_estimate": "95-99%"
    }
}
```

### 14.3 示例 3：大规模 GPU + 高压缩

```bash
# 5亿条向量，GPU 加速，尽量省内存
curl -X POST /api/v1/collections -d '{
    "name": "web_embeddings",
    "dimension": 768,
    "metric": "IP",
    "strategy": "memory_efficient",
    "estimated_vectors": 500000000,
    "hardware": "gpu"
}'

# 响应 → OPQ + IVF(HNSW粗量化) + PQ4
{
    "resolved_config": {
        "search_type": "IVF",
        "encoding": "PQ96x4",
        "preprocessing": "OPQ96",
        "index_factory_string": "OPQ96,IVF262144_HNSW32,PQ96x4",
        "requires_training": true,
        "training_vectors_recommended": 2000000
    },
    "estimated_memory": {
        "per_vector_bytes": 56,
        "total_gb": 26.08,
        "breakdown": {
            "pq_codes": "22.35 GB",
            "ivf_overhead": "3.73 GB"
        }
    },
    "capabilities": {
        "exact_results": false,
        "supports_deletion": true,
        "supports_gpu": true,
        "recall_estimate": "80-90%"
    }
}
```

### 14.4 示例 4：Advanced Mode 精细配置

```bash
# 高级用户：IVF + PQ + OPQ预处理 + RFlat精炼
curl -X POST /api/v1/collections -d '{
    "name": "image_features",
    "dimension": 512,
    "metric": "L2",
    "index_config": {
        "search_type": "IVF",
        "encoding": "PQ64x8",
        "preprocessing": "OPQ64",
        "refinement": "RFlat",
        "hardware": "cpu",
        "params": {
            "nlist": 8192,
            "nprobe": 128,
            "ivf_quantizer": "HNSW32"
        }
    }
}'

# 响应
{
    "resolved_config": {
        "search_type": "IVF",
        "encoding": "PQ64x8",
        "preprocessing": "OPQ64",
        "refinement": "RFlat",
        "index_factory_string": "OPQ64,IVF8192_HNSW32,PQ64x8,RFlat",
        "requires_training": true,
        "training_vectors_recommended": 500000
    },
    "estimated_memory": {
        "per_vector_bytes": 2112,
        "total_gb": null,
        "breakdown": {
            "pq_codes": "64 B/vec",
            "refine_vectors": "2048 B/vec (原始向量)"
        }
    },
    "capabilities": {
        "exact_results": false,
        "supports_deletion": true,
        "supports_gpu": true,
        "recall_estimate": "95-99% (with refinement)"
    }
}
```

### 14.5 示例 5：运行时调整搜索参数

```bash
# 提高 IVF 搜索精度（增加 nprobe）
curl -X PATCH /api/v1/collections/col_abc123/search_params -d '{
    "nprobe": 256
}'

# 提高 HNSW 搜索精度（增加 efSearch）
curl -X PATCH /api/v1/collections/col_def456/search_params -d '{
    "efSearch": 256
}'
```

---

## 15. 附录：FAISS 索引技术参考

### 15.1 三大搜索结构算法流程图

```
┌───────────────── IndexFlat (暴力搜索) ─────────────────┐
│                                                        │
│  query ──→ [v1] [v2] [v3] [v4] ... [vN]              │
│             ↑    ↑    ↑    ↑        ↑                 │
│            逐个计算距离，一个都不跳过                     │
│                                                        │
│  扫描量: N (100%)    复杂度: O(N × d)                  │
└────────────────────────────────────────────────────────┘

┌───────────────── IndexHNSW (图遍历搜索) ────────────────┐
│                                                         │
│  Level 2:   [A]──────────[B]          贪心跳跃          │
│              │             │                            │
│  Level 1:   [A]──[C]──[D]─[B]        逐层下降          │
│              │   │    │   │                             │
│  Level 0:   [A]-[E]-[C]-[F]-[D]-[G]-[B]  束搜索       │
│                                                         │
│  扫描量: efSearch × M (~0.1%–1%)                       │
│  复杂度: O(efSearch × M × logN)                        │
└─────────────────────────────────────────────────────────┘

┌───────────── IndexIVF (分区 + 局部搜索) ────────────────┐
│                                                         │
│  ┌─Cell 0─┐ ┌─Cell 1─┐ ┌─Cell 2─┐ ... ┌─Cell K─┐     │
│  │v1 v5   │ │v2 v6   │ │v3 v7   │     │v4 v8   │     │
│  │v9 v13  │ │v10 v14 │ │v11 v15 │     │v12 v16 │     │
│  └────────┘ └────────┘ └────────┘     └────────┘     │
│       ↑          ↑                                     │
│   nprobe=2: 只扫描最近的 2 个 cell                     │
│                                                         │
│  扫描量: nprobe/nlist × N (~1%–10%)                    │
│  复杂度: O(nlist × d + nprobe × N/nlist × d)          │
└─────────────────────────────────────────────────────────┘
```

### 15.2 编码方式内存对比 (d=768, N=100万)

| 编码 | 每向量字节 | 100万向量总计 | 压缩比 |
|------|----------|-------------|--------|
| Flat (float32) | 3,072 B | 2.86 GB | 1x |
| SQfp16 | 1,536 B | 1.43 GB | 2x |
| SQ8 | 768 B | 0.72 GB | 4x |
| SQ4 | 384 B | 0.36 GB | 8x |
| PQ96x8 | 96 B | 91.6 MB | 32x |
| PQ96x4 | 48 B | 45.8 MB | 64x |
| RaBitQ | 96 B | 91.6 MB | 32x |

### 15.3 训练数据要求

| 编码/结构 | 最低训练量 | 推荐训练量 | 说明 |
|---------|-----------|-----------|------|
| IVF (k-means) | 39 × nlist | 100 × nlist | nlist 个聚类中心 |
| SQ | 256 | 10,000+ | 学习量化范围 |
| PQ | 1,000 × M | 10,000 × M | 学习码本 |
| OPQ | 同 PQ | 同 PQ | 学习旋转矩阵 |
| RaBitQ | 1,000 | 10,000+ | 学习二值化参数 |

### 15.4 功能支持矩阵

| 功能 | Flat | HNSW+Flat | IVF+Flat | IVF+PQ | IVF+SQ |
|------|:----:|:---------:|:--------:|:------:|:------:|
| 精确结果 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 训练 | 不需要 | 不需要 | 需要 | 需要 | 需要 |
| 增量添加 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 删除向量 | ✅ | ❌ | ✅ | ✅ | ✅ |
| GPU | ✅ | ❌ | ✅ | ✅ | ✅ |
| reconstruct | ✅ | ✅ | ✅ | 近似 | 近似 |
| add_with_ids | IDMap | IDMap | ✅ | ✅ | ✅ |
| 运行时调参 | — | efSearch | nprobe | nprobe | nprobe |

### 15.5 参考资源

| 资源 | URL |
|------|-----|
| FAISS Wiki — 索引类型 | https://github.com/facebookresearch/faiss/wiki/Faiss-indexes |
| FAISS Wiki — 选择索引指南 | https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index |
| FAISS Wiki — index_factory | https://github.com/facebookresearch/faiss/wiki/The-index-factory |
| FAISS Wiki — 1M 向量基准测试 | https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors |
| HNSW 论文 (Malkov & Yashunin) | https://arxiv.org/abs/1603.09320 |
| FAISS 官方文档 | https://faiss.ai/index.html |
