# Res2CLIP 改造方案：Real-IAD 多视角异常检测比赛

> 基于 Res2CLIP (Liu et al., arXiv:2605.16171, 2026) 的改造方案，
> 适配 "Real-IAD Variety 真实工业多视角异常检测" 比赛。

---

## 目录

1. [比赛需求概览](#1-比赛需求概览)
2. [方案总览](#2-方案总览)
3. [Phase 1 — 数据准备](#3-phase-1--数据准备)
4. [Phase 2 — 单视角 Baseline](#4-phase-2--单视角-baseline)
5. [Phase 3 — 多视角推理](#5-phase-3--多视角推理)
6. [Phase 4 — Mask 上采样 (37×37 → 448×448)](#6-phase-4--mask-上采样-37337--448448)
7. [Phase 5 — 微调适配器（可选）](#7-phase-5--微调适配器可选)
8. [Phase 6 — 提交流程](#8-phase-6--提交流程)
9. [Phase 7 — 本地验证与调优](#9-phase-7--本地验证与调优)
10. [项目文件清单](#10-项目文件清单)
11. [时间规划](#11-时间规划)

---

## 1. 比赛需求概览

| 维度 | 要求 |
|------|------|
| **输入** | 每个样本 5 个视角 (0.png ~ 4.png), 分辨率 ~1800-3600px |
| **输出 1** | `submission.csv` — 每行 (group_folder, anomaly_score) |
| **输出 2** | `predicted_masks/` — 每个视角 448×448 单通道灰度 mask |
| **训练数据** | 50 类 × 20 个正常样本 = 1000 个样本 (纯正常) |
| **Test A** | 50 个已见类别 (正常+异常) |
| **Test B** | 50 个已见类别 + 50 个未见类别 |

**评估公式**：
```
S = 100 × (0.3 × S_cls + 0.5 × S_seg + 0.2 × S_zs)
```

- **S_cls (30%)**：已见类别图像级 I-AUROC / I-AP 宏平均
- **S_seg (50%)**：已见类别像素级 P-AUROC / P-AP / P-F1max 宏平均
- **S_zs (20%)**：未见类别综合得分

---

## 2. 方案总览

### 2.1 核心思路

```
样本 Sxxxx (5视角)
    │
    ├── view0 ──→ Res2CLIP* ──→ anomaly_map_0 + score_0
    ├── view1 ──→ Res2CLIP* ──→ anomaly_map_1 + score_1
    ├── view2 ──→ Res2CLIP* ──→ anomaly_map_2 + score_2
    ├── view3 ──→ Res2CLIP* ──→ anomaly_map_3 + score_3
    └── view4 ──→ Res2CLIP* ──→ anomaly_map_4 + score_4
                           │                  │
                           ▼                  ▼
               5路 upsampling          5路 score 聚合
               ↓ 448×448              ↓
               predicted_masks/       anomaly_score
```

### 2.2 选择的模式

**第一阶段（基线）**：Res2CLIP* **training-free 模式**
- 零训练成本，直接使用冻结的 CLIP ViT-L/14@336px
- 靠视觉残差 + 文本残差对齐实现异常检测
- 验证 pipeline 正确性后再考虑微调

**第二阶段（可选增强）**：多任务微调
- 使用 50 类正常样本 + 合成异常微调适配器
- 目标：提升 S_seg（分割得分，权重 50%）

### 2.3 核心改造点一览

| # | 改造点 | 原代码位置 | 改造位置 | 难度 |
|---|--------|-----------|---------|------|
| 1 | Real-IAD DataLoader | `data/dataset.py` | `data/real_iad_dataset.py` | ⭐⭐ |
| 2 | 5 视角独立推理 | `test.py` | `inference.py` | ⭐ |
| 3 | Score 多视角聚合 | `test.py` (单图) | `inference.py` | ⭐ |
| 4 | Mask 上采样 37→448 | `test.py` avg_upsample | `utils/upsample.py` | ⭐⭐ |
| 5 | 生成提交文件 | 无 | `submit.py` | ⭐ |
| 6 | 微调适配器（可选） | `train.py` | `train_real_iad.py` | ⭐⭐⭐ |
| 7 | 交叉视角注意力（进阶） | 无 | 新增模块 | ⭐⭐⭐⭐ |

---

## 3. Phase 1 — 数据准备

### 3.1 目录结构确认

```
dataset/
├── Train/
│   ├── 3_adapter/
│   │   ├── S0001/          ← 1 个样本
│   │   │   ├── 0.png       ← 5 个视角
│   │   │   ├── 1.png
│   │   │   ├── 2.png
│   │   │   ├── 3.png
│   │   │   └── 4.png
│   │   ├── S0002/
│   │   └── ...
│   ├── DVD_switch/
│   ├── ...
│   └── wireless_receiver_module/    ← 共 50 类
│       └── S0001...S0020
├── TestA/                            ← (比赛时提供，目前没有)
│   └── ...
└── TestB/                            ← (比赛时提供，目前没有)
    └── ...
```

### 3.2 步骤 1.1: 生成数据集元数据

新建 `data/generate_real_iad_meta.py`：

```python
# 扫描 Train 目录，生成 real_iad_meta.json
# 格式：
{
    "train": {
        "3_adapter": {
            "normal": ["Train/3_adapter/S0001", "Train/3_adapter/S0002", ...],
        },
        "battery": {
            "normal": ["Train/battery/S0001", ...],
        },
        ...
    }
}
```

- 记录每个类别的**所有正常样本路径**（group_folder 级别）
- 统计每个类别的图像尺寸分布（用于后续决定 resize 策略）

### 3.3 步骤 1.2: 本地验证集分割（重要！）

由于 Test A/B 未发布，需要从 50 个已见类别中**留出部分做本地验证**。

策略：

```
50 个类别
├── 35 类 → Training (用于微调)
├── 5 类  → Seen-Val (模拟 Test A: 已见类别测试)
└── 10 类 → Unseen-Val (模拟 Test B: 未见类别测试)
```

> **注意**：本地验证只能模拟**分类**指标，无法模拟**分割**指标，
> 因为训练集没有异常样本和 mask。
> 需要准备一个额外的异常检测数据集（如 MVTec/VisA）来验证分割性能。

### 3.4 步骤 1.3: 建立类别名 -> ID 映射表

```python
# data/real_iad_categories.py
SEEN_50_CATEGORIES = [
    "3_adapter", "DVD_switch", "D_sub_connector", "PLCC_socket",
    "VR_joystick", "accurate_detection_switch", "battery",
    "blade_switch", "boost_converter_module", "button_battery_holder",
    "circuit_breaker", "connector_housing_female", "crimp_st_cable_mount_box",
    "dc_jack", "dc_power_connector", "detection_switch",
    "effect_transistor", "electronic_watch_movement",
    "ffc_connector_plug", "ingot_buckle", "laser_diode",
    "lego_pin_connector_plate", "limit_switch", "lithium_battery_plug",
    "littel_fuse", "lock", "miniature_lifting_motor",
    "mobile_charging_connector", "motor_bracket", "motor_gear_reducer",
    "motor_plug", "pencil_sharpener", "pinboard_connector",
    "potentiometer", "power_jack", "power_strip_socket",
    "purple_clay_pot", "retaining_ring", "rheostat",
    "self_lock_switch", "silicon_cell_sensor", "single_switch",
    "smd_receiver_module", "suction_cup", "toy_tire",
    "travel_switch", "vacuum_switch", "vehicle_harness_conductor",
    "vibration_motor", "wireless_receiver_module",
]
```

### 3.5 步骤 1.4: 构建实时 DataLoader

新建 `data/real_iad_dataset.py`：

```python
class RealIADDataset(Dataset):
    """
    加载 Real-IAD 格式数据。
    每个样本是一个文件夹 (如 S0001)，包含 5 张图。

    Args:
        root: dataset/Train 或 dataset/TestA
        split: "train" | "test"
        categories: 要包含的类别列表
        transform: 图像预处理
        gt_transform: mask 预处理
    """

    def __getitem__(self, idx):
        # 返回:
        #   images: [5, 3, H, W]  — 5 个视角
        #   group_folder: "3_adapter/S0001"
        #   cls_name: "3_adapter"
        #   anomaly: 0/1
        #   masks: [5, 1, H, W] (全零，因为训练集无异常)
```

**关键设计决策**：

1. **是否把 5 个视角拼成 batch？**
   - 是：一次性推理 5 张图，GPU 利用率高
   - 否：逐张推理，代码更简单
   - **推荐**：组装成 batch (batch_size=5)，在 `test.py` 中 batch 化处理

2. **图像尺寸**：
   - Real-IAD 原始图像 1800-3600px
   - CLIP ViT-L/14@336px 最大支持 336px (或 Res2CLIP 默认 518px)
   - **必须 resize**，代价是丢失细节
   - 实验性方案：**Tiled processing**（后面进阶方案详述）

### 3.6 步骤 1.5: 验证数据加载

```bash
# 快速验证脚本
python -c "
from data.real_iad_dataset import RealIADDataset
from utils.utils import get_transform
import argparse

args = argparse.Namespace(image_size=518)
transform, _ = get_transform(args)
dataset = RealIADDataset(
    root='dataset/Train',
    split='train',
    categories=['3_adapter', 'battery'],
    transform=transform,
)
print(f'Dataset size: {len(dataset)}')
img, group, cls, anom = dataset[0]
print(f'Image shape: {img.shape}')  # 预期 [5, 3, 518, 518]
print(f'Group: {group}, Class: {cls}, Anomaly: {anom}')
"
```

---

## 4. Phase 2 — 单视角 Baseline

### 4.1 目标

在修改多视角之前，先在 Real-IAD 上跑通 **单视角 (view 0)** 的 Res2CLIP* training-free 推理，
确保：
- 模型加载正常（CLIP 权重下载）
- TextFeatureBank 计算正确
- Visual memory bank 构建成功
- 推理不出错

### 4.2 步骤 2.1: 直接复用原有 test.py

用少量 Real-IAD 类别替换原有 MVTec 配置来测试：

```bash
python test.py \
    --mode training-free \
    --dataset mvtec \
    --data_path ./dataset/Train \
    --image_size 518 \
    --k_shot 1 \
    --match_strategy sparse_radial_knn \
    --spatial_penalty 0.01
```

**预期问题**：原有 Dataset 类依赖 meta.json，而 Real-IAD 没有这个文件 → 报错。
我们需要先修改数据加载路径。

### 4.3 步骤 2.2: 快速打补丁，让原有代码能跑

创建一个最简单的测试脚本 `test_debug.py`：

```python
# 核心逻辑：
# 1. 加载 CLIP 模型
model, preprocess = load_clean_clip("ViT-L/14@336px", device=device)
model.visual.DAPM_replace(DPAM_layer=20)

# 2. 取一个类别的 1 个正常样本作为 reference
ref_img = load_and_preprocess("dataset/Train/3_adapter/S0001/0.png")
feat_g_ref, feat_p_ref = model.encode_image(ref_img, ALL_LAYERS)

# 3. 取另一个样本作为 query（同类别或不同类别）
query_img = load_and_preprocess("dataset/Train/3_adapter/S0002/0.png")
feat_g_q, feat_p_q = model.encode_image(query_img, ALL_LAYERS)

# 4. 计算残差 → 异常图
# ... (沿用 test.py 第 232-306 行的三路融合逻辑)

# 5. 可视化结果
import matplotlib.pyplot as plt
plt.imshow(anomaly_map)  # 正常样本应该全黑
plt.savefig("debug_map.png")
```

**预期结果**：
- 正常样本对比正常 reference → 异常图应该接近全黑（低响应）
- 不同类别对比 → 异常图应该有响应

### 4.4 验证指标

| 检查项 | 预期 | 验证方式 |
|--------|------|----------|
| 模型加载 | 成功 | 无报错 |
| 正常-正常对比 | 异常值低 | max(map) < 0.3 |
| 跨类对比 | 异常值高 | max(map) > 0.5 |
| 推理速度 | <1s/图 | time 计时 |

---

## 5. Phase 3 — 多视角推理

### 5.1 核心问题

每个样本有 5 个视角 (0-4)，需要：
1. **图像级**：5 个 anomaly_score → 1 个 sample-level score
2. **像素级**：5 个 anomaly_map → 5 个 mask（独立输出）

### 5.2 方案设计

```
┌──────────────────────────────────────────┐
│             方案对比                      │
├──────────────────────────────────────────┤
│ 方案 A: 各视角独立推理 + 后聚合 (推荐)    │
│ 方案 B: 跨视角特征融合推理                │
│ 方案 C: 3D 重建后检测 (更复杂，暂不考虑)  │
└──────────────────────────────────────────┘
```

### 5.3 方案 A（首选）：独立推理 + Score 融合

**原理**：5 个视角各自独立通过 Res2CLIP* 推理，得到 5 组 (score, map)。然后聚合。

**Score 聚合策略**（按推荐度排序）：

| 策略 | 公式 | 适用场景 | 评注 |
|------|------|----------|------|
| **A1: Max** | `score = max(s₀, ..., s₄)` | 保守检测 | 任一视角异常则判异常，最安全 |
| **A2: Mean** | `score = mean(s₀, ..., s₄)` | 均衡检测 | 平滑，但可能稀释单视角强信号 |
| **A3: Top2 Mean** | `score = mean(top2(s₀, ..., s₄))` | 折中 | 兼顾多视角异常和单视角强异常 |
| **A4: Learned** | `score = W·[s₀, ..., s₄]` | 有验证集时 | 需要真值标注 |

**推荐默认**：`A1 (Max)` — 因为大部分缺陷只在 1-2 个视角可见，
用 Max 可以最大程度保留检测信号。

**Mask 输出**：每个视角独立输出，不上采样也不融合。
因为比赛要求每个视角有独立的 mask 文件。

### 5.4 实现：新建 `inference_multi_view.py`

```python
class MultiViewInference:
    """
    多视角推理引擎。

    流程：
    1. 加载模型 & TextFeatureBank
    2. 遍历每个类别 → 构建 visual memory bank（只用第一个视角的 reference？）
    3. 遍历每个样本的每个视角 → 独立推理
    4. 聚合 5 个视角的 score
    5. 输出 submission.csv + predicted_masks/
    """
```

**关键代码框架**：

```python
def infer_one_view(model, text_bank, image, ref_features, args):
    """
    对单张图执行 Res2CLIP* 推理。

    Args:
        image: [1, 3, H, W] 预处理后的图像
        ref_features: 该类别的 reference features dict
    Returns:
        anomaly_map: [1, 1, H_feat, H_feat]
        anomaly_score: float
    """
    # 1. Encode
    feat_g, feat_p = model.encode_image(image, ALL_LAYERS, DPAM_layer=20)
    _, _, _, _, R_t = text_bank.get_features()

    # 2. Text branch
    #    s_text = <feat, R_t>
    # 3. Visual branch (KNN → residual)
    #    R_v = feat - matched_ref
    #    score_vis = ||R_v||²
    # 4. Residual branch
    #    proj = clamp(R_v @ R_t, 0)
    # 5. Fuse
    #    M = M_text * 0.1 + M_vis + M_res
    #    score = s_text + s_vis + s_res

    return anomaly_map, anomaly_score


def infer_one_sample(model, text_bank, views, cls_name, ref_features, args):
    """
    对一个样本的 5 个视角推理。

    Args:
        views: [5, 3, H, W] 预处理后的 5 个视角
    Returns:
        sample_score: float (5 视角聚合)
        per_view_maps: [5, 1, H, H] 每个视角的原始 anomaly map
    """
    scores = []
    maps = []
    for v in range(5):
        img = views[v:v+1]  # [1, 3, H, W]
        anom_map, anom_score = infer_one_view(model, text_bank, img, ref_features, args)
        scores.append(anom_score)
        maps.append(anom_map)

    # 融合策略：max + mean 组合
    sample_score = max(scores)  # 或自定义融合

    return sample_score, maps
```

### 5.5 关于 Visual Memory Bank 的跨视角策略

这是一个**重要设计决策**：

| 策略 | 做法 | 优缺点 |
|------|------|--------|
| **单视角 memory** | 只用 view0 的正样本建 memory，所有视角都匹配这个 memory | 视角差异会导致匹配噪声 |
| **全视角 memory** | 所有 5 个视角的正样本都加入 memory | memory 大 5 倍，但匹配更准 |
| **同视角 memory** | view0→view0 memory, view1→view1 memory ... | 需要每视角多张参考，memory 大 |

**推荐**：全视角 memory。因为同一类别不同视角的参考图像提供了更丰富的上下文，
对提升匹配鲁棒性有帮助。

### 5.6 实现步骤

```
步骤 3.1 ── 创建 inference_multi_view.py 主框架
步骤 3.2 ── 实现单视角推理函数 (从 test.py 搬运)
步骤 3.3 ── 实现 5 视角循环推理
步骤 3.4 ── 实现 score 聚合策略
步骤 3.5 ── 在 1 个类别上验证输出
```

---

## 6. Phase 4 — Mask 上采样 (37×37 → 448×448)

### 6.1 问题分析

Res2CLIP 使用 ViT-L/14@336px，输入 518×518 时：
- Patch size = 14
- Patch grid = 518/14 = 37 (37×37 = 1369 patches)
- 所以 anomaly map 分辨率 = 37×37

比赛要求 mask 输出 448×448 → **需要 ~12× 上采样**

### 6.2 上采样方案对比

| 方案 | 做法 | 质量 | 额外参数 | 评注 |
|------|------|------|---------|------|
| **S1: 双线性** | `F.interpolate(m, 448, bilinear)` | ★★ | 0 | 最简单，边缘模糊 |
| **S2: 双线性 + 高斯锐化** | S1 + 后处理锐化 | ★★☆ | 0 | 在 S1 基础上微调 |
| **S3: 渐进上采样** | 37→74→148→296→448 逐步插值 | ★★★ | 0 | 比一步上采样略平滑 |
| **S4: 轻量 Decoder** | 小 CNN 做 learned upsampling | ★★★★ | ~50K | 需要训练数据 |
| **S5: 像素级对齐** | 将 patch 特征投影回像素 | ★★★★★ | 参考 DINO | 复杂，需修改模型 |

**推荐顺序**：
1. **先上 S1（双线性插值）** — 5 分钟实现，跑通提交格式
2. **再升级到 S4（轻量 Decoder）** — 花 1-2 天训练，提升分割精度

### 6.3 S1 实现（快速基线）

```python
# utils/upsample.py

def upsample_anomaly_map(anomaly_map, target_size=448):
    """
    将 37×37 异常图上采样到 448×448。

    Args:
        anomaly_map: [B, 1, H_feat, H_feat] (37×37)
        target_size: 输出尺寸
    Returns:
        [B, 1, 448, 448]
    """
    import torch.nn.functional as F

    # 步骤 1: 双线性插值
    upsampled = F.interpolate(
        anomaly_map,
        size=target_size,
        mode='bilinear',
        align_corners=False,
    )

    # 步骤 2: 高斯平滑去锯齿
    kernel = get_gaussian_kernel(kernel_size=5, sigma=2).to(anomaly_map.device)
    upsampled = kernel(upsampled)

    # 步骤 3: 归一化到 [0, 1] 区间 (保持相对值)
    upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)

    return upsampled
```

### 6.4 S4 实现（轻量 Decoder）

架构设计：

```
输入: patch_features [B, 1369, 768]  (来自 ViT 层 24)

Patch Embed → Conv2D(768 → 256, 1×1)
    → ResizeConv2D(256 → 128, 2×)  → 74×74
    → ResizeConv2D(128 → 64, 2×)   → 148×148
    → ResizeConv2D(64 → 32, 2×)    → 296×296
    → ResizeConv2D(32 → 16, 2×)    → 448×448  (目标: 448 不是 2 的幂)
    → Conv2D(16 → 1, 1×1) + Sigmoid

输出: anomaly_map [B, 1, 448, 448]
```

**训练方式**：
- 这需要异常样本的 ground truth mask 来训练
- 可以在 MVTec AD 等公开数据集上预训练这个 decoder
- 或者在比赛 Test A 发布后，用 Test A 的 ground truth 训练

### 6.5 上采样效果验证

```python
# 目视验证
for each category:
    1. 取正常样本 → 推理 → 上采样 → mask
    2. 预期：全黑或接近全黑 (正常无异常)

# 定量验证（如果有 MVTec 数据）
1. 用 MVTec 数据跑 inference
2. 在原始分辨率评估 AUPRO
3. 在 448×448 评估 AUPRO
4. 对比精度损失
```

---

## 7. Phase 5 — 微调适配器（可选）

### 7.1 现状与挑战

**现状**：
- Res2CLIP* (training-free) 已经有不错的 zero-shot 能力
- 微调模式 Res2CLIP-dagger 需要**异常样本**做 ranking loss

**挑战**：
- 训练集只有正常样本 → 无法直接计算 ranking loss
- 但 `S_seg` 占 50% 权重 → 提升分割精度至关重要
- 微调适配器有望显著提升 S_seg

### 7.2 解决方案：合成异常

在正常样本上**人工合成缺陷**，生成伪 (query, reference, mask) 三元组。

#### 推荐的合成方法

| 方法 | 描述 | 效果 | 实现难度 |
|------|------|------|---------|
| **CutPaste** | 从正常区域裁剪一块，贴在另一位置 | ★★★ | ⭐ |
| **Perlin 噪声** | Perlin 噪声生成不规则瑕疵形状 | ★★★★ | ⭐⭐ |
| **DRAEM 风格** | 纹理+结构异常组合 | ★★★★★ | ⭐⭐⭐ |
| **泊松融合** | 将外部缺陷图融合到正常图像上 | ★★★★ | ⭐⭐ |

**推荐**：先用 **CutPaste**（最简单，实现快），再升级到 **Perlin + 泊松融合**。

#### CutPaste 实现（`utils/synthetic_anomaly.py`）

```python
def cut_paste_augment(image, mask_ratio=0.02):
    """
    在正常图像上合成 CutPaste 异常。

    Args:
        image: [3, H, W] 正常图像
        mask_ratio: 缺陷区域占图像比例
    Returns:
        augmented: [3, H, W] 带合成异常的图像
        anomaly_mask: [1, H, W] 缺陷区域的 GT mask
    """
    # 1. 随机裁剪一小块
    # 2. 粘贴到另一个位置
    # 3. 生成对应的二进制 mask
    # 4. 可选: 对粘贴块做颜色变换
    return augmented, mask
```

### 7.3 训练流程改造

```python
# train_real_iad.py

# 改动 1: 数据加载 - 用合成异常
for each batch:
    # 从 Real-IAD 读取正常样本对 (img, ref)
    # 在 img 上合成异常 → img_aug, synthetic_mask
    # 如果 ref 也合成异常 → ref_aug, ref_mask (可选)

# 改动 2: Loss 计算 - 基本沿用 train.py
# 总 loss = L_vis + L_text + L_res_vis/L_res_text
# 与原始 train.py 一致，因为 syntheitc_mask 提供了异常的 GT

# 改动 3: 交替训练策略
# 偶数 epoch: 训练残差视觉适配器
# 奇数 epoch: 训练残差文本适配器
# (与原始 train.py 完全一致)
```

### 7.4 超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| epochs | 20-50 | 适配器参数少，收敛快 |
| batch_size | 8-16 | 取决于 GPU 显存 |
| lr_v | 5e-4 | 视觉适配器学习率 |
| lr_t | 1e-4 | 文本适配器学习率 |
| tau | 1.0 | ranking loss margin |
| image_size | 518 | 与原始一致 |
| 合成异常比例 | 50% batch | 一半正常一半合成异常 |

### 7.5 微调流程决策树

```
有 GPU (>=24GB)?
├── 是: 可以微调
│   ├── 有 MVTec/VisA 数据?
│   │   ├── 是: 先在 MVTec 上预训练适配器
│   │   │   → 再在 Real-IAD 上用合成异常微调
│   │   └── 否: 直接在 Real-IAD 上用合成异常训练
│   └── 观察验证集指标
│       ├── 提升 > 2%: 保留微调
│       └── 提升 < 2%: 退回 training-free
│
└── 否: 放弃微调，专注 training-free + 后处理优化
```

---

## 8. Phase 6 — 提交流程

### 8.1 提交文件结构

```
submission.zip
├── submission.csv
└── predicted_masks/
    ├── 3_adapter/
    │   ├── S0001/
    │   │   ├── 0_mask.png    (448×448, 单通道灰度)
    │   │   ├── 1_mask.png
    │   │   ├── 2_mask.png
    │   │   ├── 3_mask.png
    │   │   └── 4_mask.png
    │   ├── S0002/
    │   └── ...
    ├── battery/
    └── ... (Test A + Test B 所有类别)
```

### 8.2 submission.csv 格式

```csv
group_folder,anomaly_score
3_adapter/S0001,0.0321
3_adapter/S0002,0.8765
battery/S0001,0.1248
...
```

### 8.3 实现 `submit.py`

```python
# submit.py

import os
import cv2
import pandas as pd
import zipfile

def generate_submission(results, output_path):
    """
    从推理结果生成 submission.zip。

    Args:
        results: list of dict, each with:
            - group_folder: str, e.g. "3_adapter/S0001"
            - anomaly_score: float
            - per_view_masks: list of [5, 448, 448] numpy arrays
        output_path: str, 输出 zip 路径
    """
    # 1. 写 submission.csv
    df = pd.DataFrame([
        {"group_folder": r["group_folder"], "anomaly_score": r["anomaly_score"]}
        for r in results
    ])
    df.to_csv("submission.csv", index=False)

    # 2. 写 predicted_masks/
    os.makedirs("predicted_masks", exist_ok=True)
    for r in results:
        cat, sample = r["group_folder"].split("/")
        sample_dir = os.path.join("predicted_masks", cat, sample)
        os.makedirs(sample_dir, exist_ok=True)
        for v in range(5):
            mask = r["per_view_masks"][v]      # [448, 448]
            mask = (mask * 255).astype(np.uint8) # [0, 255]
            cv2.imwrite(
                os.path.join(sample_dir, f"{v}_mask.png"),
                mask,
            )

    # 3. 打包 ZIP
    with zipfile.ZipFile(output_path, 'w') as zf:
        zf.write("submission.csv")
        for root, dirs, files in os.walk("predicted_masks"):
            for f in files:
                zf.write(os.path.join(root, f))
```

### 8.4 Mask 输出规范

```python
def save_mask(anomaly_map, save_path):
    """
    保存异常 mask。

    Args:
        anomaly_map: [H, W] numpy/tensor, 值域 [0, 1]
        save_path: str
    """
    if torch.is_tensor(anomaly_map):
        anomaly_map = anomaly_map.cpu().numpy()

    # 确保尺寸 448×448
    assert anomaly_map.shape == (448, 448), f"Expected 448×448, got {anomaly_map.shape}"

    # 量化到 uint8 [0, 255]
    mask_uint8 = (anomaly_map * 255).astype(np.uint8)

    # 保存
    cv2.imwrite(save_path, mask_uint8)
```

### 8.5 验证提交文件

```python
def validate_submission(zip_path):
    """
    验证提交文件格式。
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        files = zf.namelist()

    # 检查 submission.csv
    assert "submission.csv" in files

    # 检查 predicted_masks/ 结构
    mask_files = [f for f in files if f.startswith("predicted_masks/")]
    assert len(mask_files) > 0

    # 检查 mask 格式
    for mf in mask_files:
        assert mf.endswith("_mask.png")
        assert mf.count("/") == 3  # predicted_masks/cat/sample/N_mask.png

    print(f"✓ Valid submission: {len(mask_files)} masks, {len(files)} total files")
```

---

## 9. Phase 7 — 本地验证与调优

### 9.1 验证策略

由于比赛 Test A/B 未发布，我们需要：

| 验证方式 | 数据源 | 评估指标 | 备注 |
|---------|--------|---------|------|
| **Real-IAD 模拟** | 从 50 类中留出 5 类 | I-AUROC | 只能评分类（无 GT mask） |
| **MVTec AD** | 下载 MVTec AD 完整数据 | I-AUROC, P-AUROC, AUPRO | 可评分割，跨域验证泛化 |
| **VisA** | 下载 VisA 数据 | I-AUROC, P-AUROC, AUPRO | 同上 |
| **BTAD** | 下载 BTAD 数据 | I-AUROC, P-AUROC, AUPRO | 同上 |

**推荐**：下载 **MVTec AD** (无需额外注册) 来做完整的分割能力评估。
原论文也使用 MVTec+VisA，指标可以直接对比论文结果。

### 9.2 调优参数清单

| 参数 | 范围 | 影响 | 调优优先级 |
|------|------|------|-----------|
| `match_strategy` | `sparse_radial_knn`, `radial_3nn`, `2nn`, ... | 视觉匹配质量 | ⭐⭐⭐ |
| `spatial_penalty` | 0.001 ~ 0.1 | 空间约束强度 | ⭐⭐⭐ |
| `image_size` | 336, 518 | 输入分辨率 | ⭐⭐ |
| `visual_features_list` | `[6,12,18,24]`, `[12,24]`, etc. | 特征层选择 | ⭐⭐ |
| `res_features_list` | `[24]`, `[18,24]`, etc. | 残差层选择 | ⭐⭐ |
| `text_features_list` | `[24]`, `[12,24]` | 文本层选择 | ⭐ |
| 融合权重 (text/vis/res) | `0.1/1.0/1.0` | 三路融合比例 | ⭐⭐⭐ |
| score 聚合策略 | max, mean, top2-mean | 多视角融合 | ⭐⭐⭐⭐ |
| mask 上采样方法 | bilinear, decoder | 定位精度 | ⭐⭐⭐ |
| k_shot | 1, 2, 5 | 参考样本数 | ⭐⭐ |

### 9.3 自动化调优脚本

```python
# tune.py — 简单的网格搜索

param_grid = {
    "spatial_penalty": [0.001, 0.005, 0.01, 0.05, 0.1],
    "match_strategy": ["sparse_radial_knn", "radial_3nn", "radial_5nn"],
    "fusion_weights": [
        {"text": 0.1, "vis": 1.0, "res": 1.0},
        {"text": 0.2, "vis": 1.0, "res": 1.0},
        {"text": 0.1, "vis": 1.0, "res": 0.5},
    ],
}

best_score = 0
best_config = None
for combo in itertools.product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), combo))
    score = evaluate_on_mvtec(config)
    if score > best_score:
        best_score = score
        best_config = config
```

### 9.4 消融实验

需要回答的关键问题：

| 问题 | 实验设计 | 测量指标 |
|------|---------|---------|
| 多视角融合是否比单视角好？ | Max 融合 vs 只用 view0 | I-AUROC |
| 全视角 memory 比单视角好多少？ | 全视角 vs view0-only memory | I-AUROC, AUPRO |
| 微调后提升了多少？ | training-free vs 微调 | I-AUROC, P-AUROC |
| 合成异常的质量是否足够？ | 不同合成方法的对比 | 验证集指标 |
| 上采样方法对分割的影响？ | bilinear vs decoder | P-AUROC, AUPRO |

---

## 10. 项目文件清单

### 10.1 新建文件

```
competition_plan.md                 ← 本文档

data/
├── real_iad_dataset.py             ← Real-IAD DataLoader
├── real_iad_categories.py          ← 50 类别列表
└── generate_real_iad_meta.py       ← 元数据生成

inference/
├── multi_view_inference.py         ← 多视角推理主引擎
├── single_view_inference.py        ← 单视角推理核心
├── score_fusion.py                 ← Score 多视角聚合
└── build_memory.py                 ← Visual Memory Bank 构建

utils/
├── upsample.py                     ← Mask 上采样方法
├── synthetic_anomaly.py            ← 合成异常 (CutPaste/Perlin)
└── submit.py                       ← 提交文件生成

scripts/
├── run_baseline.sh                 ← 跑 Baseline
├── run_tuning.sh                   ← 参数调优
└── run_submit.sh                   ← 生成提交文件

configs/
├── baseline.yaml                   ← Baseline 配置
└── tuned.yaml                      ← 调优后配置
```

### 10.2 修改文件

```
test.py              → 增加 Real-IAD 支持
data/dataset.py      → 增加 Real-IAD 数据集注册
```

---

## 11. 时间规划

假设全职投入，按优先级排序：

```
Week 1: 基础设施
  Day 1-2:  Phase 1 数据准备 + DataLoader
  Day 3-4:  Phase 2 单视角 Baseline 跑通
  Day 5-7:  Phase 3 多视角推理初版

Week 2: 核心功能
  Day 1-2:  Phase 4 Mask 上采样 (S1 baseline)
  Day 3-4:  Phase 6 提交文件生成 + 格式验证
  Day 5-7:  Phase 7 在 MVTec/VisA 上调优

Week 3: 进阶优化
  Day 1-3:  Phase 5 合成异常 + 微调 (可选)
  Day 4-5:  Phase 4 轻量 Decoder (S4)
  Day 6-7:  消融实验 + 参数搜索

Week 4: 冲刺
  Day 1-3:  最终调优 + 交叉验证
  Day 4-5:  生成 Test A/B 提交文件
  Day 6-7:  文档 + 最后一次复核
```

---

## 附录

### A. 关键依赖

```
# requirements.txt 基础上增加
opencv-python>=4.8.0      # 图像 I/O
pyyaml>=6.0               # 配置管理
matplotlib>=3.7.0         # 可视化调试
```

### B. 注意事项

1. **GPU 显存**：ViT-L/14@336px 每张图约 3-4GB 显存 (batch=1)。5 个视角同时推理
   建议 batch_size=5 → 约 12-16GB 显存。如果显存不足，逐视角推理也可以。

2. **CLIP 权重**：首次运行会从 OpenAI 下载 ViT-L/14@336px.pt (~1.6GB)，
   保存在 `./clip_model/`。确保网络畅通。

3. **比赛数据存储**：7.17 GB 训练数据 + 待发布的 Test A/B。
   确保有足够磁盘空间。

4. **提交前检查**：
   - [ ] submission.csv 列名和格式正确
   - [ ] 所有 mask 是 448×448 单通道
   - [ ] mask 像素值范围 [0, 255]
   - [ ] 每个 sample 有 5 个 mask 文件
   - [ ] 没有缺失类别

5. **未见类别 (S_zs) 策略**：
   - Res2CLIP* 的文本残差是**类无关**的（使用 "object" 作为名词）
   - 这意味着它对未见类别天然泛化
   - 50% 的分数来自已见类别 (S_cls + S_seg)，50% 来自未见类别 (S_zs 也是综合评分)
   - 不需要为未见类别做特殊处理
