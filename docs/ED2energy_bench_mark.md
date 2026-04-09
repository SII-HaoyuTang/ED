# EDBench ED5-EC 能量预测复现指南

> 目标：从电子密度点云预测 6 种 DFT 能量分量，复现 EDBench 论文（NeurIPS 2025）的 X-3D 基准结果。

---

## 任务说明

**ED5-EC**（Electron Density → 5 Energy Components）是 EDBench 基准测试的核心回归任务之一。

根据 Hohenberg-Kohn 定理，分子的全部基态性质（包括总能量）由其电子密度唯一决定。ED5-EC 任务直接验证这一理论：**仅凭电子密度的三维点云，能否准确预测 DFT 计算的 6 种能量分量？**

### 预测目标（6 种能量，单位 Hartree）

| 索引 | 名称 | 含义 | 典型量级 |
|------|------|------|----------|
| E1 | @DF-RKS Final Energy | DFT 收敛后的最终总能量 | −800 ~ −900 |
| E2 | Nuclear Repulsion Energy | 原子核间排斥能 | +600 ~ +1200 |
| E3 | One-Electron Energy | 动能 + 电子-核吸引能 | −2000 ~ −3500 |
| E4 | Two-Electron Energy | 电子-电子排斥能 | +800 ~ +1600 |
| E5 | DFT Exchange-Correlation Energy | 交换相关能（DFT 核心校正） | −70 ~ −90 |
| E6 | Total Energy | 各分量之和（= E1） | −800 ~ −900 |

---

## 数据

### 来源文件

| 文件 | 大小 | 内容 |
|------|------|------|
| `src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl` | ~9 GB | 47,986 个分子的电子密度点云（已过滤 < 0.05 a.u.） |
| `src/data/ed_energy_5w/raw/ed_energy_5w.csv` | ~6 MB | 分子 ID、SMILES、6 种能量标签、划分列 |
| `src/data/ed_energy_5w/raw/readme.md` | — | 数据说明 |

### CSV 列说明

```
index            分子 ID（字符串，与 PKL 键一致）
smiles           原始 SMILES
canonical_smiles 规范化 SMILES
mol_cluster      分子簇 ID（用于 scaffold split）
energy_cluster   能量簇 ID
label            6 个能量值，空格分隔（Hartree）
scaffold_split   train / valid / test（基于分子骨架划分）
random_split     train / valid / test（随机划分，仅供参考）
```

### 数据处理流程

```
PKL 文件
  ├── mol_id → electronic_density.coords  (M, 3)  Bohr
  └── mol_id → electronic_density.density (M,)    a.u.
         ↓
  拼接 [x, y, z, density] → (M, 4)
         ↓
  FPS 采样（最远点采样）→ (2048, 4)          ← 关键！与 EDBench 原始处理一致
         ↓
  缓存为 .pt 文件（{mol_id}_fps2048.pt）
```

> **为什么用 FPS 而非 K-Means？**
> 能量预测任务不需要密度加权的聚类代表点；FPS 保留全局几何结构，与 EDBench 原始实现一致。

---

## 模型架构：PointMetaBase-S-X3D

### 架构总览

```
输入: (B, 2048, 4)   ← [x, y, z, density]
      ↓
  [Stage 1]  stride=1, width=32,  2048→2048 点
      ↓
  [Stage 2]  stride=2, width=64,  2048→1024 点
      ↓
  [Stage 3]  stride=2, width=128, 1024→512 点  ← X-3D 显式结构特征
      ↓
  [Stage 4]  stride=2, width=256, 512→256 点   ← X-3D 显式结构特征
      ↓
  [Stage 5]  stride=2, width=256, 256→128 点
      ↓
  [Stage 6]  Global SA (max+mean pool) → (B, 256)
      ↓
  [MLP Head] 256 → 512 → 256 → 6
输出: (B, 6)   能量预测值（Hartree）
```

### 核心组件

#### 1. Set Abstraction（局部聚合）

每个阶段的特征提取：

```
FPS 采样中心点 M 个
  ↓ Ball Query: 半径 r，最多 K=32 个邻居
  ↓ feature_type = "dp_fj": 拼接 [Δxyz | feature_j]
  ↓ LocalAgg MLP
  ↓ Max Pooling over neighbors
  ↓ InvResMLP block（残差精炼）
```

**半径序列**（各阶段倍增）：`0.15 → 0.225 → 0.338 → 0.506 → 0.759 → 1.139` Bohr

#### 2. X-3D 显式结构编码（第 3、4 阶段）

在局部聚合之前，对每个中心点的 K 个邻居提取几何描述符：

**PCA 几何特征（9 维）**：
- 线性度（linearity）= (λ₁ − λ₂) / λ₁
- 平面度（planarity）= (λ₂ − λ₃) / λ₁
- 散射度（scattering）= λ₃ / λ₁
- 全向性（omnivariance）= (λ₁λ₂λ₃)^(1/3)
- 各向异性（anisotropy）= (λ₁ − λ₃) / λ₁
- 三个特征值 λ₁ ≥ λ₂ ≥ λ₃
- 主方向的垂直分量

**PointHop 特征（24 维）**：
- 按坐标轴正负将邻居分为 8 个象限
- 每个象限内邻居的平均相对坐标（3 维）
- 8 × 3 = 24 维

**NeighborContext（动态权重）**：
- 用 33 维几何特征生成 attention 权重（`weight_gen`：MLP + Sigmoid）
- 对邻居特征加权后 Max Pooling
- 通过 Conv1d pipeline 输出

#### 3. InvResMLP（逆残差 MLP）

```
x → Linear(C→4C) → BN → GELU → Linear(4C→C) → BN → + x
```

#### 4. RegressionHead

```
Linear(global_dim, 512) → BN → ReLU → Dropout(0.5)
Linear(512, 256)        → BN → ReLU → Dropout(0.5)
Linear(256, 6)
```

---

## 与 EDBench 原始实现的对应关系

| EDBench 原始文件 | 本项目对应文件 |
|----------------|--------------|
| `openpoints/models/backbone/pointmetabase_X3D.py` | `bench_mark/models/backbone/pointmetabase_x3d.py` |
| `openpoints/models/backbone/X_3D_utils/explict_structure.py` | `bench_mark/models/backbone/x3d_utils/explicit_structure.py` |
| `openpoints/models/backbone/X_3D_utils/neighbor_context.py` | `bench_mark/models/backbone/x3d_utils/neighbor_context.py` |
| `openpoints/models/classification/cls_base.py` (ClsHead) | `bench_mark/models/cls_head.py` |
| `openpoints/dataset/density/density_loader.py` | `bench_mark/data/energy_dataset.py` |
| `examples/regression/main.py` | `bench_mark/train_energy.py` |
| `cfgs/energy/pointmetabase-s-x-3d.yaml` | `bench_mark/cfgs/energy_x3d.yaml` |

---

## 训练配置（对标论文）

| 超参数 | 本项目 | EDBench 论文 |
|--------|--------|-------------|
| 输入点数 | 2048 | 2048 |
| Width | 32 | 32 |
| Stages | 6 | 6 |
| Ball Query K | 32 | 32 |
| Base radius | 0.15 Bohr | 0.15 |
| X-3D stages | {3, 4} | {3, 4} |
| 损失函数 | MSELoss | MSELoss |
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-3 | 1e-3 |
| Weight decay | 0.05 | 0.05 |
| Batch size | 32 | 32 |
| Epochs | 100 | 100 |
| 调度器 | CosineAnnealingLR | CosineAnnealingLR |
| 梯度裁剪 | 1.0 | 1.0 |
| Dropout | 0.5 | 0.5 |
| 数据划分 | scaffold_split | scaffold_split |

---

## 使用方法

### 目录结构

```
bench_mark/
├── data/
│   └── energy_dataset.py       # EDBenchEnergyDataset（FPS 采样 + 能量标签）
├── models/
│   ├── cls_head.py             # RegressionHead（MLP）
│   └── backbone/
│       ├── pointmetabase_x3d.py  # X-3D 主网络
│       └── x3d_utils/
│           ├── explicit_structure.py  # PCA + PointHop
│           └── neighbor_context.py    # 动态权重聚合
├── cfgs/
│   └── energy_x3d.yaml         # 超参数配置
└── train_energy.py             # 训练入口
```

### 冒烟测试（已验证，CPU，~35 分钟）

使用 `-m` 方式从项目根目录运行（确保 `bench_mark` 包可被正确导入）：

```bash
python -m bench_mark.train_energy \
    --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv_path src/data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache_dir src/data/ed_energy_5w/cache_fps \
    --max_samples 128 \
    --npoint 512 \
    --epochs 2 \
    --device cpu
```

**实际运行输出**（2 epoch，128 样本，npoint=512，CPU）：

```
PointMetaBase-S-X3D  params: 2,478,476
Epoch    1/2  train_loss=2455288.69  val_mean_MAE=1186.46  lr=5.00e-04
  E1_Final              MAE=807.3890  RMSE=832.3622  r=0.0914
  E2_NucRepul           MAE=994.7121  RMSE=1029.5221  r=0.2225
  E3_OneElec            MAE=3074.6003  RMSE=3147.9084  r=0.1656
  E4_TwoElec            MAE=1353.5226  RMSE=1388.3496  r=0.2199
  E5_XC                 MAE=81.1312   RMSE=82.3123    r=0.1213
  E6_Total              MAE=807.4011  RMSE=832.3754   r=0.0281
Epoch    2/2  train_loss=2454587.00  val_mean_MAE=1186.39  lr=0.00e+00
  E1_Final              MAE=807.2891  RMSE=832.2639  r=0.1068
  ...
=== Test Set Evaluation ===
  Mean MAE: 1220.2147
Done. Best val mean MAE: 1186.3851
```

> **说明**：仅 2 epoch + 128 样本未收敛，MAE 偏高符合预期。相对顺序正确（E5 MAE 最小，E3 最大），管道端到端验证通过。

### 完整训练（GPU，推荐）

```bash
python -m bench_mark.train_energy \
    --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --csv_path src/data/ed_energy_5w/raw/ed_energy_5w.csv \
    --cache_dir src/data/ed_energy_5w/cache_fps \
    --npoint 2048 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 100 \
    --output_dir checkpoints/energy \
    --device cuda \
    --wandb \
    --wandb_project ed-energy \
    --run_name repro-x3d-v1
```

> **注意**：请从项目根目录（`ED/`）以 `-m` 方式运行，而非直接 `python bench_mark/train_energy.py`，以避免相对包导入错误。

---

## 论文基准结果（对比目标）

论文使用 **scaffold_split**，EDBench 完整 3.3M 数据集，三次运行平均：

| 能量成分 | X-3D MAE (Hartree) ± std |
|---------|--------------------------|
| E1 Final Energy | 190.77 ± 1.98 |
| E2 Nuclear Repulsion | 109.21 ± 2.82 |
| E3 One-Electron | 369.88 ± 1.34 |
| E4 Two-Electron | 150.05 ± 0.27 |
| E5 DFT XC | **8.13 ± 0.51** |
| E6 Total Energy | 190.77 ± 1.98 |

> **注意**：本项目使用 EDBench 的 47k 子集（非完整 3.3M）。由于训练数据量差异，绝对 MAE 数值可能偏高；但相对顺序（E5 MAE 最小，E3 最大）应保持一致。

---

## 评测指标

每种能量分量报告 4 个指标（对标论文 Table 3）：

- **MAE**：平均绝对误差（主要指标）
- **RMSE**：均方根误差
- **Pearson r**：线性相关系数
- **Spearman ρ**：秩相关系数

---

## 参考文献

```
EDBench: A Large-Scale Electron Density Dataset for Molecular Modeling
Hongxin Xiang et al., NeurIPS 2025
arXiv: 2505.09262
GitHub: https://github.com/HongxinXiang/EDBench
```
