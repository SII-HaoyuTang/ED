# 训练指南

本项目分三个阶段顺序训练：**ViSNet 预训练 → Stage 1 → Stage 2**。

---

## 项目目录结构

```
ED/
├── pretrain_visnet.py              # 第一阶段：ViSNet 预训练（QM9 U0 能量预测）
├── train_stage1.py                 # 第二阶段：等变流匹配，生成点云位置
├── train_stage2.py                 # 第三阶段：不变流匹配，生成密度值
├── inference.py                    # 推理：给定新分子采样电子云点云
│
├── src/
│   ├── model/
│   │   ├── __init__.py             # 导出 VisNetEncoder / Stage1FlowNet / Stage2FlowNet
│   │   ├── visnet_encoder.py       # VisNetEncoder 包装器（逐原子特征提取）
│   │   ├── visnet/                 # ViSNet 原始源码（仅保留必要文件）
│   │   │   └── models/
│   │   │       ├── visnet_block.py     # 核心表征模型（ViSNetBlock）
│   │   │       ├── output_modules.py  # 输出头（EquivariantScalar 等）
│   │   │       └── utils.py           # RBF / Distance / Sphere 工具
│   │   ├── stage1_flow.py          # 等变流网络（Stage 1）
│   │   ├── stage2_flow.py          # 不变流网络（Stage 2）
│   │   └── cfg.py                  # CFG 引导 + ODE 求解器
│   ├── data/
│   │   └── ed_energy_5w/
│   │       ├── raw/                # CSV 标签文件
│   │       │   └── ed_energy_5w.csv
│   │       ├── processed/          # PKL 密度点云（~9 GB）
│   │       │   └── mol_EDthresh0.05_data.pkl
│   │       └── cache/              # FPS 采样缓存（自动生成）
│   └── utils/
│       ├── eval.py
│       └── ot_cfm.py
│
├── bench_mark/                     # EDBench ED5-EC 能量预测复现（独立模块）
│   ├── data/
│   │   └── energy_dataset.py       # EDBenchEnergyDataset（FPS 采样 + 能量标签）
│   ├── models/
│   │   ├── cls_head.py             # 回归头（MLP）
│   │   └── backbone/
│   │       ├── pointmetabase_x3d.py    # X-3D 主网络
│   │       └── x3d_utils/
│   │           ├── explicit_structure.py  # PCA + PointHop 几何特征
│   │           └── neighbor_context.py    # 邻域上下文聚合
│   ├── cfgs/
│   │   └── energy_x3d.yaml         # 超参数配置（对标论文）
│   └── train_energy.py             # 训练入口
│
├── data/
│   └── qm9/                        # QM9 数据集（首次运行自动下载，~1.7 GB）
│
├── checkpoints/
│   ├── visnet_pretrained/
│   │   └── best.pt                 # ViSNet 预训练最佳检查点
│   ├── stage1/
│   │   └── final.pt                # Stage 1 最终检查点
│   └── stage2/
│       └── final.pt                # Stage 2 最终检查点
│
└── docs/
    ├── training_guide.md           # 本文档
    └── ED2energy_bench_mark.md     # EDBench 复现说明
```

---

## 环境准备

### 创建 Conda 环境（CUDA 12.1）

以下命令创建名为 `ED` 的 conda 环境，适配 CUDA 12.1。

**第一步：创建并激活环境**

```bash
conda create -n ED python=3.10 -y
conda activate ED
```

**第二步：安装 PyTorch（CUDA 12.1）**

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

验证 GPU 可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 期望输出：2.1.2+cu121   True
```

**第三步：安装 PyTorch Geometric 及其依赖**

```bash
# 核心包
pip install torch_geometric

# 编译型扩展（必须与 PyTorch 版本严格匹配）
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

**第四步：安装其他依赖**

```bash
pip install scikit-learn tqdm
pip install wandb          # 可选，用于实验跟踪（W&B）
pip install rdkit          # 可选，用于完整解析 QM9 原始 SDF 文件
                           # 不安装则使用 PyG 预处理版本（qm9_v3.pt）
```

**完整一键脚本（复制粘贴版）**

```bash
conda create -n ED python=3.10 -y && conda activate ED

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

pip install torch_geometric

pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install scikit-learn tqdm wandb rdkit
```

> **注意**：`torch_scatter` 等扩展的 whl 地址中 `torch-2.1.0` 为固定格式，
> 即使安装了 `torch==2.1.2` 也使用此链接，两者二进制兼容。



---

## 数据集路径约定

```
src/data/ed_energy_5w/
├── processed/
│   └── mol_EDthresh0.05_data.pkl   # EDBench .pkl 数据集（约 48k 分子）
└── cache/                          # 自动生成的逐分子 .pt 缓存
data/
└── qm9/                            # QM9 自动下载目录（约 1.7 GB）
checkpoints/
├── visnet_pretrained/
│   └── best.pt                     # ViSNet 预训练最佳检查点
├── stage1/
│   └── final.pt                    # Stage 1 最终检查点
└── stage2/
    └── final.pt                    # Stage 2 最终检查点
```

---

## 第一阶段：ViSNet 预训练（QM9）

在 QM9 数据集上以 U0 内能预测任务预训练 ViSNet，获得通用分子表征。

### 架构说明（2025 年更新）

本阶段改用项目内置的原始 ViSNet 源码（`src/model/visnet/`），不再依赖 PyG 内置实现，具体变更：

| 组件 | 旧版（PyG）| 新版（本地源码）|
|------|-----------|----------------|
| 表征模型 | `torch_geometric.nn.models.ViSNet` | 本地 `ViSNetBlock`（lmax=2, vertex_type=Edge）|
| 输出头 | `Scalar`（仅标量特征）| `EquivariantScalar`（标量 + 等变向量特征）|
| 先验模型 | 无 | **Atomref**（分层查找单原子 DFT 能量：① 数据集内置值 → ② `ATOMREF_TABLE` → ③ 0 初始化）|
| `hidden_channels` | 256 | **512**（与论文一致）|
| `num_layers` | 6 | **9**（与论文一致）|
| `num_rbf` | 32 | **64**（与论文一致）|
| vec 特征形状 | `(N, 3, C)` | `(N, 8, C)`（lmax=2，8 个球谐分量）|

> **注意**：vec 形状变化对 Stage 1/2 训练无影响，因为两者均忽略 vec（`_`）。
> 使用预训练权重时必须指定 `--hidden_channels 512`。

**Atomref 分层查找逻辑**

模型预测的是**原子化能**（atomization energy），即减去各元素单原子 DFT 参考能量后的残差：

| 优先级 | 来源 | 说明 |
|--------|------|------|
| ① 最高 | `dataset.atomref(target)` | 数据集内置值，与训练标签同一 DFT 计算级别 |
| ② 次之 | `ATOMREF_TABLE`（`pretrain_visnet.py` 顶部）| 代码内置的扩展元素表（B3LYP/6-31G(2df,p)），默认包含 H/C/N/O/F/Na |
| ③ 最低 | `0.0` | 完全未知元素，训练时 embedding 自适应 |

**扩展到新元素（如 Na 离子数据集）**

在 `pretrain_visnet.py` 顶部的 `ATOMREF_TABLE` 字典中添加对应原子序数和能量即可：

```python
ATOMREF_TABLE: dict[int, float] = {
    1:  -13.61,    # H
    6:  -1029.86,  # C
    7:  -1485.30,  # N
    8:  -2042.61,  # O
    9:  -2715.57,  # F
    11: -4411.90,  # Na  ← 已内置，可直接用于 Na 离子数据集微调
    # 按需添加更多元素...
}
```

**冒烟测试（验证环境，~2 分钟）**

```bash
python pretrain_visnet.py \
    --data_root data/qm9 \
    --max_samples 1000 \
    --hidden_channels 64 \
    --num_layers 2 \
    --epochs 2
```

> **服务器下载失败（403 Forbidden）**：默认从 AWS S3 下载，部分服务器会被拒。
> 脚本会自动尝试备用镜像 `https://data.pyg.org/datasets/qm9_v3.zip`。
> 若仍失败，可手动下载后用 `--qm9_url` 指定本地 HTTP 服务或可访问的镜像：
> ```bash
> # 手动下载（在可访问的机器上执行后传到服务器）
> wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
>
> # 或者指定可访问的 URL
> python pretrain_visnet.py --data_root data/qm9 --qm9_url <your-mirror-url> ...
> ```

**完整预训练（4090 GPU，论文配置）**

```bash
python pretrain_visnet.py \
    --data_root data/qm9 \
    --output_dir checkpoints/visnet_pretrained \
    --hidden_channels 512 \
    --num_layers 9 \
    --num_rbf 64 \
    --cutoff 5.0 \
    --num_heads 8 \
    --train_size 110000 \
    --val_size 10000 \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 300 \
    --lr_patience 15 \
    --lr_factor 0.8 \
    --save_every 10 \
    --device cuda
```

**启用 W&B 实验跟踪**

```bash
python pretrain_visnet.py \
    --data_root data/qm9 \
    --output_dir checkpoints/visnet_pretrained \
    --hidden_channels 512 \
    --epochs 300 \
    --device cuda \
    --wandb \
    --wandb_project ed-pretrain-visnet \
    --run_name visnet-qm9-v1
```

**关键参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | `data/qm9` | QM9 下载/缓存目录（首次运行自动下载） |
| `--hidden_channels` | **512** | 特征维度，**须与 Stage 1/2 `--hidden_channels` 保持一致** |
| `--num_layers` | **9** | 消息传递层数（论文配置）|
| `--num_rbf` | **64** | 径向基函数数量（论文配置）|
| `--train_size` | 110000 | 训练集大小（QM9 共 ~13 万分子）|
| `--epochs` | 300 | 训练轮数；val MAE ≤ 0.05 eV 为较好结果 |
| `--lr_patience` | 15 | 验证 MAE 不下降多少 epoch 后降低学习率 |
| `--wandb` | False | 开启 W&B 日志（需安装 wandb）|
| `--wandb_project` | `ed-pretrain-visnet` | W&B 项目名 |
| `--run_name` | None | W&B run 名称（留空则自动生成）|

输出：`checkpoints/visnet_pretrained/best.pt`（最佳验证 MAE 时保存）

> 检查点新增 `atomref` 字段（拟合的逐元素参考能量），`representation_model` 键格式不变，
> 与 Stage 1/2 的加载方式完全兼容。

---

## 第二阶段：Stage 1 训练（点云位置生成）

基于 VisNet 编码的原子特征，用等变流匹配生成代表性点云位置 {rⱼ}。

**冒烟测试**

```bash
python train_stage1.py \
    --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --cache_dir src/data/ed_energy_5w/cache \
    --max_samples 64 \
    --batch_size 4 \
    --epochs 2 \
    --device cpu
```

**使用预训练 ViSNet 完整训练**

```bash
python train_stage1.py \
    --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --cache_dir src/data/ed_energy_5w/cache \
    --output_dir checkpoints/stage1 \
    --pretrained_visnet checkpoints/visnet_pretrained/best.pt \
    --n_per_atom 8 \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 100 \
    --hidden_channels 512 \
    --num_layers 4 \
    --cutoff 8.0 \
    --cfg_drop_prob 0.15 \
    --save_every 10 \
    --device cuda \
    --wandb \
    --wandb_project ed-stage1 \
    --run_name stage1-v1
```

**离线预处理缓存（可选，多 epoch 训练时加速）**

```bash
python train_stage1.py \
    --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --cache_dir src/data/ed_energy_5w/cache \
    --preprocess \
    --epochs 0   # 仅预处理，不训练
```

**关键参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrained_visnet` | None | ViSNet 预训练检查点路径（推荐提供） |
| `--n_per_atom` | 8 | 每原子生成的代表点数，K = n_per_atom × N_atoms |
| `--cfg_drop_prob` | 0.15 | 训练时无分类器引导的条件丢弃概率 |
| `--cutoff` | 8.0 Bohr | Stage 1 流网络的邻域截断半径 |
| `--wandb` | False | 开启 W&B 日志 |
| `--wandb_project` | `ed-stage1` | W&B 项目名 |
| `--run_name` | None | W&B run 名称 |

输出：`checkpoints/stage1/final.pt`

---

## 第三阶段：Stage 2 训练（密度值生成）

给定 Stage 1 生成的点云位置，用不变流匹配生成对数密度值 {log ρⱼ}。

> **注意**：Stage 2 在训练时会实时调用 Stage 1 推理生成点位置，以减小训练与推理的分布偏移。

**冒烟测试**

```bash
python train_stage2.py \
    --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --cache_dir src/data/ed_energy_5w/cache \
    --stage1_ckpt checkpoints/stage1/final.pt \
    --max_samples 64 \
    --batch_size 4 \
    --epochs 2 \
    --device cpu
```

**完整训练**

```bash
python train_stage2.py \
    --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --cache_dir src/data/ed_energy_5w/cache \
    --stage1_ckpt checkpoints/stage1/final.pt \
    --output_dir checkpoints/stage2 \
    --n_per_atom 8 \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 100 \
    --hidden_channels 512 \
    --num_layers 4 \
    --num_heads 8 \
    --cutoff 8.0 \
    --cfg_drop_prob 0.15 \
    --stage1_ode_steps 20 \
    --guidance_scale 1.5 \
    --save_every 10 \
    --device cuda \
    --wandb \
    --wandb_project ed-stage2 \
    --run_name stage2-v1
```

**关键参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage1_ckpt` | 必填 | Stage 1 检查点路径（同时加载 VisNet + Stage1FlowNet） |
| `--stage1_ode_steps` | 20 | Stage 1 推理时的 ODE 步数（越大越精确，越慢） |
| `--guidance_scale` | 1.5 | Stage 1 推理时的 CFG 引导强度 |
| `--wandb` | False | 开启 W&B 日志 |
| `--wandb_project` | `ed-stage2` | W&B 项目名 |
| `--run_name` | None | W&B run 名称 |

输出：`checkpoints/stage2/final.pt`

---

## 推理

```bash
python inference.py \
    --stage1_ckpt checkpoints/stage1/final.pt \
    --stage2_ckpt checkpoints/stage2/final.pt \
    --atom_types "6 6 8 1 1 1 1" \
    --atom_coords "0.0 0.0 0.0  1.5 0.0 0.0  ..." \
    --n_samples 5 \
    --device cuda
```

---

## 完整训练流程一览

```
1. python pretrain_visnet.py   # QM9 预训练 ViSNet (~1-2 天, GPU)
        ↓
2. python train_stage1.py      # 等变流匹配：生成点云位置 (~几小时, GPU)
        ↓
3. python train_stage2.py      # 不变流匹配：生成密度值 (~几小时, GPU)
        ↓
4. python inference.py         # 给定新分子，采样电子云点云
```

---

## 检查点结构

**`checkpoints/visnet_pretrained/best.pt`**
```python
{
    "epoch": int,
    "representation_model": dict,   # VisNetEncoder 可直接 load_state_dict
    "full_model": dict,
    "mean": float,                  # QM9 U0 均值 (eV)
    "std": float,                   # QM9 U0 标准差 (eV)
    "val_mae": float,
    "args": dict,
}
```

**`checkpoints/stage1/final.pt`**
```python
{
    "visnet": dict,   # VisNetEncoder state_dict
    "flow": dict,     # Stage1FlowNet state_dict
    "args": dict,
}
```

**`checkpoints/stage2/final.pt`**
```python
{
    "stage2": dict,   # Stage2FlowNet state_dict
    "args": dict,
}
```
