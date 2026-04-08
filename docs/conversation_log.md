# 对话记录：电子云生成模型设计与实现

**日期**: 2026-04-05
**项目**: ED（Electron Density Generation）

---

## 一、项目概述讨论

**用户**: 新建科研项目，目的是基于原子坐标预测服从于该原子坐标的电子云。使用流匹配（Flow Matching）扩散原理，网络使用等变神经网络 VisNet，应当是无分类器引导的扩散模型，条件由等变网络生成的原子特征向量提供。

**助手建议**:

1. **电子云表征**：推荐原子中心基函数展开（天然等变，维度小），备选体素网格（内存开销大）或点云。
2. **等变性**：电子密度是标量场（旋转不变），但空间坐标是等变的，需确保模型满足 E(3) 对称性。
3. **网络架构**：VisNet → per-atom 等变特征 → Flow Matching 网络（Cross-Attention 注入条件）。
4. **训练数据**：推荐 QM9、MD17 或 EDBench，使用 `.cube` 格式电子密度文件。
5. **CFG**：训练时以 10-20% 概率丢弃 VisNet 条件，推理时线性组合。

---

## 二、表征方式选择

**用户**: 不确定使用高斯基展开还是直接预测点云的位置和值，不使用体素网格。数据集使用 EDBench（3M+ 数据，`.cube` 格式）。

**助手分析**:

| 方案 | 优势 | 劣势 |
|------|------|------|
| 原子中心高斯基展开 | 输出维度固定，天然等变 | 基函数选择影响精度 |
| 点云位置+密度值 | 自适应，表达能力强 | 位置等变、变长输出困难 |

**推荐**：混合方案——**固定查询点 + 预测值**（神经隐式场），查询点是输入而非生成目标。

**关键问题**：用户最终目标是生成式（给定坐标采样多种合理密度）还是确定性预测？

---

## 三、两阶段生成框架确定

**用户核心需求**:
- 给定新分子坐标，能采样出多种合理的电子密度（生成模型）
- 不想用固定查询点（大多数空间密度趋近于零，代表性差）
- 希望用扩散生成模型生成具有代表性的电子云坐标，再用另一个网络预测这些点上的密度值

**最终确定的两阶段框架**:

```
Stage 1: 等变 Flow Matching → 生成代表性点云位置 {rⱼ}
            条件: VisNet(原子坐标, 原子类型)

Stage 2: 生成式 Flow Matching → 预测每个 rⱼ 处的密度值 ρ(rⱼ)
            条件: {rⱼ} + VisNet特征
```

---

## 四、训练集构建方案

**用户需求**：点要离得尽可能远，最好用聚类提取代表性点。

**确定方案：密度加权 K-Means**

```python
# 从 .cube 提取训练点云
density = load_cube(file)
mask = density > 1e-4          # 过滤真空
weights = density[mask] / density[mask].sum()
kmeans = MiniBatchKMeans(n_clusters=K).fit(coords[mask], sample_weight=weights)
centers = kmeans.cluster_centers_  # (K, 3) 分散且覆盖高密度区
```

- **K = n_per_atom × N_atoms**（n_per_atom 默认为 8）
- K-Means 权重 = ρ(r)，聚类中心天然分散

**备选**：先密度过滤，再最远点采样（FPS）精筛。

---

## 五、设计细节确认

**Q：Stage 2 具体对什么做流匹配？**
**A（用户）**: 只生成密度值 {ρⱼ}，位置固定来自 Stage 1。

**Q：训练策略？**
**A（用户）**: 顺序训练（先训 Stage 1，再训 Stage 2）。

**Stage 2 关键设计**:
- 在**对数空间**操作：z = log(ρ + ε)，避免负值
- 训练时用 Stage 1 生成的位置（不用聚类真值位置），减少 train-test 分布偏移
- 网络：不变 Transformer（Cross-Attention + Self-Attention），条件为点位置 + VisNet 特征

---

## 六、CFG 在等变网络中的实现

**用户问题**: 流匹配中等变网络的无分类器引导是如何实现的？

**训练阶段**（`stage1_flow.py`）:
```python
h_atom = self.atom_proj(atom_feat)
if drop_condition or (self.training and torch.rand(1).item() < self.cfg_drop_prob):
    h_atom = self.null_atom_feat.expand_as(h_atom)  # 可学习的 null embedding
```

等变性自动保持——`null_atom_feat` 是标量替换，不破坏位置坐标的等变计算路径：
```python
rel = pos_tgt[edge_tgt] - pos_src[edge_src]   # 等变向量
w   = self.coord_weight(msg)                   # 不变标量权重（MLP输出）
vel_contrib = w * rel                          # 等变 × 不变 = 等变
```

**推理阶段**（两次 forward pass）:
```python
v_cond   = stage1(x_t, t, ..., drop_condition=False)
v_uncond = stage1(x_t, t, ..., drop_condition=True)
v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
```

等变性对线性组合仍然成立：`v_cond` 和 `v_uncond` 都是等变的，它们的线性组合也等变。

---

## 七、实现的完整文件结构

```
ED/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cube_parser.py       # 解析 .cube 文件（原子坐标 + 密度网格）
│   │   ├── clustering.py        # 密度加权 K-Means 提取代表性点云
│   │   └── dataset.py           # EDBenchDataset + collate_fn（支持磁盘缓存）
│   ├── model/
│   │   ├── __init__.py
│   │   ├── visnet.py            # PyG VisNet 封装（per-atom 等变特征）
│   │   ├── stage1_flow.py       # EGNN 等变流匹配（点云位置生成）
│   │   ├── stage2_flow.py       # Transformer 标量流匹配（对数密度值生成）
│   │   └── cfg.py               # CFG 推理 + Euler/RK4 ODE 求解器
│   └── utils/
│       ├── __init__.py
│       ├── ot_cfm.py            # OT-CFM 核心函数（插值、CFM 损失）
│       └── eval.py              # KDE 重建 + MAE/RMSE/积分误差评测
├── train_stage1.py              # Stage 1 训练脚本
├── train_stage2.py              # Stage 2 训练脚本（加载冻结的 Stage 1）
├── inference.py                 # 推理脚本（单分子 → n 个点云样本）
└── docs/
    └── conversation_log.md      # 本文件
```

---

## 八、关键设计决策汇总

| 决策 | 选择 | 理由 |
|------|------|------|
| 电子云表征 | 点云 {(rⱼ, ρⱼ)} | 灵活，无需选择基函数 |
| 点云提取 | 密度加权 K-Means | 分散 + 覆盖高密度区 |
| Stage 1 网络 | EGNN 等变流匹配 | 速度场 SE(3) 等变 |
| Stage 2 网络 | Transformer 不变流匹配 | 标量输出，距离特征保证不变性 |
| 密度空间 | 对数空间 z = log(ρ+ε) | 避免负值，数值稳定 |
| 训练策略 | 顺序训练 | 简单稳定；Stage 2 用 Stage 1 生成位置减少分布偏移 |
| CFG drop prob | 0.15 | 平衡有条件/无条件学习 |
| ODE 求解器 | RK4（推理）/ Euler（调试） | RK4 精度更高 |

---

## 九、主要风险与应对

| 风险 | 应对措施 |
|------|----------|
| Stage 1/2 分布偏移 | Stage 2 训练使用 Stage 1 生成的位置 |
| 密度积分不守恒（∫ρdr ≠ Nₑ） | 推理后做归一化后处理 |
| K-Means 在 3M 数据上速度慢 | 离线预处理 + 磁盘缓存（.pt 文件） |
| SE(3) 等变 flow 实现复杂 | 参考 EquiFM / DiffSBDD 已有实现 |

---

## 十、依赖

```
torch
torch_geometric   # VisNet + radius graph
scikit-learn      # MiniBatchKMeans
scipy             # KDTree（已替换为自实现的 chunk 查询）
numpy
```