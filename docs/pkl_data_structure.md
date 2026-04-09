# EDBench PKL 数据集结构说明

数据文件：`src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl`

---

## 顶层结构

```python
import pickle

with open("src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl", "rb") as f:
    data = pickle.load(f)

# type: dict，共 47,986 个分子
# 键：分子 ID（字符串），值：分子记录（dict）
{
    "482085": {...},
    "2699112": {...},
    ...
}
```

---

## 单条分子记录

每条记录是一个 `dict`，包含两个子字典：

```python
data["482085"] = {
    "mol": {
        "x":      np.ndarray,   # (N_atoms,)    int64    原子序数
        "coords": np.ndarray,   # (N_atoms, 3)  float32  原子坐标（Bohr）
    },
    "electronic_density": {
        "coords":  np.ndarray,  # (M, 3)  float32  电子密度点坐标（Bohr）
        "density": np.ndarray,  # (M,)    float32  对应点的密度值（a.u.）
    }
}
```

### 字段说明

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `mol.x` | `(N_atoms,)` | int64 | 原子序数（1=H, 6=C, 7=N, 8=O, …） |
| `mol.coords` | `(N_atoms, 3)` | float32 | 原子三维坐标，单位 Bohr，范围约 [-15, +15] |
| `electronic_density.coords` | `(M, 3)` | float32 | 电子密度采样点坐标，单位 Bohr |
| `electronic_density.density` | `(M,)` | float32 | 各采样点的电子密度值，单位 a.u. |

### 示例数值（分子 ID `482085`，28 个原子）

```
mol.x:      [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 1, 1, ...]
             ← 11个C, 3个N, 1个O, 13个H

mol.coords: shape=(28, 3), 范围约 [-6, +13] Bohr

electronic_density.coords:  shape=(6377, 3)
electronic_density.density: shape=(6377,), 值约 0.05 ~ 0.1 a.u.
```

---

## 文件名含义

`mol_EDthresh0.05_data.pkl` 中的 **`EDthresh0.05`** 表示电子密度阈值为 **0.05 a.u.**。

原始 `.cube` 文件中包含数十万个网格点，其中大量位于真空区域（密度接近 0）。数据集在生成时已过滤掉密度低于 0.05 a.u. 的点，只保留化学上有意义的高密度区域，最终每个分子约剩余数千个点（`M` 的典型值为 2000–10000）。

---

## 数据流水线

### PKL → 训练样本

`EDBenchPKLDataset._process()` 对每个分子执行以下处理，结果缓存为 `.pt` 文件：

```
pkl["mol_id"]
    ↓
1. 数据类型转换
   atom_types  = mol.x          → torch.int64   (N,)
   atom_coords = mol.coords     → torch.float32 (N, 3)
   ed_coords   = e_d.coords     → np.float32    (M, 3)
   ed_dens     = e_d.density    → np.float32    (M,)

2. 密度加权 K-Means 聚类
   K = n_per_atom × N_atoms     （默认 n_per_atom=8）
   cluster_pointcloud(ed_coords, ed_dens, K)
   → point_positions            (K, 3)  float32  聚类中心坐标
   → center_densities           (K,)    float32  最近邻源点密度

3. 对数密度
   point_log_densities = log(center_densities + 1e-10)   (K,) float32
```

单个样本输出：

```python
{
    "atom_coords":          torch.Tensor,  # (N, 3)
    "atom_types":           torch.Tensor,  # (N,)
    "point_positions":      torch.Tensor,  # (K, 3)
    "point_log_densities":  torch.Tensor,  # (K,)
}
```

### 样本 → 批次

`collate_fn()` 将可变大小的样本合并为模型输入批次：

```python
batch = {
    "atom_coords":          torch.Tensor,  # (sum_N, 3)   所有分子原子坐标拼接
    "atom_types":           torch.Tensor,  # (sum_N,)     所有原子序数拼接
    "point_positions":      torch.Tensor,  # (sum_K, 3)   所有密度点坐标拼接
    "point_log_densities":  torch.Tensor,  # (sum_K,)     所有对数密度拼接
    "atom_batch":           torch.Tensor,  # (sum_N,)     各原子所属批次索引
    "point_batch":          torch.Tensor,  # (sum_K,)     各密度点所属批次索引
}
```

`atom_batch` 和 `point_batch` 是 PyTorch Geometric 风格的批次索引，例如批次大小为 3、各分子有 [5, 7, 4] 个原子时：

```
atom_batch = [0,0,0,0,0, 1,1,1,1,1,1,1, 2,2,2,2]
```

---

## 缓存机制

PKL 文件约 9 GB，不在内存中完整加载。`EDBenchPKLDataset` 在首次访问分子时将处理结果保存为：

```
src/data/ed_energy_5w/cache/{mol_id}_n{n_per_atom}.pt
```

后续 epoch 直接读取缓存，显著加速多轮次训练。

---

## 相关源文件

| 文件 | 说明 |
|------|------|
| `src/data/dataset.py` | `EDBenchPKLDataset`、`collate_fn` 实现 |
| `src/data/clustering.py` | 密度加权 K-Means（`cluster_pointcloud`） |
| `src/data/cube_parser.py` | Gaussian `.cube` 文件解析（数据预处理阶段使用） |
