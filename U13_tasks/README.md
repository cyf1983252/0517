# U13 总发帖数 — 缺失特征补全任务

## 1. 项目概述

### 背景

在用户数据中，**数值特征缺失**是普遍问题。比如一个用户的粉丝数抓取失败、发帖数字段为空等。简单粗暴地用 0 填充或全局均值填充会严重扭曲数据分布。

### 核心思想

> **一个用户的发帖数缺了 → 找发帖数在同量级的其他用户 → 取他们的粉丝数均值来填充**

以"同量级用户更相似"为假设，用信息完整的用户去估计信息缺失的用户。

## 2. 详细处理流程

```
原始数据
    │
    ▼
Step 1: 数据清洗
    │  ├─ 数值列标准化（to_numeric → fillna → clip）
    │  └─ 衍生特征计算（log变换、ff_ratio等）
    │
    ▼
Step 2: 模拟缺失
    │  └─ 从18,698用户中随机选15%行，每列独立注入NaN
    │
    ▼
Step 3: 方法一·数量级分桶填充
    │  ├─ 对 target_col 选一个辅助列 aux_col
    │  ├─ aux_col 按 log10 分桶（每半量级一桶）
    │  └─ 缺失值 = 同桶内非缺失用户的均值
    │
    ▼
Step 4: 方法二·kNN 加权填充（推荐）
    │  ├─ 将所有特征标准化（自动检测偏态列做 log 变换）
    │  ├─ 对每个缺失用户，在完整数据中找 k 个最近邻
    │  ├─ 距离 = 标准化空间中的欧几里得距离（仅用有效维度）
    │  ├─ 排除自身（防止数据泄露）
    │  └─ 缺失值 = 邻域加权平均（权重 = 距离倒数）
    │
    ▼
Step 5: 效果评估
    ├─ 对比：填充值 vs 真实值
    ├─ 指标：MAE、RMSE
    └─ 基线：全局均值填充
```

### 两种方法原理对比

| 方面 | MagnitudeBucketing | kNN Imputer ⭐ |
|------|-------------------|---------------|
| **分组方式** | 单维对数量级分桶 | 多维空间距离度量 |
| **信息利用** | 只用 1 个辅助列 | 利用**全部**有效特征 |
| **填充方式** | 桶内简单均值 | 距离加权平均 |
| **尺度处理** | 无 | 自动 log 变换 + 标准化 |
| **粒度** | 粗（半量级一桶） | 细（连续空间最近邻） |
| **预期 MAE** | 略优于基线 | **显著优于基线** |

---

## 3. 代码结构

```
U13_tasks/
│
├── __init__.py          # 包入口，导出 MagnitudeImputer
│
├── imputer.py           核心实现
│   ├── MagnitudeImputer      # 方法一：数量级分桶
│   │   ├── __init__()        # 设置桶密度
│   │   ├── _log_bucket()     # 数值→对数分桶映射
│   │   ├── _find_aux()       # 自动选择最佳辅助列
│   │   ├── fit()             # 在完整数据上学习分桶均值
│   │   ├── transform()       # 对缺失数据执行填充
│   │   └── fit_transform()   # 一步完成
│   │
│   ├── kNNSimilarImputer     # 方法二：kNN 加权（推荐）
│   │   ├── __init__()        # 设置 k 值、随机种子
│   │   ├── _should_log()     # 检测偏态列（自动决定 log 变换）
│   │   ├── fit()             # 标准化 + 构建参考集
│   │   ├── transform()       # 对每行缺失值找 kNN + 加权填充
│   │   └── fit_transform()   # 一步完成
│   │
│   ├── evaluate_imputation() # 评估函数（MAE/RMSE/基线对比）
│   └── print_comparison()    # 多方法对比表格打印
│
├── main.py              运行入口
│   ├── load_data()           # 加载 + 清洗
│   ├── inject_missing()      # 模拟注入 15% 缺失值
│   └── main()                # 编排完整流程
│
├── README.md            本文件
│
└── output/              运行输出
    ├── imputation_results.csv     # 分桶法填充结果
    ├── imputation_comparison.csv  # 多方法对比结果
    └── run_log.txt                # 终端输出日志
```

### 各文件作用

| 文件 | 作用 |
|------|------|
| `imputer.py` | 核心算法实现。两种填充方法 + 评估工具 |
| `main.py` | 运行入口。编排数据加载→模拟缺失→填充→评估→对比 |
| `output/imputation_comparison.csv` | 每用户每列的 true/knn/bucketing 对比 |

---

## 4. 运行说明

### 环境要求

```bash
pip install numpy pandas scikit-learn
```

### 运行

```bash
cd U13_tasks
python3 main.py
```

约 **3-5 秒**完成。

### 输出解读

终端会打印：

```
Loaded: 18698 users
Injected 19628 missing values (15%)
```

然后两种方法的对比表格：

```
  followers_count:
    MagnitudeBucketing    MAE=2139117    Baseline=2187743     +2.2%
    kNN Imputer           MAE=331163     Baseline=2216511    +85.1%
```

- **MAE**：平均绝对误差（越小越好）
- **Baseline**：全局均值填充的 MAE
- **Improve**：`(Baseline - MAE) / Baseline`，正数表示比基线好


## 7. 总结

| 项目 | 说明 |
|------|------|
| **最佳方法** | kNN 加权填充（平均改善 **+65.5%**） |
| **最适用场景** | 中小用户（粉丝 < 10万）的数值特征缺失 |
| **局限** | 极值用户（> 500万粉）样本稀疏，填充误差大 |
| **后续优化** | 引入文本相似度、MICE 迭代回归、分群体建模 |
