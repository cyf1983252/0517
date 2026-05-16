# 主流水线运行提示词

复制以下内容发给执行 agent：

---

## 任务：运行用户属性预测流水线并报告结果

### 项目说明

跨平台（Twitter + Weibo）用户属性预测系统。数据集已补全、已打标。

目标：评估模型在 7 个属性预测任务上的召回率/准确率是否达标。

### 第一步：进入项目目录

```bash
cd <项目根目录>  # 实际路径以挂载为准
```

### 第二步：安装依赖

```bash
pip install scikit-learn pandas numpy sentence-transformers --break-system-packages
```

装完后，删除项目自带的 sklearn 兼容层避免干扰：

```bash
rm -rf sklearn/ _sklearn_compat.py
```

### 第三步：确认配置

编辑 `config.py`，确认当前优化后的配置：

```python
EMBED_MODE = "dual"                         # 双路融合：TF-IDF 400d + MiniLM 384d
STATIC_EMBED_DIM = 834                       # 784+9+9+17+1+14
FUSION_DIM = 962                             # 834+64+64
```

**本次优化：双路融合（TF-IDF + MiniLM 叠加）**

原理：TF-IDF 的字符 n-gram 精确匹配关键词，MiniLM 的语义向量覆盖无关键词的难例。
RandomForest 每棵树自动选最优维度——对 interest_sub 偏好 TF-IDF，对模糊描述偏好 MiniLM。

**已完成的优化：**

① **信号特征与打标器完全同步**
   - 关键词从打标器 `STRONG_KW` / `SUB_KW` / `_OCCUPATION_CHECKS` 直接导入
   - **新增 SUB_SIGNALS 17d** ← 直接对应 interest_sub 的 18 类分类
   - 匹配使用 `word_match`（词边界正则，防误配）

② **MiniLM 预训练嵌入**
   - 384d 多语言语义向量，替代 TF-IDF 200d
   - 首次运行自动下载模型（~118MB）

③ **增强数值特征（14d）**
   - 分位数桶 3d + 交叉特征 1d + 语言比例 2d + verified 1d

### 第四步：运行流水线

```bash
python3 main.py
```

运行过程：
- 加载缓存 → 特征编码（这次用新配置重新编码）→ 预测 → 评估
- **注意：** 因嵌入维度变了，会自动重新 fit 编码器
- 预计运行时间：约 2-3 分钟（首次需下载 MiniLM 模型 ~118MB）

### 第五步：保存终端输出

把终端全部输出保存到 `output/pipeline_result.txt`。

---

## 你需要反馈的内容

运行完毕后，请告诉我以下信息：

### 1. 三个场景的指标

```
  A (w/ correction):  PASS  or  NOT MET   平均召回率=?  平均准确率=?
  B (baseline):       PASS  or  NOT MET   平均召回率=?  平均准确率=?
  C (gated):          PASS  or  NOT MET   平均召回率=?  平均准确率=?
```

### 2. 逐任务明细（场景 B 或 A 的表格）

哪个任务最高？哪个最低？有没有哪个明显拖后腿？

```
Task                        Recall  Precision       F1
-----------------------------------------------------
interest_domain             ?.????     ?.????   ?.????
interest_sub                ?.????     ?.????   ?.????
active_period               ?.????     ?.????   ?.????
occupation_type             ?.????     ?.????   ?.????
age_group                   ?.????     ?.????   ?.????
account_scale               ?.????     ?.????   ?.????
content_language            ?.????     ?.????   ?.????
-----------------------------------------------------
** AVERAGE **               ?.????     ?.????
```

### 3. 改进建议

本轮是双路融合（TF-IDF + MiniLM），前一轮纯 TF-IDF 结果：
- 平均召回率 77.0%，平均准确率 88.0%（场景A）
- interest_sub 74.6%，occupation_type 55.7%
看双路融合能否进一步拉高。

---

## 流水线架构速览

```
[labeling/]    →  AI打标（已缓存）
[features/]    →  静态834d（TF-IDF 400d + MiniLM 384d + 信号35d + 数值14d）
                 + 动态64d + 社交64d → 门控融合962d
[anomaly/]     →  IsolationForest + kNN投影修正
[prediction/]  →  RandomForest 多任务分类
```

关键文件：

| 文件 | 说明 |
|------|------|
| `config.py` | 全局配置 |
| `data_loader.py` | 数据加载 + 打标 |
| `labeling/labeler.py` | 增强型打标器 |
| `features/fusion.py` | 质量感知门控融合 |
| `prediction/model.py` | 多任务 RandomForest |
| `standalone/completion/` | 缺失补全（独立模块） |
| `U13_tasks/` | 缺失特征补全专项任务 |
| `output/labeled_data.csv` | 已缓存的打标数据 |
| `output/train_indices.npy` | 训练集索引 |
| `output/test_indices.npy` | 测试集索引 |
