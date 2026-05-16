# 用户多元异质特征表示学习系统

用户属性预测。TF-IDF + 显式信号 + RandomForest。

## 两个任务

| 任务 | 入口 | 说明 |
|------|------|------|
| **属性预测** | `python main.py` | 7 项用户属性多任务分类，三场景评估 |
| **U13 缺失补全** | `python U13_tasks/main.py` | 数量级分桶均值填充，MAE vs 基线对比 |

---

## 处理流程

### 属性预测流水线

```
labeled_data.csv ──→ [缓存加载] ──→ train/test 切分 ──→ bootstrap 扩充
                                                              │
┌─────────────────────────────────────────────────────────────┘
│
├─ 静态编码 (454d): TF-IDF SVD + 主题信号 + 职业信号 + 子兴趣信号 + 数值
├─ 动态编码 (64d):  短/中/长三尺度行为信号
├─ 社交编码 (64d):  邻域/社区/全局结构信号
│
└─→ 门控融合 (582d) ──→ [异常检测+修正] ──→ [7任务预测] ──→ 三场景评估
```

### U13 缺失补全

```
原始 CSV ──→ 注入 15% 缺失 ──→ 对数分桶 → 同桶均值填充 ──→ MAE vs 基线对比
```

---

## 目录结构

```
├── config.py                全局配置（维度、阈值、路径）
├── main.py                  主流水线入口（缓存→编码→检测→预测→评估）
├── data_loader.py           数据加载 + 清洗 + 衍生特征
├── utils.py                 工具（文本归一化、指标计算）
├── requirements.txt         依赖
├── .gitignore               Git 排除规则
│
├── features/                ── 特征编码层 ──
│   ├── text_embedder.py     TF-IDF + SVD 文本嵌入
│   ├── static_encoder.py    静态特征（嵌入+信号+属性+数值）
│   ├── dynamic_encoder.py   动态行为特征（短/中/长尺度）
│   ├── social_encoder.py    社交结构特征（邻域/社区/全局）
│   └── fusion.py            质量感知门控融合
│
├── labeling/                ── AI 打标层（已缓存，仅首次运行）──
│   ├── labeler.py           增强型打标器
│   ├── keywords.py          关键词库（STRONG_KW/SUB_KW/WEAK_KW）
│   └── rules.py             规则推断（年龄/活跃时段）
│
├── anomaly/                 ── 异常治理层 ──
│   ├── detector.py          IsolationForest 四轴异常检测
│   └── corrector.py         kNN 加权投影修正
│
├── prediction/              ── 属性预测层 ──
│   ├── model.py             RandomForest 多任务分类
│   └── evaluate.py          评估器（召回率/准确率/F1）
│
├── U13_tasks/               ── U13 缺失补全专项 ──
│   ├── main.py              入口：注入→分桶→填充→评估
│   ├── imputer.py           MagnitudeImputer 数量级分桶均值填充
```

---

## 特征编码详情

```
静态特征 454d ──┬── TF-IDF SVD         400d   char_wb ngram(2,5)
                ├── TOPIC_SIGNALS         9d   兴趣关键词命中（与打标器同源）
                ├── OCCUPATION_SIGNALS    9d   职业关键词命中
                ├── SUB_SIGNALS          18d   子兴趣关键词命中
                ├── platform              1d   平台标识
                └── numeric              14d   log值 + 分位桶 + 交叉特征
                                              + CJK/ASCII比例 + verified

动态特征  64d ──── 短/中/长三尺度行为信号
社交特征  64d ──── 邻域/社区/全局结构信号

融合    582d ──── 质量感知自适应门控加权
```

---

## 预测任务（7 项）

| 任务 | 类别数 | 生成方式 |
|------|--------|---------|
| interest_domain | 10 | 关键词匹配（描述×2 + 昵称 + 认证） |
| interest_sub | 18 | 细粒度关键词匹配 |
| active_period | 4 | 发帖密度 + 粉关比决策树 |
| occupation_type | 10 | 关键词匹配 |
| age_group | 4 | 粉丝量级 + 发帖数决策树 |
| account_scale | 5 | 粉丝数五档分桶 |
| content_language | 3 | CJK/ASCII 字符比例 |

---

## 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TFIDF_MAX_FEATURES | 3000 | TF-IDF 最大特征数 |
| TFIDF_SVD_DIM | 400 | SVD 降维目标 |
| STATIC_EMBED_DIM | 454 | 静态特征维度 |
| FUSION_DIM | 582 | 融合向量维度 |
| MISSING_RATE | 0.20 | 测试集缺失注入比例 |
| ANOMALY_RATE | 0.08 | 测试集异常注入比例 |
| TARGET_RECALL | 0.75 | 召回率达标线 |
| TARGET_PRECISION | 0.80 | 准确率达标线 |

---

## 当前结果

目标：平均召回率 ≥ 75%，平均准确率 ≥ 80%

| 任务 | 召回率 | 准确率 | F1 |
|------|--------|--------|-----|
| account_scale | 0.952 | 0.952 | 0.952 |
| content_language | 0.940 | 0.914 | 0.926 |
| interest_sub | 0.829 | 0.904 | 0.857 |
| age_group | 0.816 | 0.944 | 0.860 |
| interest_domain | 0.755 | 0.886 | 0.803 |
| active_period | 0.621 | 0.828 | 0.629 |
| occupation_type | 0.586 | 0.746 | 0.630 |
| **平均** | **0.786** | **0.882** | ✅ |

三个评估场景（修正后/基线/置信门控）全部 PASS。
