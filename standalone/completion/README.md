# 缺失特征补全模块

从主流水线 `main.py` 中拆出的独立模块。

## 功能

- **EvidenceBuilder** — 四源证据构造（时间/语义/结构/事件锚）
- **TransferCompletionModel** — 迁移推理补全（Ridge + RandomForest）
- **ConsistencyChecker** — 三维一致性校核（时间/语义/结构）

## 使用场景

数据集存在缺失值时，在特征编码之后、属性预测之前使用。

## 原流水线位置

`main.py` Step 4：缺失特征补全 → 一致性校核

## 移除原因

当前数据集已是补全后的，无需此步骤。保留代码以便在需要时复用。
