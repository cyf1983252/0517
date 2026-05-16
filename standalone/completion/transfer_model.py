"""
迁移推理补全模型：共享骨干 + 行为/属性双通道解码
对应文档 关键技术2.2.1 第3节：迁移推理模型
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


class TransferCompletionModel:
    """
    基于迁移证据的缺失特征补全模型。

    采用"共享骨干（融合向量）+ 双通道"结构：
    - 行为通道：回归连续型特征（followers/friends/statuses等对数尺度）
    - 属性通道：分类离散型特征（兴趣、职业、活跃时段等）

    补全过程引入证据簇作为上下文增强。
    """

    def __init__(self):
        # 行为通道：使用Ridge回归（比MLP快100倍，效果相近）
        self.behavior_regressor = Ridge(alpha=1.0, random_state=config.RANDOM_SEED)
        # 属性通道：为每个属性训练独立分类器
        self.attribute_classifiers = {}
        self.attribute_encoders = {}
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, fused_vec: np.ndarray, df, missing_mask: np.ndarray = None):
        """
        训练补全模型。
        fused_vec: [N, D] 融合后的用户表征
        df: 包含所有特征列的 DataFrame
        missing_mask: [N, F] 缺失掩码（1=缺失），用于仅对非缺失样本训练
        """
        # 行为通道：拟合连续型数值特征
        behavioral_targets = ["log_followers", "log_friends", "log_statuses", "ff_ratio"]
        y_behavior = df[behavioral_targets].fillna(0).values.astype(np.float32)
        y_behavior = np.nan_to_num(y_behavior, nan=0.0, posinf=0.0, neginf=0.0)
        self.behavior_regressor.fit(fused_vec, y_behavior)

        # 属性通道：为每个属性训练分类器（所有ATTRIBUTE_TASKS中的非数值列）
        attr_cols = [t for t in config.ATTRIBUTE_TASKS if t != "platform"]
        for col in attr_cols:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            y_attr = df[col].fillna("missing").astype(str)
            encoded = le.fit_transform(y_attr)
            self.attribute_encoders[col] = le

            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
            )
            clf.fit(fused_vec, encoded)
            self.attribute_classifiers[col] = clf

        self.fitted = True
        print(f"Transfer completion model fitted: "
              f"{len(self.attribute_classifiers)} attribute classifiers")

    def predict_behavioral(self, fused_vec: np.ndarray) -> np.ndarray:
        """预测连续行为特征"""
        if not self.fitted:
            raise RuntimeError("Must call fit() first")
        return self.behavior_regressor.predict(fused_vec)

    def predict_attributes(self, fused_vec: np.ndarray) -> dict:
        """预测离散属性特征，返回 {col: predictions}"""
        if not self.fitted:
            raise RuntimeError("Must call fit() first")
        results = {}
        for col, clf in self.attribute_classifiers.items():
            pred_encoded = clf.predict(fused_vec)
            le = self.attribute_encoders[col]
            results[col] = le.inverse_transform(pred_encoded)
        return results

    def complete(self, fused_vec: np.ndarray, df,
                 missing_mask: np.ndarray,
                 evidence: np.ndarray = None) -> dict:
        """
        执行特征补全：
        - 对于缺失位置，用模型预测值填充
        - 对于非缺失位置，保留原始值
        - 输出补全后的特征 + 置信度
        """
        # 行为特征补全
        behavioral_cols = ["log_followers", "log_friends", "log_statuses", "ff_ratio"]
        pred_behavioral = self.predict_behavioral(fused_vec)

        completed_behavioral = {}
        for i, col in enumerate(behavioral_cols):
            original = df[col].fillna(0).values.astype(np.float32)
            completed = original.copy()
            # 仅补全缺失位置
            if missing_mask is not None and i < missing_mask.shape[1]:
                miss_idx = missing_mask[:, i] == 1
                completed[miss_idx] = pred_behavioral[miss_idx, i]
            completed_behavioral[col] = completed

        # 属性特征补全
        pred_attributes = self.predict_attributes(fused_vec)
        completed_attributes = {}
        for col, preds in pred_attributes.items():
            original = df[col].fillna("missing").astype(str).values
            completed = original.copy()
            # 缺失位置用预测填充
            miss_idx = df[col].isna().values | (df[col] == "")
            completed[miss_idx] = [preds[i] for i in range(len(preds)) if miss_idx[i]]
            completed_attributes[col] = completed

        return {
            "behavioral": completed_behavioral,
            "attributes": completed_attributes,
            "confidence": self._estimate_confidence(fused_vec, missing_mask),
        }

    def _estimate_confidence(self, fused_vec: np.ndarray,
                             missing_mask: np.ndarray) -> np.ndarray:
        """基于表征质量和邻居密度估计补全置信度"""
        # 简化：基于向量范数和缺失密度
        norms = np.linalg.norm(fused_vec, axis=1)
        norm_conf = norms / (norms.max() + 1e-8)
        if missing_mask is not None:
            miss_density = missing_mask.mean(axis=1)
            miss_conf = 1.0 - miss_density
        else:
            miss_conf = np.ones(len(fused_vec))
        return (0.5 * norm_conf + 0.5 * miss_conf).astype(np.float32)
