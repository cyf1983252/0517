"""
多任务属性预测器 — flat 分路专家 + class_weight 平衡
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import config


class AttributePredictor:
    """多任务分类器（独立分路专家）"""

    def __init__(self):
        self.label_encoders = {}
        self.models = {}
        self.fitted = False

    def fit(self, fused_vec: np.ndarray, df,
            quality_signals: np.ndarray = None):
        task_names = [t for t in config.ATTRIBUTE_TASKS if t != "platform"]

        for task in task_names:
            if task not in df.columns:
                continue

            le = LabelEncoder()
            y = df[task].fillna("missing").astype(str).values
            y_encoded = le.fit_transform(y)
            self.label_encoders[task] = le

            n_classes = len(le.classes_)
            if n_classes > 10:
                n_est, max_d = 300, 20
            elif n_classes > 5:
                n_est, max_d = 250, 16
            elif n_classes > 3:
                n_est, max_d = 200, 14
            else:
                n_est, max_d = 150, 12

            if quality_signals is not None:
                sample_weight = np.clip(quality_signals, 0.3, 2.0)
            else:
                sample_weight = None

            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                class_weight="balanced_subsample",
                random_state=config.RANDOM_SEED,
            )
            clf.fit(fused_vec, y_encoded, sample_weight=sample_weight)
            self.models[task] = clf

        self.fitted = True
        print(f"Predictor fitted: {len(self.models)} tasks (flat)")

    def predict(self, fused_vec: np.ndarray) -> dict:
        if not self.fitted:
            raise RuntimeError("Must call fit()")
        results = {}
        for task, clf in self.models.items():
            le = self.label_encoders[task]
            results[task] = le.inverse_transform(clf.predict(fused_vec))
        return results

    def predict_proba(self, fused_vec: np.ndarray) -> dict:
        if not self.fitted:
            raise RuntimeError("Must call fit()")
        return {t: clf.predict_proba(fused_vec) for t, clf in self.models.items()}
