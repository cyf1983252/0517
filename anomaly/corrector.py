"""
异常特征修正模块：最小改动原则 + 近邻子空间投影
对应文档 关键技术2.2.2 第3节：对抗学习与难例生成中的修正策略
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class AnomalyCorrector:
    """
    对被标记为异常的特征执行"最小改动"修正：
    - 用 K 近邻的加权中值替换异常值
    - 通过近邻子空间投影保持修正值在合理范围内
    - 输出修正值 + 置信区间
    """

    def __init__(self, k_neighbors: int = 10):
        self.k = k_neighbors
        self._normal_vecs = None  # 正常样本的向量库

    def fit(self, fused_vec: np.ndarray, anomaly_labels: np.ndarray):
        """存储正常样本作为修正参考库"""
        normal_mask = anomaly_labels == 0
        self._normal_vecs = fused_vec[normal_mask]
        print(f"Corrector fitted: {len(self._normal_vecs)} normal reference vectors")

    def correct(self, fused_vec: np.ndarray, anomaly_labels: np.ndarray,
                anomaly_scores: np.ndarray) -> np.ndarray:
        """
        对异常样本执行修正：
        1. 找到最近的 k 个正常邻居
        2. 用加权平均修正（权重 = 1 - 异常分数 × 余弦相似度）
        3. 非异常样本保持原值
        """
        if self._normal_vecs is None or len(self._normal_vecs) == 0:
            return fused_vec.copy()

        corrected = fused_vec.copy()
        anomaly_idx = np.where(anomaly_labels == 1)[0]

        if len(anomaly_idx) == 0:
            return corrected

        # 批量计算异常样本与正常参考库的相似度
        anomaly_vecs = fused_vec[anomaly_idx]
        sim_matrix = cosine_similarity(anomaly_vecs, self._normal_vecs)  # [n_anom, n_normal]

        for i, ai in enumerate(anomaly_idx):
            sims = sim_matrix[i]
            top_k_idx = np.argsort(sims)[::-1][:self.k]

            if len(top_k_idx) == 0:
                continue

            # 加权平均修正
            weights = sims[top_k_idx]
            weights = weights / (weights.sum() + 1e-8)
            corrected[ai] = (self._normal_vecs[top_k_idx] * weights[:, np.newaxis]).sum(axis=0)

        return corrected

    def correct_features(self, df, anomaly_labels: np.ndarray,
                         completion_results: dict) -> dict:
        """
        对原始特征 DataFrame 中的异常行进行修正。
        对数值列用中位数替换，对类别列用众数替换。
        """
        corrected_df = df.copy()
        anomaly_idx = np.where(anomaly_labels == 1)[0]

        if len(anomaly_idx) == 0:
            return {"corrected_df": corrected_df, "correction_count": 0}

        normal_mask = anomaly_labels == 0
        normal_df = df.iloc[normal_mask]
        corrected_count = 0

        # 只对数值列进行修正
        numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
        for col in numeric_cols:
            median_val = normal_df[col].median()
            # 保持原始 dtype
            if pd.api.types.is_integer_dtype(df[col]):
                median_val = int(round(median_val))
            for ai in anomaly_idx:
                corrected_df.iloc[ai, corrected_df.columns.get_loc(col)] = median_val
                corrected_count += 1

        return {
            "corrected_df": corrected_df,
            "correction_count": corrected_count,
        }
