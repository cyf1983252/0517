"""
异常特征检测模块：时间/语义/结构/索引四轴一致性驱动检测
对应文档 关键技术2.2.2 第1-2节：异常模式刻画与多源检测
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config


class AnomalyDetector:
    """
    多维一致性驱动的异常检测器。

    在四个轴上构建异常证据向量 ψ = [rtime, revent, rnbr, rindex, ...]
    并输出融合异常分数 S。
    """

    def __init__(self, contamination: float = 0.08):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=config.RANDOM_SEED,
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def _build_evidence_vector(self, fused_vec: np.ndarray,
                               semantic_vec: np.ndarray,
                               social_vec: np.ndarray) -> np.ndarray:
        """
        构建四轴异常证据向量：
        - rtime: 相邻帧差异（时间残差）
        - revent: 语义锚内偏离（语义残差）
        - rnbr: 邻域拉普拉斯差（结构残差）
        - rindex: 重构残差（向量空间离群度）
        """
        n, d = fused_vec.shape

        # rtime: 与前一帧的差异幅度
        prev = np.roll(fused_vec, shift=1, axis=0)
        rtime = np.linalg.norm(fused_vec - prev, axis=1)

        # revent: 与语义均值的马氏距离
        sem_mean = semantic_vec.mean(axis=0)
        sem_std = semantic_vec.std(axis=0) + 1e-8
        revent = np.linalg.norm((semantic_vec - sem_mean) / sem_std, axis=1)

        # rnbr: 与社交均值的偏离
        soc_mean = social_vec.mean(axis=0)
        soc_std = social_vec.std(axis=0) + 1e-8
        rnbr = np.linalg.norm((social_vec - soc_mean) / soc_std, axis=1)

        # rindex: 向量空间中的离群度（到kNN质心的距离）
        # 简化为：到全局均值的距离
        fused_mean = fused_vec.mean(axis=0)
        rindex = np.linalg.norm(fused_vec - fused_mean, axis=1)

        # 附加信号
        interval_var = np.zeros(n)   # 简化占位
        entropy = np.zeros(n)
        diversity = np.ones(n) * 0.5

        psi = np.column_stack([
            rtime, revent, rnbr, rindex,
            interval_var, entropy, diversity,
        ])
        return psi

    def fit(self, fused_vec: np.ndarray, semantic_vec: np.ndarray,
            social_vec: np.ndarray):
        """在训练数据上拟合检测器"""
        psi = self._build_evidence_vector(fused_vec, semantic_vec, social_vec)
        psi_scaled = self.scaler.fit_transform(psi)
        self.isolation_forest.fit(psi_scaled)
        self.fitted = True
        print(f"Anomaly detector fitted: contamination={self.contamination}")

    def detect(self, fused_vec: np.ndarray, semantic_vec: np.ndarray,
               social_vec: np.ndarray) -> np.ndarray:
        """
        检测异常，返回：
        - anomaly_scores: [0,1] 之间的异常分数（越高越异常）
        - anomaly_labels: 0=正常, 1=异常
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() first")
        psi = self._build_evidence_vector(fused_vec, semantic_vec, social_vec)
        psi_scaled = self.scaler.transform(psi)

        # IsolationForest 返回 -1(异常)/1(正常)，转换为 0/1
        raw_labels = self.isolation_forest.predict(psi_scaled)
        labels = (raw_labels == -1).astype(int)

        # 异常分数：使用决策函数（越低越异常）取反
        raw_scores = self.isolation_forest.decision_function(psi_scaled)
        scores = 1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)

        return scores, labels
