"""
多模态融合模块：静态-动态协同学习 + 质量感知门控
对应文档 关键技术2.1.1 第3节：静态—动态特征协同学习与门控融合
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

import config


def _pad_to_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """将向量补齐或截断到目标维度"""
    if vec.shape[1] < target_dim:
        pad = np.zeros((vec.shape[0], target_dim - vec.shape[1]), dtype=vec.dtype)
        return np.hstack([vec, pad])
    elif vec.shape[1] > target_dim:
        return vec[:, :target_dim]
    return vec


class MultiModalFusion:
    """
    质量感知门控融合：将静态、动态、社交三类特征向量
    通过可学习的门控机制融合为统一的用户表征。
    z = α * zs + β * zd + γ * zsocial
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        self.alpha = 0.45
        self.beta = 0.30
        self.gamma = 0.25
        self.target_dim = config.FUSION_DIM

    def _compute_quality_signals(self, static_vec, dynamic_vec, social_vec):
        """基于表征质量计算门控调整因子"""
        static_var = np.var(static_vec, axis=1).mean()
        dynamic_var = np.var(dynamic_vec, axis=1).mean()
        social_var = np.var(social_vec, axis=1).mean()
        total_var = static_var + dynamic_var + social_var + 1e-8

        q_static = static_var / total_var
        q_dynamic = dynamic_var / total_var
        q_social = social_var / total_var

        alpha = self.alpha * (0.7 + 0.6 * q_static)
        beta = self.beta * (0.7 + 0.6 * q_dynamic)
        gamma = self.gamma * (0.7 + 0.6 * q_social)

        total = alpha + beta + gamma + 1e-8
        return alpha / total, beta / total, gamma / total

    def fit(self, static_vec: np.ndarray, dynamic_vec: np.ndarray,
            social_vec: np.ndarray):
        """拟合融合层的标准化参数"""
        # 先补齐到统一维度
        s = _pad_to_dim(static_vec, self.target_dim)
        d = _pad_to_dim(dynamic_vec, self.target_dim)
        soc = _pad_to_dim(social_vec, self.target_dim)

        fused = self.alpha * s + self.beta * d + self.gamma * soc
        self.scaler.fit(fused)
        self.fitted = True
        print(f"Fusion fitted: inputs ({static_vec.shape[1]}/{dynamic_vec.shape[1]}/{social_vec.shape[1]}) → {self.target_dim}D")

    def transform(self, static_vec: np.ndarray, dynamic_vec: np.ndarray,
                  social_vec: np.ndarray) -> np.ndarray:
        """门控融合三个视图的表征"""
        if not self.fitted:
            raise RuntimeError("Must call fit() before transform()")

        s = _pad_to_dim(static_vec, self.target_dim)
        d = _pad_to_dim(dynamic_vec, self.target_dim)
        soc = _pad_to_dim(social_vec, self.target_dim)

        a, b, g = self._compute_quality_signals(s, d, soc)
        fused = a * s + b * d + g * soc
        fused = self.scaler.transform(fused)
        return fused.astype(np.float32)

    def fit_transform(self, static_vec, dynamic_vec, social_vec) -> np.ndarray:
        s = _pad_to_dim(static_vec, self.target_dim)
        d = _pad_to_dim(dynamic_vec, self.target_dim)
        soc = _pad_to_dim(social_vec, self.target_dim)
        fused = self.alpha * s + self.beta * d + self.gamma * soc
        self.scaler.fit(fused)
        self.fitted = True
        print(f"Fusion fitted: inputs ({static_vec.shape[1]}/{dynamic_vec.shape[1]}/{social_vec.shape[1]}) → {self.target_dim}D")
        return self.transform(static_vec, dynamic_vec, social_vec)
