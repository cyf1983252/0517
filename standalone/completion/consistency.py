"""
一致性约束模块：时间/语义/结构三维一致性
对应文档 关键技术2.2.1 第4节：弱监督融合中的三维一致性约束
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _align_dims(v1: np.ndarray, v2: np.ndarray) -> tuple:
    """将两个矩阵补齐到相同维度"""
    d1, d2 = v1.shape[1], v2.shape[1]
    target = max(d1, d2)
    if d1 < target:
        pad = np.zeros((v1.shape[0], target - d1), dtype=v1.dtype)
        v1 = np.hstack([v1, pad])
    if d2 < target:
        pad = np.zeros((v2.shape[0], target - d2), dtype=v2.dtype)
        v2 = np.hstack([v2, pad])
    return v1, v2


class ConsistencyChecker:
    """
    对补全结果执行三维一致性校核：
    - 时间一致性：相邻窗口不应突变（平滑约束）
    - 语义一致性：同话题锚内分布应对齐
    - 结构一致性：强关联邻居应保持协同
    """

    def __init__(self, smooth_weight: float = 0.1, semantic_weight: float = 0.3,
                 structural_weight: float = 0.2):
        self.smooth_w = smooth_weight
        self.semantic_w = semantic_weight
        self.structural_w = structural_weight

    def check_time_consistency(self, current: np.ndarray,
                               previous: np.ndarray) -> np.ndarray:
        """
        时间一致性分数：衡量当前窗口与前一窗口的变化幅度。
        score = max(0, 1 - |Δ|/σ)，越接近1越一致
        """
        diff = np.abs(current - previous)
        sigma = np.std(previous, axis=0) + 1e-8
        scores = np.mean(np.clip(1.0 - diff / (sigma * 3.0), 0, 1), axis=1)
        return scores

    def check_semantic_consistency(self, vec: np.ndarray,
                                   anchor_prototypes: np.ndarray) -> np.ndarray:
        """
        语义一致性分数：衡量目标向量与事件/话题锚原型的相似度
        """
        if len(anchor_prototypes) == 0:
            return np.ones(len(vec)) * 0.5
        v, a = _align_dims(vec, anchor_prototypes)
        sim = cosine_similarity(v, a)
        return sim.max(axis=1)

    def check_structural_consistency(self, vec: np.ndarray,
                                     neighbor_vecs: np.ndarray) -> np.ndarray:
        """
        结构一致性分数：衡量个体与邻域均值的偏差
        """
        if len(neighbor_vecs) == 0:
            return np.ones(len(vec)) * 0.5
        v, n = _align_dims(vec, neighbor_vecs)
        sim = cosine_similarity(v, n)
        return sim.mean(axis=1)

    def compute_overall_consistency(self, fused_vec: np.ndarray,
                                    semantic_vec: np.ndarray,
                                    social_vec: np.ndarray) -> np.ndarray:
        """
        综合三维一致性分数：
        S_consistency = 0.35*temporal + 0.35*semantic + 0.30*structural
        """
        n = len(fused_vec)

        # 时间一致性：使用相邻索引作为"前后时间片"的近似
        prev_vec = np.roll(fused_vec, shift=1, axis=0)
        time_score = self.check_time_consistency(fused_vec, prev_vec)

        # 语义一致性：使用自身语义向量作为锚原型
        sem_score = self.check_semantic_consistency(fused_vec, semantic_vec)

        # 结构一致性：使用社交向量邻域
        soc_score = self.check_structural_consistency(fused_vec, social_vec)

        overall = (0.35 * time_score + 0.35 * sem_score + 0.30 * soc_score)
        return np.clip(overall, 0, 1)
