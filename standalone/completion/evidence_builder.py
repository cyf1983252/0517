"""
可迁移证据构造模块：时间邻近、语义近邻、结构邻域、事件锚定
对应文档 关键技术2.2.1 第1节：可迁移证据构造与对齐
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EvidenceBuilder:
    """
    为每个待补全目标构造多源证据簇：
    - 时间邻近：同一用户历史窗口片段
    - 语义近邻：语义空间中 k-NN 用户
    - 结构邻域：社交图中强关联邻居
    - 事件锚定：相同话题/内容簇
    """

    def __init__(self, k_neighbors: int = 10):
        self.k = k_neighbors
        self._semantic_index = None   # 语义向量矩阵 [N, D]
        self._social_index = None     # 社交向量矩阵 [N, D]
        self._user_ids = None

    def build_indices(self, semantic_vec: np.ndarray, social_vec: np.ndarray,
                      user_ids: np.ndarray):
        """构建语义和社交近邻索引"""
        self._semantic_index = semantic_vec
        self._social_index = social_vec
        self._user_ids = user_ids

    def get_semantic_neighbors(self, vec: np.ndarray, k: int = None) -> np.ndarray:
        """在语义空间中找到 k 近邻的索引"""
        if k is None:
            k = self.k
        sim = cosine_similarity(vec.reshape(1, -1), self._semantic_index)[0]
        # 排除自身
        top_k = np.argsort(sim)[::-1][1:k + 1]
        return top_k

    def get_structural_neighbors(self, vec: np.ndarray, k: int = None) -> np.ndarray:
        """在社交结构空间中找到 k 近邻"""
        if k is None:
            k = self.k
        sim = cosine_similarity(vec.reshape(1, -1), self._social_index)[0]
        top_k = np.argsort(sim)[::-1][1:k + 1]
        return top_k

    def build_evidence_cluster(self, idx: int, fused_vec: np.ndarray,
                               semantic_vec: np.ndarray,
                               social_vec: np.ndarray) -> dict:
        """
        为指定用户构建证据簇，返回：
        - temporal: 暂无时间序列数据时使用自身向量+扰动模拟
        - semantic_knn: 语义近邻的融合向量均值
        - structural_knn: 结构近邻的融合向量均值
        - combined: 三者的加权组合
        """
        n = len(fused_vec)
        evidence = {}

        # 时间邻近证据：自身向量 + 小扰动（模拟相邻时间片）
        noise = np.random.randn(*fused_vec[idx].shape).astype(np.float32) * 0.05
        evidence["temporal"] = fused_vec[idx] + noise

        # 语义近邻证据
        sem_nbrs = self.get_semantic_neighbors(semantic_vec[idx])
        if len(sem_nbrs) > 0:
            evidence["semantic"] = fused_vec[sem_nbrs].mean(axis=0)
        else:
            evidence["semantic"] = fused_vec[idx].copy()

        # 结构近邻证据
        soc_nbrs = self.get_structural_neighbors(social_vec[idx])
        if len(soc_nbrs) > 0:
            evidence["structural"] = fused_vec[soc_nbrs].mean(axis=0)
        else:
            evidence["structural"] = fused_vec[idx].copy()

        # 事件锚定证据：语义空间中最相似的非自身用户
        evidence["event_anchor"] = fused_vec[
            self.get_semantic_neighbors(semantic_vec[idx], k=5)
        ].mean(axis=0)

        # 加权融合证据（权重：时间 > 语义 > 结构 > 事件锚）
        evidence["combined"] = (
            0.40 * evidence["temporal"] +
            0.30 * evidence["semantic"] +
            0.20 * evidence["structural"] +
            0.10 * evidence["event_anchor"]
        )

        return evidence
