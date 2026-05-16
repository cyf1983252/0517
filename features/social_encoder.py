"""
社交结构特征编码器：图结构建模、邻域表示、社区发现
对应文档 关键技术2.1.2 的社交结构特征独立编码
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


class SocialFeatureEncoder:
    """
    从粉丝/关注/列表等关系指标中提取社交结构特征。
    在无真实图结构的情况下，用统计特征近似社交网络位置。
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def _extract_social_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取多层次社交结构特征"""
        followers = df["followers_count"].fillna(0).values.astype(np.float32)
        friends = df["friends_count"].fillna(0).values.astype(np.float32)
        statuses = df["statuses_count"].fillna(0).values.astype(np.float32)
        listed = df.get("listed_count", pd.Series([0] * len(df))).fillna(0).values.astype(np.float32)

        ff_ratio = followers / np.maximum(friends, 1)
        eng_rate = statuses / np.maximum(followers, 1)

        # 邻域级：局部关系密度
        neighborhood = np.stack([
            np.log1p(friends) / 10.0,                               # 关注规模
            np.log1p(listed) / 5.0,                                 # 被列表收录（权威度）
            np.clip(friends / np.maximum(followers + friends, 1), 0, 1),  # 互关倾向
        ], axis=1)

        # 社区级：在平台中的位置
        community = np.stack([
            np.log1p(followers) / 15.0,                             # 粉丝层级
            np.log1p(ff_ratio) / 5.0,                               # 影响力
            np.clip(eng_rate * 100, 0, 1),                          # 互动率（压缩）
        ], axis=1)

        # 全局级：跨平台可比的结构特征
        global_feat = np.stack([
            np.clip(np.log1p(followers + friends + statuses) / 15.0, 0, 1),  # 综合规模
            np.clip(ff_ratio / (ff_ratio + 10), 0, 1),                        # 归一化粉关比
            np.clip(listed / np.maximum(followers, 1), 0, 1),                  # 权威度比率
        ], axis=1)

        return np.hstack([neighborhood, community, global_feat]).astype(np.float32)

    def fit(self, df: pd.DataFrame):
        feats = self._extract_social_features(df)
        self.scaler.fit(feats)
        self.fitted = True
        print(f"Social encoder fitted: {feats.shape[1]} feature dimensions")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Must call fit() before transform()")
        feats = self._extract_social_features(df)
        scaled = self.scaler.transform(feats)

        target_dim = config.SOCIAL_EMBED_DIM
        if scaled.shape[1] < target_dim:
            pad = np.zeros((scaled.shape[0], target_dim - scaled.shape[1]))
            scaled = np.hstack([scaled, pad])
        elif scaled.shape[1] > target_dim:
            scaled = scaled[:, :target_dim]
        return scaled.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)
