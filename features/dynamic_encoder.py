"""
动态行为特征编码器：行为事件归一、时序窗口聚合、多尺度表示
对应文档 关键技术2.1.1 的动态行为建模部分
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


class DynamicFeatureEncoder:
    """
    从社交指标中提取动态行为特征。
    由于原始数据没有真实行为日志，基于 followers/friends/statuses
    等指标模拟短/中/长三个时间尺度的行为特征。
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def _extract_dynamic_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        提取多尺度行为信号：
        - 短期（周）：基于 statuses 增量的活跃度模拟
        - 中期（月）：基于关注/粉丝比的变化模拟
        - 长期（年）：基于账号特征的综合节奏
        """
        n = len(df)
        # 基础指标
        followers = df["followers_count"].fillna(0).values.astype(np.float32)
        friends = df["friends_count"].fillna(0).values.astype(np.float32)
        statuses = df["statuses_count"].fillna(0).values.astype(np.float32)
        ff_ratio = followers / np.maximum(friends, 1)

        # — 短期信号（高频变化） —
        short_term = np.stack([
            np.log1p(statuses) / np.maximum(np.log1p(followers), 1),       # 单位粉丝的发帖率
            np.log1p(friends) / np.maximum(np.log1p(followers), 1),        # 互关倾向
            np.clip(np.log1p(statuses) / 10.0, 0, 1),                      # 发帖活跃度（压缩）
        ], axis=1)

        # — 中期信号（阶段主题） —
        mid_term = np.stack([
            np.log1p(followers) / 15.0,                                     # 影响力级别
            np.log1p(ff_ratio) / 5.0,                                       # 粉关比
            np.clip(friends / np.maximum(followers, 1), 0, 1),             # 关注倾向
        ], axis=1)

        # — 长期信号（稳态习惯） —
        long_term = np.stack([
            np.log1p(statuses) / np.maximum(np.log1p(followers + friends + 1), 1),  # 综合活跃
            np.ones(n) * 0.5,                                                         # 平台粘性（占位）
            1.0 - np.exp(-np.log1p(statuses) / 10.0),                                 # 饱和型活跃度
        ], axis=1)

        return np.hstack([short_term, mid_term, long_term]).astype(np.float32)

    def fit(self, df: pd.DataFrame):
        signals = self._extract_dynamic_signals(df)
        self.scaler.fit(signals)
        self.fitted = True
        print(f"Dynamic encoder fitted: {signals.shape[1]} signal dimensions")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Must call fit() before transform()")
        signals = self._extract_dynamic_signals(df)
        scaled = self.scaler.transform(signals)

        # 补齐到目标维度
        target_dim = config.DYNAMIC_EMBED_DIM
        if scaled.shape[1] < target_dim:
            pad = np.zeros((scaled.shape[0], target_dim - scaled.shape[1]))
            scaled = np.hstack([scaled, pad])
        elif scaled.shape[1] > target_dim:
            scaled = scaled[:, :target_dim]
        return scaled.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)
