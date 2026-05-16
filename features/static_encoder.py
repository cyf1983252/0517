"""
静态特征编码器 v5

特征组成：
  TF-IDF/Sentence-Embedding → 主题信号(9d) → 职业信号(10d) → 属性(1d) → 数值(11d)

关键改进：
  1. 信号特征与打标器同步 — 使用 labeling/labeler.py 的 word_match + 同一关键词库
  2. 数值特征增强 — 增加分位数桶 + 交叉特征
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import config
from .text_embedder import TextEmbedder
from labeling.labeler import word_match
from labeling.keywords import STRONG_KW, SUB_KW, WEAK_KW
from labeling.labeler import _OCCUPATION_CHECKS

# ============================================================
# 显式信号关键词（与打标器同步）
# 使用 labeling/labeler.py 的 word_match 做词边界匹配
# ============================================================

# 兴趣信号：从打标器 STRONG_KW 提取前 20 个最具代表性的关键词
# 按类别排序以保持维度顺序一致
TOPIC_SIGNALS = {cat: kws[:20] for cat, kws in STRONG_KW.items()}

# 职业信号：从打标器的 _OCCUPATION_CHECKS 提取关键词
OCCUPATION_SIGNALS = {}
for kw_list, label in _OCCUPATION_CHECKS:
    if label not in OCCUPATION_SIGNALS:
        OCCUPATION_SIGNALS[label] = []
    OCCUPATION_SIGNALS[label].extend(kw_list[:15])

# 子兴趣信号（18d）：直接从打标器 SUB_KW 导入
SUB_SIGNALS = {cat: kws[:10] for cat, kws in SUB_KW.items()}


def _compute_enhanced_numeric(df: pd.DataFrame) -> np.ndarray:
    """
    增强数值特征（14维）

    基础（7d）:
      log_followers, log_friends, log_statuses,
      ff_ratio, post_rate, interaction_density, authority_score

    数值增强（4d）:
      fol_decile, fri_decile, sta_decile — 分位数桶（0~9）
      log_fol_x_sta — 粉丝×发帖交叉特征

    语言特征（2d）:
      cjk_ratio, ascii_ratio — 字符占比

    认证特征（1d）:
      verified — 是否认证（0/1），occupation_type 强信号
    """
    n = len(df)
    num_data = np.zeros((n, 14), dtype=np.float32)

    # 基础7维
    base_cols = ["log_followers", "log_friends", "log_statuses",
                 "ff_ratio", "post_rate", "interaction_density", "authority_score"]
    for i, col in enumerate(base_cols):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
            num_data[:, i] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    # 分位数桶（7-9）
    fol = np.maximum(df.get("followers_count", pd.Series(0)).fillna(0).values, 0)
    fri = np.maximum(df.get("friends_count", pd.Series(0)).fillna(0).values, 0)
    sta = np.maximum(df.get("statuses_count", pd.Series(0)).fillna(0).values, 0)

    for i, arr in enumerate([fol, fri, sta]):
        col_idx = 7 + i
        non_zero = arr > 0
        if non_zero.any():
            buckets = np.zeros(n, dtype=np.int32)
            ranks = np.argsort(np.argsort(arr[non_zero]))
            buckets[non_zero] = np.clip(ranks * 10 // max(non_zero.sum(), 1), 0, 9)
            num_data[:, col_idx] = buckets.astype(np.float32)

    # 交叉特征：log_followers × log_statuses（10）
    log_fol = num_data[:, 0]
    log_sta = num_data[:, 2]
    num_data[:, 10] = log_fol * log_sta

    # 语言字符比例（11-12）：修复 ngram(2,5) 丢失的单字符语言信号
    descs = df["description_norm"].fillna("").astype(str).tolist()
    for i, desc in enumerate(descs):
        if len(desc) > 0:
            cjk = sum(1 for c in desc if '一' <= c <= '鿿')
            ascii_c = sum(1 for c in desc if c.isascii() and c.isalpha())
            total = len(desc)
            num_data[i, 11] = cjk / total
            num_data[i, 12] = ascii_c / total

    # 认证特征（13）：verified，occupation_type 强信号
    if "verified" in df.columns:
        num_data[:, 13] = pd.to_numeric(df["verified"], errors="coerce").fillna(0).values

    return num_data


class StaticFeatureEncoder:
    """静态特征编码器"""

    def __init__(self):
        self.embedder = TextEmbedder()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.fitted = False
        self.embed_dim = 0
        self._extra_dim = 0

    def _extract_signals(self, df: pd.DataFrame, signal_map: dict) -> np.ndarray:
        """
        使用 word_match（词边界匹配）提取显式信号。
        与打标器 labeler.py 使用相同的关键词和匹配逻辑。
        """
        descs = df["description_norm"].fillna("").tolist()
        names = df.get("name_norm", df["screen_name_norm"]).fillna("").tolist()
        signals = np.zeros((len(df), len(signal_map)), dtype=np.float32)
        for i, (desc, name) in enumerate(zip(descs, names)):
            combined = f"{desc} {desc} {name}".lower()
            for j, (cat, keywords) in enumerate(signal_map.items()):
                signals[i, j] = float(sum(1 for kw in keywords if word_match(kw, combined)))
        return signals

    def fit(self, df: pd.DataFrame):
        desc_texts = df["description_norm"].fillna("").tolist()
        self.embedder.fit(desc_texts)
        self.embed_dim = self.embedder.output_dim

        for col in ["platform"]:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].fillna("missing").astype(str))
                self.label_encoders[col] = le

        topic_dim = len(TOPIC_SIGNALS)
        occ_dim = len(OCCUPATION_SIGNALS)
        sub_dim = len(SUB_SIGNALS)
        attr_dim = len(self.label_encoders)
        num_dim = 14
        self._extra_dim = topic_dim + occ_dim + sub_dim + attr_dim + num_dim

        self.fitted = True
        print(f"Static encoder [TF-IDF]: embed={self.embed_dim}d "
              f"+ topic={topic_dim}+occ={occ_dim}+sub={sub_dim} "
              f"+ attr={attr_dim}+num={num_dim} "
              f"= {self.embed_dim + self._extra_dim}d → target={config.STATIC_EMBED_DIM}d")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Must call fit()")

        # 1. 文本嵌入
        desc_texts = df["description_norm"].fillna("").tolist()
        embed_vec = self.embedder.transform(desc_texts)

        # 2. 显式信号（与打标器同步）
        topic_sig = self._extract_signals(df, TOPIC_SIGNALS)
        occ_sig = self._extract_signals(df, OCCUPATION_SIGNALS)
        sub_sig = self._extract_signals(df, SUB_SIGNALS)  # 18d，直接对应 interest_sub 分类

        # 3. 属性嵌入
        attr_parts = []
        for col in ["platform"]:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                vals = df[col].fillna("missing").astype(str)
                encoded = np.array([le.transform([v])[0] if v in le.classes_ else 0
                                    for v in vals], dtype=np.float32)
                if le.classes_.shape[0] > 1:
                    encoded = 2.0 * encoded / (le.classes_.shape[0] - 1) - 1.0
                attr_parts.append(encoded.reshape(-1, 1))

        # 4. 增强数值特征（11维）
        num_data = _compute_enhanced_numeric(df)

        # 拼接
        attr_mat = np.hstack(attr_parts) if attr_parts else np.zeros((len(df), 0))
        parts = [embed_vec, topic_sig, occ_sig, sub_sig, attr_mat, num_data]
        combined = np.hstack([p for p in parts if p.shape[1] > 0])

        # 标准化
        scaler_mean = getattr(self.scaler, "mean_", None)
        if scaler_mean is None:
            combined = self.scaler.fit_transform(combined)
        else:
            combined = self.scaler.transform(combined)

        # 补齐到目标维度（通常刚好，不需要补零）
        target_dim = config.STATIC_EMBED_DIM
        if combined.shape[1] < target_dim:
            pad = np.zeros((combined.shape[0], target_dim - combined.shape[1]))
            combined = np.hstack([combined, pad])
        elif combined.shape[1] > target_dim:
            combined = combined[:, :target_dim]

        return combined.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)
