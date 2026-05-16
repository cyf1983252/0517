"""
数据加载与增强模块 v3

流程：加载 → 清洗 → 衍生特征 → AI打标 → 返回
打标使用 labeling/ 模块，每行独立推理，无跨用户泄露。

注意：bootstrap 扩充在 main.py 中 train/test 切分之后进行，
     以确保增强数据不会泄露到测试集中。
"""
import re
import numpy as np
import pandas as pd
from typing import Tuple

import config
from labeling import label_dataframe


def _clean_text(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)
        df[f"{col}_norm"] = df[col].apply(
            lambda x: re.sub(r"\s+", " ", str(x).lower().strip()))
    else:
        df[f"{col}_norm"] = ""


def _clean_numeric(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype(int)


def _compute_derived_features(df: pd.DataFrame):
    df["ff_ratio"] = df["followers_count"] / df["friends_count"].clip(lower=1)
    df["log_followers"] = np.log1p(df["followers_count"])
    df["log_friends"] = np.log1p(df["friends_count"])
    df["log_statuses"] = np.log1p(df["statuses_count"])
    df["post_rate"] = df["statuses_count"] / df["followers_count"].clip(lower=1)
    df["interaction_density"] = df["followers_count"] / (df["friends_count"] + df["statuses_count"] + 1)
    if "listed_count" in df.columns:
        df["authority_score"] = df["listed_count"] / df["followers_count"].clip(lower=1)
    else:
        df["authority_score"] = 0


def load_raw_and_label() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载原始 CSV → 清洗 → 衍生特征 → AI打标

    返回:
        tw: Twitter DataFrame (8,669 行，未扩充)
        wb: Weibo DataFrame (10,029 行)
    """
    tw = pd.read_csv(config.TWITTER_FILE, engine="python")
    wb = pd.read_csv(config.WEIBO_FILE, engine="python")

    tw = tw.rename(columns={
        "id": "user_id", "profile_image_url_https": "profile_url",
        "rest_id": "twitter_rest_id"})
    wb = wb.rename(columns={"id": "user_id", "uid": "weibo_uid"})

    for df, platform in [(tw, "twitter"), (wb, "weibo")]:
        df["platform"] = platform
        for col in ["screen_name", "name", "description", "location", "verified_reason"]:
            _clean_text(df, col)
        for col in ["followers_count", "friends_count", "statuses_count",
                     "favourites_count", "listed_count", "media_count",
                     "total_interactions", "comment_cnt", "repost_cnt", "like_cnt"]:
            _clean_numeric(df, col)
        _compute_derived_features(df)

    # ---- AI 打标（每用户独立推理，无跨用户泄露） ----
    print("Labeling Twitter users...")
    tw = label_dataframe(tw, verbose=True)
    print("Labeling Weibo users...")
    wb = label_dataframe(wb, verbose=True)

    print(f"Twitter: {len(tw):,} (labeled)  |  Weibo: {len(wb):,} (labeled)")
    return tw, wb
