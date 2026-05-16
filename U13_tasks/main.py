"""
U13 缺失特征补全 — 多方法对比评估

对比两种填充方法（核心思想：找同量级用户取均值）：
  1. MagnitudeImputer  — 数量级分桶（快速基线）
  2. kNNSimilarImputer — kNN 加权相似度（推荐）
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from U13_tasks.imputer import (MagnitudeImputer,
                                kNNSimilarImputer,
                                evaluate_imputation,
                                print_comparison)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    tw = pd.read_csv(config.TWITTER_FILE, engine="python")
    wb = pd.read_csv(config.WEIBO_FILE, engine="python")
    tw = tw.rename(columns={"id": "user_id", "rest_id": "twitter_rest_id"})
    wb = wb.rename(columns={"id": "user_id", "uid": "weibo_uid"})
    tw["platform"] = "twitter"
    wb["platform"] = "weibo"
    for col in ["followers_count", "friends_count", "statuses_count"]:
        for df in [tw, wb]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype(int)
    for df in [tw, wb]:
        df["ff_ratio"] = df["followers_count"] / df["friends_count"].clip(lower=1)
        df["log_followers"] = np.log1p(df["followers_count"])
        df["log_friends"] = np.log1p(df["friends_count"])
        df["log_statuses"] = np.log1p(df["statuses_count"])
    df_all = pd.concat([tw, wb], ignore_index=True)
    print(f"Loaded: {len(df_all)} users")
    return df_all


def inject_missing(df, rate=0.15, cols=None):
    if cols is None:
        cols = ["followers_count", "friends_count", "statuses_count",
                "log_followers", "log_friends", "log_statuses", "ff_ratio"]
    np.random.seed(42)
    df_orig = df.copy()
    df_miss = df.copy()
    n, n_cols = len(df_miss), len(cols)
    mask = np.zeros((n, n_cols), dtype=np.int8)
    for i, col in enumerate(cols):
        if col not in df_miss.columns:
            continue
        idx = np.random.choice(n, size=int(n * rate), replace=False)
        mask[idx, i] = 1
        df_miss.loc[idx, col] = np.nan
    print(f"Injected {mask.sum()} missing values ({rate*100:.0f}%)")
    return df_miss, df_orig, mask


def main():
    print("=" * 60)
    print("U13 Missing Feature Completion — Method Comparison")
    print("=" * 60)

    df = load_data()
    cols = ["followers_count", "friends_count", "statuses_count",
            "log_followers", "log_friends", "log_statuses", "ff_ratio"]
    df_miss, df_orig, mask = inject_missing(df, rate=0.15, cols=cols)

    all_results = {}

    # Method 1: Magnitude bucketing
    print("\n--- Method 1: MagnitudeImputer ---")
    imputer1 = MagnitudeImputer(buckets_per_order=2)
    imputer1.fit(df_orig, target_cols=cols)
    filled1 = imputer1.transform(df_miss)
    res1 = evaluate_imputation(filled1, df_orig, mask, cols)
    all_results["1 MagnitudeBucketing"] = [res1]

    # Method 2: kNN similarity (recommended)
    print("\n--- Method 2: kNNSimilarImputer ---")
    imputer2 = kNNSimilarImputer(k=20)
    imputer2.fit(df_orig, target_cols=cols)
    filled2 = imputer2.transform(df_miss)
    res2 = evaluate_imputation(filled2, df_orig, mask, cols)
    all_results["2 kNN Imputer"] = [res2]

    # Comparison
    print_comparison(all_results)

    # Save detailed results
    out_df = df_orig[["user_id", "platform"] + cols].copy()
    for col in cols:
        if col in filled2.columns:
            out_df[f"{col}_true"] = df_orig[col].values
            out_df[f"{col}_knn"] = filled2[col].values
            out_df[f"{col}_bucketing"] = filled1[col].values
    out_path = os.path.join(OUTPUT_DIR, "imputation_comparison.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nDetailed results: {out_path}")

    # Best method
    best_name, best_imp = None, -999
    for name, rlist in all_results.items():
        avg = np.mean([r["improvement_pct"] for r in rlist[0].values()])
        print(f"  {name}: avg improvement = {avg:+.1f}%")
        if avg > best_imp:
            best_imp = avg
            best_name = name
    print(f"\nBest: {best_name} ({best_imp:+.1f}%)")


if __name__ == "__main__":
    main()
