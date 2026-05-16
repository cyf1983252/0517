"""
主流水线（缓存版）：用户属性预测

流程：
  1. 加载数据 → 检查缓存 → 有则直接读，无则打标+切分并缓存
  2. 多模态特征编码
  3. 异常检测与修正
  4. 属性预测 + 三种场景评估

缓存策略（一次性步骤）：
  - output/labeled_data.csv   — 全部用户打标结果
  - output/train_indices.npy  — 训练集索引
  - output/cache_meta.txt     — 缓存元数据（打标+切分已完成的标记）

首次运行自动生成缓存，后续直接复用。
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from data_loader import load_raw_and_label
from features import (StaticFeatureEncoder, DynamicFeatureEncoder,
                      SocialFeatureEncoder, MultiModalFusion)
from anomaly import AnomalyDetector, AnomalyCorrector
from prediction import AttributePredictor, Evaluator


CACHE_DIR = config.OUTPUT_DIR
LABELED_CACHE = os.path.join(CACHE_DIR, "labeled_data.csv")
TRAIN_IDX_CACHE = os.path.join(CACHE_DIR, "train_indices.npy")
TEST_IDX_CACHE = os.path.join(CACHE_DIR, "test_indices.npy")
CACHE_META = os.path.join(CACHE_DIR, "cache_meta.txt")


# ============================================================
# 缓存管理
# ============================================================
def _cache_exists() -> bool:
    return (os.path.exists(LABELED_CACHE) and
            os.path.exists(TRAIN_IDX_CACHE) and
            os.path.exists(TEST_IDX_CACHE) and
            os.path.exists(CACHE_META))


def _save_cache(df_all: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_all.to_csv(LABELED_CACHE, index=False, encoding="utf-8-sig")
    np.save(TRAIN_IDX_CACHE, train_idx)
    np.save(TEST_IDX_CACHE, test_idx)
    with open(CACHE_META, "w") as f:
        f.write(f"labeled_users={len(df_all)}\n")
        f.write(f"train_size={len(train_idx)}\n")
        f.write(f"test_size={len(test_idx)}\n")
    print(f"  [Cache saved] {len(df_all)} users, "
          f"{len(train_idx)} train + {len(test_idx)} test")


def _load_cache():
    df_all = pd.read_csv(LABELED_CACHE, encoding="utf-8-sig")
    train_idx = np.load(TRAIN_IDX_CACHE)
    test_idx = np.load(TEST_IDX_CACHE)
    return df_all, train_idx, test_idx


# ============================================================
# 辅助：bootstrap 扩充（仅对训练集）
# ============================================================
def _bootstrap_platform(df: pd.DataFrame, target: int = 10000,
                        seed: int = 42) -> pd.DataFrame:
    if len(df) >= target:
        return df
    needed = target - len(df)
    extra = df.sample(n=needed, replace=True, random_state=seed)
    for col in ["followers_count", "friends_count", "statuses_count"]:
        noise = np.random.randint(-30, 31, size=len(extra))
        extra[col] = np.maximum(extra[col].astype(float).values + noise, 0).astype(int)
    extra["ff_ratio"] = extra["followers_count"] / extra["friends_count"].clip(lower=1)
    extra["log_followers"] = np.log1p(extra["followers_count"])
    extra["log_friends"] = np.log1p(extra["friends_count"])
    extra["log_statuses"] = np.log1p(extra["statuses_count"])
    n_exist = df["user_id"].nunique()
    extra["user_id"] = [f"aug_{n_exist + i}" for i in range(needed)]
    return pd.concat([df, extra], ignore_index=True)


# ============================================================
# 数据准备：加载 or 缓存
# ============================================================
def _prepare_data():
    """
    加载数据，利用缓存避免重复打标+切分。

    Returns:
        train_df: 训练集（已 bootstrap 扩充）
        test_df:  测试集（原始，无增强）
    """
    if _cache_exists():
        print("=" * 60)
        print("Loading cached data (skip labeling + split)")
        print("=" * 60)
        df_all, train_idx, test_idx = _load_cache()
        print(f"  Loaded {len(df_all)} users from cache")

        train_df_raw = df_all.iloc[train_idx].reset_index(drop=True)
        test_df = df_all.iloc[test_idx].reset_index(drop=True)
    else:
        print("=" * 60)
        print("First run: loading, labeling, and splitting data")
        print("=" * 60)
        tw_raw, wb_raw = load_raw_and_label()
        df_all = pd.concat([tw_raw, wb_raw], ignore_index=True)
        print(f"  Combined: {len(df_all):,} users")

        train_df_raw, test_df = train_test_split(
            df_all, test_size=0.2,
            random_state=config.RANDOM_SEED,
            stratify=df_all["platform"],
        )
        train_idx = train_df_raw.index.values
        test_idx = test_df.index.values
        _save_cache(df_all, train_idx, test_idx)

        train_df_raw = train_df_raw.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    print(f"  Train (raw): {len(train_df_raw):,}  |  Test: {len(test_df):,}")

    # ---- 仅对训练集 bootstrap ----
    print("\n--- Bootstrap Training Set ---")
    train_tw = train_df_raw[train_df_raw["platform"] == "twitter"]
    train_wb = train_df_raw[train_df_raw["platform"] == "weibo"]
    train_tw = _bootstrap_platform(train_tw, target=config.TARGET_SIZE, seed=config.RANDOM_SEED)
    train_df = pd.concat([train_tw, train_wb], ignore_index=True).reset_index(drop=True)
    print(f"  Total train: {len(train_df):,}")

    return train_df, test_df


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Step 1: 数据准备（缓存 or 打标+切分）
    train_df, test_df = _prepare_data()

    # ============================================================
    # Step 2: 多模态特征编码
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: Multi-modal Feature Encoding")
    print("=" * 60)
    enc_static = StaticFeatureEncoder()
    enc_dynamic = DynamicFeatureEncoder()
    enc_social = SocialFeatureEncoder()
    fusion = MultiModalFusion()

    v_static = enc_static.fit_transform(train_df)
    v_dynamic = enc_dynamic.fit_transform(train_df)
    v_social = enc_social.fit_transform(train_df)
    v_fused_train = fusion.fit_transform(v_static, v_dynamic, v_social)

    v_static_test = enc_static.transform(test_df)
    v_dynamic_test = enc_dynamic.transform(test_df)
    v_social_test = enc_social.transform(test_df)
    v_fused_test = fusion.transform(v_static_test, v_dynamic_test, v_social_test)

    # ============================================================
    # Step 3: 异常检测与修正
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: Anomaly Detection & Correction")
    print("=" * 60)
    detector = AnomalyDetector(contamination=config.ANOMALY_RATE)
    detector.fit(v_fused_train, v_static, v_social)
    anom_scores, anom_labels = detector.detect(v_fused_test, v_static_test, v_social_test)

    corrector = AnomalyCorrector(k_neighbors=10)
    corrector.fit(v_fused_train, np.zeros(len(train_df)))
    v_fused_corrected = corrector.correct(v_fused_test, anom_labels, anom_scores)

    print(f"  Anomalies detected: {anom_labels.sum()} / {len(anom_labels)} "
          f"({anom_labels.mean()*100:.1f}%)")

    # ============================================================
    # Step 4: 属性预测 + 评估
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 4: Attribute Prediction & Evaluation")
    print("=" * 60)

    predictor = AttributePredictor()
    train_quality = np.ones(len(train_df))
    predictor.fit(v_fused_train, train_df, quality_signals=train_quality)

    # 场景A：异常修正后
    print("\n--- Scenario A: After Anomaly Correction ---")
    pred_a = predictor.predict(v_fused_corrected)
    eva_a = Evaluator()
    eva_a.evaluate(pred_a, test_df)
    passed_a = eva_a.report()

    # 场景B：原始基线
    print("\n--- Scenario B: Without Correction (baseline) ---")
    pred_b = predictor.predict(v_fused_test)
    eva_b = Evaluator()
    eva_b.evaluate(pred_b, test_df)
    passed_b = eva_b.report()

    # 场景C：置信门控
    print("\n--- Scenario C: Confidence-gated (threshold=0.4) ---")
    proba = predictor.predict_proba(v_fused_corrected)
    confs = {t: p.max(axis=1) for t, p in proba.items()}
    eva_c = Evaluator()
    eva_c.evaluate_with_confidence_gate(pred_a, confs, test_df, 0.4)
    passed_c = eva_c.report()

    # ============================================================
    # Step 5: 导出
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 5: Export Results")
    print("=" * 60)
    out = test_df[["user_id", "screen_name", "platform"]].copy()
    for task in config.ATTRIBUTE_TASKS:
        if task in pred_a:
            out[f"true_{task}"] = test_df[task].values[:len(out)]
            out[f"pred_{task}"] = pred_a[task][:len(out)]
    out["anomaly_score"] = anom_scores[:len(out)]
    out["anomaly_label"] = anom_labels[:len(out)]
    path = os.path.join(CACHE_DIR, "prediction_results.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")

    print("\n" + "=" * 60)
    print("  FINAL")
    print(f"  A (w/ correction):  {'PASS' if passed_a else 'NOT MET'}")
    print(f"  B (baseline):       {'PASS' if passed_b else 'NOT MET'}")
    print(f"  C (gated):          {'PASS' if passed_c else 'NOT MET'}")
    print(f"  Target: recall >= {config.TARGET_RECALL*100:.0f}%, "
          f"precision >= {config.TARGET_PRECISION*100:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
