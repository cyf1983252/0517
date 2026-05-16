
"""
Missing feature completion — magnitude-based imputation methods.
Core idea: find users of similar magnitude, use their values.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class MagnitudeImputer:
    """Magnitude bucketing: group users by log-order, fill with group mean."""

    def __init__(self, buckets_per_order: int = 2):
        self.buckets_per_order = buckets_per_order
        self._bucket_map: Dict[str, Dict] = {}

    @staticmethod
    def _log_bucket(value, bins_per_order):
        if value <= 0:
            return -1
        return int(np.log10(value) * bins_per_order)

    def _find_aux(self, df, target):
        candidates = {
            "followers_count": ["statuses_count", "friends_count"],
            "friends_count": ["statuses_count", "followers_count"],
            "statuses_count": ["followers_count", "friends_count"],
            "log_followers": ["log_statuses", "log_friends"],
            "log_friends": ["log_statuses", "log_followers"],
            "log_statuses": ["log_followers", "log_friends"],
            "ff_ratio": ["log_followers", "log_friends"],
        }
        return candidates.get(target, ["statuses_count"])[0]

    def fit(self, df, target_cols=None):
        if target_cols is None:
            target_cols = [c for c in df.columns if c in [
                "followers_count", "friends_count", "statuses_count",
                "log_followers", "log_friends", "log_statuses", "ff_ratio"]]
        for col in target_cols:
            if col not in df.columns or df[col].nunique() < 2:
                continue
            aux = self._find_aux(df, col)
            if aux not in df.columns:
                continue
            valid = df[col].notna() & df[aux].notna()
            if valid.sum() < 10:
                continue
            bfn = np.vectorize(lambda v: self._log_bucket(v, self.buckets_per_order))
            buckets = bfn(df.loc[valid, aux].values)
            means = {}
            for b in np.unique(buckets):
                mask = buckets == b
                vals = df.loc[valid.values, col].values[mask]
                if len(vals) > 0:
                    means[int(b)] = float(np.mean(vals))
            self._bucket_map[col] = {"aux_col": aux, "bucket_means": means}
        return self

    def transform(self, df):
        result = df.copy()
        for target_col, info in self._bucket_map.items():
            aux_col = info["aux_col"]
            means = info["bucket_means"]
            missing = result[target_col].isna()
            if missing.sum() == 0:
                continue
            for idx in result.index[missing]:
                aux_val = result.loc[idx, aux_col]
                if pd.isna(aux_val) or aux_val <= 0:
                    continue
                b = self._log_bucket(aux_val, self.buckets_per_order)
                if b in means:
                    result.loc[idx, target_col] = means[b]
        return result

    def fit_transform(self, df, target_cols=None):
        self.fit(df, target_cols)
        return self.transform(df)


class kNNSimilarImputer:
    """
    kNN weighted imputation — find users of similar magnitude.

    Core idea:
        "发帖数缺了 -> 找发帖数在同量级的用户 -> 取他们的粉丝数均值"

    For each missing value:
    1. Use available features to find k most similar users
    2. Similarity = inverse Euclidean distance (standardized + log space)
    3. Fill with distance-weighted average of neighbors
    4. Auto-excludes self to prevent data leakage
    """

    def __init__(self, k: int = 20, random_state: int = 42):
        self.k = k
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._X_train = None
        self._cols: List[str] = []
        self._log_cols: Dict[str, bool] = {}

    def _should_log(self, values):
        vals = values[~np.isnan(values)]
        if len(vals) < 10:
            return False
        mean, med = np.mean(vals), np.median(vals)
        skew = abs(mean - med) / max(np.std(vals), 1e-8)
        return skew > 2.0 and np.min(vals) >= 0

    def fit(self, df, target_cols=None):
        if target_cols is None:
            target_cols = [c for c in df.columns if c in [
                "followers_count", "friends_count", "statuses_count",
                "log_followers", "log_friends", "log_statuses", "ff_ratio"]]
        self._cols = [c for c in target_cols if c in df.columns]
        X_raw = df[self._cols].values.astype(np.float64)
        complete = ~np.any(np.isnan(X_raw), axis=1)
        if complete.sum() < 10:
            raise ValueError(f"Not enough complete rows: {complete.sum()}")
        for i, col in enumerate(self._cols):
            self._log_cols[col] = self._should_log(X_raw[complete, i])
        X_proc = X_raw[complete].copy()
        for i, col in enumerate(self._cols):
            if self._log_cols[col]:
                X_proc[:, i] = np.log1p(np.maximum(X_proc[:, i], 0))
        self._X_train = self._scaler.fit_transform(X_proc)
        nl = sum(self._log_cols.values())
        print(f"  kNNSimilarImputer: {len(self._cols)} cols, "
              f"{complete.sum()} rows, {nl} log-transformed")
        return self

    def transform(self, df):
        result = df.copy()
        cols = self._cols
        X_raw = result[cols].values.astype(np.float64)
        missing_mask = np.isnan(X_raw)

        for row_idx in range(len(result)):
            row_mask = missing_mask[row_idx]
            if not row_mask.any():
                continue
            valid_dims = ~row_mask
            if valid_dims.sum() == 0:
                continue
            row_proc = X_raw[row_idx].copy()
            for i, col in enumerate(cols):
                if self._log_cols[col]:
                    row_proc[i] = np.log1p(max(row_proc[i], 0))
            row_scaled = self._scaler.transform(row_proc.reshape(1, -1))[0]
            train_valid = self._X_train[:, valid_dims]
            row_valid = row_scaled[valid_dims]
            dists = np.sqrt(np.mean((train_valid - row_valid) ** 2, axis=1))
            k_plus = min(self.k + 1, len(dists))
            nearest = np.argsort(dists)[:k_plus]
            nearest = nearest[dists[nearest] > 1e-10][:self.k]
            if len(nearest) == 0:
                continue
            weights = 1.0 / (dists[nearest] + 1e-10)
            weights = weights / weights.sum()
            for dim in np.where(row_mask)[0]:
                nv = self._X_train[nearest, dim]
                filled_scaled = np.sum(nv * weights)
                mean = self._scaler.mean_[dim]
                std = self._scaler.scale_[dim]
                filled_trans = filled_scaled * std + mean
                if self._log_cols[cols[dim]]:
                    filled_orig = np.expm1(max(filled_trans, 0))
                else:
                    filled_orig = filled_trans
                result.loc[result.index[row_idx], cols[dim]] = float(filled_orig)
        return result

    def fit_transform(self, df, target_cols=None):
        self.fit(df, target_cols)
        return self.transform(df)


def evaluate_imputation(df_filled, df_original, mask, cols, method_name=""):
    results = {}
    for i, col in enumerate(cols):
        if i >= mask.shape[1] or col not in df_filled.columns:
            continue
        miss = mask[:, i] == 1
        if miss.sum() == 0:
            continue
        t = df_original[col].values[miss]
        p = df_filled[col].values[miss]
        valid = ~np.isnan(p.astype(float))
        if valid.sum() == 0:
            continue
        mae = float(np.mean(np.abs(t[valid] - p[valid])))
        rmse = float(np.sqrt(np.mean((t[valid] - p[valid]) ** 2)))
        base = float(np.mean(np.abs(t[valid] - df_original[col].mean())))
        impr = (base - mae) / max(base, 1e-8) * 100
        results[col] = {"mae": mae, "rmse": rmse, "baseline_mae": base,
                        "improvement_pct": impr, "filled": int(valid.sum())}
    return results


def print_comparison(all_results):
    print(f"\n{'='*70}")
    print("Method Comparison Summary")
    print(f"{'='*70}")
    all_cols = set()
    for results_list in all_results.values():
        for r in results_list:
            all_cols.update(r.keys())
    for col in sorted(all_cols):
        print(f"\n  {col}:")
        print(f"  {'Method':<25s} {'MAE':>12s} {'Baseline':>12s} {'Improve':>10s}")
        print(f"  {'-'*59}")
        for method_name, results_list in all_results.items():
            for r in results_list:
                if col in r:
                    res = r[col]
                    print(f"  {method_name:<25s} {res['mae']:>12.2f} "
                          f"{res['baseline_mae']:>12.2f} "
                          f"{res['improvement_pct']:>+9.1f}%")
