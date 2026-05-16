"""
评估模块：召回率、准确率、F1计算，分层验证，置信门控
对应文档 六、关键指标测试方法
"""
import numpy as np
import pandas as pd
from typing import Dict, List

import config
from utils import compute_metrics, print_metrics_table


class Evaluator:
    """
    用户属性预测评估器。
    指标：召回率≥75%，准确率≥80%
    """

    def __init__(self):
        self.results = {}

    def evaluate(self, predictions: Dict[str, np.ndarray],
                 ground_truth_df: pd.DataFrame,
                 task_names: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        评估所有任务的召回率、准确率、F1。
        返回 {task_name: {"recall": float, "precision": float, "f1": float}}
        """
        if task_names is None:
            task_names = config.ATTRIBUTE_TASKS

        task_names = [t for t in task_names if t in predictions and t in ground_truth_df.columns]

        all_metrics = {}
        for task in task_names:
            y_true = ground_truth_df[task].fillna("missing").astype(str).values
            y_pred = np.array(predictions[task])

            # 确保维度一致
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

            metrics = compute_metrics(y_true, y_pred, average="macro")
            all_metrics[task] = metrics

        self.results = all_metrics
        return all_metrics

    def evaluate_with_confidence_gate(self, predictions: Dict[str, np.ndarray],
                                      confidences: Dict[str, np.ndarray],
                                      ground_truth_df: pd.DataFrame,
                                      confidence_threshold: float = 0.5) -> Dict:
        """
        分层验证：仅对高置信预测评估（置信门控）。
        低于阈值的预测标记为"不确定"，不参与指标计算。
        """
        task_names = [t for t in predictions if t in ground_truth_df.columns]

        all_metrics = {}
        gated_counts = {}
        for task in task_names:
            y_true = ground_truth_df[task].fillna("missing").astype(str).values
            y_pred = np.array(predictions[task])
            conf = confidences.get(task, np.ones(len(y_pred)))

            min_len = min(len(y_true), len(y_pred), len(conf))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            conf = conf[:min_len]

            # 置信门控
            high_conf_mask = conf >= confidence_threshold
            gated_counts[task] = high_conf_mask.sum()

            if high_conf_mask.sum() > 0:
                metrics = compute_metrics(
                    y_true[high_conf_mask],
                    y_pred[high_conf_mask],
                    average="macro",
                )
            else:
                metrics = {"recall": 0.0, "precision": 0.0, "f1": 0.0}

            all_metrics[task] = metrics

        self.results = all_metrics
        return {"metrics": all_metrics, "gated_counts": gated_counts}

    def report(self) -> bool:
        """打印评估报告，返回是否达标"""
        print("\n" + "=" * 60)
        print("  EVALUATION REPORT — User Attribute Prediction")
        print("=" * 60)
        avg_r, avg_p = print_metrics_table(self.results)

        print(f"\n  Targets: Recall >= {config.TARGET_RECALL}, "
              f"Precision >= {config.TARGET_PRECISION}")
        passed = avg_r >= config.TARGET_RECALL and avg_p >= config.TARGET_PRECISION
        status = "PASSED" if passed else "NOT YET MET"
        print(f"  Average Recall:    {avg_r:.4f}  (target >= {config.TARGET_RECALL})")
        print(f"  Average Precision: {avg_p:.4f}  (target >= {config.TARGET_PRECISION})")
        print(f"  Status: {status}")
        print("=" * 60)
        return passed
