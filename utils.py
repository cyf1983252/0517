"""
工具函数：文本归一化、评估指标、编码辅助等
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import recall_score, precision_score, f1_score


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s一-鿿]", "", text)
    return text


def detect_language(text: str) -> str:
    """简易语种检测：zh / en / mixed"""
    if not text:
        return "unknown"
    cjk = len(re.findall(r"[一-鿿]", text))
    ascii_chars = len(re.findall(r"[a-zA-Z]", text))
    if cjk > ascii_chars:
        return "zh"
    elif ascii_chars > cjk:
        return "en"
    return "mixed"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Dict[str, float]:
    """计算召回率、准确率、F1"""
    return {
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def print_metrics_table(results: Dict[str, Dict[str, float]]):
    """格式化打印多任务评估指标"""
    print(f"\n{'Task':<25} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print("-" * 53)
    recalls, precisions = [], []
    for task, metrics in results.items():
        r, p, f = metrics["recall"], metrics["precision"], metrics["f1"]
        print(f"{task:<25} {r:>8.4f} {p:>10.4f} {f:>8.4f}")
        recalls.append(r)
        precisions.append(p)
    print("-" * 53)
    avg_r = np.mean(recalls)
    avg_p = np.mean(precisions)
    print(f"{'** AVERAGE **':<25} {avg_r:>8.4f} {avg_p:>10.4f}")
    return avg_r, avg_p
