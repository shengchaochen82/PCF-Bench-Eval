"""Classification metrics."""

from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter


def compute_accuracy(predictions: List[Any], ground_truths: List[Any]) -> float:
    """Compute accuracy."""
    if not predictions or not ground_truths:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)


def compute_topk_accuracy(predictions: List[List[Any]], ground_truths: List[Any], k: int = 3) -> float:
    """Compute Top-K accuracy."""
    if not predictions or not ground_truths:
        return 0.0
    correct = sum(g in p[:k] for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)


def compute_confusion_matrix(predictions: List[Any], ground_truths: List[Any], labels: List[Any]) -> np.ndarray:
    """Compute confusion matrix."""
    n = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    matrix = np.zeros((n, n), dtype=int)
    for pred, gt in zip(predictions, ground_truths):
        if pred in label_to_idx and gt in label_to_idx:
            matrix[label_to_idx[gt], label_to_idx[pred]] += 1
    
    return matrix


def compute_balanced_accuracy(predictions: List[Any], ground_truths: List[Any], labels: Optional[List[Any]] = None) -> float:
    """Compute balanced accuracy (average of per-class recalls)."""
    if not predictions or not ground_truths:
        return 0.0
    
    if labels is None:
        labels = list(set(ground_truths))
    
    recalls = []
    for label in labels:
        # Get indices where ground truth is this label
        gt_indices = [i for i, g in enumerate(ground_truths) if g == label]
        if not gt_indices:
            continue
        # Count correct predictions for this class
        correct = sum(1 for i in gt_indices if predictions[i] == label)
        recalls.append(correct / len(gt_indices))
    
    return np.mean(recalls) if recalls else 0.0


def compute_f1_scores(predictions: List[Any], ground_truths: List[Any], labels: List[Any]) -> Dict[str, float]:
    """Compute F1 scores (macro and micro)."""
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n = len(labels)
    
    # Initialize counters
    tp = np.zeros(n)
    fp = np.zeros(n)
    fn = np.zeros(n)
    
    for pred, gt in zip(predictions, ground_truths):
        if pred in label_to_idx and gt in label_to_idx:
            pred_idx = label_to_idx[pred]
            gt_idx = label_to_idx[gt]
            
            if pred == gt:
                tp[pred_idx] += 1
            else:
                fp[pred_idx] += 1
                fn[gt_idx] += 1
    
    # Compute per-class F1
    f1_per_class = []
    for i in range(n):
        if tp[i] + fp[i] > 0:
            precision = tp[i] / (tp[i] + fp[i])
        else:
            precision = 0.0
        
        if tp[i] + fn[i] > 0:
            recall = tp[i] / (tp[i] + fn[i])
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        f1_per_class.append(f1)
    
    # Macro F1
    f1_macro = np.mean(f1_per_class)
    
    # Micro F1
    total_tp = np.sum(tp)
    total_fp = np.sum(fp)
    total_fn = np.sum(fn)
    
    if total_tp + total_fp > 0:
        micro_precision = total_tp / (total_tp + total_fp)
    else:
        micro_precision = 0.0
    
    if total_tp + total_fn > 0:
        micro_recall = total_tp / (total_tp + total_fn)
    else:
        micro_recall = 0.0
    
    if micro_precision + micro_recall > 0:
        f1_micro = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        f1_micro = 0.0
    
    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_per_class": {labels[i]: f1_per_class[i] for i in range(n)}
    }
