"""Regression metrics."""

from typing import List
import numpy as np


def compute_mae(predictions: List[float], ground_truths: List[float]) -> float:
    """Compute Mean Absolute Error."""
    if not predictions or not ground_truths:
        return float('inf')
    return float(np.mean([abs(p - g) for p, g in zip(predictions, ground_truths)]))


def compute_rmse(predictions: List[float], ground_truths: List[float]) -> float:
    """Compute Root Mean Squared Error."""
    if not predictions or not ground_truths:
        return float('inf')
    return float(np.sqrt(np.mean([(p - g)**2 for p, g in zip(predictions, ground_truths)])))


def compute_r2(predictions: List[float], ground_truths: List[float]) -> float:
    """Compute R² score."""
    if not predictions or not ground_truths:
        return 0.0
    
    mean_gt = np.mean(ground_truths)
    ss_tot = sum((g - mean_gt)**2 for g in ground_truths)
    ss_res = sum((g - p)**2 for g, p in zip(ground_truths, predictions))
    
    if ss_tot == 0:
        return 0.0
    
    return float(1 - (ss_res / ss_tot))


def compute_relative_error(predictions: List[float], ground_truths: List[float]) -> float:
    """Compute mean relative error."""
    if not predictions or not ground_truths:
        return float('inf')
    
    rel_errors = []
    for p, g in zip(predictions, ground_truths):
        if g != 0:
            rel_errors.append(abs(p - g) / abs(g))
    
    return float(np.mean(rel_errors)) if rel_errors else float('inf')


def compute_log_mae(predictions: List[float], ground_truths: List[float]) -> float:
    """Compute MAE in log space (for values spanning multiple orders of magnitude)."""
    if not predictions or not ground_truths:
        return float('inf')
    
    log_errors = []
    for p, g in zip(predictions, ground_truths):
        if p > 0 and g > 0:
            log_errors.append(abs(np.log10(p) - np.log10(g)))
    
    return float(np.mean(log_errors)) if log_errors else float('inf')


def compute_nmae(predictions: List[float], ground_truths: List[float]) -> float:
    """Compute Normalized MAE (MAE / range of ground truths).
    
    Normalizes by the range (max - min) of ground truth values so that
    the metric is scale-independent and comparable across tasks.
    Returns value in [0, inf); lower is better.
    """
    if not predictions or not ground_truths:
        return float('inf')
    gt_range = float(max(ground_truths) - min(ground_truths))
    if gt_range < 1e-12:
        # All ground truths are identical; fall back to relative error
        gt_mean = float(np.mean(ground_truths))
        if abs(gt_mean) < 1e-12:
            return float('inf')
        return float(np.mean([abs(p - g) for p, g in zip(predictions, ground_truths)])) / abs(gt_mean)
    mae = float(np.mean([abs(p - g) for p, g in zip(predictions, ground_truths)]))
    return mae / gt_range


def compute_nrmse(predictions: List[float], ground_truths: List[float]) -> float:
    """Compute Normalized RMSE (RMSE / range of ground truths).
    
    Normalizes by the range (max - min) of ground truth values so that
    the metric is scale-independent and comparable across tasks.
    Returns value in [0, inf); lower is better.
    """
    if not predictions or not ground_truths:
        return float('inf')
    gt_range = float(max(ground_truths) - min(ground_truths))
    if gt_range < 1e-12:
        gt_mean = float(np.mean(ground_truths))
        if abs(gt_mean) < 1e-12:
            return float('inf')
        rmse = float(np.sqrt(np.mean([(p - g)**2 for p, g in zip(predictions, ground_truths)])))
        return rmse / abs(gt_mean)
    rmse = float(np.sqrt(np.mean([(p - g)**2 for p, g in zip(predictions, ground_truths)])))
    return rmse / gt_range
