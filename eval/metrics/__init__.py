"""Evaluation metrics for PCF benchmarking."""

from .classification import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_f1_scores,
    compute_topk_accuracy
)
from .regression import (
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_relative_error,
    compute_log_mae,
    compute_nmae,
    compute_nrmse
)
from .generation import (
    compute_bleu,
    compute_rouge,
    compute_keyword_overlap,
    compute_technical_term_coverage
)
from .code import (
    check_syntax,
    check_executability,
    extract_code_from_markdown,
    check_code_completeness
)

__all__ = [
    # Classification
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_f1_scores",
    "compute_topk_accuracy",
    # Regression
    "compute_mae",
    "compute_rmse",
    "compute_r2",
    "compute_relative_error",
    "compute_log_mae",
    "compute_nmae",
    "compute_nrmse",
    # Generation
    "compute_bleu",
    "compute_rouge",
    "compute_keyword_overlap",
    "compute_technical_term_coverage",
    # Code
    "check_syntax",
    "check_executability",
    "extract_code_from_markdown",
    "check_code_completeness",
]
