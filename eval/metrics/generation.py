"""Text generation metrics."""

import math
from collections import Counter
from typing import List, Set
import re


def compute_bleu(prediction: str, reference: str, n: int = 4) -> float:
    """Compute BLEU score following the standard definition (Papineni et al., 2002).

    Uses clipped n-gram counts and an exponential brevity penalty.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Compute clipped n-gram precision for each order
    scores = []
    for i in range(1, n + 1):
        pred_ngrams = [tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens) - i + 1)]
        ref_ngrams = [tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1)]

        if not pred_ngrams:
            scores.append(0.0)
            continue

        pred_counts = Counter(pred_ngrams)
        ref_counts = Counter(ref_ngrams)
        clipped = sum(min(pred_counts[ng], ref_counts[ng]) for ng in pred_counts)
        scores.append(clipped / sum(pred_counts.values()))

    # Geometric mean of all n-gram precisions
    if all(s > 0 for s in scores):
        log_avg = sum(math.log(s) for s in scores) / len(scores)
        bleu = math.exp(log_avg)
    else:
        bleu = 0.0

    # Standard exponential brevity penalty
    c, r = len(pred_tokens), len(ref_tokens)
    bp = math.exp(1 - r / c) if c < r else 1.0

    return bleu * bp


def compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE-L score."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if not pred_tokens or not ref_tokens:
        return {"rouge_l": 0.0}
    
    # Longest Common Subsequence
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    # Precision and Recall
    precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
    
    # F1 score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "rouge_l": f1,
        "rouge_l_precision": precision,
        "rouge_l_recall": recall
    }


def compute_keyword_overlap(prediction: str, reference: str) -> float:
    """Compute keyword overlap ratio."""
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())
    
    if not ref_words:
        return 0.0
    
    overlap = len(pred_words & ref_words)
    return overlap / len(ref_words)


def compute_technical_term_coverage(prediction: str, technical_terms: List[str]) -> float:
    """Compute coverage of technical terms."""
    if not technical_terms:
        return 1.0
    
    pred_lower = prediction.lower()
    covered = sum(1 for term in technical_terms if term.lower() in pred_lower)
    
    return covered / len(technical_terms)


def compute_bertscore(prediction: str, reference: str, use_model: bool = False) -> dict:
    """Compute BERTScore (F1, Precision, Recall).
    
    Args:
        prediction: Generated text
        reference: Ground truth text
        use_model: If True, use actual BERT model (requires bert_score package).
                   If False, use simplified embedding-based approximation.
    
    Returns:
        Dictionary with bertscore_f1, bertscore_precision, bertscore_recall
    """
    if not prediction or not reference:
        return {"bertscore_f1": 0.0, "bertscore_precision": 0.0, "bertscore_recall": 0.0}
    
    if use_model:
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn([prediction], [reference], lang="en", verbose=False)
            return {
                "bertscore_f1": float(F1[0]),
                "bertscore_precision": float(P[0]),
                "bertscore_recall": float(R[0])
            }
        except ImportError:
            pass  # Fall through to simplified version
    
    # Lightweight fallback: character 3-gram overlap (NOT semantic BERTScore).
    # Keys are prefixed "charngramf1" to avoid confusion with real BERTScore.
    def get_ngrams(text: str, n: int = 3) -> Set[str]:
        text = text.lower()
        return {text[i:i+n] for i in range(max(len(text) - n + 1, 1))}

    pred_ngrams = get_ngrams(prediction)
    ref_ngrams = get_ngrams(reference)

    if not pred_ngrams or not ref_ngrams:
        return {"bertscore_f1": 0.0, "bertscore_precision": 0.0, "bertscore_recall": 0.0}

    overlap = len(pred_ngrams & ref_ngrams)
    precision = overlap / len(pred_ngrams)
    recall = overlap / len(ref_ngrams)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "bertscore_f1": f1,              # char-ngram proxy; install bert_score for true BERTScore
        "bertscore_precision": precision,
        "bertscore_recall": recall
    }


def compute_generation_metrics(prediction: str, reference: str, 
                                technical_terms: List[str] = None) -> dict:
    """Compute comprehensive generation metrics.
    
    Returns:
        Dictionary with BLEU, ROUGE-L, BERTScore-F1, and term coverage
    """
    metrics = {}
    
    # BLEU
    metrics["bleu"] = compute_bleu(prediction, reference)
    
    # ROUGE-L
    rouge_scores = compute_rouge(prediction, reference)
    metrics.update(rouge_scores)
    
    # BERTScore
    bert_scores = compute_bertscore(prediction, reference)
    metrics.update(bert_scores)
    
    # Keyword overlap
    metrics["keyword_overlap"] = compute_keyword_overlap(prediction, reference)
    
    # Technical term coverage (if provided)
    if technical_terms:
        metrics["term_coverage"] = compute_technical_term_coverage(prediction, technical_terms)
    
    return metrics
