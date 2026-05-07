"""Physics understanding tasks for PCF benchmarking."""

from typing import Any, Dict
import numpy as np
import re
from .base import BaseTask


# Wavelength values that commonly appear in prompt echoes ("at 1550 nm")
# and must be excluded from naive number extraction.
_WAVELENGTH_ECHOES = {1310, 1550}

# Loss floor (dB/km). 2D-FDTD residual loss below this value reflects the
# simulator's numerical-noise floor (~1e-12 dB/m, see paper appendix) rather
# than physical design differences, so both ground-truth and prediction are
# clamped here before log-space scoring.
LOSS_FLOOR_DB_PER_KM = 1e-9


# A number written as either scientific notation, "× 10^x" form, or a plain
# decimal. Captures the full numeric span so we can re-parse it with float().
_NUM_PATTERN = (
    r'(?P<sci>\d+\.?\d*\s*[×x*]\s*10\s*\^?\s*\(?\s*-?\d+\s*\)?)'
    r'|(?P<exp>\d+\.?\d*[eE][+-]?\d+)'
    r'|(?P<dec>\d+\.?\d+|\d+)'
)


def _parse_number(span: str) -> float:
    """Convert a numeric substring (sci / exp / decimal) to float."""
    span = span.strip()
    m = re.fullmatch(
        r'(\d+\.?\d*)\s*[×x*]\s*10\s*\^?\s*\(?\s*(-?\d+)\s*\)?',
        span, re.IGNORECASE,
    )
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))
    return float(span)


def _extract_loss_value(text: str, wavelength: int) -> float:
    """Robustly extract a loss value in dB/km from a model response.

    Strategy (priority order):
        1. Number explicitly tagged with ``dB/km`` or ``dB/m`` — wins even
           when sci-notation appears earlier without a unit.
        2. First scientific-notation number not followed by ``nm`` (so a
           prompt echo like ``1550 nm`` cannot be mistaken for the answer).
        3. Bare number fallback, skipping prompt-echoed wavelengths
           (1310 / 1550) and ``<N> nm`` patterns.

    ``dB/m`` answers are converted to ``dB/km`` (×1000) so downstream
    metrics are comparable to the simulated ground-truth.
    """
    if not text:
        return 0.0
    try:
        # 1) Unit-tagged value: "<N> dB/km" or "<N> dB/m" anywhere in text.
        m = re.search(
            rf'({_NUM_PATTERN})\s*dB\s*/\s*(km|m)\b',
            text, re.IGNORECASE,
        )
        if m:
            v = _parse_number(m.group(1))
            return v if m.group(m.lastindex).lower() == "km" else v * 1000.0

        # 2) Scientific notation, but reject if immediately followed by "nm"
        #    (would be a wavelength like "1550 nm" rather than a loss value).
        for m in re.finditer(r'(?P<sci>\d+\.?\d*\s*[×x*]\s*10\s*\^?\s*\(?\s*-?\d+\s*\)?)|(?P<exp>\d+\.?\d*[eE][+-]?\d+)', text):
            tail = text[m.end(): m.end() + 4].lstrip().lower()
            if tail.startswith("nm"):
                continue
            return _parse_number(m.group(0))

        # 3) Bare-number fallback. Skip the prompt-echoed wavelength values
        #    and any number directly followed by "nm".
        for m in re.finditer(r'\d+\.?\d+|\d+', text):
            n = float(m.group(0))
            tail = text[m.end(): m.end() + 4].lstrip().lower()
            if tail.startswith("nm"):
                continue
            if int(n) == n and int(n) in _WAVELENGTH_ECHOES:
                continue
            return n
    except Exception:
        pass
    return 0.0


class LossPredictionTask(BaseTask):
    """Predict loss value at specific wavelength.
    
    Uses the appropriate field image based on wavelength:
    - 1310nm: field_1310.png
    - 1550nm: field_1550.png
    """
    
    def __init__(self, data_dir: str, model_interface: Any, wavelength: int = 1550):
        super().__init__(data_dir, model_interface)
        self.wavelength = wavelength
        # Map wavelength to image config
        self.image_config = f"field_{wavelength}_only"
    
    def get_task_name(self) -> str:
        return f"loss_prediction_{self.wavelength}"
    
    def get_task_description(self) -> str:
        return f"Predict confinement loss at {self.wavelength}nm wavelength"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "sim" not in data or "metrics" not in data["sim"]:
            return None
        
        loss_key = f"loss_{self.wavelength}"
        if loss_key not in data["sim"]["metrics"]:
            return None
        
        gt_loss_raw = float(data["sim"]["metrics"][loss_key])
        gt_loss = max(gt_loss_raw, LOSS_FLOOR_DB_PER_KM)

        # Use the appropriate field image for the wavelength
        image_path, prompt_prefix = self.prepare_task_images(data, self.image_config)

        prompt = f"{prompt_prefix}Estimate the confinement loss of this PCF at {self.wavelength}nm from the field distribution pattern. Provide a single numerical value in dB/km. Use scientific notation if needed (e.g. 1.5e-3)."

        pred_str = self.model.query(prompt, image_path=image_path, max_tokens=128)
        pred_loss_raw = _extract_loss_value(pred_str, self.wavelength)
        pred_loss = max(pred_loss_raw, LOSS_FLOOR_DB_PER_KM)

        # Errors are computed on clamped values so log-space metrics are
        # always finite; without the floor, residual-noise samples (gt ≈ 0)
        # produce inf log-error and dominate aggregate metrics.
        log_abs_error = abs(np.log10(pred_loss) - np.log10(gt_loss))
        rel_error = abs(pred_loss - gt_loss) / gt_loss
        abs_error = abs(pred_loss - gt_loss)
        
        return {
            "sample_id": sample_id,
            "wavelength": self.wavelength,
            "ground_truth": gt_loss,
            "prediction": pred_loss,
            "absolute_error": abs_error,
            "relative_error": rel_error,
            "log_absolute_error": log_abs_error
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        gts = [r["ground_truth"] for r in valid_results]
        abs_errors = [r["absolute_error"] for r in valid_results]
        rel_errors = [r["relative_error"] for r in valid_results if r["relative_error"] != float('inf')]
        log_errors = [r["log_absolute_error"] for r in valid_results if r["log_absolute_error"] != float('inf')]
        
        # Normalized metrics (scale-independent)
        gt_range = max(gts) - min(gts) if len(gts) > 1 else abs(np.mean(gts))
        mae_val = float(np.mean(abs_errors))
        rmse_val = float(np.sqrt(np.mean([e**2 for e in abs_errors])))
        nmae = mae_val / gt_range if gt_range > 1e-12 else float('inf')
        nrmse = rmse_val / gt_range if gt_range > 1e-12 else float('inf')
        
        return {
            "mae": mae_val,
            "rmse": rmse_val,
            "nmae": nmae,
            "nrmse": nrmse,
            "mean_relative_error": np.mean(rel_errors) if rel_errors else float('inf'),
            "log_mae": np.mean(log_errors) if log_errors else float('inf'),
            "num_samples": len(valid_results)
        }


class ApplicationClassificationTask(BaseTask):
    """Classify PCF application based on performance."""
    
    def get_task_name(self) -> str:
        return "application_classification"
    
    def get_task_description(self) -> str:
        return "Classify PCF application: telecom, sensing, or experimental"
    
    # All application categories found in the dataset
    APPLICATION_CATEGORIES = [
        "telecom", "sensing", "experimental",
        "polarization_maintaining", "high_power_delivery",
        "space_division_multiplexing", "nonlinear_optics",
        "plasmonic_sensing", "orbital_angular_momentum",
        "optical_switching", "robust_bandgap_guidance",
        "ultra_low_loss_telecom", "unknown",
    ]
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "annotation" not in data or "level2" not in data["annotation"]:
            return None
        
        level2 = data["annotation"]["level2"]
        conversations = level2.get("conversations", [])
        
        # Extract ground truth: find the human question about application,
        # then take the next gpt answer and parse "application is XXX"
        gt_app = None
        for i, conv in enumerate(conversations):
            if conv.get("from") == "human" and "application" in conv.get("value", "").lower():
                if i + 1 < len(conversations) and conversations[i + 1].get("from") == "gpt":
                    answer = conversations[i + 1].get("value", "")
                    match = re.search(r'application is (\S+)', answer)
                    if match:
                        gt_app = match.group(1).rstrip(',.')
                break
        
        if gt_app is None:
            return None
        
        # Build prompt with all known categories
        categories_str = ", ".join(self.APPLICATION_CATEGORIES)
        prompt = (
            f"Based on the performance characteristics of this PCF structure, "
            f"what is its most suitable application? Choose from: {categories_str}. "
            f"Answer with only the category name."
        )
        
        pred_app = self.model.query(prompt, image_path=data["field_image"], max_tokens=64)
        pred_app = pred_app.strip().lower().replace(" ", "_")
        
        # Try to match prediction to a known category (longest match first
        # to avoid "telecom" prematurely matching "ultra_low_loss_telecom")
        matched_pred = pred_app
        for cat in sorted(self.APPLICATION_CATEGORIES, key=len, reverse=True):
            if cat in pred_app:
                matched_pred = cat
                break
        
        correct = (matched_pred == gt_app)
        
        return {
            "sample_id": sample_id,
            "ground_truth": gt_app,
            "prediction": matched_pred,
            "correct": correct
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        correct = sum(r["correct"] for r in valid_results)
        total = len(valid_results)
        
        # Compute per-class accuracy and F1
        classes = list(set(r["ground_truth"] for r in valid_results))
        predictions = [r["prediction"] for r in valid_results]
        ground_truths = [r["ground_truth"] for r in valid_results]
        
        per_class = {}
        f1_scores = []
        for cls in classes:
            cls_results = [r for r in valid_results if r["ground_truth"] == cls]
            if cls_results:
                cls_correct = sum(r["correct"] for r in cls_results)
                per_class[f"accuracy_{cls}"] = cls_correct / len(cls_results)
            
            # F1 per class
            tp = sum(1 for p, g in zip(predictions, ground_truths) if p == cls and g == cls)
            fp = sum(1 for p, g in zip(predictions, ground_truths) if p == cls and g != cls)
            fn = sum(1 for p, g in zip(predictions, ground_truths) if p != cls and g == cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        metrics = {
            "accuracy": correct / total if total > 0 else 0.0,
            "macro_f1": float(macro_f1),
            "num_samples": total
        }
        metrics.update(per_class)
        
        return metrics


class ModeClassificationTask(BaseTask):
    """Classify guidance mode."""
    
    def get_task_name(self) -> str:
        return "mode_classification"
    
    def get_task_description(self) -> str:
        return "Classify guidance mode: single-mode, multi-mode, or leaky"
    
    MODE_CATEGORIES = ["single_mode", "multi_mode", "leaky"]
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "annotation" not in data or "level2" not in data["annotation"]:
            return None
        
        level2 = data["annotation"]["level2"]
        conversations = level2.get("conversations", [])
        
        # Extract ground truth mode from GPT answer to the mode question
        gt_mode = None
        for i, conv in enumerate(conversations):
            if conv.get("from") == "human" and ("single-mode" in conv.get("value", "").lower() or "multi-mode" in conv.get("value", "").lower()):
                if i + 1 < len(conversations) and conversations[i + 1].get("from") == "gpt":
                    answer = conversations[i + 1].get("value", "")
                    match = re.search(r'more likely (\S+)', answer)
                    if match:
                        gt_mode = match.group(1).rstrip(',.')
                break
        
        if gt_mode is None:
            return None
        
        prompt = "What is the guidance mode of this PCF structure? Choose from: single_mode (supports only fundamental mode), multi_mode (supports multiple modes), or leaky (high loss, poor confinement). Answer with only the mode type."
        
        pred_mode = self.model.query(prompt, image_path=data["field_image"], max_tokens=64)
        pred_mode = pred_mode.strip().lower().replace("-", "_").replace(" ", "_")
        
        # Match to known category
        matched_pred = pred_mode
        for cat in self.MODE_CATEGORIES:
            if cat in pred_mode:
                matched_pred = cat
                break
        
        correct = (matched_pred == gt_mode)
        
        return {
            "sample_id": sample_id,
            "ground_truth": gt_mode,
            "prediction": matched_pred,
            "correct": correct
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        correct = sum(r["correct"] for r in valid_results)
        total = len(valid_results)
        
        # Compute F1 (macro)
        predictions = [r["prediction"] for r in valid_results]
        ground_truths = [r["ground_truth"] for r in valid_results]
        labels = list(set(ground_truths))
        
        f1_scores = []
        for label in labels:
            tp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g != label)
            fn = sum(1 for p, g in zip(predictions, ground_truths) if p != label and g == label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "macro_f1": float(macro_f1),
            "num_samples": total
        }


class MechanismReasoningTask(BaseTask):
    """Evaluate mechanism reasoning quality."""
    
    def get_task_name(self) -> str:
        return "mechanism_reasoning"
    
    def get_task_description(self) -> str:
        return "Evaluate quality of physical mechanism explanation"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "annotation" not in data or "level3" not in data["annotation"]:
            return None
        
        level3 = data["annotation"]["level3"]
        
        # Extract ground truth mechanism keywords
        gt_text = ""
        for conv in level3.get("conversations", []):
            if conv.get("from") == "gpt":
                gt_text += conv.get("value", "") + " "
        
        if not gt_text:
            return None
        
        # Expanded mechanism keywords matching actual GT content
        mechanism_keywords = [
            # Core guidance mechanisms
            "mTIR", "modified total internal reflection", "total internal reflection",
            "PBG", "photonic bandgap", "photonic band gap", "bandgap",
            "ARF", "anti-resonant", "antiresonant", "anti resonant",
            "NANF", "nested antiresonant",
            # Physical phenomena
            "confinement", "guidance", "leakage", "tunneling",
            "scattering", "dispersion", "birefringence", "mode field",
            "resonance", "reflection", "refraction", "diffraction",
            # Mode properties
            "fundamental mode", "higher-order mode", "evanescent",
            "effective index", "neff", "cutoff",
        ]
        
        gt_mechanisms = [kw for kw in mechanism_keywords if kw.lower() in gt_text.lower()]
        
        prompt = "Explain the physical mechanism of light guidance in this PCF structure in 3-5 sentences. What principle enables light confinement? Mention the specific guidance mechanism (e.g., mTIR, photonic bandgap, anti-resonant reflection)."
        
        pred_text = self.model.query(prompt, image_path=data["struct_image"], max_tokens=512)
        
        # Check mechanism keyword coverage
        pred_mechanisms = [kw for kw in mechanism_keywords if kw.lower() in pred_text.lower()]
        
        # Compute keyword overlap
        if gt_mechanisms:
            keyword_recall = len(set(pred_mechanisms) & set(gt_mechanisms)) / len(set(gt_mechanisms))
        else:
            keyword_recall = 0.0
        
        # Check if any valid mechanism is mentioned
        has_mechanism = len(pred_mechanisms) > 0
        
        return {
            "sample_id": sample_id,
            "gt_mechanisms": gt_mechanisms,
            "pred_mechanisms": pred_mechanisms,
            "keyword_recall": keyword_recall,
            "has_mechanism": has_mechanism,
            "response_length": len(pred_text)
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        keyword_recalls = [r["keyword_recall"] for r in valid_results]
        has_mechanism_rate = sum(r["has_mechanism"] for r in valid_results) / len(valid_results)
        avg_length = np.mean([r["response_length"] for r in valid_results])
        
        return {
            "mean_keyword_recall": np.mean(keyword_recalls),
            "mechanism_mention_rate": has_mechanism_rate,
            "avg_response_length": avg_length,
            "num_samples": len(valid_results)
        }
