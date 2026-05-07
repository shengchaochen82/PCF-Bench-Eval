"""Inverse Design tasks for PCF benchmarking.

This module implements actionable design assistance tasks:
- PS (Parameter Selection): Classification of which parameter to optimize
- DP (Design Direction Prediction): Binary decision on increase/decrease
- CI (Closed-loop Improvement): Suggest numeric parameter modifications
"""

from typing import Any, Dict, List, Optional
import numpy as np
import re
from .base import BaseTask


class ParameterSelectionTask(BaseTask):
    """Parameter Selection (PS) - Classification task for design optimization.
    
    Given structure and field images, predict which geometric parameter 
    should be prioritized for optimization.
    Metrics: Accuracy, Macro-F1
    """
    
    PARAMETERS = [
        "pitch_um",       # Λ - lattice pitch
        "hole_diameter",  # d - hole diameter  
        "core_diameter",  # core size
        "strut_thickness", # t - strut/wall thickness
        "ring_radius",    # ring tube radius
        "num_rings",      # number of cladding rings
        "gap_width"       # gap between elements
    ]
    
    def __init__(self, data_dir: str, model_interface: Any, variant: int = 1):
        super().__init__(data_dir, model_interface)
        self.variant = variant
        self.image_config = "struct_field_1550"
    
    def get_task_name(self) -> str:
        return f"parameter_selection_v{self.variant}"
    
    def get_task_description(self) -> str:
        return "Select which geometric parameter to optimize for improved performance"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "params" not in data:
            return None
        
        # Determine ground truth parameter based on family AND loss level.
        # High-loss samples likely suffer from a dominant design flaw whose
        # most sensitive parameter differs from the general family default.
        params = data["params"]
        family = params.get("family", "")
        loss = float(data.get("sim", {}).get("metrics", {}).get("loss_1550", 1.0) or 1.0)

        # Default per-family priority (moderate-loss regime)
        param_priority = {
            "anti_resonant_arf": "strut_thickness",
            "nested_antiresonant": "strut_thickness",
            "hollow_core_pbg": "pitch_um",
            "solid_core_hex": "hole_diameter",
            "high_birefringence_pm": "hole_diameter",
            "suspended_cobweb": "strut_thickness",
            "multi_core": "core_diameter",
            "vortex_ring_core": "ring_radius",
            "kagome": "strut_thickness",
        }

        if loss > 0.1:
            # Very high loss: for anti-resonant families the strut is almost
            # certainly near a resonance; for index-guiding fibers the
            # air-filling fraction (hole_diameter) is the bottleneck.
            high_loss_priority = {
                "anti_resonant_arf": "strut_thickness",
                "nested_antiresonant": "strut_thickness",
                "kagome": "strut_thickness",
                "suspended_cobweb": "strut_thickness",
                "hollow_core_pbg": "hole_diameter",
                "solid_core_hex": "hole_diameter",
                "high_birefringence_pm": "hole_diameter",
                "multi_core": "pitch_um",
            }
            gt_param = high_loss_priority.get(family, param_priority.get(family, "pitch_um"))
        else:
            gt_param = param_priority.get(family, "pitch_um")
        
        image_path, prompt_prefix = self.prepare_task_images(data, self.image_config)
        
        if self.variant == 1:
            prompt = f"""{prompt_prefix}To reduce the confinement loss observed here, which geometric parameter should be prioritized for optimization? Choose from: pitch (Λ), hole diameter (d), strut thickness (t), core diameter, ring radius, number of rings, or gap width. Answer with the parameter name first, then briefly explain why."""
        else:
            prompt = f"""{prompt_prefix}The mode is leaking into the struts. Which structural parameter controls this resonance behavior? Answer with only the parameter name."""
        
        max_tok = 256 if self.variant == 1 else 64
        response = self.model.query(prompt, image_path=image_path, max_tokens=max_tok)
        response_lower = response.lower()
        
        # Map response to parameter
        param_mapping = {
            "pitch": "pitch_um",
            "Λ": "pitch_um",
            "lambda": "pitch_um",
            "hole": "hole_diameter",
            "diameter": "hole_diameter",
            "d": "hole_diameter",
            "strut": "strut_thickness",
            "thickness": "strut_thickness",
            "wall": "strut_thickness",
            "t": "strut_thickness",
            "core": "core_diameter",
            "ring": "ring_radius",
            "radius": "ring_radius",
            "num": "num_rings",
            "number": "num_rings",
            "gap": "gap_width"
        }
        
        pred_param = "unknown"
        # Sort by key length descending so "strut thickness" matches before "t"
        for key, value in sorted(param_mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
            if re.search(r'\b' + re.escape(key) + r'\b', response_lower):
                pred_param = value
                break
        
        correct = (pred_param == gt_param)
        
        return {
            "sample_id": sample_id,
            "family": family,
            "ground_truth": gt_param,
            "prediction": pred_param,
            "correct": correct,
            "response": response[:500]
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        predictions = [r["prediction"] for r in valid_results]
        ground_truths = [r["ground_truth"] for r in valid_results]
        
        # Accuracy
        accuracy = sum(r["correct"] for r in valid_results) / len(valid_results)
        
        # Macro F1
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
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "num_samples": len(valid_results)
        }


class DesignDirectionTask(BaseTask):
    """Design Direction Prediction (DP) - Binary decision task.
    
    Predict whether a parameter should be increased or decreased 
    to achieve a design goal.
    Metrics: Accuracy, F1
    """
    
    PARAM_ALIASES = {
        "pitch_um": ["pitch_um", "pitch"],
        "hole_diameter": ["hole_d_um", "hole_d", "hole_diameter", "d"],
        "core_diameter": ["core_d_um", "core_d", "core_diameter", "core_radius_um"],
        "strut_thickness": ["strut_thickness", "strut_um", "wall_thickness"],
        "ring_radius": ["ring_radius_um", "ring_radius", "tube_radius_um"],
        "num_rings": ["rings", "num_rings", "n_rings"],
        "gap_width": ["gap_width_um", "gap_width", "gap"],
    }
    
    # ── Family-aware physics rules ──────────────────────────────────────────
    # Sources: Pryamikov et al. (OE 2011), Poletti (OE 2014), Litchinitser
    # et al. (OL 2002), Russell (Science 2003).  The rules encode first-order
    # design intuition; non-monotonic regimes (ARF resonances, PBG bandgap
    # edges) are documented in comments.

    @staticmethod
    def _physics_direction(family: str, target_param: str, loss: float) -> str:
        """Return the physics-motivated direction ('increase' or 'decrease')
        to reduce confinement loss for the given family and parameter.

        High-loss threshold: 0.01 dB/km (above → needs improvement).
        """
        high_loss = loss > 0.01

        # ── Anti-resonant fibers (ARF, nested) ────────────────────────────
        # Strut thickness is non-monotonic (resonances at t = mλ/2√(n²-1)),
        # but thinner walls broaden the anti-resonant band, so 'decrease' is
        # the correct first-order heuristic when loss is high.  For size
        # parameters (core diameter, ring radius, pitch), increasing them
        # reduces loss by enlarging the air core.
        if family in ("anti_resonant_arf", "nested_antiresonant"):
            if target_param == "strut_thickness":
                return "decrease" if high_loss else "increase"
            if target_param == "gap_width":
                return "decrease" if high_loss else "increase"
            return "increase" if high_loss else "decrease"

        # ── Kagome / suspended-cobweb ─────────────────────────────────────
        # Inhibited-coupling guidance: thinner struts → fewer cladding states
        # to couple to → lower loss (smoother than ARF).
        if family in ("kagome", "suspended_cobweb"):
            if target_param in ("strut_thickness", "gap_width"):
                return "decrease" if high_loss else "increase"
            return "increase" if high_loss else "decrease"

        # ── Hollow-core PBG ───────────────────────────────────────────────
        # Guidance exists only within the bandgap.  Increasing d/Λ widens
        # the gap and reliably reduces loss.  Pitch shifts the bandgap
        # position — "increase pitch" is a reasonable default because most
        # HC-PBG operate near the short-wavelength bandgap edge at 1550 nm.
        if family == "hollow_core_pbg":
            if target_param in ("hole_diameter", "pitch_um"):
                return "increase" if high_loss else "decrease"
            if target_param in ("strut_thickness", "gap_width"):
                return "decrease" if high_loss else "increase"
            return "increase" if high_loss else "decrease"

        # ── Solid-core index-guiding (hexagonal, PM, multi-core, etc.) ────
        # Monotonic: larger pitch and larger holes → stronger confinement.
        if target_param in ("pitch_um", "hole_diameter", "core_diameter",
                            "ring_radius", "num_rings"):
            return "increase" if high_loss else "decrease"
        if target_param in ("strut_thickness", "gap_width"):
            return "decrease" if high_loss else "increase"

        # Fallback for unknown param/family
        return "increase" if high_loss else "decrease"

    def __init__(self, data_dir: str, model_interface: Any,
                 target_param: str = "pitch_um", variant: int = 1):
        super().__init__(data_dir, model_interface)
        self.target_param = target_param
        self.variant = variant
        self.image_config = "struct_field_1550"
    
    def _find_param_value(self, params: Dict[str, Any]) -> Optional[float]:
        """Find parameter value using aliases."""
        aliases = self.PARAM_ALIASES.get(self.target_param, [self.target_param])
        for alias in aliases:
            if alias in params:
                return float(params[alias])
        return None
    
    def get_task_name(self) -> str:
        return f"design_direction_{self.target_param}_v{self.variant}"
    
    def get_task_description(self) -> str:
        return f"Predict whether {self.target_param} should increase or decrease"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "params" not in data or "sim" not in data:
            return None
        
        params = data["params"].get("params", {})
        current_value = self._find_param_value(params)
        if current_value is None:
            return None
        
        family = data["params"].get("family", "")
        loss = data["sim"].get("metrics", {}).get("loss_1550", 1.0)
        loss = float(loss) if loss else 1.0

        gt_direction = self._physics_direction(family, self.target_param, loss)
        
        image_path, prompt_prefix = self.prepare_task_images(data, self.image_config)
        
        param_display = self.target_param.replace("_", " ").replace("um", "")
        
        if self.variant == 1:
            prompt = f"""{prompt_prefix}The confinement loss is {loss:.2e} dB/km. Should the {param_display} be increased or decreased to improve confinement? Answer with 'increase' or 'decrease' first, then briefly explain."""
        else:
            prompt = f"""{prompt_prefix}To reduce the confinement loss, should the {param_display} be increased or decreased? Answer 'increase' or 'decrease'."""
        
        max_tok = 256 if self.variant == 1 else 64
        response = self.model.query(prompt, image_path=image_path, max_tokens=max_tok)
        response_lower = response.lower()
        
        if "increase" in response_lower:
            pred_direction = "increase"
        elif "decrease" in response_lower or "reduce" in response_lower:
            pred_direction = "decrease"
        else:
            pred_direction = "unknown"
        
        correct = (pred_direction == gt_direction)
        
        return {
            "sample_id": sample_id,
            "param": self.target_param,
            "current_value": current_value,
            "ground_truth": gt_direction,
            "prediction": pred_direction,
            "correct": correct,
            "response": response[:500]
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        predictions = [r["prediction"] for r in valid_results]
        ground_truths = [r["ground_truth"] for r in valid_results]
        
        # Accuracy
        accuracy = sum(r["correct"] for r in valid_results) / len(valid_results)
        
        # F1 for binary classification
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == "increase" and g == "increase")
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == "increase" and g == "decrease")
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p == "decrease" and g == "increase")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            # Binary F1/precision/recall for the positive class (= "increase").
            # Documented as such in app_task_prompts.tex Task 21/22 metric line.
            "binary_f1": f1,
            "binary_precision": precision,
            "binary_recall": recall,
            "num_samples": len(valid_results)
        }


class ClosedLoopImprovementTask(BaseTask):
    """Closed-loop Improvement (CI) - Numeric parameter modification task.
    
    Suggest a specific numeric change to a parameter to achieve 
    a performance improvement target.
    Metrics: Success Rate, Mean ΔProxy, Median ΔProxy
    """
    
    PARAM_ALIASES = {
        "pitch_um": ["pitch_um", "pitch"],
        "hole_diameter": ["hole_d_um", "hole_d", "hole_diameter", "d"],
        "core_diameter": ["core_d_um", "core_d", "core_diameter", "core_radius_um"],
        "strut_thickness": ["strut_thickness", "strut_um", "wall_thickness"],
        "ring_radius": ["ring_radius_um", "ring_radius", "tube_radius_um"],
        "num_rings": ["rings", "num_rings", "n_rings"],
        "gap_width": ["gap_width_um", "gap_width", "gap"],
    }
    
    def __init__(self, data_dir: str, model_interface: Any, 
                 target_param: str = "pitch_um", target_improvement: float = 0.1,
                 variant: int = 1):
        super().__init__(data_dir, model_interface)
        self.target_param = target_param
        self.target_improvement = target_improvement
        self.variant = variant
        self.image_config = "struct_field_1550"
    
    def _find_param_value(self, params: Dict[str, Any]) -> Optional[float]:
        aliases = self.PARAM_ALIASES.get(self.target_param, [self.target_param])
        for alias in aliases:
            if alias in params:
                return float(params[alias])
        return None
    
    def get_task_name(self) -> str:
        return f"closed_loop_improvement_{self.target_param}_v{self.variant}"
    
    def get_task_description(self) -> str:
        return f"Suggest numeric modification to {self.target_param} for improvement"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "params" not in data or "sim" not in data:
            return None
        
        params = data["params"].get("params", {})
        current_value = self._find_param_value(params)
        if current_value is None:
            return None
        
        current_loss = float(data["sim"].get("metrics", {}).get("loss_1550", 1.0))
        
        image_path, prompt_prefix = self.prepare_task_images(data, self.image_config)
        
        param_display = self.target_param.replace("_", " ").replace("um", "μm")
        
        if self.variant == 1:
            prompt = f"""{prompt_prefix}The current loss is {current_loss:.2e} dB/km. Suggest a new value for {param_display} (currently {current_value:.2f} μm) to reduce the loss. Provide only the new numeric value in μm."""
        else:
            prompt = f"""{prompt_prefix}Current parameter: {param_display} = {current_value:.2f} μm. Propose a new value for {param_display} to suppress the higher-order mode shown in the image. Answer with only the numeric value in μm."""
        
        response = self.model.query(prompt, image_path=image_path, max_tokens=128)
        
        # Extract numeric value from response
        try:
            numbers = re.findall(r'(\d+\.?\d*)', response)
            if numbers:
                pred_value = float(numbers[0])
            else:
                pred_value = current_value
        except:
            pred_value = current_value
        
        # Compute delta
        delta = pred_value - current_value
        relative_change = abs(delta) / current_value if current_value != 0 else 0
        
        # Success criteria: reasonable change (not too extreme, not trivial)
        reasonable_change = 0.02 <= relative_change <= 0.5  # 2-50% change

        # Use the same family-aware physics rules as DesignDirection
        family = data["params"].get("family", "")
        gt_direction = DesignDirectionTask._physics_direction(
            family, self.target_param, current_loss)
        direction_correct = (delta > 0) == (gt_direction == "increase")

        success = reasonable_change and direction_correct
        
        # Proxy metric: normalized distance from a reasonable 10% change
        # Lower is better
        target_delta = 0.10 * current_value
        delta_proxy = abs(abs(delta) - target_delta) / current_value if current_value > 0 else float('inf')
        
        return {
            "sample_id": sample_id,
            "param": self.target_param,
            "current_value": current_value,
            "predicted_value": pred_value,
            "delta": delta,
            "relative_change": relative_change,
            "success": success,
            "delta_proxy": delta_proxy,
            "response": response[:500]
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        # Success Rate
        success_rate = sum(r["success"] for r in valid_results) / len(valid_results)
        
        # Mean and Median ΔProxy
        delta_proxies = [r["delta_proxy"] for r in valid_results]
        mean_delta_proxy = np.mean(delta_proxies)
        median_delta_proxy = np.median(delta_proxies)
        
        # Additional stats
        relative_changes = [r["relative_change"] for r in valid_results]
        
        return {
            "success_rate": success_rate,
            "mean_delta_proxy": mean_delta_proxy,
            "median_delta_proxy": median_delta_proxy,
            "mean_relative_change": np.mean(relative_changes),
            "num_samples": len(valid_results)
        }
