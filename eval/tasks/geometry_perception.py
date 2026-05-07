"""Geometry perception tasks for PCF benchmarking."""

import math
from typing import Any, Dict, Optional
import numpy as np
from .base import BaseTask


def compute_fov_diameter(family: str, params: Dict[str, Any]) -> Optional[float]:
    """Compute the field-of-view diameter (in µm) that the structure image spans.
    
    Mirrors the viewport `lim` calculation in render/struct_renderer.py.
    Returns 2*lim (full diameter) or None if the family is unrecognized.
    """
    # Backward-compatible aliases (same as renderer)
    if family == "solid_core_mtir":
        family = "solid_core_hex"
    if family == "hollow_core_arf":
        family = "anti_resonant_arf"
    
    try:
        if family in ("solid_core_hex", "hollow_core_pbg", "high_birefringence_pm",
                      "multi_core", "fractal_quasicrystal", "kagome_approx",
                      "octagonal_lattice", "elliptical_core", "hybrid_plasmonic",
                      "all_solid_bandgap", "twisted_hexagonal"):
            pitch = float(params["pitch_um"])
            rings = int(params["rings"])
            lim = (rings + 1) * pitch
        elif family == "anti_resonant_arf":
            tube_r = float(params["tube_radius_um"])
            core_r = float(params["core_radius_um"])
            ring_r = core_r + 2.0 * tube_r
            lim = ring_r + tube_r + core_r
        elif family == "suspended_cobweb":
            core_r = float(params["core_radius_um"])
            hole_r = float(params["outer_hole_radius_um"])
            ring_r = core_r + hole_r + 1.0
            lim = ring_r + hole_r + core_r
        elif family == "square_rectangular":
            pitch = float(params["pitch_um"])
            nx = int(params["nx"])
            ny = int(params["ny"])
            lim = max(nx, ny) * pitch / 2.0 + pitch
        elif family == "vortex_ring_core":
            r_out = float(params["ring_outer_radius_um"])
            lim = r_out + 2.0
        elif family == "dual_core_coupler":
            pitch = float(params["pitch_um"])
            rings = int(params["rings"])
            core_sep = float(params["core_separation_um"])
            lim = max((rings + 1) * pitch, core_sep + pitch * 2)
        elif family == "nested_antiresonant":
            outer_tube_r = float(params["outer_tube_radius_um"])
            core_r = float(params["core_radius_um"])
            ring_r = core_r + outer_tube_r * 1.5
            lim = ring_r + outer_tube_r + core_r
        elif family == "graded_index_multiring":
            pitch = float(params["pitch_um"])
            lim = pitch + 1.0
        else:
            return None
        return round(2.0 * lim, 1)
    except (KeyError, ValueError):
        return None


class FamilyClassificationTask(BaseTask):
    """Classify PCF family from structure image."""

    # Canonical snake_case family identifiers (those used in params.json).
    CANONICAL_FAMILIES = (
        "solid_core_hex", "hollow_core_pbg", "anti_resonant_arf",
        "high_birefringence_pm", "suspended_cobweb", "multi_core",
        "square_rectangular", "vortex_ring_core", "fractal_quasicrystal",
        "octagonal_lattice", "dual_core_coupler", "nested_antiresonant",
        "elliptical_core", "hybrid_plasmonic", "all_solid_bandgap",
        "twisted_hexagonal", "graded_index_multiring",
    )

    # Paper-style → canonical aliases.  Any normalised response that
    # equals an alias maps directly; otherwise we fall back to a
    # longest-substring search over the canonical list.
    FAMILY_ALIASES = {
        "solid_core_hexagonal": "solid_core_hex",
        "fractal_quasi_crystal": "fractal_quasicrystal",
        "nested_anti_resonant": "nested_antiresonant",
        "graded_index_multi_ring": "graded_index_multiring",
        "anti_resonant": "anti_resonant_arf",
    }

    @classmethod
    def _normalise_family(cls, response: str) -> str:
        """Map an arbitrary model response to a canonical family identifier.

        Matches the procedure documented in app_task_prompts.tex Task 1:
        lowercase, strip enclosing punctuation, replace ``-/whitespace``
        with underscores, drop parens, then alias-lookup or longest
        canonical-substring match.
        """
        if not response:
            return ""
        text = response.strip().lower()
        # Drop parenthesised tokens (e.g. "Anti-Resonant (ARF)" → "anti-resonant arf").
        text = text.replace("(", " ").replace(")", " ")
        # Convert separators to underscores; collapse repeats.
        for ch in ("-", "/", "\t", "\n", " "):
            text = text.replace(ch, "_")
        while "__" in text:
            text = text.replace("__", "_")
        text = text.strip("_")
        if text in cls.FAMILY_ALIASES:
            return cls.FAMILY_ALIASES[text]
        if text in cls.CANONICAL_FAMILIES:
            return text
        # Longest canonical substring match (prevents "anti_resonant_arf"
        # matching the shorter "multi_core" inside e.g. "anti-resonant
        # ARF (note: not multi-core)").
        hits = [c for c in cls.CANONICAL_FAMILIES if c in text]
        if hits:
            return max(hits, key=len)
        return text  # leave as-is for downstream debugging

    def get_task_name(self) -> str:
        return "family_classification"

    def get_task_description(self) -> str:
        return "Classify PCF structure into one of 17 families from image"

    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        # Ground truth
        gt_family = data["params"]["family"]
        
        # Prompt model
        # Human-readable family list per the appendix specification of Task 1;
        # the extraction step lower-cases + underscore-normalises the response
        # before matching against the canonical 17 family identifiers, so the
        # surface-form of the prompt does not constrain the parser.
        prompt = (
            "What is the family type of this photonic crystal fiber structure? "
            "Choose from: Solid-Core Hexagonal, Hollow-Core PBG, Anti-Resonant (ARF), "
            "High-Birefringence PM, Suspended Cobweb, Multi-Core, Square/Rectangular, "
            "Vortex/Ring-Core, Fractal/Quasi-Crystal, Octagonal Lattice, "
            "Dual-Core Coupler, Nested Anti-Resonant, Elliptical Core, "
            "Hybrid Plasmonic, All-Solid Bandgap, Twisted Hexagonal, "
            "Graded-Index Multi-Ring. Answer with only the family name."
        )
        
        pred_family = self.model.query(prompt, image_path=data["struct_image"], max_tokens=64)

        # Normalise the response (paper appendix Task 1 specifies
        # "lowercase + underscore normalisation; longest-substring match
        #  to the 17 family labels"), then map paper-style aliases
        # ("Solid-Core Hexagonal", "Anti-Resonant (ARF)", ...) back to
        # the canonical snake_case family identifiers in params.json.
        pred_family = self._normalise_family(pred_family)
        correct = (pred_family == gt_family)
        
        return {
            "sample_id": sample_id,
            "ground_truth": gt_family,
            "prediction": pred_family,
            "correct": correct
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        if not self.results:
            return {}
        
        correct = sum(r["correct"] for r in self.results)
        total = len(self.results)
        
        # Compute F1 (macro)
        predictions = [r["prediction"] for r in self.results]
        ground_truths = [r["ground_truth"] for r in self.results]
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


class ParameterPredictionTask(BaseTask):
    """Predict geometric parameters from structure image."""
    
    def __init__(self, data_dir: str, model_interface: Any, param_name: str = "pitch_um"):
        super().__init__(data_dir, model_interface)
        self.param_name = param_name
    
    def get_task_name(self) -> str:
        return f"parameter_prediction_{self.param_name}"
    
    def get_task_description(self) -> str:
        return f"Predict {self.param_name} parameter from structure image"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        # Check if parameter exists
        if self.param_name not in data["params"]["params"]:
            return None
        
        gt_value = float(data["params"]["params"][self.param_name])
        
        # Build prompt (family hint retained as contextual information;
        # FOV diameter removed to prevent GT-derived scale leakage)
        family = data["params"]["family"]
        family_hint = family.replace('_', ' ')
        prompt = (
            f"This is a {family_hint} photonic crystal fiber. "
            f"What is the value of {self.param_name} in this PCF structure? "
            f"Provide only the numerical value in micrometers (e.g., 1.35)."
        )

        pred_str = self.model.query(prompt, image_path=data["struct_image"], max_tokens=128)

        # Extract number
        try:
            import re
            numbers = re.findall(r'\d+\.?\d*', pred_str)
            pred_value = float(numbers[0]) if numbers else 0.0
        except:
            pred_value = 0.0
        
        abs_error = abs(pred_value - gt_value)
        rel_error = abs_error / gt_value if gt_value != 0 else float('inf')
        
        return {
            "sample_id": sample_id,
            "param_name": self.param_name,
            "ground_truth": gt_value,
            "prediction": pred_value,
            "absolute_error": abs_error,
            "relative_error": rel_error
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        gts = [r["ground_truth"] for r in valid_results]
        abs_errors = [r["absolute_error"] for r in valid_results]
        rel_errors = [r["relative_error"] for r in valid_results if r["relative_error"] != float('inf')]
        
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
            "num_samples": len(valid_results)
        }


class StructureCountingTask(BaseTask):
    """Count structural elements (rings, tubes, etc.)."""
    
    def __init__(self, data_dir: str, model_interface: Any, count_param: str = "rings"):
        super().__init__(data_dir, model_interface)
        self.count_param = count_param
    
    def get_task_name(self) -> str:
        return f"structure_counting_{self.count_param}"
    
    def get_task_description(self) -> str:
        return f"Count {self.count_param} in PCF structure from image"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if self.count_param not in data["params"]["params"]:
            return None
        
        gt_count = int(data["params"]["params"][self.count_param])
        
        family_hint = data["params"]["family"].replace('_', ' ')
        prompt = (
            f"This is a {family_hint} photonic crystal fiber. "
            f"How many {self.count_param} are in this PCF structure? "
            f"Provide only the integer number (e.g., 4)."
        )

        pred_str = self.model.query(prompt, image_path=data["struct_image"], max_tokens=128)
        
        try:
            import re
            numbers = re.findall(r'\d+', pred_str)
            pred_count = int(numbers[0]) if numbers else 0
        except:
            pred_count = 0
        
        exact_match = (pred_count == gt_count)
        abs_error = abs(pred_count - gt_count)
        
        return {
            "sample_id": sample_id,
            "count_param": self.count_param,
            "ground_truth": gt_count,
            "prediction": pred_count,
            "exact_match": exact_match,
            "absolute_error": abs_error
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        exact_matches = sum(r["exact_match"] for r in valid_results)
        abs_errors = [r["absolute_error"] for r in valid_results]
        
        return {
            "exact_match_accuracy": exact_matches / len(valid_results),
            "mae": np.mean(abs_errors),
            "num_samples": len(valid_results)
        }


class TopologyRecognitionTask(BaseTask):
    """Recognize lattice topology type."""
    
    def get_task_name(self) -> str:
        return "topology_recognition"
    
    def get_task_description(self) -> str:
        return "Recognize lattice topology (hexagonal, square, octagonal, etc.)"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        family = data["params"]["family"]
        
        # Map family to topology
        topology_map = {
            "solid_core_hex": "hexagonal",
            "hollow_core_pbg": "hexagonal",
            "high_birefringence_pm": "hexagonal",
            "square_rectangular": "square",
            "octagonal_lattice": "octagonal",
            "twisted_hexagonal": "hexagonal",
            "kagome_approx": "kagome",
        }
        
        gt_topology = topology_map.get(family, "other")
        
        prompt = "What is the lattice topology of this PCF structure? Choose from: hexagonal, square, octagonal, kagome, other. Answer with only the topology type."
        
        pred_topology = self.model.query(prompt, image_path=data["struct_image"], max_tokens=64)
        pred_topology = pred_topology.strip().lower()
        
        correct = (pred_topology == gt_topology)
        
        return {
            "sample_id": sample_id,
            "family": family,
            "ground_truth": gt_topology,
            "prediction": pred_topology,
            "correct": correct
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        if not self.results:
            return {}
        
        correct = sum(r["correct"] for r in self.results)
        total = len(self.results)
        
        # Compute F1 (macro)
        predictions = [r["prediction"] for r in self.results]
        ground_truths = [r["ground_truth"] for r in self.results]
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
