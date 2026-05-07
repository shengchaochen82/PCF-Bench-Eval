"""Multi-modal Physical Reasoning tasks for PCF benchmarking.

This module implements higher-level reasoning tasks over heterogeneous visual evidence:
- WC (Wavelength Comparison): Comparative judgment between wavelengths
- MD (Mechanism Diagnosis): Multi-class classification of loss mechanisms
- EE (Evidence-based Explanation): Physics-grounded text generation
"""

from typing import Any, Dict, List
import numpy as np
import re
from .base import BaseTask


class WavelengthComparisonTask(BaseTask):
    """Wavelength Comparison (WC) - Comparative judgment between wavelengths.
    
    Uses all three images: struct + field_1310 + field_1550
    Metrics: Accuracy, Balanced Accuracy
    """
    
    def __init__(self, data_dir: str, model_interface: Any, variant: int = 1):
        """
        Args:
            variant: 1 = with structure image, 2 = field images only
        """
        super().__init__(data_dir, model_interface)
        self.variant = variant
        self.image_config = "all_three" if variant == 1 else "both_fields"
    
    def get_task_name(self) -> str:
        return f"wavelength_comparison_v{self.variant}"
    
    def get_task_description(self) -> str:
        return "Compare optical field distributions at different wavelengths"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "sim" not in data or "metrics" not in data["sim"]:
            return None
        
        metrics = data["sim"]["metrics"]
        
        # Get loss values at both wavelengths
        loss_1310 = metrics.get("loss_1310", metrics.get("confinement_loss_1310"))
        loss_1550 = metrics.get("loss_1550", metrics.get("confinement_loss_1550"))
        
        if loss_1310 is None or loss_1550 is None:
            return None
        
        loss_1310 = float(loss_1310)
        loss_1550 = float(loss_1550)
        
        # Ground truth: which wavelength has higher loss
        gt_higher = "1310" if loss_1310 > loss_1550 else "1550"
        
        # Prepare images
        image_path, prompt_prefix = self.prepare_task_images(data, self.image_config)
        
        if self.variant == 1:
            prompt = f"""{prompt_prefix}Compare the optical field distributions at 1310nm and 1550nm. Which wavelength exhibits higher confinement loss? Answer with the wavelength number (1310 or 1550) first, then briefly explain why."""
        else:
            prompt = f"""{prompt_prefix}Which wavelength exhibits higher confinement loss: 1310nm or 1550nm? Answer with only the wavelength number (1310 or 1550)."""
        
        max_tok = 256 if self.variant == 1 else 64
        response = self.model.query(prompt, image_path=image_path, max_tokens=max_tok)
        
        # Extract prediction: find which wavelength is associated with "higher"/"greater"/"more" loss
        response_lower = response.lower()
        # Look for explicit patterns like "1310nm has higher" or "higher loss at 1550"
        match_1310 = re.search(r'1310\s*(?:nm)?\s*(?:has|exhibits?|shows?|with)\s+(?:higher|greater|more|larger)', response_lower)
        match_1550 = re.search(r'1550\s*(?:nm)?\s*(?:has|exhibits?|shows?|with)\s+(?:higher|greater|more|larger)', response_lower)
        higher_at_1310 = re.search(r'(?:higher|greater|more|larger)\s+(?:confinement\s+)?loss\s+(?:at|for)\s+1310', response_lower)
        higher_at_1550 = re.search(r'(?:higher|greater|more|larger)\s+(?:confinement\s+)?loss\s+(?:at|for)\s+1550', response_lower)
        
        if match_1310 or higher_at_1310:
            pred_higher = "1310"
        elif match_1550 or higher_at_1550:
            pred_higher = "1550"
        else:
            # Fallback: first wavelength mentioned in response (case-insensitive)
            first_1310 = response_lower.find("1310")
            first_1550 = response_lower.find("1550")
            if first_1310 >= 0 and (first_1550 < 0 or first_1310 < first_1550):
                pred_higher = "1310"
            elif first_1550 >= 0:
                pred_higher = "1550"
            else:
                pred_higher = "unknown"
        
        correct = (pred_higher == gt_higher)
        
        return {
            "sample_id": sample_id,
            "ground_truth": gt_higher,
            "prediction": pred_higher,
            "correct": correct,
            "loss_1310": loss_1310,
            "loss_1550": loss_1550,
            "response": response[:500]
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        predictions = [r["prediction"] for r in valid_results]
        ground_truths = [r["ground_truth"] for r in valid_results]
        
        # Accuracy
        correct = sum(r["correct"] for r in valid_results)
        accuracy = correct / len(valid_results)
        
        # Balanced Accuracy
        labels = ["1310", "1550"]
        recalls = []
        for label in labels:
            label_results = [r for r in valid_results if r["ground_truth"] == label]
            if label_results:
                label_correct = sum(r["correct"] for r in label_results)
                recalls.append(label_correct / len(label_results))
        
        balanced_acc = np.mean(recalls) if recalls else 0.0
        
        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "num_samples": len(valid_results)
        }


class MechanismDiagnosisTask(BaseTask):
    """Mechanism Diagnosis (MD) - Multi-class classification of loss mechanisms.
    
    Uses struct + field image
    Metrics: Macro-F1, Top-1 Accuracy, Top-3 Accuracy
    """
    
    LOSS_MECHANISMS = [
        "material_absorption",
        "macro_bending", 
        "resonant_coupling",
        "tunneling_leakage",
        "surface_scattering",
        "mode_mismatch",
        "structural_irregularity"
    ]
    
    def __init__(self, data_dir: str, model_interface: Any, variant: int = 1):
        super().__init__(data_dir, model_interface)
        self.variant = variant
        self.image_config = "struct_field_1550"
    
    def get_task_name(self) -> str:
        return f"mechanism_diagnosis_v{self.variant}"
    
    def get_task_description(self) -> str:
        return "Diagnose dominant loss mechanism from structure and field images"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "annotation" not in data:
            return None
        
        # Extract ground truth mechanism from annotations
        gt_mechanism = None
        
        # Try to find mechanism in level3 annotations
        if "level3" in data["annotation"]:
            level3 = data["annotation"]["level3"]
            for conv in level3.get("conversations", []):
                value = conv.get("value", "").lower()
                for mech in self.LOSS_MECHANISMS:
                    if mech.replace("_", " ") in value or mech.replace("_", "-") in value:
                        gt_mechanism = mech
                        break
                if gt_mechanism:
                    break
        
        # Skip samples without a reliable GT mechanism (no fabricated defaults)
        if gt_mechanism is None:
            return None
        
        image_path, prompt_prefix = self.prepare_task_images(data, self.image_config)
        
        if self.variant == 1:
            prompt = f"""{prompt_prefix}Diagnose the dominant loss mechanism from the structure and field pattern. Choose from: material absorption, macro-bending, resonant coupling, tunneling leakage, surface scattering, mode mismatch, structural irregularity. State the mechanism first, then briefly explain."""
        else:
            prompt = f"""{prompt_prefix}Diagnose the dominant loss mechanism. Choose from: material absorption, macro-bending, resonant coupling, tunneling leakage, surface scattering, mode mismatch, structural irregularity. Answer with only the mechanism name."""
        
        max_tok = 512 if self.variant == 1 else 128
        response = self.model.query(prompt, image_path=image_path, max_tokens=max_tok)
        response_lower = response.lower()
        
        # Extract predictions (may have multiple).
        # Use word-boundary regex to avoid false positives (e.g. "not surface scattering").
        pred_mechanisms = []
        for mech in self.LOSS_MECHANISMS:
            mech_variants = [
                mech.replace("_", " "),
                mech.replace("_", "-"),
                mech
            ]
            for variant in mech_variants:
                # Word-boundary match to reduce false positives
                if re.search(r'\b' + re.escape(variant) + r'\b', response_lower):
                    pred_mechanisms.append(mech)
                    break

        # Primary prediction is the first mentioned
        pred_mechanism = pred_mechanisms[0] if pred_mechanisms else "unknown"

        correct = (pred_mechanism == gt_mechanism)
        # Top-3: check if GT is anywhere in the first 3 predictions
        top3_correct = gt_mechanism in pred_mechanisms[:3]
        
        return {
            "sample_id": sample_id,
            "ground_truth": gt_mechanism,
            "prediction": pred_mechanism,
            "all_predictions": pred_mechanisms,
            "correct": correct,
            "top3_correct": top3_correct,
            "response": response[:500]
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        predictions = [r["prediction"] for r in valid_results]
        ground_truths = [r["ground_truth"] for r in valid_results]
        
        # Top-1 Accuracy
        top1_acc = sum(r["correct"] for r in valid_results) / len(valid_results)
        
        # Top-3 Accuracy
        top3_acc = sum(r["top3_correct"] for r in valid_results) / len(valid_results)
        
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
            "macro_f1": macro_f1,
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
            "num_samples": len(valid_results)
        }


class EvidenceBasedExplanationTask(BaseTask):
    """Evidence-based Explanation (EE) - Physics-grounded text generation.
    
    Uses struct + field image
    Metrics: BERTScore-F1, ROUGE-L
    """
    
    def __init__(self, data_dir: str, model_interface: Any, variant: int = 1):
        super().__init__(data_dir, model_interface)
        self.variant = variant
        self.image_config = "struct_field_1550" if variant == 1 else "field_1550_only"
    
    def get_task_name(self) -> str:
        return f"evidence_explanation_v{self.variant}"
    
    def get_task_description(self) -> str:
        return "Generate evidence-based explanation citing visual features"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "annotation" not in data:
            return None
        
        # Extract ground truth explanation from level3
        gt_explanation = ""
        if "level3" in data["annotation"]:
            level3 = data["annotation"]["level3"]
            for conv in level3.get("conversations", []):
                if conv.get("from") == "gpt":
                    gt_explanation += conv.get("value", "") + " "
        
        if not gt_explanation.strip():
            return None
        
        # Get mode label for context
        mode_label = "fundamental mode"
        if "level2" in data["annotation"]:
            layer2_text = str(data["annotation"]["level2"])
            if "multi-mode" in layer2_text.lower():
                mode_label = "multi-mode"
            elif "higher-order" in layer2_text.lower() or "hom" in layer2_text.lower():
                mode_label = "Higher-Order Mode (HOM)"
        
        image_path, prompt_prefix = self.prepare_task_images(data, self.image_config)
        
        if self.variant == 1:
            prompt = f"""{prompt_prefix}The mode is classified as {mode_label}. Provide an evidence-based explanation (3-5 sentences) citing specific visual features in the cladding region."""
        else:
            prompt = f"""{prompt_prefix}Why is this mode considered a Higher-Order Mode (HOM)? Point out the visual evidence in the field distribution in 2-3 sentences."""
        
        max_tok = 512 if self.variant == 1 else 256
        response = self.model.query(prompt, image_path=image_path, max_tokens=max_tok)
        
        # Compute metrics
        from eval.metrics.generation import compute_rouge, compute_bertscore

        rouge_scores = compute_rouge(response, gt_explanation)
        bert_scores = compute_bertscore(response, gt_explanation)
        
        # Expanded evidence keywords for visual feature citation
        evidence_keywords = [
            "cladding", "core", "intensity", "field", "leakage", "pattern",
            "distribution", "bright", "dark", "ring", "structure", "visible",
            "penetration", "confinement", "lobe", "symmetric", "asymmetric",
            "tube", "hole", "boundary", "interface", "evanescent", "decay",
            "concentrated", "spread", "confined", "extending", "surrounding",
        ]
        evidence_count = sum(1 for kw in evidence_keywords if kw.lower() in response.lower())
        
        return {
            "sample_id": sample_id,
            "gt_length": len(gt_explanation),
            "pred_length": len(response),
            "rouge_l": rouge_scores["rouge_l"],
            "bertscore_f1": bert_scores["bertscore_f1"],
            "evidence_keywords": evidence_count,
            "response": response[:500]
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        rouge_scores = [r["rouge_l"] for r in valid_results]
        bert_scores = [r["bertscore_f1"] for r in valid_results]
        evidence_counts = [r["evidence_keywords"] for r in valid_results]
        
        return {
            "bertscore_f1": np.mean(bert_scores),
            "rouge_l": np.mean(rouge_scores),
            "avg_evidence_keywords": np.mean(evidence_counts),
            "num_samples": len(valid_results)
        }
