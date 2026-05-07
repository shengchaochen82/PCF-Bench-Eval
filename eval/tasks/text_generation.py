"""Text generation tasks for PCF benchmarking."""

from typing import Any, Dict
import re
from .base import BaseTask


class GeometryDescriptionTask(BaseTask):
    """Generate geometry description from image."""
    
    def get_task_name(self) -> str:
        return "geometry_description"
    
    def get_task_description(self) -> str:
        return "Generate accurate geometry description from PCF structure image"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "annotation" not in data or "level1" not in data["annotation"]:
            return None
        
        level1 = data["annotation"]["level1"]
        
        # Extract ground truth description
        gt_desc = ""
        for conv in level1.get("conversations", []):
            if conv.get("from") == "gpt":
                gt_desc = conv.get("value", "")
                break
        
        if not gt_desc:
            return None
        
        prompt = "Describe the geometric structure of this photonic crystal fiber in 3-5 sentences. Include: lattice type, core structure, number of rings, and key geometric parameters."
        
        pred_desc = self.model.query(prompt, image_path=data["struct_image"], max_tokens=512)
        
        # Domain-specific technical terms for PCF geometry description
        tech_terms = [
            # Lattice types
            "hexagonal", "square", "octagonal", "kagome", "triangular",
            # Core types
            "solid-core", "solid core", "hollow-core", "hollow core",
            "anti-resonant", "antiresonant",
            # Structural elements
            "pitch", "hole", "ring", "tube", "core", "cladding",
            "lattice", "strut", "rod", "capillary", "wall",
            # Family-specific
            "bandgap", "birefringence", "plasmonic", "nested",
            "elliptical", "graded", "fractal", "vortex", "dual-core",
            "twisted", "suspended", "cobweb", "multi-core",
            # Geometry parameters
            "diameter", "radius", "thickness", "spacing", "period",
            "symmetry", "defect",
            # Guidance
            "confinement", "guidance", "mTIR", "total internal reflection",
        ]
        
        gt_terms = [term for term in tech_terms if term.lower() in gt_desc.lower()]
        pred_terms = [term for term in tech_terms if term.lower() in pred_desc.lower()]
        
        # Compute term coverage
        if gt_terms:
            term_recall = len(set(pred_terms) & set(gt_terms)) / len(set(gt_terms))
        else:
            term_recall = 0.0
        
        # Compute ROUGE-L and BertScore
        from eval.metrics.generation import compute_rouge, compute_bertscore
        rouge_scores = compute_rouge(pred_desc, gt_desc)
        bert_scores = compute_bertscore(pred_desc, gt_desc)
        
        return {
            "sample_id": sample_id,
            "gt_length": len(gt_desc),
            "pred_length": len(pred_desc),
            "gt_terms": gt_terms,
            "pred_terms": pred_terms,
            "term_recall": term_recall,
            "rouge_l": rouge_scores["rouge_l"],
            "bertscore_f1": bert_scores["bertscore_f1"]
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        import numpy as np
        term_recalls = [r["term_recall"] for r in valid_results]
        pred_lengths = [r["pred_length"] for r in valid_results]
        rouge_scores = [r["rouge_l"] for r in valid_results]
        bert_scores = [r["bertscore_f1"] for r in valid_results]
        
        return {
            "bertscore_f1": float(np.mean(bert_scores)),
            "rouge_l": float(np.mean(rouge_scores)),
            "mean_term_recall": float(np.mean(term_recalls)),
            "avg_pred_length": float(np.mean(pred_lengths)),
            "num_samples": len(valid_results)
        }


class ConversationQATask(BaseTask):
    """Multi-turn conversation QA."""
    
    def get_task_name(self) -> str:
        return "conversation_qa"
    
    def get_task_description(self) -> str:
        return "Answer questions about PCF structure and performance"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "annotation" not in data or "level1" not in data["annotation"]:
            return None

        # Use the first Q&A pair from level1 (open-ended geometry/mechanism
        # description), per the appendix specification of Task 12.  L1's prose
        # answer is what BERTScore-F1 / ROUGE-L are intended to score; L2 is
        # numerical lookup and would saturate those metrics meaninglessly.
        level1 = data["annotation"]["level1"]
        conversations = level1.get("conversations", [])

        if len(conversations) < 2:
            return None

        # Take first Q&A pair
        question = None
        gt_answer = None

        for i, conv in enumerate(conversations):
            if conv.get("from") == "human":
                question = conv.get("value", "")
                if i + 1 < len(conversations) and conversations[i+1].get("from") == "gpt":
                    gt_answer = conversations[i+1].get("value", "")
                    break

        if not question or not gt_answer:
            return None

        pred_answer = self.model.query(question, image_path=data["field_image"], max_tokens=256)
        
        # Check if key information is preserved
        # Extract numbers from both answers
        gt_numbers = re.findall(r'\d+\.?\d*', gt_answer)
        pred_numbers = re.findall(r'\d+\.?\d*', pred_answer)
        
        # Domain-specific keyword overlap (not raw words, which include stopwords)
        domain_terms = [
            "loss", "confinement", "neff", "effective index", "refractive",
            "single-mode", "single_mode", "multi-mode", "multi_mode",
            "leaky", "well-confined", "well_confined",
            "telecom", "sensing", "bandgap", "anti-resonant",
            "birefringence", "dispersion", "nonlinear",
            "dB/km", "dB", "nm", "um", "μm",
            "wavelength", "mode", "fiber", "core", "cladding",
            "guidance", "application", "polarization",
        ]
        gt_lower = gt_answer.lower()
        pred_lower = pred_answer.lower()
        gt_domain = set(t for t in domain_terms if t in gt_lower)
        pred_domain = set(t for t in domain_terms if t in pred_lower)
        keyword_overlap = len(gt_domain & pred_domain) / len(gt_domain) if gt_domain else 0.0
        
        # Compute ROUGE-L and BertScore
        from eval.metrics.generation import compute_rouge, compute_bertscore
        rouge_scores = compute_rouge(pred_answer, gt_answer)
        bert_scores = compute_bertscore(pred_answer, gt_answer)
        
        return {
            "sample_id": sample_id,
            "question": question,
            "gt_numbers_count": len(gt_numbers),
            "pred_numbers_count": len(pred_numbers),
            "keyword_overlap": keyword_overlap,
            "rouge_l": rouge_scores["rouge_l"],
            "bertscore_f1": bert_scores["bertscore_f1"]
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        import numpy as np
        keyword_overlaps = [r["keyword_overlap"] for r in valid_results]
        rouge_scores = [r["rouge_l"] for r in valid_results]
        bert_scores = [r["bertscore_f1"] for r in valid_results]
        
        return {
            "bertscore_f1": float(np.mean(bert_scores)),
            "rouge_l": float(np.mean(rouge_scores)),
            "mean_keyword_overlap": float(np.mean(keyword_overlaps)),
            "num_samples": len(valid_results)
        }


class ReportGenerationTask(BaseTask):
    """Generate complete analysis report."""
    
    def get_task_name(self) -> str:
        return "report_generation"
    
    def get_task_description(self) -> str:
        return "Generate comprehensive PCF analysis report"
    
    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Dict[str, Any]:
        data = self.load_sample_data(sample_dir)
        
        if "annotation" not in data:
            return None
        
        image_path, prompt_prefix = self.prepare_task_images(data, "struct_field_1550")
        
        prompt = f"""{prompt_prefix}Generate a structured analysis report (300-500 words) for this photonic crystal fiber. Cover these four sections:
1. Geometry: lattice type, core structure, key parameters
2. Guidance mechanism: physical principle of light confinement
3. Performance: loss, mode properties, dispersion characteristics
4. Applications: suitable use cases based on the design"""
        
        pred_report = self.model.query(prompt, image_path=image_path, max_tokens=512)
        
        # Check report completeness with expanded domain-specific keywords
        required_sections = [
            ("geometry", ["structure", "lattice", "core", "hole", "pitch",
                          "ring", "cladding", "diameter", "hexagonal", "tube"]),
            ("mechanism", ["guidance", "confinement", "total internal reflection",
                           "mTIR", "bandgap", "PBG", "anti-resonant", "ARF",
                           "resonance", "reflection", "leakage"]),
            ("performance", ["loss", "mode", "dispersion", "birefringence",
                              "confinement loss", "effective index", "neff",
                              "single-mode", "multi-mode", "bandwidth"]),
            ("application", ["telecom", "sensing", "communication", "nonlinear",
                              "polarization", "high power", "laser", "amplifier",
                              "spectroscopy", "medical", "industrial"])
        ]
        
        section_scores = {}
        for section_name, keywords in required_sections:
            score = sum(1 for kw in keywords if kw.lower() in pred_report.lower()) / len(keywords)
            section_scores[f"{section_name}_coverage"] = score
        
        overall_completeness = sum(section_scores.values()) / len(section_scores)
        
        return {
            "sample_id": sample_id,
            "report_length": len(pred_report),
            **section_scores,
            "overall_completeness": overall_completeness
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        valid_results = [r for r in self.results if r is not None]
        if not valid_results:
            return {}
        
        import numpy as np
        
        completeness_scores = [r["overall_completeness"] for r in valid_results]
        report_lengths = [r["report_length"] for r in valid_results]
        
        # Compute average section coverages
        section_metrics = {}
        for key in ["geometry_coverage", "mechanism_coverage", "performance_coverage", "application_coverage"]:
            values = [r[key] for r in valid_results if key in r]
            if values:
                section_metrics[f"avg_{key}"] = np.mean(values)
        
        metrics = {
            "mean_completeness": np.mean(completeness_scores),
            "avg_report_length": np.mean(report_lengths),
            "num_samples": len(valid_results)
        }
        metrics.update(section_metrics)
        
        return metrics
