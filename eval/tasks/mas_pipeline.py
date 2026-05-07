"""
Multi-Agent System (MAS) Pipeline Evaluation for PCFBench.

Evaluates a sequential 4-stage pipeline that mirrors the real photonic
inverse-design workflow:

    Perception → Understanding → Reasoning → Design

Two modes:
  - Oracle:   each stage receives ground-truth context from prior stages.
  - Cascaded: each stage receives *predicted* context from the prior agent.

The cascaded mode exposes error propagation through the pipeline.
"""

from typing import Any, Dict, List, Optional
import json
import re
import numpy as np
from .base import BaseTask


# ── Stage prompt templates ────────────────────────────────────────────────

STAGE1_PERCEPTION_PROMPT = """\
Analyze this photonic crystal fiber cross-section image and provide:
1. Family type (choose from: Solid-Core Hexagonal, Hollow-Core PBG, Anti-Resonant ARF, \
High-Birefringence PM, Suspended Cobweb, Multi-Core, Square/Rectangular, \
Vortex/Ring-Core, Fractal/Quasi-Crystal, Octagonal Lattice, Dual-Core Coupler, \
Nested Anti-Resonant, Elliptical Core, Hybrid Plasmonic, All-Solid Bandgap, \
Twisted Hexagonal, Graded-Index Multi-Ring)
2. Estimated pitch in micrometers
3. Number of cladding rings
4. Lattice topology (hexagonal, square, octagonal, kagome, or other)

Respond in this exact format:
Family: <family name>
Pitch: <value> um
Rings: <integer>
Topology: <type>"""

STAGE2_UNDERSTANDING_PROMPT = """\
You previously identified this fiber as: {perception_context}

Now analyze the mode-field image at 1550 nm and provide:
1. Estimated confinement loss in dB/km (use scientific notation)
2. Mode type: single-mode, multi-mode, or leaky
3. Primary guidance mechanism (e.g., modified TIR, photonic bandgap, anti-resonant reflection, inhibited coupling)

Respond in this exact format:
Loss: <value> dB/km
Mode: <type>
Mechanism: <mechanism name>"""

STAGE3_REASONING_PROMPT = """\
Based on your prior analysis:
{perception_context}
{understanding_context}

Now examine both the structure and field images together. Provide:
1. Which wavelength (1310 nm or 1550 nm) likely has higher confinement loss, and why?
2. What is the dominant loss mechanism for this fiber?
3. A brief evidence-based explanation (2-3 sentences) citing visual features.

Respond in this exact format:
Higher loss wavelength: <1310 or 1550>
Dominant mechanism: <mechanism>
Explanation: <your explanation>"""

STAGE4_DESIGN_PROMPT = """\
Based on the full analysis pipeline:
{perception_context}
{understanding_context}
{reasoning_context}

The fiber currently has a confinement loss of {loss_value} dB/km.
Your task is to propose a design improvement:
1. Which parameter should be adjusted first? (pitch, hole diameter, strut thickness, core diameter, ring radius, gap width, or number of rings)
2. Should it be increased or decreased?
3. Suggest a specific new value for the parameter you selected (if the current value is known).

Respond in this exact format:
Parameter: <parameter name>
Direction: <increase or decrease>
Reasoning: <brief physics justification>"""


# ── Parsing helpers ───────────────────────────────────────────────────────

def _parse_kv(text: str, key: str) -> str:
    """Extract value for a key from 'Key: value' formatted text."""
    pattern = rf'{key}\s*:\s*(.+)'
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_number(text: str) -> Optional[float]:
    """Extract first number (including scientific notation) from text."""
    m = re.search(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)', text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _normalize_family(name: str) -> str:
    """Normalize family name to code-style identifier."""
    return name.strip().lower().replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")


# ══════════════════════════════════════════════════════════════════════════
#  MAS Pipeline Task
# ══════════════════════════════════════════════════════════════════════════

class MASPipelineTask(BaseTask):
    """Multi-Agent System pipeline: Perception → Understanding → Reasoning → Design.

    Each stage queries the VLM with accumulated context from prior stages.

    Args:
        mode: "cascaded" (predicted context) or "oracle" (ground-truth context).
    """

    VALID_MODES = ("cascaded", "oracle")

    def __init__(self, data_dir: str, model_interface: Any, mode: str = "cascaded"):
        super().__init__(data_dir, model_interface)
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.mode = mode

    def get_task_name(self) -> str:
        return f"mas_pipeline_{self.mode}"

    def get_task_description(self) -> str:
        return (f"Multi-agent pipeline ({self.mode}): "
                "Perception → Understanding → Reasoning → Design")

    # ── Stage 1: Perception ───────────────────────────────────────────────

    def _run_perception(self, data: Dict) -> Dict[str, Any]:
        """Identify family, estimate parameters, count rings, classify topology."""
        image_path = data.get("struct_image")
        if not image_path:
            return {"success": False, "context": ""}

        response = self.model.query(
            STAGE1_PERCEPTION_PROMPT,
            image_path=image_path,
            max_tokens=256
        )

        result = {
            "family": _parse_kv(response, "Family"),
            "pitch": _parse_number(_parse_kv(response, "Pitch")),
            "rings": _parse_number(_parse_kv(response, "Rings")),
            "topology": _parse_kv(response, "Topology"),
            "raw_response": response[:500],
            "success": True,
        }

        # Build context string for downstream stages
        result["context"] = (
            f"Family: {result['family']}. "
            f"Pitch: {result['pitch']} um. "
            f"Rings: {int(result['rings']) if result['rings'] else 'unknown'}. "
            f"Topology: {result['topology']}."
        )
        return result

    def _oracle_perception(self, data: Dict) -> Dict[str, Any]:
        """Build perception context from ground truth."""
        params = data.get("params", {})
        pp = params.get("params", {})
        family = params.get("family", "unknown")
        pitch = pp.get("pitch_um", "unknown")
        rings = pp.get("rings") or pp.get("num_rings") or pp.get("n_rings") or "unknown"

        context = f"Family: {family}. Pitch: {pitch} um. Rings: {rings}."
        return {
            "family": family,
            "pitch": pitch,
            "rings": rings,
            "topology": "ground_truth",
            "context": context,
            "success": True,
        }

    # ── Stage 2: Understanding ────────────────────────────────────────────

    def _run_understanding(self, data: Dict, perception_ctx: str) -> Dict[str, Any]:
        """Predict loss, classify mode, identify guidance mechanism."""
        image_path = data.get("field_1550_image") or data.get("field_image")
        if not image_path:
            return {"success": False, "context": ""}

        prompt = STAGE2_UNDERSTANDING_PROMPT.format(
            perception_context=perception_ctx
        )
        response = self.model.query(prompt, image_path=image_path, max_tokens=256)

        loss_str = _parse_kv(response, "Loss")
        loss_val = _parse_number(loss_str)
        mode_str = _parse_kv(response, "Mode")
        mechanism = _parse_kv(response, "Mechanism")

        result = {
            "loss_pred": loss_val,
            "mode_pred": mode_str.lower().replace(" ", "_") if mode_str else "",
            "mechanism_pred": mechanism,
            "raw_response": response[:500],
            "success": True,
        }
        result["context"] = (
            f"Loss (1550 nm): {loss_val if loss_val else 'unknown'} dB/km. "
            f"Mode: {result['mode_pred']}. "
            f"Mechanism: {mechanism}."
        )
        return result

    def _oracle_understanding(self, data: Dict) -> Dict[str, Any]:
        """Build understanding context from ground truth."""
        sim = data.get("sim", {})
        metrics = sim.get("metrics", {})
        loss = metrics.get("loss_1550") or metrics.get("confinement_loss_1550")
        loss = float(loss) if loss else None

        context = f"Loss (1550 nm): {loss} dB/km."
        return {
            "loss_pred": loss,
            "mode_pred": "ground_truth",
            "mechanism_pred": "ground_truth",
            "context": context,
            "success": True,
        }

    # ── Stage 3: Reasoning ────────────────────────────────────────────────

    def _run_reasoning(self, data: Dict, perception_ctx: str,
                       understanding_ctx: str) -> Dict[str, Any]:
        """Comparative reasoning and mechanism diagnosis."""
        image_path, prompt_prefix = self.prepare_task_images(data, "struct_field_1550")

        prompt = STAGE3_REASONING_PROMPT.format(
            perception_context=perception_ctx,
            understanding_context=understanding_ctx
        )
        response = self.model.query(
            prompt_prefix + prompt, image_path=image_path, max_tokens=512
        )

        higher_wl = _parse_kv(response, "Higher loss wavelength")
        mechanism = _parse_kv(response, "Dominant mechanism")
        explanation = _parse_kv(response, "Explanation")

        result = {
            "higher_loss_wl": higher_wl,
            "diagnosed_mechanism": mechanism,
            "explanation": explanation[:300],
            "raw_response": response[:500],
            "success": True,
        }
        result["context"] = (
            f"Higher loss at: {higher_wl} nm. "
            f"Dominant mechanism: {mechanism}. "
            f"Explanation: {explanation[:150]}."
        )
        return result

    # ── Stage 4: Design ───────────────────────────────────────────────────

    def _run_design(self, data: Dict, perception_ctx: str,
                    understanding_ctx: str, reasoning_ctx: str,
                    loss_value: float) -> Dict[str, Any]:
        """Propose parameter modification (closed-loop design)."""
        image_path, prompt_prefix = self.prepare_task_images(data, "struct_field_1550")

        loss_str = f"{loss_value:.2e}" if loss_value is not None else "unknown"
        prompt = STAGE4_DESIGN_PROMPT.format(
            perception_context=perception_ctx,
            understanding_context=understanding_ctx,
            reasoning_context=reasoning_ctx,
            loss_value=loss_str
        )
        response = self.model.query(
            prompt_prefix + prompt, image_path=image_path, max_tokens=256
        )

        param = _parse_kv(response, "Parameter").lower().replace(" ", "_")
        direction = _parse_kv(response, "Direction").lower()
        reasoning = _parse_kv(response, "Reasoning")

        # Normalize direction
        if "increase" in direction:
            direction = "increase"
        elif "decrease" in direction or "reduce" in direction:
            direction = "decrease"
        else:
            direction = "unknown"

        return {
            "selected_param": param,
            "direction": direction,
            "design_reasoning": reasoning[:300],
            "raw_response": response[:500],
            "success": True,
        }

    # ── Main evaluate_sample ──────────────────────────────────────────────

    def evaluate_sample(self, sample_dir: str, sample_id: str) -> Optional[Dict[str, Any]]:
        data = self.load_sample_data(sample_dir)

        if "params" not in data:
            return None

        # Ground truth for evaluation
        params = data.get("params", {})
        pp = params.get("params", {})
        family_gt = params.get("family", "")
        sim = data.get("sim", {})
        metrics_gt = sim.get("metrics", {})
        loss_1550_raw = metrics_gt.get("loss_1550") or metrics_gt.get("confinement_loss_1550")
        loss_gt = float(loss_1550_raw) if loss_1550_raw is not None else None

        # Skip samples without loss data (cannot evaluate design)
        if loss_gt is None:
            return None

        # ── Stage 1: Perception ───────────────────────────────────────────
        if self.mode == "oracle":
            s1 = self._oracle_perception(data)
        else:
            s1 = self._run_perception(data)
        if not s1["success"]:
            return None

        # Evaluate perception — use substring matching to handle
        # abbreviation mismatches (GT "solid_core_hex" vs pred "solid_core_hexagonal")
        family_pred = _normalize_family(s1["family"])
        family_gt_norm = _normalize_family(family_gt)
        family_correct = (family_pred == family_gt_norm
                          or family_gt_norm in family_pred
                          or family_pred in family_gt_norm)

        rings_gt = pp.get("rings") or pp.get("num_rings") or pp.get("n_rings")
        rings_pred = int(s1["rings"]) if s1["rings"] else None
        rings_correct = (rings_pred == int(rings_gt)) if rings_gt and rings_pred else False

        # ── Stage 2: Understanding ────────────────────────────────────────
        if self.mode == "oracle":
            s2 = self._oracle_understanding(data)
        else:
            s2 = self._run_understanding(data, s1["context"])
        if not s2["success"]:
            return None

        # Evaluate loss prediction
        loss_pred = s2["loss_pred"]
        if loss_pred is not None and loss_pred > 0 and loss_gt > 0:
            log_mae = abs(np.log10(max(loss_pred, 1e-15)) - np.log10(max(loss_gt, 1e-15)))
        else:
            log_mae = None

        # ── Stage 3: Reasoning ────────────────────────────────────────────
        s3 = self._run_reasoning(data, s1["context"], s2["context"])

        # Evaluate wavelength comparison
        loss_1310_raw = metrics_gt.get("loss_1310") or metrics_gt.get("confinement_loss_1310")
        loss_1310 = float(loss_1310_raw) if loss_1310_raw is not None else 0.0
        wl_gt = "1310" if loss_1310 > loss_gt else "1550"
        wl_pred = s3.get("higher_loss_wl", "").strip().replace(" nm", "").replace("nm", "")
        wl_correct = (wl_pred == wl_gt)

        # ── Stage 4: Design ───────────────────────────────────────────────
        # Use the loss from the pipeline (cascaded) or GT (oracle)
        pipeline_loss = (loss_pred if (self.mode == "cascaded" and loss_pred is not None)
                         else loss_gt)

        s4 = self._run_design(
            data, s1["context"], s2["context"], s3["context"],
            pipeline_loss
        )

        # Evaluate design direction using physics-aware GT
        from .inverse_design import DesignDirectionTask
        selected_param = s4["selected_param"]
        direction_pred = s4["direction"]

        # Get GT direction for the predicted parameter
        if selected_param and direction_pred != "unknown":
            gt_direction = DesignDirectionTask._physics_direction(
                family_gt, selected_param, loss_gt)
            direction_correct = (direction_pred == gt_direction)
        else:
            direction_correct = False

        # ── Pipeline-level metrics ────────────────────────────────────────
        # End-to-end success: perception correct AND direction correct
        e2e_success = family_correct and direction_correct

        # Consistency: does the design parameter match what's expected for
        # the predicted family (not just the GT family)?
        if family_pred and selected_param and direction_pred != "unknown":
            pred_gt_dir = DesignDirectionTask._physics_direction(
                family_pred, selected_param, pipeline_loss)
            consistency = (direction_pred == pred_gt_dir)
        else:
            consistency = False

        return {
            "sample_id": sample_id,
            "mode": self.mode,
            # Stage 1
            "s1_family_pred": family_pred,
            "s1_family_gt": family_gt,
            "s1_family_correct": family_correct,
            "s1_rings_correct": rings_correct,
            # Stage 2
            "s2_loss_pred": loss_pred,
            "s2_loss_gt": loss_gt,
            "s2_log_mae": log_mae,
            # Stage 3
            "s3_wl_correct": wl_correct,
            "s3_mechanism": s3.get("diagnosed_mechanism", ""),
            # Stage 4
            "s4_param": selected_param,
            "s4_direction": direction_pred,
            "s4_direction_correct": direction_correct,
            # Pipeline
            "e2e_success": e2e_success,
            "consistency": consistency,
        }

    # ── Metrics aggregation ───────────────────────────────────────────────

    def compute_metrics(self) -> Dict[str, float]:
        valid = [r for r in self.results if r is not None]
        if not valid:
            return {}

        n = len(valid)

        # Per-stage accuracy
        s1_family_acc = sum(r["s1_family_correct"] for r in valid) / n
        s1_rings_acc = sum(r["s1_rings_correct"] for r in valid) / n
        s3_wl_acc = sum(r["s3_wl_correct"] for r in valid) / n
        s4_dir_acc = sum(r["s4_direction_correct"] for r in valid) / n

        # Stage 2: loss log-MAE (skip None)
        log_maes = [r["s2_log_mae"] for r in valid if r["s2_log_mae"] is not None]
        s2_log_mae = float(np.mean(log_maes)) if log_maes else float("inf")

        # Pipeline metrics
        e2e_success = sum(r["e2e_success"] for r in valid) / n
        consistency = sum(r["consistency"] for r in valid) / n

        # Error cascade: P(design wrong | perception wrong)
        perc_wrong = [r for r in valid if not r["s1_family_correct"]]
        if perc_wrong:
            cascade_rate = 1.0 - sum(r["s4_direction_correct"] for r in perc_wrong) / len(perc_wrong)
        else:
            cascade_rate = 0.0

        # Error cascade: P(design wrong | perception correct)
        perc_right = [r for r in valid if r["s1_family_correct"]]
        if perc_right:
            design_given_correct_perc = sum(r["s4_direction_correct"] for r in perc_right) / len(perc_right)
        else:
            design_given_correct_perc = 0.0

        return {
            # Stage-wise
            "s1_family_accuracy": float(s1_family_acc),
            "s1_rings_accuracy": float(s1_rings_acc),
            "s2_loss_log_mae": float(s2_log_mae),
            "s3_wavelength_accuracy": float(s3_wl_acc),
            "s4_direction_accuracy": float(s4_dir_acc),
            # Pipeline
            "e2e_success_rate": float(e2e_success),
            "pipeline_consistency": float(consistency),
            "error_cascade_rate": float(cascade_rate),
            "design_acc_given_correct_perception": float(design_given_correct_perc),
            # Meta
            "num_samples": n,
            "mode": self.mode,
        }
