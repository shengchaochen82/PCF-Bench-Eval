"""
PCFBench Multi-Agent Pipeline for MASLab.

A 4-stage sequential pipeline for photonic crystal fiber inverse design:
    Stage 1 (Perception)    → Family, params, rings, topology
    Stage 2 (Understanding) → Loss, mode, mechanism
    Stage 3 (Reasoning)     → Wavelength comparison, mechanism diagnosis
    Stage 4 (Design)        → Parameter selection, direction, justification

Each stage receives images + accumulated textual context from prior stages.
Integrates with MASLab's base class for token tracking and API management.
"""

import os
import sys
import base64
import re

# Add parent path for MASLab imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from methods.mas_base.mas_base import MAS


def _encode_image(image_path: str) -> str:
    """Encode image to base64 data URL for vision API."""
    import mimetypes
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _parse_kv(text: str, key: str) -> str:
    """Extract value for 'Key: value' pattern."""
    m = re.search(rf'{key}\s*:\s*(.+)', text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_number(text: str):
    """Extract first number (including scientific notation)."""
    m = re.search(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)', text)
    try:
        return float(m.group(1)) if m else None
    except (ValueError, AttributeError):
        return None


class PCFBenchPipeline(MAS):
    """4-stage PCFBench pipeline integrated with MASLab."""

    def __init__(self, general_config, method_config_name=None):
        method_config_name = method_config_name or "config_main"
        super().__init__(general_config, method_config_name)

        self.stages = self.method_config.get("stages", {})
        self.pipeline_mode = self.method_config.get("pipeline_mode", "cascaded")

    def _call_vision(self, system_prompt: str, user_text: str,
                     image_paths: list, temperature: float = 0.1,
                     max_tokens: int = 256) -> str:
        """Call VLM with images via vision-compatible message format."""
        content = [{"type": "text", "text": user_text}]
        for img_path in image_paths:
            if os.path.isfile(img_path):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": _encode_image(img_path)}
                })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        old_temp = self.model_temperature
        old_max = self.model_max_tokens
        self.model_temperature = temperature
        self.model_max_tokens = max_tokens
        try:
            response = self.call_llm(messages=messages)
        finally:
            self.model_temperature = old_temp
            self.model_max_tokens = old_max
        return response

    # ── Stage implementations ─────────────────────────────────────────────

    def _stage_perception(self, struct_path: str) -> dict:
        """Stage 1: Identify family, estimate parameters, count rings."""
        cfg = self.stages.get("perception", {})
        prompt = (
            "Analyze this photonic crystal fiber cross-section image.\n\n"
            "Provide:\n"
            "1. Family type (e.g., Solid-Core Hexagonal, Anti-Resonant ARF, "
            "Hollow-Core PBG, Kagome, etc.)\n"
            "2. Estimated pitch in micrometers\n"
            "3. Number of cladding rings\n"
            "4. Lattice topology (hexagonal, square, octagonal, kagome, other)\n\n"
            "Respond in this exact format:\n"
            "Family: <name>\nPitch: <value> um\nRings: <integer>\nTopology: <type>"
        )
        response = self._call_vision(
            cfg.get("system_prompt", "You are a photonic fiber expert."),
            prompt, [struct_path],
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 256)
        )

        family = _parse_kv(response, "Family")
        pitch = _parse_number(_parse_kv(response, "Pitch"))
        rings = _parse_number(_parse_kv(response, "Rings"))
        topology = _parse_kv(response, "Topology")

        context = (f"Family: {family}. Pitch: {pitch} um. "
                   f"Rings: {int(rings) if rings else 'unknown'}. "
                   f"Topology: {topology}.")

        return {
            "family": family, "pitch": pitch, "rings": rings,
            "topology": topology, "context": context,
            "raw_response": response[:500]
        }

    def _stage_understanding(self, field_path: str, perception_ctx: str) -> dict:
        """Stage 2: Predict loss, classify mode, identify mechanism."""
        cfg = self.stages.get("understanding", {})
        prompt = (
            f"Prior analysis: {perception_ctx}\n\n"
            "Now analyze this mode-field image at 1550 nm.\n\n"
            "Provide:\n"
            "1. Estimated confinement loss in dB/km (scientific notation)\n"
            "2. Mode type: single-mode, multi-mode, or leaky\n"
            "3. Primary guidance mechanism\n\n"
            "Respond in this exact format:\n"
            "Loss: <value> dB/km\nMode: <type>\nMechanism: <name>"
        )
        response = self._call_vision(
            cfg.get("system_prompt", "You are a photonic physics expert."),
            prompt, [field_path],
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 256)
        )

        loss_val = _parse_number(_parse_kv(response, "Loss"))
        mode = _parse_kv(response, "Mode")
        mechanism = _parse_kv(response, "Mechanism")

        context = (f"Loss (1550 nm): {loss_val} dB/km. "
                   f"Mode: {mode}. Mechanism: {mechanism}.")

        return {
            "loss_pred": loss_val, "mode_pred": mode,
            "mechanism_pred": mechanism, "context": context,
            "raw_response": response[:500]
        }

    def _stage_reasoning(self, struct_path: str, field_path: str,
                         perception_ctx: str, understanding_ctx: str) -> dict:
        """Stage 3: Wavelength comparison, mechanism diagnosis."""
        cfg = self.stages.get("reasoning", {})
        prompt = (
            f"Prior analysis:\n{perception_ctx}\n{understanding_ctx}\n\n"
            "Examine both the structure and field images together.\n\n"
            "Provide:\n"
            "1. Which wavelength (1310 or 1550 nm) has higher loss?\n"
            "2. Dominant loss mechanism\n"
            "3. Brief evidence-based explanation (2-3 sentences)\n\n"
            "Respond in this format:\n"
            "Higher loss wavelength: <1310 or 1550>\n"
            "Dominant mechanism: <name>\n"
            "Explanation: <text>"
        )
        response = self._call_vision(
            cfg.get("system_prompt", "You are a photonic diagnostician."),
            prompt, [struct_path, field_path],
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 512)
        )

        higher_wl = _parse_kv(response, "Higher loss wavelength")
        mechanism = _parse_kv(response, "Dominant mechanism")
        explanation = _parse_kv(response, "Explanation")

        context = (f"Higher loss at: {higher_wl} nm. "
                   f"Dominant mechanism: {mechanism}. "
                   f"Explanation: {explanation[:150]}.")

        return {
            "higher_loss_wl": higher_wl, "diagnosed_mechanism": mechanism,
            "explanation": explanation[:300], "context": context,
            "raw_response": response[:500]
        }

    def _stage_design(self, struct_path: str, field_path: str,
                      perception_ctx: str, understanding_ctx: str,
                      reasoning_ctx: str, loss_value: float) -> dict:
        """Stage 4: Propose design modification."""
        cfg = self.stages.get("design", {})
        loss_str = f"{loss_value:.2e}" if loss_value else "unknown"
        prompt = (
            f"Full analysis pipeline:\n{perception_ctx}\n"
            f"{understanding_ctx}\n{reasoning_ctx}\n\n"
            f"Current confinement loss: {loss_str} dB/km.\n\n"
            "Propose a design improvement:\n"
            "1. Which parameter to adjust? (pitch, hole diameter, "
            "strut thickness, core diameter, ring radius, gap width, "
            "or number of rings)\n"
            "2. Increase or decrease?\n"
            "3. Brief physics justification\n\n"
            "Respond in this format:\n"
            "Parameter: <name>\nDirection: <increase or decrease>\n"
            "Reasoning: <text>"
        )
        response = self._call_vision(
            cfg.get("system_prompt", "You are a photonic design engineer."),
            prompt, [struct_path, field_path],
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens", 256)
        )

        param = _parse_kv(response, "Parameter").lower().replace(" ", "_")
        direction = _parse_kv(response, "Direction").lower()
        reasoning = _parse_kv(response, "Reasoning")

        if "increase" in direction:
            direction = "increase"
        elif "decrease" in direction or "reduce" in direction:
            direction = "decrease"
        else:
            direction = "unknown"

        return {
            "selected_param": param, "direction": direction,
            "design_reasoning": reasoning[:300],
            "raw_response": response[:500]
        }

    # ── Main inference ────────────────────────────────────────────────────

    def inference(self, sample: dict) -> dict:
        """
        Run the full 4-stage pipeline.

        Expected sample keys:
            query:        str (the overall task description)
            struct_image: str (path to structure cross-section PNG)
            field_image:  str (path to mode-field at 1550 nm PNG)
            loss_gt:      float (ground-truth loss, used in oracle mode)

        Returns:
            {"response": str, "pipeline": dict with all stage outputs}
        """
        struct_path = sample.get("struct_image", "")
        field_path = sample.get("field_image", "")

        # Stage 1: Perception
        s1 = self._stage_perception(struct_path)

        # Stage 2: Understanding
        s2 = self._stage_understanding(field_path, s1["context"])

        # Stage 3: Reasoning
        s3 = self._stage_reasoning(
            struct_path, field_path, s1["context"], s2["context"])

        # Stage 4: Design
        pipeline_loss = (s2.get("loss_pred") if s2.get("loss_pred") is not None
                         else sample.get("loss_gt", 1.0))
        s4 = self._stage_design(
            struct_path, field_path,
            s1["context"], s2["context"], s3["context"],
            pipeline_loss)

        # Compose final response
        response = (
            f"[Perception] {s1['context']}\n"
            f"[Understanding] {s2['context']}\n"
            f"[Reasoning] {s3['context']}\n"
            f"[Design] Parameter: {s4['selected_param']}, "
            f"Direction: {s4['direction']}. {s4['design_reasoning']}"
        )

        return {
            "response": response,
            "pipeline": {
                "stage1_perception": s1,
                "stage2_understanding": s2,
                "stage3_reasoning": s3,
                "stage4_design": s4,
            }
        }
