"""Lazy / fault-tolerant MASLab method registry.

Each method is imported in its own try block so an optional dependency
missing from one method (e.g. wikipediaapi for ChatDev under Python 3.9)
does not prevent other methods from loading. ``method2class`` is built
from the modules that imported successfully; missing entries raise a
clear error only when the user asks for a method we couldn't load.
"""

import warnings

method2class: dict = {}
_load_errors: dict = {}


def _safe_register(name, importer):
    try:
        cls = importer()
    except Exception as e:  # noqa: BLE001
        _load_errors[name] = repr(e)
        return
    method2class[name] = cls


# ── Single-agent baselines ─────────────────────────────────────────────────
_safe_register("vanilla", lambda: __import__("methods.mas_base",
                                             fromlist=["MAS"]).MAS)
_safe_register("cot", lambda: __import__("methods.cot",
                                         fromlist=["CoT"]).CoT)
_safe_register("self_consistency",
               lambda: __import__("methods.self_consistency",
                                  fromlist=["SelfConsistency"]).SelfConsistency)

# ── Debate / role-playing ──────────────────────────────────────────────────
_safe_register("llm_debate",
               lambda: __import__("methods.llm_debate",
                                  fromlist=["LLM_Debate_Main"]).LLM_Debate_Main)
_safe_register("mad", lambda: __import__("methods.mad",
                                         fromlist=["MAD_Main"]).MAD_Main)
_safe_register("camel", lambda: __import__("methods.camel",
                                           fromlist=["CAMEL_Main"]).CAMEL_Main)

# ── Verification / dynamic networks ────────────────────────────────────────
_safe_register("dylan", lambda: __import__("methods.dylan",
                                           fromlist=["DyLAN_Main"]).DyLAN_Main)
_safe_register("dylan_humaneval",
               lambda: __import__("methods.dylan",
                                  fromlist=["DyLAN_HumanEval"]).DyLAN_HumanEval)
_safe_register("dylan_math", lambda: __import__("methods.dylan",
                                                fromlist=["DyLAN_MATH"]).DyLAN_MATH)
_safe_register("dylan_mmlu", lambda: __import__("methods.dylan",
                                                fromlist=["DyLAN_MMLU"]).DyLAN_MMLU)
_safe_register("mav", lambda: __import__("methods.mav",
                                         fromlist=["MAV_Main"]).MAV_Main)
_safe_register("mav_gpqa", lambda: __import__("methods.mav",
                                              fromlist=["MAV_GPQA"]).MAV_GPQA)
_safe_register("mav_humaneval",
               lambda: __import__("methods.mav",
                                  fromlist=["MAV_HumanEval"]).MAV_HumanEval)
_safe_register("mav_math", lambda: __import__("methods.mav",
                                              fromlist=["MAV_MATH"]).MAV_MATH)
_safe_register("mav_mmlu", lambda: __import__("methods.mav",
                                              fromlist=["MAV_MMLU"]).MAV_MMLU)

# ── Graph / orchestration ──────────────────────────────────────────────────
_safe_register("macnet", lambda: __import__("methods.macnet",
                                            fromlist=["MacNet_Main"]).MacNet_Main)
_safe_register("macnet_srdd",
               lambda: __import__("methods.macnet",
                                  fromlist=["MacNet_SRDD"]).MacNet_SRDD)
_safe_register("agentverse",
               lambda: __import__("methods.agentverse",
                                  fromlist=["AgentVerse_Main"]).AgentVerse_Main)
_safe_register("agentverse_humaneval",
               lambda: __import__("methods.agentverse",
                                  fromlist=["AgentVerse_HumanEval"]).AgentVerse_HumanEval)
_safe_register("agentverse_mgsm",
               lambda: __import__("methods.agentverse",
                                  fromlist=["AgentVerse_MGSM"]).AgentVerse_MGSM)
_safe_register("autogen",
               lambda: __import__("methods.autogen",
                                  fromlist=["AutoGen_Main"]).AutoGen_Main)
_safe_register("evomac",
               lambda: __import__("methods.evomac",
                                  fromlist=["EvoMAC_Main"]).EvoMAC_Main)

# ── Optional methods (not in the paper's 13-baseline set) ──────────────────
_safe_register("chatdev_srdd",
               lambda: __import__("methods.chatdev",
                                  fromlist=["ChatDev_SRDD"]).ChatDev_SRDD)
_safe_register("mapcoder_humaneval",
               lambda: __import__("methods.mapcoder",
                                  fromlist=["MapCoder_HumanEval"]).MapCoder_HumanEval)
_safe_register("mapcoder_mbpp",
               lambda: __import__("methods.mapcoder",
                                  fromlist=["MapCoder_MBPP"]).MapCoder_MBPP)

# ── PCFBench native pipeline ───────────────────────────────────────────────
_safe_register("pcfbench_pipeline",
               lambda: __import__("methods.pcfbench_pipeline",
                                  fromlist=["PCFBenchPipeline"]).PCFBenchPipeline)


# Re-export MAS base directly for callers that import it explicitly.
try:
    from .mas_base import MAS  # noqa: F401
except Exception:  # pragma: no cover
    MAS = None  # type: ignore


def get_method_class(method_name, dataset_name=None):
    method_name = method_name.lower()

    if method_name in _load_errors and method_name not in method2class:
        raise RuntimeError(
            f"MAS method '{method_name}' failed to import: "
            f"{_load_errors[method_name]}"
        )

    matched = [n for n in method2class if method_name in n]
    if not matched:
        avail = sorted(method2class.keys())
        skipped = sorted(_load_errors.keys())
        msg = f"No MAS method matches '{method_name}'. Available: {avail}."
        if skipped:
            msg += f" (Skipped due to import errors: {skipped}.)"
        raise ValueError(msg)

    if dataset_name is not None:
        dataset_name = dataset_name.lower()
        ds_matched = [n for n in matched if n.split("_")[-1] in dataset_name]
        if ds_matched:
            method_name = ds_matched[0]
            if len(ds_matched) > 1:
                warnings.warn(
                    f"Multiple methods match {dataset_name}: {ds_matched}. "
                    f"Using {method_name}.",
                    stacklevel=2,
                )
            return method2class[method_name]

    return method2class[matched[0]]
