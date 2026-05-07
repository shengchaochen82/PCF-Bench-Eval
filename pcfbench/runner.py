"""Single-model evaluation orchestrator.

Loads tasks from ``eval/tasks/``, runs each through a worker pool sharing a
:class:`ProviderClient`, and emits a :class:`pcfbench.reporting.RunResult`.
MAS baselines are dispatched separately via the ``pcfbench mas-eval`` CLI
subcommand (see :mod:`scripts.mas.run_mas_benchmark`).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from pcfbench.config import EvalConfig
from pcfbench.providers import ProviderClient
from pcfbench.reporting import RunResult, write_outputs


# ── Task registry ─────────────────────────────────────────────────────────

def _import_legacy_tasks():
    """Import every task class from ``eval/tasks`` and return name → factory.

    Some task classes are parameterised (LossPredictionTask by wavelength,
    WavelengthComparisonTask by variant, DesignDirectionTask by target
    param, MASPipelineTask by mode, ...).  We expose one registry entry per
    fully-bound task so configs can name them directly.

    Returns
    -------
    dict[str, callable]
        Name → factory ``f(data_dir, provider) -> BaseTask`` mapping.
    """
    repo_root = Path(__file__).resolve().parent.parent
    eval_dir = repo_root / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tasks import geometry_perception as gp
    from tasks import physics_understanding as pu
    from tasks import text_generation as tg
    from tasks import code_generation_complete as cgc
    from tasks import multimodal_reasoning as mr
    from tasks import inverse_design as idd
    from tasks import mas_pipeline as mp

    # Each entry is (canonical_name, factory(data_dir, provider) -> task).
    spec = [
        ("family_classification",
         lambda d, m: gp.FamilyClassificationTask(d, m)),
        ("parameter_prediction_pitch",
         lambda d, m: gp.ParameterPredictionTask(d, m, "pitch_um")),
        ("parameter_prediction_hole_d",
         lambda d, m: gp.ParameterPredictionTask(d, m, "hole_d_um")),
        ("structure_counting_rings",
         lambda d, m: gp.StructureCountingTask(d, m, "rings")),
        ("topology_recognition",
         lambda d, m: gp.TopologyRecognitionTask(d, m)),
        ("loss_prediction_1310",
         lambda d, m: pu.LossPredictionTask(d, m, 1310)),
        ("loss_prediction_1550",
         lambda d, m: pu.LossPredictionTask(d, m, 1550)),
        ("application_classification",
         lambda d, m: pu.ApplicationClassificationTask(d, m)),
        ("mode_classification",
         lambda d, m: pu.ModeClassificationTask(d, m)),
        ("mechanism_reasoning",
         lambda d, m: pu.MechanismReasoningTask(d, m)),
        ("geometry_description",
         lambda d, m: tg.GeometryDescriptionTask(d, m)),
        ("conversation_qa",
         lambda d, m: tg.ConversationQATask(d, m)),
        ("report_generation",
         lambda d, m: tg.ReportGenerationTask(d, m)),
        ("simulation_code_generation_complete",
         lambda d, m: cgc.SimulationCodeGenerationTask(d, m)),
        ("parameter_modification_complete",
         lambda d, m: cgc.ParameterModificationTask(d, m)),
        ("code_refactoring",
         lambda d, m: cgc.CodeRefactoringTask(d, m)),
        ("wavelength_comparison_v1",
         lambda d, m: mr.WavelengthComparisonTask(d, m, variant=1)),
        ("wavelength_comparison_v2",
         lambda d, m: mr.WavelengthComparisonTask(d, m, variant=2)),
        ("mechanism_diagnosis_v1",
         lambda d, m: mr.MechanismDiagnosisTask(d, m, variant=1)),
        ("mechanism_diagnosis_v2",
         lambda d, m: mr.MechanismDiagnosisTask(d, m, variant=2)),
        ("evidence_explanation_v1",
         lambda d, m: mr.EvidenceBasedExplanationTask(d, m, variant=1)),
        ("evidence_explanation_v2",
         lambda d, m: mr.EvidenceBasedExplanationTask(d, m, variant=2)),
        ("parameter_selection_v1",
         lambda d, m: idd.ParameterSelectionTask(d, m, variant=1)),
        ("design_direction_pitch",
         lambda d, m: idd.DesignDirectionTask(d, m, target_param="pitch_um", variant=1)),
        ("design_direction_hole_d",
         lambda d, m: idd.DesignDirectionTask(d, m, target_param="hole_diameter", variant=1)),
        ("closed_loop_improvement_pitch",
         lambda d, m: idd.ClosedLoopImprovementTask(d, m, target_param="pitch_um", variant=1)),
        ("mas_pipeline_cascaded",
         lambda d, m: mp.MASPipelineTask(d, m, mode="cascaded")),
        ("mas_pipeline_oracle",
         lambda d, m: mp.MASPipelineTask(d, m, mode="oracle")),
    ]
    return dict(spec)


def _slugify_dataset(dataset: str) -> str:
    """Stable, filesystem-safe slug for a parquet source."""
    import hashlib
    import re
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset).strip("_")[:60] or "dataset"
    digest = hashlib.sha1(dataset.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


def _resolve_cache_dir(config: EvalConfig) -> Path:
    if config.cache_dir is not None:
        return config.cache_dir
    return config.output_dir / ".cache" / _slugify_dataset(config.dataset)


def _ensure_materialized(config: EvalConfig) -> Path:
    """Ensure a per-sample directory exists for the configured dataset.

    Returns the directory path. If it already contains ``PCF_*`` subdirs
    we trust them and skip re-materialization.
    """
    from pcfbench.fetch import materialize

    cache_dir = _resolve_cache_dir(config).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in cache_dir.iterdir()
                if p.is_dir() and p.name.startswith("PCF_")]
    if existing:
        print(f"[data] cache hit: {len(existing)} samples in {cache_dir}")
        return cache_dir

    print(f"[data] materializing {config.dataset} → {cache_dir}")
    is_local = (
        any(ch in config.dataset for ch in "*?[") or
        config.dataset.endswith(".parquet") or
        "/" in config.dataset and Path(config.dataset).expanduser().exists()
    )
    n = materialize(
        out_dir=cache_dir,
        repo_id=None if is_local else config.dataset,
        shards_glob=config.dataset if is_local else None,
    )
    print(f"[data] materialized {n} samples")
    return cache_dir


_SAMPLE_SHUFFLE_SEED = 42  # deterministic across runs, breaks alphabetical bias


def _resolve_sample_ids(
    data_dir: Path,
    subset_manifest: Optional[Path],
    task_name: str,
    max_samples: Optional[int],
) -> List[Tuple[str, str]]:
    """Return (sample_dir_path, sample_id) pairs for one task.

    The single source of truth for sample selection: optional subset manifest
    keyed by task, otherwise every ``PCF_*`` directory under ``data_dir``.
    A small ``max_samples`` then yields a family-diverse subset rather than
    the alphabetically-first one (e.g. all ``all_solid_bandgap``).
    """
    if subset_manifest:
        with open(subset_manifest) as f:
            manifest = json.load(f)
        ids = manifest.get("tasks", {}).get(task_name, {}).get("sample_ids", [])
        if not ids:
            ids = manifest.get(task_name, [])
        out = [(str(data_dir / sid), sid) for sid in ids
               if (data_dir / sid).is_dir()]
    else:
        all_pairs = [(str(p), p.name)
                     for p in sorted(data_dir.iterdir())
                     if p.is_dir() and p.name.startswith("PCF_")]
        import random as _rand
        rng = _rand.Random(_SAMPLE_SHUFFLE_SEED)
        out = list(all_pairs)
        rng.shuffle(out)
    if max_samples and len(out) > max_samples:
        out = out[:max_samples]
    return out


# ── Main entry ────────────────────────────────────────────────────────────

def run_eval(config: EvalConfig) -> RunResult:
    """Run a full single-model evaluation per ``config``."""
    registry = _import_legacy_tasks()
    if config.tasks == "all":
        task_names = sorted(registry.keys())
    else:
        task_names = list(config.tasks)
        unknown = [t for t in task_names if t not in registry]
        if unknown:
            raise ValueError(f"Unknown tasks: {unknown}. "
                             f"Available: {sorted(registry.keys())}")

    provider = ProviderClient(config)
    print(f"[provider] {config.provider.name}/{config.model} "
          f"({provider.n_keys} keys)")
    print(f"[tasks] {len(task_names)}: {task_names}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = _ensure_materialized(config)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config.run_name}_{timestamp}"

    task_records: List[Dict[str, Any]] = []
    t0_total = time.time()

    for tname in task_names:
        factory = registry[tname]
        task = factory(str(data_dir), provider)
        samples = _resolve_sample_ids(
            data_dir, config.subset_manifest, tname, config.max_samples,
        )
        # Hand the runner-resolved IDs to the task body so the legacy
        # in-task discovery short-circuits to this list.
        task.sample_ids = [sid for _, sid in samples]
        if not samples:
            print(f"[skip] {tname}: no samples")
            continue
        print(f"\n[task] {tname}  N={len(samples)}")

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=config.runner.num_workers) as pool:
            futs = {
                pool.submit(_safe_evaluate, task, sd, sid): sid
                for sd, sid in samples
            }
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc=tname, unit="s"):
                _ = fut.result()  # already accumulated in task.results

        dt = time.time() - t0
        try:
            metrics = task.compute_metrics()
        except Exception as e:  # noqa: BLE001
            metrics = {"error": str(e)}
        task_records.append({
            "task": tname,
            "n_samples": len(samples),
            "wall_time_s": dt,
            "metrics": metrics,
            "results": task.results,
        })
        print(f"[task] {tname} done in {dt:.1f}s -> {metrics}")

    result = RunResult(
        run_id=run_id,
        config=config.model_dump(),
        wall_time_s=time.time() - t0_total,
        tasks=task_records,
    )
    write_outputs(result, output_dir)
    return result


def _safe_evaluate(task, sample_dir: str, sample_id: str) -> Optional[Dict[str, Any]]:
    """Wrap task.evaluate_sample with exception isolation."""
    try:
        rec = task.evaluate_sample(sample_dir, sample_id)
        if rec is not None:
            task.results.append(rec)
        return rec
    except Exception as e:  # noqa: BLE001
        task.results.append({
            "sample_id": sample_id,
            "error": f"{type(e).__name__}: {e}",
        })
        return None
