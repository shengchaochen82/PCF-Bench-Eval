"""Stable run-result schema + serialisers (JSON, CSV leaderboard, Markdown).

The :class:`RunResult` JSON is the canonical artifact for cross-experiment
comparison and downstream leaderboard scripts.  CSV and Markdown derive from
it deterministically.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class RunResult:
    run_id: str
    config: Dict[str, Any]
    wall_time_s: float
    tasks: List[Dict[str, Any]] = field(default_factory=list)


def write_outputs(result: RunResult, output_dir: Path) -> None:
    """Write run.json (full), leaderboard.csv (per-task metric flat),
    and summary.md (markdown table)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{result.run_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, default=_json_default)

    csv_path = output_dir / f"{result.run_id}.csv"
    _write_leaderboard_csv(result, csv_path)

    md_path = output_dir / f"{result.run_id}.md"
    _write_markdown_summary(result, md_path)

    print(f"[out] wrote {json_path.name}, {csv_path.name}, {md_path.name}")


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def _write_leaderboard_csv(result: RunResult, path: Path) -> None:
    """One row per task; columns are union of metric keys."""
    metric_keys: List[str] = []
    seen = set()
    for t in result.tasks:
        for k in (t.get("metrics") or {}).keys():
            if k not in seen:
                seen.add(k)
                metric_keys.append(k)
    fieldnames = ["run_id", "task", "n_samples", "wall_time_s", *metric_keys]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in result.tasks:
            row = {
                "run_id": result.run_id,
                "task": t.get("task"),
                "n_samples": t.get("n_samples"),
                "wall_time_s": round(t.get("wall_time_s", 0.0), 2),
            }
            for k in metric_keys:
                v = (t.get("metrics") or {}).get(k)
                row[k] = _fmt_metric(v)
            w.writerow(row)


def _write_markdown_summary(result: RunResult, path: Path) -> None:
    cfg = result.config
    lines = [
        f"# {result.run_id}",
        "",
        f"- model: `{cfg.get('model')}`",
        f"- provider: `{(cfg.get('provider') or {}).get('name')}`",
        f"- dataset: `{cfg.get('dataset')}`",
        f"- total wall time: **{result.wall_time_s:.1f} s**",
        "",
        "## Per-task metrics",
        "",
    ]
    for t in result.tasks:
        m = t.get("metrics") or {}
        bits = ", ".join(f"{k}={_fmt_metric(v)}" for k, v in m.items()
                         if not isinstance(v, (list, dict)))
        lines.append(
            f"- **{t['task']}**: N={t['n_samples']}, "
            f"wall={t['wall_time_s']:.1f}s — {bits}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fmt_metric(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if abs(v) >= 1e3 or (abs(v) < 1e-3 and v != 0):
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)
