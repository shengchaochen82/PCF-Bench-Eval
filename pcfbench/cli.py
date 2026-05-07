"""``pcfbench`` command-line interface.

Subcommands::

    fetch      Materialize a HF release into the per-sample eval layout.
    eval       Run a configured evaluation.
    tasks      List registered task names (with optional filter).
    init       Scaffold a new YAML config from a template.
    validate   Type-check a config without running.
    score      Re-render a previously written run.json into CSV + Markdown.
    compare    Diff two run.json files and print a leaderboard delta.
    info       Print environment diagnostics (provider keys, workers, ...).
    version    Print pcfbench version.

Run ``pcfbench <subcommand> --help`` for per-command flags.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pcfbench import __version__
from pcfbench.config import EvalConfig, load_config


CONSOLE = Console()
REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = REPO_ROOT / "examples"


# ── version ───────────────────────────────────────────────────────────────

def _cmd_version(_: argparse.Namespace) -> int:
    CONSOLE.print(f"[bold cyan]pcfbench[/bold cyan] [dim]v[/dim]{__version__}")
    return 0


# ── tasks ─────────────────────────────────────────────────────────────────

def _cmd_tasks(args: argparse.Namespace) -> int:
    from pcfbench.runner import _import_legacy_tasks
    names = sorted(_import_legacy_tasks().keys())
    rows = [(n, _task_category(n)) for n in names]
    if args.filter:
        f = args.filter.lower()
        rows = [(n, c) for n, c in rows if f in n.lower() or f in c.lower()]
    table = Table(title=f"Registered Tasks ({len(rows)})",
                  title_style="bold")
    table.add_column("name", style="cyan")
    table.add_column("category", style="dim")
    for n, c in rows:
        table.add_row(n, c)
    CONSOLE.print(table)
    return 0


def _task_category(name: str) -> str:
    if name.startswith(("family_", "parameter_prediction", "structure_counting",
                        "topology_")):
        return "geometry perception"
    if name.startswith(("loss_prediction", "application_classification",
                        "mode_classification", "mechanism_reasoning")):
        return "physics understanding"
    if name in {"geometry_description", "conversation_qa", "report_generation"}:
        return "text generation"
    if name in {"simulation_code_generation_complete",
                "parameter_modification_complete", "code_refactoring"}:
        return "code generation"
    if name.startswith(("wavelength_comparison", "mechanism_diagnosis",
                        "evidence_explanation")):
        return "multimodal reasoning"
    if name.startswith(("parameter_selection", "design_direction",
                        "closed_loop")):
        return "inverse design"
    if name.startswith("mas_"):
        return "multi-agent"
    return "other"


# ── init ──────────────────────────────────────────────────────────────────

def _cmd_init(args: argparse.Namespace) -> int:
    src = TEMPLATES / f"{args.template}.yaml"
    if not src.is_file():
        available = sorted(p.stem for p in TEMPLATES.glob("*.yaml"))
        CONSOLE.print(f"[red]template not found: {args.template}[/red]")
        CONSOLE.print(f"available templates: {', '.join(available)}")
        return 2
    dest = Path(args.path)
    # If args.path lacks a config extension, treat it as a directory.
    if dest.suffix.lower() not in {".yaml", ".yml", ".json"}:
        dest = dest / f"{args.name}.yaml"
    if dest.exists() and not args.force:
        CONSOLE.print(f"[red]refusing to overwrite {dest}[/red]; pass --force.")
        return 2
    text = src.read_text(encoding="utf-8")
    text = text.replace("PLACEHOLDER", args.name)
    if args.dataset:
        text = text.replace("PCF-Bench/PCF-Bench", args.dataset)
    text = text.replace("run_name: " + src.stem,
                        f"run_name: {args.name}", 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    CONSOLE.print(Panel(
        f"[green]✓[/green] wrote [cyan]{dest}[/cyan]\n\n"
        f"next: [bold]pcfbench validate {dest}[/bold] && "
        f"[bold]pcfbench eval --config {dest}[/bold]",
        title="config scaffolded", border_style="green",
    ))
    return 0


# ── validate ──────────────────────────────────────────────────────────────

def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except Exception as e:  # noqa: BLE001
        CONSOLE.print(Panel(f"[red]{type(e).__name__}: {e}[/red]",
                            title="validation failed", border_style="red"))
        return 1
    table = Table(title=f"Config: {Path(args.config).name}", title_style="bold")
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("run_name", cfg.run_name)
    table.add_row("provider", f"{cfg.provider.name}  (timeout={cfg.provider.timeout_s}s, retries={cfg.provider.max_retries})")
    table.add_row("model", cfg.model)
    table.add_row("dataset", cfg.dataset)
    table.add_row("cache_dir",
                  str(cfg.cache_dir) if cfg.cache_dir else "(auto)")
    table.add_row("output_dir", str(cfg.output_dir))
    table.add_row("tasks",
                  "all" if cfg.tasks == "all" else f"{len(cfg.tasks)} listed")
    table.add_row("max_samples", str(cfg.max_samples or "(no cap)"))
    table.add_row("subset_manifest", str(cfg.subset_manifest or "(none)"))
    table.add_row("num_workers", str(cfg.runner.num_workers))
    table.add_row("rpm_limit", str(cfg.runner.rpm_limit))
    table.add_row("temperature", f"{cfg.runner.temperature}")
    table.add_row("api_keys_env",
                  ", ".join(cfg.api_keys_env) if cfg.api_keys_env else "(provider default)")
    CONSOLE.print(table)
    CONSOLE.print("[green]✓ config is valid[/green]")
    return 0


# ── eval ──────────────────────────────────────────────────────────────────

def _cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.override_model:
        cfg.model = args.override_model
    if args.override_dataset:
        cfg.dataset = args.override_dataset
    if args.override_max_samples is not None:
        cfg.max_samples = args.override_max_samples
    if args.dry_run:
        return _cmd_validate(argparse.Namespace(config=args.config))

    CONSOLE.print(Panel(
        f"[bold]{cfg.run_name}[/bold]\n"
        f"provider: [cyan]{cfg.provider.name}[/cyan]   "
        f"model: [magenta]{cfg.model}[/magenta]\n"
        f"dataset: [dim]{cfg.dataset}[/dim]\n"
        f"output: [dim]{cfg.output_dir}[/dim]",
        title="pcfbench eval", border_style="cyan",
    ))

    from pcfbench.runner import run_eval
    run_eval(cfg)
    return 0


# ── score ─────────────────────────────────────────────────────────────────

def _cmd_score(args: argparse.Namespace) -> int:
    from dataclasses import fields
    from pcfbench.reporting import RunResult, write_outputs
    p = Path(args.results)
    if not p.is_file():
        CONSOLE.print(f"[red]not a file: {p}[/red]")
        return 2
    raw = json.loads(p.read_text())
    valid_keys = {f.name for f in fields(RunResult)}
    rr = RunResult(**{k: v for k, v in raw.items() if k in valid_keys})
    out_dir = Path(args.output_dir) if args.output_dir else p.parent
    write_outputs(rr, out_dir)
    return 0


# ── compare ───────────────────────────────────────────────────────────────

def _cmd_compare(args: argparse.Namespace) -> int:
    a = _load_run(args.left)
    b = _load_run(args.right)
    tasks_a = {t["task"]: t for t in a["tasks"]}
    tasks_b = {t["task"]: t for t in b["tasks"]}
    common = sorted(set(tasks_a.keys()) & set(tasks_b.keys()))

    table = Table(title=f"{Path(args.left).stem}  →  {Path(args.right).stem}",
                  title_style="bold")
    table.add_column("task", style="cyan")
    table.add_column("metric", style="dim")
    table.add_column(Path(args.left).stem, justify="right")
    table.add_column(Path(args.right).stem, justify="right")
    table.add_column("Δ", justify="right")

    for t in common:
        ma = tasks_a[t].get("metrics") or {}
        mb = tasks_b[t].get("metrics") or {}
        for k in sorted(set(ma) & set(mb)):
            va, vb = ma[k], mb[k]
            if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
                continue
            delta = vb - va
            arrow = "[green]↑[/green]" if delta > 0 else "[red]↓[/red]" if delta < 0 else "[dim]·[/dim]"
            table.add_row(t, k, _f(va), _f(vb), f"{arrow} {_f(delta, signed=True)}")
    CONSOLE.print(table)
    only_a = sorted(set(tasks_a) - set(tasks_b))
    only_b = sorted(set(tasks_b) - set(tasks_a))
    if only_a:
        CONSOLE.print(f"[dim]only in left: {', '.join(only_a)}[/dim]")
    if only_b:
        CONSOLE.print(f"[dim]only in right: {', '.join(only_b)}[/dim]")
    return 0


def _load_run(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _f(v: Any, signed: bool = False) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if abs(v) >= 1e3 or (abs(v) < 1e-3 and v != 0):
            s = f"{v:+.3e}" if signed else f"{v:.3e}"
        else:
            s = f"{v:+.4f}" if signed else f"{v:.4f}"
        return s
    return str(v)


# ── info ──────────────────────────────────────────────────────────────────

def _cmd_info(_: argparse.Namespace) -> int:
    # Best-effort .env loading from common locations (no override).
    try:
        from dotenv import load_dotenv
        for cand in (REPO_ROOT / ".env", Path.cwd() / ".env"):
            if cand.is_file():
                load_dotenv(cand, override=False)
    except ImportError:
        pass

    table = Table(title="pcfbench environment diagnostic", title_style="bold")
    table.add_column("provider", style="cyan")
    table.add_column("key env vars detected", justify="right")
    for prov in ("yunwu", "siliconflow", "zhipuai", "openrouter", "openai"):
        try:
            from tools.env_utils import get_api_keys
            keys = list(get_api_keys(prov) or [])
        except Exception:
            keys = []
        cell = (f"[green]{len(keys)}[/green]" if keys
                else "[red]0[/red]")
        table.add_row(prov, cell)
    CONSOLE.print(table)
    other = Table(title="runtime", title_style="bold")
    other.add_column("setting", style="cyan")
    other.add_column("value")
    other.add_row("repo root", str(REPO_ROOT))
    other.add_row("templates dir",
                  str(TEMPLATES) if TEMPLATES.is_dir() else "(missing)")
    other.add_row("python", sys.version.split()[0])
    try:
        import openai
        other.add_row("openai", openai.__version__)
    except ImportError:
        other.add_row("openai", "[red]not installed[/red]")
    try:
        import pydantic
        other.add_row("pydantic", pydantic.__version__)
    except ImportError:
        other.add_row("pydantic", "[red]not installed[/red]")
    CONSOLE.print(other)
    return 0


# ── parser construction ───────────────────────────────────────────────────

EPILOG = textwrap.dedent("""\
    Examples
    --------
      pcfbench init my_run --dataset PCF-Bench/PCF-Bench
      pcfbench validate my_run.yaml
      pcfbench eval --config my_run.yaml
      pcfbench eval --config my_run.yaml --override-model gpt-4o-mini
      pcfbench fetch --shards-glob '/data/shards/full-*.parquet' --out-dir ./PCF-Bench
      pcfbench mas-eval --method dylan --data_dir ./cache --max_samples 100
      pcfbench tasks --filter physics
      pcfbench score results/my_run_2026...Z.json
      pcfbench compare results/run_a.json results/run_b.json
      pcfbench info
""")


# ── mas-eval ──────────────────────────────────────────────────────────────

def _cmd_mas_eval(args: argparse.Namespace) -> int:
    """Run a MASLab baseline + score it. Wraps the standalone scripts in
    ``scripts/mas/`` so users have a single entrypoint."""
    import subprocess
    repo_root = REPO_ROOT
    runner = repo_root / "scripts" / "mas" / "run_mas_benchmark.py"
    scorer = repo_root / "scripts" / "mas" / "score_maslab_results.py"

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(runner),
        "--method", args.method,
        "--model", args.model,
        "--maslab_config", args.maslab_config,
        "--data_dir", str(Path(args.data_dir).expanduser()),
        "--num_workers", str(args.num_workers),
        "--output_dir", str(out_dir),
    ]
    if args.max_samples is not None:
        cmd += ["--max_samples", str(args.max_samples)]
    if args.method_config:
        cmd += ["--method_config", args.method_config]
    if args.subset_manifest:
        cmd += ["--subset_manifest", str(Path(args.subset_manifest).expanduser())]

    CONSOLE.print(Panel(
        f"method=[cyan]{args.method}[/cyan]  model=[magenta]{args.model}[/magenta]\n"
        f"data=[dim]{args.data_dir}[/dim]\nout=[dim]{out_dir}[/dim]",
        title="pcfbench mas-eval", border_style="cyan",
    ))

    try:
        rc = subprocess.run(cmd, check=False).returncode
    except FileNotFoundError:
        CONSOLE.print(f"[red]runner script not found: {runner}[/red]")
        return 2
    if rc != 0:
        CONSOLE.print(f"[red]runner exited with code {rc}[/red]")
        return rc

    if args.no_score:
        return 0

    # Find the JSON the runner just produced (most-recent under out_dir)
    candidates = sorted(out_dir.glob(f"MASLab_{args.method}_*_{args.model.split('/')[-1]}_*.json"))
    if not candidates:
        candidates = sorted(out_dir.glob(f"MASLab_{args.method}_*.json"))
    if not candidates:
        CONSOLE.print("[yellow]could not locate runner output for scoring[/yellow]")
        return 0
    latest = candidates[-1]
    CONSOLE.print(f"[dim]scoring {latest.name} ...[/dim]")
    score_rc = subprocess.run(
        [sys.executable, str(scorer), str(latest)], check=False,
    ).returncode
    return score_rc


# ── fetch ─────────────────────────────────────────────────────────────────

def _cmd_fetch(args: argparse.Namespace) -> int:
    from pcfbench.fetch import materialize
    out_dir = Path(args.out_dir).expanduser().resolve()
    n = materialize(
        out_dir=out_dir,
        repo_id=args.repo_id,
        shards_glob=args.shards_glob,
        max_samples=args.max_samples,
        streaming=args.streaming,
    )
    CONSOLE.print(
        f"[green]✓[/green] materialized [bold]{n}[/bold] samples → "
        f"[cyan]{out_dir}[/cyan]"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcfbench",
        description="PCF-Bench evaluation harness: configuration-driven, "
                    "multi-key, leaderboard-ready.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    p = sub.add_parser("version", help="Print pcfbench version.")
    p.set_defaults(fn=_cmd_version)

    p = sub.add_parser(
        "fetch",
        help="Materialize a PCF-Bench HF release into per-sample directories.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo-id",
                     help="Hub repo id, e.g. PCF-Bench/PCF-Bench.")
    src.add_argument("--shards-glob",
                     help="Local glob, e.g. /path/PCF-Bench-HF/data/full-*.parquet.")
    p.add_argument("--out-dir", required=True,
                   help="Destination data_dir for the eval pipeline.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Stop after N samples (default: all).")
    p.add_argument("--streaming", action="store_true",
                   help="Use HF streaming mode (no local cache, lower memory).")
    p.set_defaults(fn=_cmd_fetch)

    p = sub.add_parser("tasks", help="List registered task names.")
    p.add_argument("--filter", help="Substring filter on task name.")
    p.set_defaults(fn=_cmd_tasks)

    p = sub.add_parser("init", help="Scaffold a new YAML config.")
    p.add_argument("name",
                   help="run_name; used in filename if path is a directory.")
    p.add_argument("--template", default="single_model",
                   help="Template stem in examples/ (default: single_model).")
    p.add_argument("--path", default=".",
                   help="Output file or directory (default: cwd).")
    p.add_argument("--dataset", default=None,
                   help="Override the template's dataset (HF repo or parquet glob).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing file.")
    p.set_defaults(fn=_cmd_init)

    p = sub.add_parser("validate",
                       help="Type-check and pretty-print a config.")
    p.add_argument("config", help="Path to config YAML or JSON.")
    p.set_defaults(fn=_cmd_validate)

    p = sub.add_parser("eval", help="Run a configured evaluation.")
    p.add_argument("--config", required=True,
                   help="Path to config YAML or JSON (see examples/).")
    p.add_argument("--override-model", default=None,
                   help="Replace cfg.model (e.g. for sweeping).")
    p.add_argument("--override-dataset", default=None,
                   help="Replace cfg.dataset (HF repo or parquet glob).")
    p.add_argument("--override-max-samples", type=int, default=None,
                   help="Replace cfg.max_samples.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate the config but do not call any model.")
    p.set_defaults(fn=_cmd_eval)

    p = sub.add_parser(
        "score", help="Re-render an existing run.json into CSV + Markdown.")
    p.add_argument("results", help="Path to a run.json from a previous run.")
    p.add_argument("--output-dir", default=None,
                   help="Destination directory (default: alongside input).")
    p.set_defaults(fn=_cmd_score)

    p = sub.add_parser(
        "compare", help="Diff two run.json files (leaderboard delta).")
    p.add_argument("left", help="Baseline run.json.")
    p.add_argument("right", help="Comparison run.json.")
    p.set_defaults(fn=_cmd_compare)

    p = sub.add_parser("info", help="Environment diagnostic.")
    p.set_defaults(fn=_cmd_info)

    p = sub.add_parser(
        "mas-eval",
        help="Run a MASLab multi-agent baseline (DyLAN, MacNet, AutoGen, ...) "
             "on PCFBench data and score it.",
    )
    p.add_argument("--method", required=True,
                   help="MAS method name (vanilla, cot, self_consistency, "
                        "llm_debate, mad, camel, dylan, mav, macnet, "
                        "agentverse, autogen, evomac, pcfbench_pipeline).")
    p.add_argument("--method_config", default=None,
                   help="Optional method-specific config name.")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="VLM backbone (default: gpt-4o-mini).")
    p.add_argument("--maslab_config", default="yunwu_config.json",
                   help="Filename in MAS/model_api_configs/.")
    p.add_argument("--data_dir", required=True,
                   help="Materialized PCF samples directory "
                        "(typically the cache_dir from 'pcfbench fetch' or "
                        "the auto-cache under <output_dir>/.cache/<slug>).")
    p.add_argument("--subset_manifest", default=None,
                   help="Optional JSON listing the sample IDs to restrict to.")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output_dir", default="results/mas")
    p.add_argument("--no-score", action="store_true",
                   help="Skip the post-run scoring step.")
    p.set_defaults(fn=_cmd_mas_eval)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        CONSOLE.print("[yellow]interrupted[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
