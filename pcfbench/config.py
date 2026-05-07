"""Typed configuration schema for the PCF-Bench evaluation harness.

The dataset is always specified as a parquet source — either a HuggingFace
repo id or a local glob over parquet shards. The runner materializes it into
a per-sample cache directory on first use; subsequent runs reuse the cache.

Example YAML::

    run_name: gpt-4o-mini-baseline
    dataset: PCF-Bench/PCF-Bench
    output_dir: results/
    provider: yunwu
    model: gpt-4o-mini
    tasks: [family_classification, loss_prediction_1550]
    max_samples: 200
    runner:
      num_workers: 16
      rpm_limit: 600
    subset_manifest: examples/subset_v1.json
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


PROVIDERS = ("yunwu", "siliconflow", "zhipuai", "openrouter", "openai")


class ProviderConfig(BaseModel):
    """LLM provider settings."""
    model_config = ConfigDict(extra="forbid")

    name: Literal[PROVIDERS] = "yunwu"
    base_url: Optional[str] = Field(
        default=None,
        description="Override the provider default (only set for self-hosted endpoints).",
    )
    timeout_s: int = Field(default=120, ge=10, le=900)
    max_retries: int = Field(default=3, ge=0, le=10)


class RunnerConfig(BaseModel):
    """Concurrency and rate-limit knobs."""
    model_config = ConfigDict(extra="forbid")

    num_workers: int = Field(default=10, ge=1, le=512)
    rpm_limit: int = Field(default=300, ge=1, description="Aggregate RPM ceiling")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=64, le=32768)


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""
    model_config = ConfigDict(extra="forbid")

    # Identity
    run_name: str = Field(
        default="pcfbench_run",
        description="Identifier used in output filenames.",
    )

    # Data — parquet input only
    dataset: str = Field(
        description=(
            "Parquet source: either a HuggingFace repo id "
            "(e.g. 'PCF-Bench/PCF-Bench') or a local glob over parquet shards "
            "(e.g. '/data/PCF-Bench-HF/data/full-*.parquet')."
        ),
    )
    cache_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Where to materialize the parquet shards into per-sample directories. "
            "Defaults to '<output_dir>/.cache/<dataset_slug>' if unset."
        ),
    )
    output_dir: Path = Field(default=Path("results"))
    subset_manifest: Optional[Path] = Field(
        default=None,
        description="Stratified-subset manifest JSON. Overrides max_samples per task.",
    )
    max_samples: Optional[int] = Field(
        default=None,
        description="Per-task sample cap. Ignored if subset_manifest is set.",
    )

    # Model
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    model: str = Field(description="Model name as recognised by provider.")
    api_keys_env: List[str] = Field(
        default_factory=list,
        description=(
            "Environment-variable names holding API keys. "
            "Empty means use the provider's default lookup."
        ),
    )

    # Tasks
    tasks: List[str] | Literal["all"] = Field(
        default="all",
        description=(
            "List of task names or the literal 'all'. "
            "Run `pcfbench tasks` to see available names."
        ),
    )

    # Concurrency / sampling
    runner: RunnerConfig = Field(default_factory=RunnerConfig)

    @field_validator("output_dir", "cache_dir")
    @classmethod
    def _expand(cls, v: Optional[Path]) -> Optional[Path]:
        return None if v is None else Path(v).expanduser()


def load_config(path: str | Path) -> EvalConfig:
    """Parse YAML or JSON into a validated :class:`EvalConfig`."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        raw: Dict[str, Any] = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        import json
        raw = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")
    return EvalConfig.model_validate(raw)
