"""Materialize a PCF-Bench HuggingFace release into the per-sample directory
layout that ``BaseTask.load_sample_data`` expects.

The released parquet shards embed every text and image field. The eval pipeline
in ``eval/tasks/base.py`` reads from per-sample directories::

    <data_dir>/PCF_<family>_<id>/
    ├── params.json
    ├── sim.json
    ├── annotation.json
    └── images/{struct.png, struct.svg, field_1310.png, field_1550.png}

This module bridges the two by streaming the dataset row-by-row and writing the
expected files. Either pass a Hub repo id (``--repo-id PCF-Bench/PCF-Bench``)
for the canonical release or a local parquet glob (``--shards-glob``) for an
offline copy.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Iterable, Optional


def _iter_dataset(
    repo_id: Optional[str],
    shards_glob: Optional[str],
    streaming: bool,
) -> Iterable[dict]:
    from datasets import load_dataset
    if repo_id and shards_glob:
        raise ValueError("Pass exactly one of --repo-id or --shards-glob.")
    if not repo_id and not shards_glob:
        raise ValueError("Provide --repo-id or --shards-glob.")

    if repo_id:
        ds = load_dataset(repo_id, split="full", streaming=streaming)
    else:
        ds = load_dataset("parquet", data_files=shards_glob,
                          split="train", streaming=streaming)
    return ds


def _write_sample(row: dict, out_root: Path) -> Path:
    pcf_id = row["pcf_id"]
    sample_dir = out_root / pcf_id
    img_dir = sample_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # The eval task implementations expect a nested layout where geometric
    # parameters live under params.params (legacy 1K-pool schema), with the
    # top level carrying only family + identifiers.
    geometry = json.loads(row["geometry"])
    params = {
        "family": row["family"],
        "params": geometry,
        "pcf_id": row["pcf_id"],
    }
    (sample_dir / "params.json").write_text(json.dumps(params, indent=2))

    physics = json.loads(row["physics"])
    sim = {
        "metrics": {k: physics[k] for k in
                    ("loss_1310", "loss_1550", "neff_1310", "neff_1550")
                    if k in physics},
        "intermediates": {
            "difficulty": row["difficulty"],
            "mode_count_proxy": physics.get("mode_count_proxy"),
        },
    }
    (sample_dir / "sim.json").write_text(json.dumps(sim, indent=2))

    # Eval tasks consume ShareGPT-style ``conversations`` lists at every
    # level, so package each level's human + gpt strings as paired turns.
    annotation = {
        "pcf_id": pcf_id,
        "level1": {
            "conversations": [
                {"from": "human", "value": row["level1_human"]},
                {"from": "gpt", "value": row["level1_gpt"]},
            ],
            "meta": {"family": row["family"]},
        },
        "level2": {
            "conversations": json.loads(row["level2"]),
            "meta": {"family": row["family"]},
        },
        "level3": {
            "conversations": [
                {"from": "human", "value": row["level3_human"]},
                {"from": "gpt", "value": row["level3_gpt"]},
            ],
            "meta": {"family": row["family"], "difficulty": row["difficulty"]},
        },
        "level4": {
            "conversations": [
                {"from": "human", "value": row["level4_human"]},
                {"from": "gpt", "value": row["level4_gpt"]},
            ],
            "code": row["level4_code"],
            "validation_errors": json.loads(row["level4_validation_errors"]),
        },
    }
    (sample_dir / "annotation.json").write_text(json.dumps(annotation, indent=2))

    for col, fname in [
        ("image_struct", "struct.png"),
        ("image_field_1310", "field_1310.png"),
        ("image_field_1550", "field_1550.png"),
    ]:
        img = row[col]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        (img_dir / fname).write_bytes(buf.getvalue())

    (img_dir / "struct.svg").write_text(row["image_struct_svg"])
    return sample_dir


def materialize(
    out_dir: Path,
    repo_id: Optional[str] = None,
    shards_glob: Optional[str] = None,
    max_samples: Optional[int] = None,
    streaming: bool = False,
    progress_every: int = 100,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = _iter_dataset(repo_id, shards_glob, streaming)

    if hasattr(ds, "__len__"):
        total = len(ds) if max_samples is None else min(max_samples, len(ds))
    else:
        total = max_samples  # may be None for full streaming

    written = 0
    for row in ds:
        if max_samples is not None and written >= max_samples:
            break
        _write_sample(row, out_dir)
        written += 1
        if written % progress_every == 0 or written == total:
            elapsed = time.time() - t0
            rate = written / elapsed if elapsed else 0
            tail = f"/{total}" if total else ""
            print(f"  [{written}{tail}] {rate:.1f} samples/s")
    return written
