"""PCF-Bench evaluation suite.

A unified, configuration-driven evaluation harness for vision-language model
performance on photonic-crystal-fiber design tasks.  Wraps the legacy ``eval/``
task implementations behind a single CLI and a typed configuration schema.

Quickstart::

    pcfbench eval --config examples/single_model.yaml

See ``README.md`` for the full architecture and ``examples/`` for sample
configurations.
"""
from __future__ import annotations

__version__ = "0.1.0"

from pcfbench.config import EvalConfig, load_config

__all__ = [
    "EvalConfig",
    "load_config",
    "__version__",
]
