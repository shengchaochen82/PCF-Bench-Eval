"""I/O helpers used by the legacy task implementations under ``eval/tasks/``."""
import json
from typing import Any, Dict


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
