"""Small helpers for JSON reports and reproducible metadata."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars into JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def update_json(path: str | Path, section: str, payload: dict[str, Any]) -> Path:
    """Merge one section into an existing report without dropping other sections."""
    report = load_json(path)
    report[section] = json_safe(payload)
    report.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    return write_json(path, report)


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_metadata() -> dict[str, Any]:
    import lightgbm
    import pandas
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pandas.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "git_commit": git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
