"""Dataset storage and provenance metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import ModelConfig
from .constants import ELEMENTS, METALS

FORMAT_VERSION = 1


def config_hash(config: ModelConfig) -> str:
    """Return a stable hash of the full configuration."""
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def git_commit() -> str | None:
    """Return the current git commit, or ``None`` outside a repository."""
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            env={"GIT_TERMINAL_PROMPT": "0"},
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def build_meta(
    config: ModelConfig,
    ledger: dict[str, int],
    n_drawn: int,
    n_accepted: int,
    feature_names: tuple[str, ...],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON metadata written next to a dataset."""
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "config": config.to_dict(),
        "elements": list(ELEMENTS),
        "metals": list(METALS),
        "feature_elements": list(config.feature_elements),
        "reference": config.reference,
        "feature_names": list(feature_names),
        "source_names": ["PISN", "POPIII", "POPII", "SNIA"],
        "units": {
            "mass": "solar masses per event or per cloud",
            "abundance": "log10(mass fraction / solar mass fraction)",
            "ratio": "dex",
            "time": "Myr",
            "metallicity": "mass fraction",
        },
        "cuts": {
            "r1": "n_pisn + n_iii > 0",
            "r2": "z_star > 0 and z_star <= z_max (and z_star >= z_cr if set)",
            "r3": "Fe and at least min_valid_features feature elements above the relative floor",
        },
        "ledger": ledger,
        "n_drawn": int(n_drawn),
        "n_accepted": int(n_accepted),
        "acceptance": float(n_accepted / n_drawn) if n_drawn else 0.0,
    }
    if extra:
        meta.update(extra)
    return meta


def save_dataset(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> Path:
    """Write arrays to ``.npz`` and metadata to ``.meta.json``."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    meta_path = path.with_suffix(".meta.json")
    with meta_path.open("w") as handle:
        json.dump(meta, handle, indent=2)
    return path


def load_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load arrays and metadata from a dataset pair."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with path.with_suffix(".meta.json").open() as handle:
        meta = json.load(handle)
    return arrays, meta
