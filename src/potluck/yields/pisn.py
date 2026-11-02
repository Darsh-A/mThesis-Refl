"""PISN table factory."""

from __future__ import annotations

from src.load_data import load_hw2002, load_takahashi

from .base import PISNTable


def load_pisn_table(name: str) -> PISNTable:
    """Load a named PISN yield table."""
    key = name.strip()
    if key.upper() == "HW2002":
        return PISNTable.from_entries(load_hw2002(), "HW2002")

    if key.lower().startswith("takahashi"):
        models = load_takahashi()
        model = key.split("-", 1)[1].upper() if "-" in key else "MR"
        if model not in models:
            raise ValueError(
                f"Unknown Takahashi model {model!r}; choose from {sorted(models)}"
            )
        return PISNTable.from_entries(models[model], f"Takahashi-{model}")

    raise ValueError(f"Unknown PISN table {name!r}")
