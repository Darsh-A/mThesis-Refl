"""Same-seed comparison of the Nomoto13 and Limongi18 Pop II tables.

Nomoto13 stops at Z = 1e-3; Limongi18 reaches its [Fe/H] = -3 node at
Z = 3.236e-5.  This module generates one mock pool per table from the same
seed and reports where the two disagree.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from ..config import ModelConfig
from ..dataset import generate_dataset

LIMONGI_Z_NODES = (3.236e-5, 3.236e-4, 3.236e-3, 1.345e-2)
NOMOTO_Z_FLOOR = 1.0e-3
SOURCES = ("nomoto13", "limongi18")
COLORS = {"nomoto13": "tab:blue", "limongi18": "tab:orange"}
LABELS = {"nomoto13": "Nomoto13", "limongi18": "Limongi18"}


def _config(source: str, base: ModelConfig | None = None) -> ModelConfig:
    config = base if base is not None else ModelConfig()
    if source == "nomoto13":
        return replace(
            config,
            yields=replace(
                config.yields,
                popii_table="Nomoto13",
                popii_z_nodes=None,
                popii_z_grid_size=24,
            ),
        )
    return replace(
        config,
        yields=replace(
            config.yields,
            popii_table="Limongi18",
            popii_z_nodes=LIMONGI_Z_NODES,
            popii_z_grid_size=len(LIMONGI_Z_NODES),
        ),
    )


def _summarise(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, Any]:
    feh = arrays["feh"].astype(float)
    f_zp = arrays["f_zp"].astype(float)
    z_star = arrays["z_star"].astype(float)
    names = tuple(meta["feature_names"])
    median: dict[str, float] = {}
    p16: dict[str, float] = {}
    p84: dict[str, float] = {}
    for index, name in enumerate(names):
        values = arrays["features"][:, index].astype(float)
        values = values[np.isfinite(values)]
        if values.size:
            median[name] = float(np.median(values))
            p16[name] = float(np.percentile(values, 16))
            p84[name] = float(np.percentile(values, 84))
    return {
        "n_mocks": int(feh.size),
        "feh_median": float(np.nanmedian(feh)),
        "feh_p16": float(np.nanpercentile(feh, 16)),
        "feh_p84": float(np.nanpercentile(feh, 84)),
        "f_zp_median": float(np.nanmedian(f_zp)),
        "f_zp_positive_fraction": float(np.mean(f_zp >= 0.5)),
        "z_star_p16": float(np.nanpercentile(z_star, 16)),
        "z_star_median": float(np.nanmedian(z_star)),
        "z_star_p84": float(np.nanpercentile(z_star, 84)),
        "feature_names": list(names),
        "element_median": median,
        "element_p16": p16,
        "element_p84": p84,
    }


def _design(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.hstack(
        [
            arrays["features"].astype(float),
            arrays["ratio_mask"].astype(float),
        ]
    )


def _separability(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    seed: int,
    per_pool: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    picks = []
    for arrays in (first, second):
        size = arrays["features"].shape[0]
        picks.append(
            rng.choice(size, size=min(size, per_pool), replace=False)
        )
    x = np.vstack([_design(first)[picks[0]], _design(second)[picks[1]]])
    y = np.concatenate([np.zeros(picks[0].size), np.ones(picks[1].size)])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.zeros_like(y, dtype=float)
    for train, test in folds.split(x, y):
        model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(x[train], y[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
    fpr, tpr, _ = roc_curve(y, scores)
    return {
        "auc": float(roc_auc_score(y, scores)),
        "fpr": fpr,
        "tpr": tpr,
        "n_per_pool": int(picks[0].size),
    }


def _plot(
    summaries: dict[str, dict[str, Any]],
    arrays: dict[str, dict[str, np.ndarray]],
    roc: dict[str, Any],
    path: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.axisbelow": True,
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    axis = axes[0, 0]
    for source in SOURCES:
        feh = arrays[source]["feh"].astype(float)
        feh = feh[np.isfinite(feh)]
        axis.hist(
            feh,
            bins=45,
            histtype="step",
            density=True,
            lw=1.8,
            color=COLORS[source],
            label=f"{LABELS[source]}: median {np.median(feh):.2f}",
        )
        axis.axvline(np.median(feh), color=COLORS[source], ls=":", lw=1.2)
    axis.set_xlabel("[Fe/H]")
    axis.set_ylabel("density")
    axis.set_title("Final [Fe/H]")
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    names = summaries["nomoto13"]["feature_names"]
    positions = np.arange(len(names))
    for source in SOURCES:
        values = [
            summaries[source]["element_median"].get(name, np.nan) for name in names
        ]
        axis.plot(
            positions,
            values,
            marker="o",
            ms=4,
            lw=1.5,
            color=COLORS[source],
            label=LABELS[source],
        )
    axis.axhline(0.0, color="0.5", lw=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=45, ha="right")
    axis.set_ylabel("median [X/Fe]")
    axis.set_title("Median [X/Fe] by element")
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    for source in SOURCES:
        f_zp = arrays[source]["f_zp"].astype(float)
        f_zp = f_zp[np.isfinite(f_zp)]
        axis.hist(
            f_zp,
            bins=40,
            histtype="step",
            density=True,
            lw=1.8,
            color=COLORS[source],
            label=(
                f"{LABELS[source]}: "
                f"{np.mean(f_zp >= 0.5):.1%} positive"
            ),
        )
    axis.axvline(0.5, color="black", ls="--", lw=1.0)
    axis.set_xlabel(r"$f_Z^P$")
    axis.set_ylabel("density")
    axis.set_title(r"$f_Z^P$ distribution (dashed: 0.5)")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    axis.plot(roc["fpr"], roc["tpr"], color="tab:purple", lw=1.8)
    axis.plot([0.0, 1.0], [0.0, 1.0], color="0.6", ls=":", lw=1.0)
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.set_title(f"Cross-table classifier (5-fold AUC {roc['auc']:.3f})")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    fig.suptitle("Nomoto13 versus Limongi18 with the same seed", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_popii_metallicity(
    n_samples: int = 20000,
    seed: int = 42,
    outdir: str | Path = "src/potluck/tests/v9",
    config: ModelConfig | None = None,
    auc_per_pool: int = 8000,
) -> dict[str, Any]:
    """Generate one pool per table and write the comparison."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    metas: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        pool_config = _config(source, config)
        pool_arrays, pool_meta = generate_dataset(
            pool_config, n_samples=n_samples, seed=seed
        )
        arrays[source] = pool_arrays
        metas[source] = pool_meta
        summaries[source] = _summarise(pool_arrays, pool_meta)

    floors = {"nomoto13": NOMOTO_Z_FLOOR, "limongi18": LIMONGI_Z_NODES[0]}
    for source in SOURCES:
        z_star = arrays[source]["z_star"].astype(float)
        summaries[source]["clamp_fraction"] = float(
            np.mean(z_star <= floors[source])
        )
        summaries[source]["z_floor"] = float(floors[source])
        summaries[source]["accepted"] = int(metas[source]["ledger"]["accepted"])
        summaries[source]["drawn"] = int(metas[source]["ledger"]["drawn"])

    roc = _separability(
        arrays["nomoto13"], arrays["limongi18"], seed=seed, per_pool=auc_per_pool
    )
    _plot(summaries, arrays, roc, outdir / "popii_metallicity.png")

    names = summaries["nomoto13"]["feature_names"]
    shift = {
        name: float(
            summaries["limongi18"]["element_median"][name]
            - summaries["nomoto13"]["element_median"][name]
        )
        for name in names
        if name in summaries["limongi18"]["element_median"]
        and name in summaries["nomoto13"]["element_median"]
    }
    result: dict[str, Any] = {
        "n_samples": int(n_samples),
        "seed": int(seed),
        "limongi_z_nodes": list(LIMONGI_Z_NODES),
        "pools": summaries,
        "element_median_shift": shift,
        "separability": {
            "auc": roc["auc"],
            "n_per_pool": roc["n_per_pool"],
            "features": "features + ratio_mask",
        },
    }
    with (outdir / "popii_metallicity.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="src/potluck/tests/v9")
    parser.add_argument("--auc-per-pool", type=int, default=8000)
    args = parser.parse_args(argv)
    result = run_popii_metallicity(
        n_samples=args.n,
        seed=args.seed,
        outdir=args.outdir,
        auc_per_pool=args.auc_per_pool,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
