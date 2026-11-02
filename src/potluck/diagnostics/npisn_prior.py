"""Test how the number-of-PISN prior reshapes the ``f_Z^P`` distribution.

``f_Z^P`` is the PISN share of the metal mass, so it is exactly zero whenever
no PISN is drawn. The fraction of mocks at zero is therefore set by the
``n_pisn`` prior (and by the acceptance cuts), not by the yields. This module
generates the same model twice with the same seed, changing only that prior,
and compares the resulting label distributions.
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
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp

from ..config import ModelConfig
from ..dataset import generate_dataset

UNIFORM_PROBS = (0.25, 0.25, 0.25, 0.25)
PAIRS = (("default", "tab:blue"), ("uniform", "tab:orange"))
GROUPS = (1, 2, 3)


def _summarise(
    arrays: dict[str, np.ndarray], prior_probs: tuple[float, ...]
) -> dict[str, Any]:
    """Return the ``n_pisn`` counts and the ``f_Z^P`` moments per group."""
    f_zp = arrays["f_zp"].astype(float)
    n_pisn = arrays["n_pisn"].astype(int)
    counts = np.bincount(n_pisn, minlength=4)
    by_group: dict[str, Any] = {}
    for group in range(4):
        values = f_zp[n_pisn == group]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        by_group[str(group)] = {
            "count": int(values.size),
            "fraction": float(values.size / f_zp.size),
            "median": float(np.median(values)),
            "p16": float(np.percentile(values, 16)),
            "p84": float(np.percentile(values, 84)),
            "positive_fraction": float(np.mean(values >= 0.5)),
        }
    finite = f_zp[np.isfinite(f_zp)]
    return {
        "n_mocks": int(f_zp.size),
        "n_pisn_counts": counts.tolist(),
        "prior_probs": list(prior_probs),
        "zero_fraction": float(np.mean(finite == 0.0)),
        "positive_fraction": float(np.mean(finite >= 0.5)),
        "quantiles": {
            f"p{q}": float(np.percentile(finite, q)) for q in (1, 5, 16, 50, 84, 95, 99)
        },
        "by_group": by_group,
    }


def _plot(
    base: dict[str, np.ndarray],
    uniform: dict[str, np.ndarray],
    summaries: dict[str, dict[str, Any]],
    path: Path,
) -> Path:
    """Compare the two ``n_pisn`` priors in four panels."""
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
    width = 0.38
    positions = np.arange(4)
    for offset, (name, color) in zip((-width / 2, width / 2), PAIRS):
        counts = np.asarray(summaries[name]["n_pisn_counts"], dtype=float)
        fractions = counts / counts.sum()
        axis.bar(
            positions + offset,
            fractions,
            width,
            color=color,
            label=name,
            alpha=0.85,
        )
        prior = np.asarray(summaries[name]["prior_probs"], dtype=float)
        axis.scatter(
            positions + offset,
            prior,
            marker="_",
            s=120,
            color="black",
            linewidth=1.2,
            zorder=5,
        )
        for position, fraction, count in zip(positions, fractions, counts):
            axis.text(
                position + offset,
                fraction + 0.006,
                f"{count:,.0f}",
                ha="center",
                fontsize=6.5,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels(["0", "1", "2", "3"])
    axis.set_xlabel(r"$n_{\rm PISN}$")
    axis.set_ylabel("accepted fraction")
    axis.set_title("Accepted counts and prior probabilities")
    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D([], [], marker="_", color="black", ls="", ms=10, label="prior")
    )
    labels.append("prior")
    axis.legend(handles, labels, fontsize=8)
    axis.set_ylim(0.0, 0.45)

    axis = axes[0, 1]
    bins = np.linspace(0.0, 1.0, 41)
    for name, color in PAIRS:
        f_zp = (base if name == "default" else uniform)["f_zp"].astype(float)
        axis.hist(
            f_zp[np.isfinite(f_zp)],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            color=color,
            label=f"{name}: {summaries[name]['positive_fraction']:.1%} positive",
        )
        axis.axvline(
            summaries[name]["quantiles"]["p50"], color=color, ls=":", lw=1.0
        )
    axis.axvline(0.5, color="black", ls="--", lw=1.0)
    axis.set_xlabel(r"$f_Z^P$")
    axis.set_ylabel("density")
    axis.set_title(r"$f_Z^P$ distribution (dotted: median, dashed: 0.5)")
    axis.legend(fontsize=8, loc="upper right")

    axis = axes[1, 0]
    for offset, (name, color) in zip((-0.12, 0.12), PAIRS):
        for group in GROUPS:
            stats = summaries[name]["by_group"][str(group)]
            axis.errorbar(
                group + offset,
                stats["median"],
                yerr=[
                    [stats["median"] - stats["p16"]],
                    [stats["p84"] - stats["median"]],
                ],
                fmt="o",
                ms=6,
                color=color,
                capsize=3,
                lw=1.4,
            )
    axis.set_xticks(GROUPS)
    axis.set_xticklabels([str(group) for group in GROUPS])
    axis.set_xlabel(r"$n_{\rm PISN}$")
    axis.set_ylabel(r"$f_Z^P$")
    axis.set_title(r"$f_Z^P$ versus $n_{\rm PISN}$")

    axis = axes[1, 1]
    for offset, (name, color) in zip((-0.12, 0.12), PAIRS):
        rates = [summaries[name]["by_group"][str(group)]["positive_fraction"] for group in GROUPS]
        axis.plot(
            np.asarray(GROUPS) + offset,
            rates,
            "o-",
            color=color,
            lw=1.8,
            ms=6,
        )
        for group, rate in zip(GROUPS, rates):
            axis.annotate(
                f"{rate:.0%}",
                (group + offset, rate),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
            )
    axis.set_xticks(GROUPS)
    axis.set_xticklabels([str(group) for group in GROUPS])
    axis.set_xlabel(r"$n_{\rm PISN}$")
    axis.set_ylabel(r"fraction with $f_Z^P \geq 0.5$")
    axis.set_title(r"Positive fraction versus $n_{\rm PISN}$")
    axis.set_ylim(0.0, 1.0)

    fig.suptitle(
        r"Same model with two $n_{\rm PISN}$ priors",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def run_npisn_prior(
    n_samples: int = 20000,
    seed: int = 42,
    outdir: str | Path = "src/potluck/tests/v8",
    config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Generate both priors, compare the label, and write figure and JSON."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base_config = config if config is not None else ModelConfig()
    uniform_config = replace(
        base_config,
        prior=replace(base_config.prior, n_pisn_probs=UNIFORM_PROBS),
    )

    base, _ = generate_dataset(base_config, n_samples=n_samples, seed=seed)
    uniform, _ = generate_dataset(uniform_config, n_samples=n_samples, seed=seed)
    summaries = {
        "default": _summarise(base, base_config.prior.n_pisn_probs),
        "uniform": _summarise(uniform, UNIFORM_PROBS),
    }

    ks_by_group = {}
    for group in GROUPS:
        base_mask = base["n_pisn"] == group
        uniform_mask = uniform["n_pisn"] == group
        ks = ks_2samp(
            base["f_zp"][base_mask].astype(float),
            uniform["f_zp"][uniform_mask].astype(float),
        )
        ks_by_group[str(group)] = {
            "statistic": float(ks.statistic),
            "p_value": float(ks.pvalue),
        }
    result = {
        "n_samples": int(n_samples),
        "seed": int(seed),
        "priors": {
            "default": list(base_config.prior.n_pisn_probs),
            "uniform": list(UNIFORM_PROBS),
        },
        "default": summaries["default"],
        "uniform": summaries["uniform"],
        "ks_by_group": ks_by_group,
    }

    _plot(base, uniform, summaries, outdir / "npisn_prior.png")
    with (outdir / "npisn_prior.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the prior comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="src/potluck/tests/v8")
    args = parser.parse_args(argv)
    result = run_npisn_prior(args.n, seed=args.seed, outdir=args.outdir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
