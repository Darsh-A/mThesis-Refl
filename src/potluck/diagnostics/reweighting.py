"""Importance-reweight a mock pool to the Li+ catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..calibration import CORE_SUPPORT_ELEMENTS
from ..constants import LI_FEATURE_ELEMENTS, REFERENCE
from ..io import load_dataset
from ..observation import LIRecords, load_li_records

PARAMETERS = (
    "mdil",
    "mstar_ii",
    "n_pisn",
    "n_iii",
    "delta_myr",
    "t_myr",
    "feh",
    "lambda_iii",
    "z_star",
    "popii_n",
)


def _active_ratio_elements(excluded: tuple[str, ...]) -> tuple[str, ...]:
    """Return ratio elements that were modelled for this pool."""
    return tuple(
        element
        for element in LI_FEATURE_ELEMENTS
        if element != REFERENCE and element not in excluded
    )


def _real_ratios(
    records: LIRecords,
    elements: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return real ratios, detection flags, and errors in element order."""
    columns = [LI_FEATURE_ELEMENTS.index(element) for element in elements]
    return records.xfe[:, columns], records.flags[:, columns], records.sigma[:, columns]


def _mock_ratios(
    arrays: dict[str, np.ndarray],
    elements: tuple[str, ...],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """Return noiseless mock ratios in element order."""
    names = list(feature_names)
    return np.column_stack(
        [
            arrays["features"][:, names.index(f"{element}/Fe")]
            for element in elements
        ]
    )


def _log_importance(
    real_ratios: np.ndarray,
    real_flags: np.ndarray,
    real_sigma: np.ndarray,
    mock_ratios: np.ndarray,
    columns: list[int],
    min_elements: int,
) -> tuple[np.ndarray, int]:
    """Return per-mock log mixture likelihood and the number of stars used."""
    n_mock = mock_ratios.shape[0]
    log_weight = np.full(n_mock, -np.inf, dtype=float)
    used = 0
    log_two_pi = np.log(2.0 * np.pi)
    for star in range(real_ratios.shape[0]):
        selected = [
            column
            for column in columns
            if real_flags[star, column] > 0
            and np.isfinite(real_ratios[star, column])
            and np.isfinite(real_sigma[star, column])
            and real_sigma[star, column] > 0.0
        ]
        if len(selected) < min_elements:
            continue
        value = real_ratios[star, selected]
        sigma = real_sigma[star, selected]
        difference = (mock_ratios[:, selected] - value[None, :]) / sigma[None, :]
        finite = np.isfinite(difference)
        dimension = finite.sum(axis=1)
        d2 = np.where(finite, np.square(difference), 0.0).sum(axis=1)
        log_norm = -0.5 * (
            dimension * log_two_pi
            + np.where(finite, np.log(np.square(sigma))[None, :], 0.0).sum(axis=1)
        )
        log_likelihood = np.where(
            dimension >= min_elements,
            log_norm - 0.5 * d2,
            -np.inf,
        )
        log_weight = np.logaddexp(log_weight, log_likelihood)
        used += 1
    if used:
        log_weight -= np.log(used)
    return log_weight, used


def _normalize(log_weight: np.ndarray) -> np.ndarray:
    """Return normalized importance weights, or zeros if none are finite."""
    weight = np.zeros_like(log_weight, dtype=float)
    finite = np.isfinite(log_weight)
    if not finite.any():
        return weight
    peak = float(np.max(log_weight[finite]))
    weight[finite] = np.exp(log_weight[finite] - peak)
    total = weight.sum()
    if total <= 0.0:
        return np.zeros_like(weight)
    return weight / total


def _weighted_quantile(
    values: np.ndarray,
    weight: np.ndarray,
    quantiles: tuple[float, ...] = (0.16, 0.5, 0.84),
) -> list[float]:
    """Return weighted quantiles from the step inverse CDF."""
    finite = np.isfinite(values) & np.isfinite(weight) & (weight > 0.0)
    if not finite.any():
        return [float("nan")] * len(quantiles)
    value = values[finite]
    mass = weight[finite]
    order = np.argsort(value)
    value = value[order]
    mass = mass[order]
    cumulative = np.cumsum(mass)
    target = np.clip(np.asarray(quantiles, dtype=float), 0.0, 1.0) * cumulative[-1]
    index = np.searchsorted(cumulative, target, side="left")
    index = np.clip(index, 0, value.size - 1)
    return [float(value[i]) for i in index]


def _summarize_weights(
    log_weight: np.ndarray,
    used: int,
    arrays: dict[str, np.ndarray],
    top_fraction: float = 0.01,
) -> dict[str, Any]:
    """Return ESS, weight shares, and parameter shifts for one scheme."""
    weight = _normalize(log_weight)
    total = float(weight.sum())
    if total <= 0.0:
        return {
            "n_stars": used,
            "ess": 0.0,
            "ess_fraction": 0.0,
            "max_weight": 0.0,
            "top_fraction": top_fraction,
            "top_share": 0.0,
            "parameters": {},
        }
    ess = float(1.0 / np.sum(np.square(weight)))
    order = np.argsort(weight)[::-1]
    n_top = max(1, int(round(top_fraction * weight.size)))
    top = order[:n_top]
    top_share = float(weight[top].sum())
    parameters: dict[str, Any] = {}
    for name in PARAMETERS:
        if name not in arrays:
            continue
        values = np.asarray(arrays[name], dtype=float)
        prior = _weighted_quantile(values, np.ones_like(values))
        posterior = _weighted_quantile(values, weight)
        top_values = values[top]
        parameters[name] = {
            "prior_p16": prior[0],
            "prior_median": prior[1],
            "prior_p84": prior[2],
            "posterior_p16": posterior[0],
            "posterior_median": posterior[1],
            "posterior_p84": posterior[2],
            "top_median": float(np.nanmedian(top_values)),
        }
    return {
        "n_stars": used,
        "ess": ess,
        "ess_fraction": ess / weight.size,
        "max_weight": float(weight[order[0]]),
        "top_fraction": top_fraction,
        "n_top": n_top,
        "top_share": top_share,
        "parameters": parameters,
    }


def _null_ess(
    arrays: dict[str, np.ndarray],
    elements: tuple[str, ...],
    min_elements: int,
    mock_ratios: np.ndarray,
    n_stars: int = 385,
    n_draws: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Reweight the pool to mock stars to calibrate the ESS scale."""
    ratio_elements = tuple(
        element for element in LI_FEATURE_ELEMENTS if element != REFERENCE
    )
    columns = [ratio_elements.index(element) for element in elements]
    observed = arrays["obs_features"][:, columns]
    flags = arrays["obs_flags"][:, columns]
    sigma = arrays["obs_sigma"][:, columns]
    eligible = np.isfinite(observed).sum(axis=1) >= min_elements
    candidates = np.flatnonzero(eligible)
    n_stars = int(min(n_stars, max(1, candidates.size - 1)))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_draws):
        chosen = rng.choice(candidates, size=n_stars, replace=False)
        keep = np.ones(observed.shape[0], dtype=bool)
        keep[chosen] = False
        log_weight, _ = _log_importance(
            observed[chosen],
            flags[chosen],
            sigma[chosen],
            mock_ratios[keep],
            list(range(len(elements))),
            min_elements,
        )
        weight = _normalize(log_weight)
        total = float(weight.sum())
        values.append(float(1.0 / np.sum(np.square(weight))) if total > 0.0 else 0.0)
    return {
        "n_draws": int(n_draws),
        "n_stars": n_stars,
        "values": values,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _schemes(
    elements: tuple[str, ...],
    core: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Return element subsets used to locate the constraining elements."""
    schemes: dict[str, tuple[str, ...]] = {"all": elements}
    if "Cr" in elements:
        schemes["all_no_Cr"] = tuple(e for e in elements if e != "Cr")
    if "Ni" in elements:
        schemes["all_no_Ni"] = tuple(e for e in elements if e != "Ni")
    if core:
        schemes["core"] = core
        for element in core:
            schemes[f"core_no_{element}"] = tuple(e for e in core if e != element)
        for element in core:
            schemes[element] = (element,)
    return schemes


def run_reweighting(
    dataset: str | Path | tuple[dict[str, np.ndarray], dict[str, Any]],
    outdir: str | Path = "src/potluck/tests/v6",
    min_elements: int = 4,
    core_elements: tuple[str, ...] | None = None,
    top_fraction: float = 0.01,
    verbose: bool = False,
) -> dict[str, Any]:
    """Reweight a mock pool to Li+ and report the effective sample size."""
    if isinstance(dataset, (str, Path)):
        arrays, meta = load_dataset(dataset)
    else:
        arrays, meta = dataset
    excluded = tuple(meta.get("observation_excluded_elements", ()))
    elements = _active_ratio_elements(excluded)
    core_available = tuple(
        element
        for element in (core_elements or CORE_SUPPORT_ELEMENTS)
        if element in elements
    )
    real_ratios, real_flags, real_sigma = _real_ratios(load_li_records(), elements)
    mock_ratios = _mock_ratios(arrays, elements, meta["feature_names"])
    schemes = _schemes(elements, core_available)
    summary: dict[str, Any] = {
        "n_mocks": int(mock_ratios.shape[0]),
        "n_real": int(real_ratios.shape[0]),
        "elements": elements,
        "excluded_elements": excluded,
        "core_elements": core_available,
        "min_elements": min_elements,
        "schemes": {},
    }
    for name, chosen in schemes.items():
        columns = [elements.index(element) for element in chosen]
        required = 1 if len(chosen) == 1 else min_elements
        log_weight, used = _log_importance(
            real_ratios,
            real_flags,
            real_sigma,
            mock_ratios,
            columns,
            required,
        )
        result = _summarize_weights(
            log_weight,
            used,
            arrays,
            top_fraction=top_fraction,
        )
        result["elements"] = chosen
        summary["schemes"][name] = result
        if verbose:
            print(
                f"{name}: ess={result['ess']:.2f} "
                f"({result['ess_fraction']:.2e}), stars={used}",
                flush=True,
            )
    null = _null_ess(
        arrays,
        elements,
        min_elements,
        mock_ratios,
        n_stars=real_ratios.shape[0],
        seed=1,
    )
    summary["null"] = null
    if verbose:
        print(
            f"null: ess={null['mean']:.2f} "
            f"(median {null['median']:.2f}), stars={null['n_stars']}",
            flush=True,
        )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "reweighting.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _plot(summary, outdir / "reweighting.png")
    return summary


def _plot(summary: dict[str, Any], path: Path) -> Path:
    """Plot ESS by scheme, weights, and parameter shifts."""
    schemes = summary["schemes"]
    names = list(schemes)
    ess = np.asarray([schemes[name]["ess"] for name in names])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].barh(names, ess, color="tab:blue")
    axes[0, 0].set_xscale("log")
    axes[0, 0].axvline(
        max(summary["null"]["mean"], 1e-6),
        color="black",
        ls="--",
        lw=1,
        label="mock null",
    )
    axes[0, 0].set_xlabel("effective sample size")
    axes[0, 0].set_title("ESS by element subset")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].invert_yaxis()

    weight = np.asarray(
        [schemes["all"]["max_weight"], schemes["all"]["top_share"]]
    )
    axes[0, 1].bar(["max weight", "top 1% share"], weight, color="tab:orange")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_title("all-element weight concentration")

    core_names = [name for name in ("Na", "Mg", "Ca", "Cr", "Ni") if name in schemes]
    core_ess = np.asarray([schemes[name]["ess"] for name in core_names])
    axes[1, 0].bar(core_names, core_ess, color="tab:green")
    if np.all(core_ess > 0.0):
        axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("ESS")
    axes[1, 0].set_title("single-element ESS")

    parameters = [name for name in ("mdil", "mstar_ii", "n_pisn", "n_iii") if name in schemes["all"]["parameters"]]
    for index, name in enumerate(parameters):
        entry = schemes["all"]["parameters"][name]
        prior = entry["prior_median"]
        posterior = entry["posterior_median"]
        axes[1, 1].plot([0, 1], [prior, posterior], marker="o", label=name)
    axes[1, 1].set_xticks([0, 1], ["prior", "posterior"])
    axes[1, 1].set_title("all-element median shift")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        f"Li+ reweighting: {summary['n_mocks']} mocks, "
        f"ESS(all)={schemes['all']['ess']:.2f}, "
        f"mock-null={summary['null']['mean']:.2f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the Li+ reweighting diagnostic from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="dataset .npz file")
    parser.add_argument("--outdir", default="src/potluck/tests/v6")
    parser.add_argument("--min-elements", type=int, default=4)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    summary = run_reweighting(
        args.dataset,
        outdir=args.outdir,
        min_elements=args.min_elements,
        top_fraction=args.top_fraction,
        verbose=args.verbose,
    )
    print(
        json.dumps(
            {
                "n_mocks": summary["n_mocks"],
                "ess_all": summary["schemes"]["all"]["ess"],
                "ess_core": summary["schemes"].get("core", {}).get("ess"),
                "ess_null": summary["null"]["mean"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
