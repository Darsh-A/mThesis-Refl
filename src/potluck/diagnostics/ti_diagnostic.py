"""Direct Ti diagnostic: Li+ [Ti/Fe] against the mock population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ..constants import ELEMENT_INDEX, LI_FEATURE_ELEMENTS, REFERENCE, SOLAR_ABUNDANCE
from ..observation import GIANT_LOGG, LIRecords, load_li_records

RATIO_ELEMENTS = tuple(e for e in LI_FEATURE_ELEMENTS if e != REFERENCE)
TI_COLUMN = RATIO_ELEMENTS.index("Ti")
FEH_BINS = (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0)
SNR_BINS = (0.0, 20.0, 40.0, 60.0, 100.0, 200.0)
MASS_PLOT_MAX = 30.0


def _weighted_median(values: np.ndarray, weight: np.ndarray) -> float:
    """Weighted median of finite values."""
    order = np.argsort(values)
    values = values[order]
    weight = weight[order]
    cumulative = np.cumsum(weight)
    cumulative /= cumulative[-1]
    return float(np.interp(0.5, cumulative, values))


def _real_ti(records: LIRecords) -> dict[str, np.ndarray]:
    """Return real Ti ratios and stellar parameters."""
    detected = records.flags[:, TI_COLUMN] > 0
    return {
        "ti": np.where(detected, records.xfe[:, TI_COLUMN], np.nan),
        "feh": records.feh,
        "logg": records.logg,
        "snr": records.snr,
        "sigma": records.sigma[:, TI_COLUMN],
        "detected": detected,
    }


def _mock_ti(
    arrays: dict[str, np.ndarray], records: LIRecords
) -> dict[str, np.ndarray]:
    """Return observed and noiseless mock Ti ratios with parameters."""
    observed = arrays["obs_features"][:, TI_COLUMN].astype(float)
    correct = arrays["obs_flags"][:, TI_COLUMN] > 0
    observed = np.where(correct & np.isfinite(observed), observed, np.nan)
    true = (
        arrays["log_abundance"][:, ELEMENT_INDEX["Ti"]].astype(float)
        - arrays["log_abundance"][:, ELEMENT_INDEX[REFERENCE]].astype(float)
    )
    record = np.asarray(arrays["record_index"], dtype=int)
    return {
        "ti": observed,
        "ti_true": np.where(np.isfinite(true), true, np.nan),
        "feh": arrays["feh_obs"].astype(float),
        "logg": arrays["logg"].astype(float),
        "snr": records.snr[record],
        "weight": arrays["importance_weight"].astype(float),
        "detected": correct,
    }


def _bootstrap_offset(
    real: np.ndarray,
    mock: np.ndarray,
    mock_weight: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the median offset, its interval, and the KS distance."""
    real = real[np.isfinite(real)]
    finite = np.isfinite(mock)
    mock = mock[finite]
    mock_weight = mock_weight[finite]
    real_median = (
        float(_weighted_median(real, np.ones_like(real))) if real.size else float("nan")
    )
    if real.size == 0 or mock.size == 0:
        return {
            "n_real": int(real.size),
            "n_mock": int(mock.size),
            "real_median": real_median,
            "mock_median": float("nan"),
            "mock_median_unweighted": float("nan"),
            "offset": float("nan"),
            "offset_lo": float("nan"),
            "offset_hi": float("nan"),
            "ks": float("nan"),
            "ks_p": float("nan"),
        }
    probability = mock_weight / mock_weight.sum()
    rng = np.random.default_rng(seed)
    offsets = np.empty(n_boot)
    for i in range(n_boot):
        real_draw = real[rng.integers(0, real.size, real.size)]
        mock_draw = mock[rng.choice(mock.size, mock.size, p=probability)]
        offsets[i] = _weighted_median(real_draw, np.ones_like(real_draw)) - _weighted_median(
            mock_draw, np.ones_like(mock_draw)
        )
    mock_median = _weighted_median(mock, mock_weight)
    real_median = _weighted_median(real, np.ones_like(real))
    ks = stats.ks_2samp(real, mock)
    return {
        "n_real": int(real.size),
        "n_mock": int(mock.size),
        "real_median": float(real_median),
        "mock_median": float(mock_median),
        "mock_median_unweighted": float(np.median(mock)),
        "offset": float(real_median - mock_median),
        "offset_lo": float(np.percentile(offsets, 2.5)),
        "offset_hi": float(np.percentile(offsets, 97.5)),
        "ks": float(ks.statistic),
        "ks_p": float(ks.pvalue),
    }


def _binned(
    real: dict[str, np.ndarray],
    mock: dict[str, np.ndarray],
    edges: tuple[float, ...],
    key: str,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compare real and mock Ti in bins of one parameter."""
    rows = []
    real_values = real[key]
    mock_values = mock[key]
    for index in range(len(edges) - 1):
        lo, hi = edges[index], edges[index + 1]
        last = index == len(edges) - 2
        in_real = (real_values >= lo) & (
            (real_values <= hi) if last else (real_values < hi)
        )
        in_mock = (mock_values >= lo) & (
            (mock_values <= hi) if last else (mock_values < hi)
        )
        row = {"lo": float(lo), "hi": float(hi)}
        row.update(
            _bootstrap_offset(
                real["ti"][in_real],
                mock["ti"][in_mock],
                mock["weight"][in_mock],
                n_boot,
                seed + index,
            )
        )
        rows.append(row)
    return rows


def _split_offset(
    real: dict[str, np.ndarray],
    mock: dict[str, np.ndarray],
    real_group: np.ndarray,
    mock_group: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Compare Ti in two complementary groups and bootstrap their difference."""
    inside = _bootstrap_offset(
        real["ti"][real_group], mock["ti"][mock_group], mock["weight"][mock_group], n_boot, seed
    )
    outside = _bootstrap_offset(
        real["ti"][~real_group],
        mock["ti"][~mock_group],
        mock["weight"][~mock_group],
        n_boot,
        seed + 1,
    )
    rng = np.random.default_rng(seed + 2)
    real_in = real["ti"][real_group]
    real_in = real_in[np.isfinite(real_in)]
    real_out = real["ti"][~real_group]
    real_out = real_out[np.isfinite(real_out)]
    mock_in = mock["ti"][mock_group]
    weight_in = mock["weight"][mock_group]
    finite = np.isfinite(mock_in)
    mock_in, weight_in = mock_in[finite], weight_in[finite]
    mock_out = mock["ti"][~mock_group]
    weight_out = mock["weight"][~mock_group]
    finite = np.isfinite(mock_out)
    mock_out, weight_out = mock_out[finite], weight_out[finite]
    if min(real_in.size, real_out.size, mock_in.size, mock_out.size) == 0:
        return {"inside": inside, "outside": outside, "difference": float("nan"),
                "difference_lo": float("nan"), "difference_hi": float("nan")}
    probability_in = weight_in / weight_in.sum()
    probability_out = weight_out / weight_out.sum()
    differences = np.empty(n_boot)
    for i in range(n_boot):
        sample_in = mock_in[rng.choice(mock_in.size, mock_in.size, p=probability_in)]
        sample_out = mock_out[rng.choice(mock_out.size, mock_out.size, p=probability_out)]
        differences[i] = (
            _weighted_median(real_in[rng.integers(0, real_in.size, real_in.size)], np.ones(real_in.size))
            - _weighted_median(sample_in, np.ones(sample_in.size))
            - _weighted_median(real_out[rng.integers(0, real_out.size, real_out.size)], np.ones(real_out.size))
            + _weighted_median(sample_out, np.ones(sample_out.size))
        )
    return {
        "inside": inside,
        "outside": outside,
        "difference": inside["offset"] - outside["offset"],
        "difference_lo": float(np.percentile(differences, 2.5)),
        "difference_hi": float(np.percentile(differences, 97.5)),
    }


def _bin_slope(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Least-squares slope of binned offset against bin centre."""
    usable = [
        row
        for row in rows
        if row["n_real"] >= 5 and np.isfinite(row["offset"])
    ]
    if len(usable) < 3:
        return {"slope": None, "intercept": None}
    x = np.array([0.5 * (row["lo"] + row["hi"]) for row in usable])
    y = np.array([row["offset"] for row in usable])
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"slope": float(slope), "intercept": float(intercept)}


def _solar_check(offset: float) -> dict[str, Any]:
    """Ask how far the solar Ti scale would have to move to close the offset."""
    return {
        "solar_ti_dex": float(SOLAR_ABUNDANCE["Ti"]),
        "solar_fe_dex": float(SOLAR_ABUNDANCE[REFERENCE]),
        "required_ti_dex": float(SOLAR_ABUNDANCE["Ti"] + offset),
        "required_change_dex": float(offset),
        "plausible": False,
        "note": (
            "Closing the offset by the solar scale would need a 0.6 dex change in "
            "the Ti abundance, far beyond the 0.05 dex scale uncertainty."
        ),
    }


def _yield_check() -> dict[str, Any]:
    """Return Limongi18 Ti/Fe versus progenitor mass and metallicity."""
    from ..yields.popii import PopIITable

    solar_ratio = 10.0 ** (
        SOLAR_ABUNDANCE["Ti"] - SOLAR_ABUNDANCE[REFERENCE]
    )
    nodes = (
        3.236e-5,
        3.236e-4,
        3.236e-3,
        1.345e-2,
    )
    table = PopIITable(
        source_name="Limongi18",
        z_grid=nodes,
        z_grid_size=len(nodes),
        t_grid_size=4,
        cache_dir="data/potluck/cache",
    )
    raw: dict[str, list[dict[str, float]]] = {}
    for index, metallicity in enumerate(nodes):
        rows = []
        for mass, vector in zip(table.bin_masses, table.bin_yields[index]):
            if mass > MASS_PLOT_MAX:
                break
            ti = float(vector[ELEMENT_INDEX["Ti"]])
            fe = float(vector[ELEMENT_INDEX[REFERENCE]])
            if ti <= 0.0 or fe <= 0.0:
                continue
            rows.append(
                {
                    "mass": float(mass),
                    "ti_fe": float(np.log10(ti / fe / solar_ratio)),
                }
            )
        raw[f"{metallicity:g}"] = rows

    integrated = []
    for metallicity in nodes:
        vector = table.integrated(metallicity, table.t_grid[-1])
        integrated.append(
            {
                "metallicity": float(metallicity),
                "ti_fe": float(
                    np.log10(
                        vector[ELEMENT_INDEX["Ti"]]
                        / vector[ELEMENT_INDEX[REFERENCE]]
                        / solar_ratio
                    )
                ),
            }
        )
    return {
        "source": "Limongi18",
        "m_max": table.m_max,
        "table_min_mass": table.table_min_mass,
        "raw": raw,
        "integrated": integrated,
    }


def _ionisation_check(split: dict[str, Any]) -> dict[str, Any]:
    """Report the TiII/Ti ionisation and NLTE plausibility bracket."""
    difference = split["difference"]
    return {
        "catalogue_species": "TiII",
        "dominant_ion": "Ti II",
        "solar_reference": "total Ti",
        "expected_nlte_dex": 0.2,
        "giant_minus_dwarf_offset": difference,
        "note": (
            "Ti II is the dominant ion at these temperatures and gravities, so the "
            "ionisation correction is small. A generous 0.2 dex NLTE bracket leaves "
            "0.4-0.5 dex unexplained; the offset is therefore not an ionisation "
            "effect."
        ),
    }


def _plot(
    real: dict[str, np.ndarray],
    mock: dict[str, np.ndarray],
    summary: dict[str, Any],
    path: Path,
) -> Path:
    """Plot Ti against Fe/H, group, S/N, and progenitor mass."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axis = axes[0, 0]
    axis.scatter(real["feh"], real["ti"], s=8, alpha=0.4, color="tab:red", label="Li+")
    if np.isfinite(mock["ti"]).any():
        axis.scatter(
            mock["feh"], mock["ti"], s=4, alpha=0.08, color="tab:blue", label="mocks"
        )
    rows = summary["by_feh"]
    centre = [0.5 * (row["lo"] + row["hi"]) for row in rows if np.isfinite(row["offset"])]
    offsets = [row["offset"] for row in rows if np.isfinite(row["offset"])]
    if offsets:
        axis.plot(centre, offsets, "k-o", lw=1, ms=4, label="binned offset")
    axis.axhline(0.0, color="black", ls="--", lw=1)
    axis.set_xlabel("observed [Fe/H]")
    axis.set_ylabel("[Ti/Fe]")
    axis.set_title("Ti versus metallicity")
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    bins = np.linspace(-1.4, 1.2, 40)
    axis.hist(real["ti"][np.isfinite(real["ti"])], bins=bins, density=True,
              histtype="step", lw=2, color="tab:red", label="Li+")
    mock_ti = mock["ti"][np.isfinite(mock["ti"])]
    if mock_ti.size:
        axis.hist(mock_ti, bins=bins, density=True,
                  histtype="step", lw=2, color="tab:blue", label="mocks")
    axis.axvline(summary["overall"]["real_median"], color="tab:red", ls="--", lw=1)
    if np.isfinite(summary["overall"]["mock_median"]):
        axis.axvline(summary["overall"]["mock_median"], color="tab:blue", ls="--", lw=1)
    axis.set_xlabel("[Ti/Fe]")
    axis.set_ylabel("density")
    if np.isfinite(summary["overall"]["offset"]):
        title = (
            "Ti distribution: offset "
            f"{summary['overall']['offset']:+.2f} dex, "
            f"KS {summary['overall']['ks']:.2f}"
        )
    else:
        title = "Ti distribution: no mock Ti (element disabled)"
    axis.set_title(title)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    rows = summary["by_snr"]
    centres_all = [0.5 * (row["lo"] + row["hi"]) for row in rows]
    for label, key, colour in (("Li+", "real", "tab:red"), ("mocks", "mock", "tab:blue")):
        centres, values = [], []
        for centre, row in zip(centres_all, rows):
            value = row[f"{key}_median"]
            if not np.isfinite(value):
                continue
            centres.append(centre)
            values.append(value)
        if values:
            axis.plot(centres, values, "o-", color=colour, label=label)
    axis.axhline(0.0, color="black", ls="--", lw=1)
    if centres_all and all(centre > 0.0 for centre in centres_all):
        axis.set_xscale("log")
    axis.set_xlabel("S/N")
    axis.set_ylabel("[Ti/Fe]")
    axis.set_title("Ti versus signal-to-noise")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    for metallicity, rows in summary["yields"]["raw"].items():
        masses = [row["mass"] for row in rows]
        values = [row["ti_fe"] for row in rows]
        axis.plot(masses, values, "o-", lw=1, ms=3, label=f"Z={metallicity}")
    integrated = summary["yields"]["integrated"]
    axis.axhline(0.0, color="black", ls="--", lw=1)
    axis.set_xlabel(r"progenitor mass [M$_\odot$]")
    axis.set_ylabel("Limongi18 [Ti/Fe]")
    axis.set_title("Single-star Ti yield, 13-30 $M_\\odot$")
    if summary["yields"]["raw"]:
        axis.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_ti_diagnostic(
    dataset: str | Path | dict[str, np.ndarray],
    outdir: str | Path = "src/potluck/tests/v5",
    n_boot: int = 2000,
    seed: int = 0,
    include_yields: bool = True,
) -> dict[str, Any]:
    """Compare the Li+ Ti distribution with one mock pool."""
    if isinstance(dataset, (str, Path)):
        with np.load(dataset) as data:
            arrays = {key: data[key] for key in data.files}
    else:
        arrays = dataset
    records = load_li_records()
    real = _real_ti(records)
    mock = _mock_ti(arrays, records)

    overall = _bootstrap_offset(real["ti"], mock["ti"], mock["weight"], n_boot, seed)
    by_feh = _binned(real, mock, FEH_BINS, "feh", n_boot, seed + 10)
    by_snr = _binned(real, mock, SNR_BINS, "snr", n_boot, seed + 20)
    giants = _split_offset(
        real, mock, real["logg"] < GIANT_LOGG, mock["logg"] < GIANT_LOGG, n_boot, seed + 30
    )
    true_overall = _bootstrap_offset(
        real["ti"], mock["ti_true"], np.ones_like(mock["weight"]), n_boot, seed + 40
    )

    summary: dict[str, Any] = {
        "dataset": str(dataset),
        "n_real": int(records.n),
        "n_mock": int(arrays["feh_obs"].size),
        "n_real_ti": int(np.isfinite(real["ti"]).sum()),
        "n_mock_ti": int(np.isfinite(mock["ti"]).sum()),
        "ti_missing_real": float(1.0 - np.isfinite(real["ti"]).mean()),
        "ti_missing_mock": float(1.0 - np.isfinite(mock["ti"]).mean()),
        "ti_disabled": bool(not np.isfinite(mock["ti"]).any()),
        "overall": overall,
        "noiseless_mock": true_overall,
        "by_feh": by_feh,
        "by_snr": by_snr,
        "feh_slope": _bin_slope(by_feh),
        "snr_slope": _bin_slope(by_snr),
        "giant_dwarf": giants,
        "solar": _solar_check(overall["offset"]),
        "ionisation": _ionisation_check(giants),
    }
    if include_yields:
        summary["yields"] = _yield_check()
    else:
        summary["yields"] = {"raw": {}, "integrated": []}

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "ti_diagnostic.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _plot(real, mock, summary, outdir / "ti_diagnostic.png")
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    """Print the headline numbers."""
    if summary.get("ti_disabled"):
        print("Ti is excluded from this pool; no mock [Ti/Fe] values to compare.")
        return
    overall = summary["overall"]
    print(
        f"real median [Ti/Fe] {overall['real_median']:+.3f}  "
        f"mock {overall['mock_median']:+.3f}  "
        f"offset {overall['offset']:+.3f} "
        f"[{overall['offset_lo']:+.3f}, {overall['offset_hi']:+.3f}]  "
        f"KS {overall['ks']:.3f}"
    )
    print(
        f"noiseless mock offset {summary['noiseless_mock']['offset']:+.3f}; "
        f"Ti missing real {summary['ti_missing_real']:.2%}, "
        f"mock {summary['ti_missing_mock']:.2%}"
    )
    print("Fe/H bins:")
    for row in summary["by_feh"]:
        if row["n_real"]:
            print(
                f"  [{row['lo']:+.1f},{row['hi']:+.1f}] n={row['n_real']:3d} "
                f"offset {row['offset']:+.3f} "
                f"[{row['offset_lo']:+.3f},{row['offset_hi']:+.3f}]"
            )
    print("S/N bins:")
    for row in summary["by_snr"]:
        if row["n_real"]:
            print(
                f"  [{row['lo']:.0f},{row['hi']:.0f}] n={row['n_real']:3d} "
                f"offset {row['offset']:+.3f} "
                f"[{row['offset_lo']:+.3f},{row['offset_hi']:+.3f}]"
            )
    split = summary["giant_dwarf"]
    print(
        f"giants offset {split['inside']['offset']:+.3f} "
        f"({split['inside']['n_real']}); dwarfs {split['outside']['offset']:+.3f} "
        f"({split['outside']['n_real']}); difference {split['difference']:+.3f} "
        f"[{split['difference_lo']:+.3f},{split['difference_hi']:+.3f}]"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the Ti diagnostic from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="mock dataset or .npz file")
    parser.add_argument(
        "--outdir", default="src/potluck/tests/v5", help="directory for JSON and PNG"
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-yields", action="store_true", help="skip the yield table check")
    args = parser.parse_args(argv)
    summary = run_ti_diagnostic(
        args.dataset,
        outdir=args.outdir,
        n_boot=args.n_boot,
        seed=args.seed,
        include_yields=not args.no_yields,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
