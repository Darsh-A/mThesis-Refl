"""Classifier diagnostics for the Li+ observation model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from ..io import load_dataset
from .ceiling import observed_matrix

SCORE_BINS = np.linspace(0.0, 1.0, 61)
NIII_LABELS = ("0", "1-3", "4-10", "11-30", "31-100", ">100")
NIII_EDGES = (-0.5, 0.5, 3.5, 10.5, 30.5, 100.5, np.inf)
FEH_EDGES = (-3.8, -3.5, -3.0, -2.5, -2.0, -1.7)


def oof_scores(
    arrays: dict[str, np.ndarray], folds: int = 3, seed: int = 0
) -> np.ndarray:
    """Return out-of-fold P(y=1|x) scores."""
    x = observed_matrix(arrays)
    y = arrays["y_f_05"].astype(int)
    weight = arrays["importance_weight"].astype(float)
    splits = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    ).split(x, y)
    probability = np.zeros(len(y))
    for train, test in splits:
        model = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
        )
        model.fit(x[train], y[train], sample_weight=weight[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    return probability


def _weighted_roc(
    y: np.ndarray, score: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(y, score, sample_weight=weight)


def _threshold_at_tpr(
    y: np.ndarray, score: np.ndarray, weight: np.ndarray, tpr: float
) -> tuple[float, float, float]:
    fpr, tpr_values, thresholds = _weighted_roc(y, score, weight)
    index = min(np.searchsorted(tpr_values, tpr), len(tpr_values) - 1)
    return float(thresholds[index]), float(tpr_values[index]), float(fpr[index])


def _threshold_at_fpr(
    y: np.ndarray, score: np.ndarray, weight: np.ndarray, fpr: float
) -> tuple[float, float, float]:
    fpr_values, tpr_values, thresholds = _weighted_roc(y, score, weight)
    index = min(np.searchsorted(fpr_values, fpr), len(fpr_values) - 1)
    return float(thresholds[index]), float(tpr_values[index]), float(fpr_values[index])


def _precision(tpr: float, fpr: float, prevalence: float) -> float:
    numerator = tpr * prevalence
    denominator = numerator + fpr * (1.0 - prevalence)
    return numerator / denominator if denominator > 0.0 else 0.0


def _weighted_quantile(
    values: np.ndarray, weight: np.ndarray, quantiles: tuple[float, ...]
) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weight = weight[order]
    cumulative = np.cumsum(weight)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, values)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _score_histogram(
    y: np.ndarray, score: np.ndarray, weight: np.ndarray, outdir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, colour in ((0, "tab:blue"), (1, "tab:orange")):
        selected = y == label
        counts, _ = np.histogram(
            score[selected], bins=SCORE_BINS, weights=weight[selected]
        )
        counts = counts / counts.sum()
        ax.step(
            SCORE_BINS[:-1],
            counts,
            where="post",
            color=colour,
            label=f"true y={label} (n={selected.sum():,})",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Classifier score  P(y = 1 | x)")
    ax.set_ylabel("Fraction of class per score bin")
    ax.set_title("Score distribution by true class (Li+ weights)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    path = outdir / "01_score_histogram.png"
    _save(fig, path)
    return path


def _roc_plot(
    y: np.ndarray, score: np.ndarray, weight: np.ndarray, outdir: Path
) -> tuple[Path, float]:
    fpr, tpr, _ = _weighted_roc(y, score, weight)
    auc = roc_auc_score(y, score, sample_weight=weight)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for ax in axes:
        ax.plot(fpr, tpr, color="tab:blue", lw=2, label=f"weighted ROC (AUC={auc:.4f})")
        ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="random")
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate (recall)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="lower right")
    axes[0].set_title("Full ROC")
    axes[1].set_title("Low-FPR region")
    axes[1].set_xlim(0.0, 0.05)
    axes[1].set_ylim(0.0, 1.0)
    for target in (0.001, 0.005, 0.01):
        cut, tpr_at, fpr_at = _threshold_at_fpr(y, score, weight, target)
        axes[1].plot(fpr_at, tpr_at, "o", ms=5, color="tab:red")
        axes[1].annotate(
            f"FPR={fpr_at:.3f}\nTPR={tpr_at:.3f}",
            (fpr_at, tpr_at),
            textcoords="offset points",
            xytext=(8, -18),
            fontsize=8,
        )
    fig.suptitle("Out-of-fold classifier ROC (100k mocks, Li+ weights)")
    path = outdir / "02_roc.png"
    _save(fig, path)
    return path, auc


def _precision_plot(
    y: np.ndarray, score: np.ndarray, weight: np.ndarray, outdir: Path
) -> Path:
    fpr, tpr, _ = _weighted_roc(y, score, weight)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for prevalence, colour in ((2.5e-3, "tab:green"), (1.0e-3, "tab:purple")):
        precision = np.array(
            [_precision(t, f, prevalence) for t, f in zip(tpr, fpr)]
        )
        ax.plot(tpr, precision, color=colour, lw=2, label=f"π={prevalence:g}")
        ax.axhline(
            prevalence,
            color=colour,
            ls=":",
            lw=1,
            alpha=0.7,
        )
    ax.set_xlabel("Recall (true-positive rate)")
    ax.set_ylabel("Precision at assumed prevalence")
    ax.set_title("Precision–recall trade-off for a rare class")
    for target in (0.001, 0.01):
        cut, tpr_at, fpr_at = _threshold_at_fpr(y, score, weight, target)
        precision_at = _precision(tpr_at, fpr_at, 2.5e-3)
        ax.plot(tpr_at, precision_at, "o", ms=5, color="tab:red")
        ax.annotate(
            f"FPR={target:g}\nTPR={tpr_at:.3f}\nprec={precision_at:.3f}",
            (tpr_at, precision_at),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=7,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.05,
        "Red points: operating points at FPR = 0.1% and 1%, π = 2.5e-3",
        transform=ax.transAxes,
        fontsize=7,
        color="0.35",
    )
    path = outdir / "03_precision_recall.png"
    _save(fig, path)
    return path


def _calibration_plot(
    y: np.ndarray, score: np.ndarray, weight: np.ndarray, outdir: Path
) -> tuple[Path, float]:
    edges = np.linspace(0.0, 1.0, 11)
    index = np.clip(np.digitize(score, edges) - 1, 0, len(edges) - 2)
    mean_predicted = []
    observed = []
    errors = []
    counts = []
    for bin_index in range(len(edges) - 1):
        selected = index == bin_index
        if not np.any(selected):
            continue
        w = weight[selected]
        total = w.sum()
        if total <= 0.0:
            continue
        predicted = np.average(score[selected], weights=w)
        actual = np.average(y[selected], weights=w)
        mean_predicted.append(predicted)
        observed.append(actual)
        counts.append(total)
        errors.append(np.sqrt(max(actual * (1.0 - actual), 1e-12) / total))
    mean_predicted = np.asarray(mean_predicted)
    observed = np.asarray(observed)
    errors = np.asarray(errors)
    counts = np.asarray(counts)
    brier = np.average((score - y) ** 2, weights=weight)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="perfect calibration")
    ax.errorbar(
        mean_predicted,
        observed,
        yerr=errors,
        fmt="o",
        color="tab:blue",
        ms=6,
        capsize=3,
        label="weighted bins",
    )
    for x, y_value, n in zip(mean_predicted, observed, counts):
        ax.annotate(
            f"{n:.0f}",
            (x, y_value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            color="0.35",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(f"Calibration (weighted Brier={brier:.4f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    path = outdir / "04_calibration.png"
    _save(fig, path)
    return path, brier


def _score_vs_fz(
    arrays: dict[str, np.ndarray], score: np.ndarray, outdir: Path
) -> Path:
    fz = arrays["f_zp"].astype(float)
    weight = arrays["importance_weight"].astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    hb = ax.hexbin(
        fz,
        score,
        C=weight,
        reduce_C_function=np.sum,
        gridsize=55,
        bins="log",
        cmap="viridis",
        mincnt=1,
    )
    edges = np.array([-0.01, 0.01, 0.25, 0.50, 0.75, 0.99, 1.01])
    centres = []
    medians = []
    lower = []
    upper = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (fz >= low) & (fz < high)
        if np.count_nonzero(selected) < 10:
            continue
        q = _weighted_quantile(
            score[selected], weight[selected], (0.25, 0.5, 0.75)
        )
        centres.append(0.5 * (low + high))
        medians.append(q[1])
        lower.append(q[1] - q[0])
        upper.append(q[2] - q[1])
    ax.errorbar(
        centres,
        medians,
        yerr=[lower, upper],
        fmt="o",
        color="white",
        mec="black",
        ms=5,
        capsize=3,
        label="weighted median and IQR",
    )
    ax.set_xlabel("True PISN metal fraction  f_Z")
    ax.set_ylabel("Classifier score")
    ax.set_title("Score vs true PISN contribution")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.0)
    fig.colorbar(hb, ax=ax, label="Weighted mock count")
    ax.legend(frameon=False, loc="upper left")
    path = outdir / "05_score_vs_fz.png"
    _save(fig, path)
    return path


def _strata_plot(
    arrays: dict[str, np.ndarray], score: np.ndarray, outdir: Path
) -> Path:
    weight = arrays["importance_weight"].astype(float)
    y = arrays["y_f_05"].astype(int)
    niii = arrays["n_iii"].astype(float)
    feh = arrays["feh_obs"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    fig.subplots_adjust(wspace=0.52, bottom=0.22)

    niii_index = np.digitize(niii, NIII_EDGES[1:-1])
    niii_centres = np.arange(len(NIII_LABELS))
    medians = []
    lower = []
    upper = []
    rates = []
    for i in range(len(NIII_LABELS)):
        selected = niii_index == i
        q = _weighted_quantile(
            score[selected], weight[selected], (0.25, 0.5, 0.75)
        )
        medians.append(q[1])
        lower.append(q[1] - q[0])
        upper.append(q[2] - q[1])
        rates.append(np.average(y[selected], weights=weight[selected]))
    axes[0].errorbar(
        niii_centres,
        medians,
        yerr=[lower, upper],
        fmt="o",
        color="tab:blue",
        capsize=3,
    )
    axes[0].set_xticks(niii_centres, NIII_LABELS)
    axes[0].set_xlabel("Pop III CCSN count  n_III")
    axes[0].set_ylabel("Classifier score (median, IQR)")
    axes[0].set_title("Score vs Pop III count")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    twin = axes[0].twinx()
    twin.plot(niii_centres, rates, "s--", color="tab:orange", ms=4, lw=1)
    twin.set_ylim(0.0, 1.0)
    twin.tick_params(axis="y", colors="tab:orange", labelsize=8)

    feh_index = np.digitize(feh, FEH_EDGES[1:-1])
    feh_centres = 0.5 * (np.asarray(FEH_EDGES[:-1]) + np.asarray(FEH_EDGES[1:]))
    medians = []
    lower = []
    upper = []
    rates = []
    for i in range(len(FEH_EDGES) - 1):
        selected = feh_index == i
        q = _weighted_quantile(
            score[selected], weight[selected], (0.25, 0.5, 0.75)
        )
        medians.append(q[1])
        lower.append(q[1] - q[0])
        upper.append(q[2] - q[1])
        rates.append(np.average(y[selected], weights=weight[selected]))
    axes[1].errorbar(
        feh_centres,
        medians,
        yerr=[lower, upper],
        fmt="o",
        color="tab:blue",
        capsize=3,
    )
    axes[1].set_xlabel("Observed [Fe/H]")
    axes[1].set_ylabel("Classifier score (median, IQR)")
    axes[1].set_title("Score vs observed metallicity")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    twin = axes[1].twinx()
    twin.plot(feh_centres, rates, "s--", color="tab:orange", ms=4, lw=1)
    twin.set_ylim(0.0, 1.0)
    twin.tick_params(axis="y", colors="tab:orange", labelsize=8)

    giant = arrays["is_giant"].astype(bool)
    labels = ("dwarf", "giant")
    positions = np.arange(2)
    medians = []
    lower = []
    upper = []
    rates = []
    for i, selected in enumerate((~giant, giant)):
        q = _weighted_quantile(
            score[selected], weight[selected], (0.25, 0.5, 0.75)
        )
        medians.append(q[1])
        lower.append(q[1] - q[0])
        upper.append(q[2] - q[1])
        rates.append(np.average(y[selected], weights=weight[selected]))
    axes[2].errorbar(
        positions,
        medians,
        yerr=[lower, upper],
        fmt="o",
        color="tab:blue",
        capsize=3,
    )
    axes[2].set_xticks(positions, labels)
    axes[2].set_ylabel("Classifier score (median, IQR)")
    axes[2].set_title("Score by evolutionary state")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(alpha=0.25)
    twin = axes[2].twinx()
    twin.plot(positions, rates, "s--", color="tab:orange", ms=4, lw=1)
    twin.set_ylim(0.0, 1.0)
    twin.tick_params(axis="y", colors="tab:orange", labelsize=8)

    fig.suptitle("Where the classifier score changes")
    fig.text(
        0.5,
        -0.02,
        "Blue: weighted median score and IQR (left axis). "
        "Orange dashed: positive fraction (right axis).",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    path = outdir / "06_strata.png"
    _save(fig, path)
    return path


def _confusion_plot(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    thresholds: dict[str, float],
    outdir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    fig.subplots_adjust(wspace=0.42)
    for ax, (name, threshold) in zip(axes, thresholds.items()):
        predicted = score >= threshold
        matrix = np.zeros((2, 2))
        for true_value in (0, 1):
            for pred_value in (0, 1):
                selected = (y == true_value) & (predicted == pred_value)
                matrix[true_value, pred_value] = weight[selected].sum()
        row = matrix.sum(axis=1, keepdims=True)
        fraction = np.divide(
            matrix, row, out=np.zeros_like(matrix), where=row > 0
        )
        image = ax.imshow(fraction, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{fraction[i, j]:.3f}\n({matrix[i, j]:,.0f})",
                    ha="center",
                    va="center",
                    color="black" if fraction[i, j] < 0.6 else "white",
                    fontsize=9,
                )
        ax.set_xticks((0, 1), ("predicted\nnegative", "predicted\npositive"))
        ax.set_yticks((0, 1), ("true\nnegative", "true\npositive"))
        ax.set_title(f"{name}  (cut={threshold:.3f})")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Weighted confusion matrices (positive = PISN-dominated, f_Z >= 0.5)\n"
        "Color and annotations: row fraction (weighted count)"
    )
    path = outdir / "07_confusion.png"
    _save(fig, path)
    return path


def make_plots(dataset: str | Path, outdir: str | Path) -> dict[str, str]:
    """Train OOF scores and write the classifier diagnostic figures."""
    arrays, _ = load_dataset(dataset)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    y = arrays["y_f_05"].astype(int)
    weight = arrays["importance_weight"].astype(float)
    score = oof_scores(arrays)

    paths = {}
    paths["score_histogram"] = str(
        _score_histogram(y, score, weight, outdir)
    )
    roc_path, auc = _roc_plot(y, score, weight, outdir)
    paths["roc"] = str(roc_path)
    paths["precision_recall"] = str(
        _precision_plot(y, score, weight, outdir)
    )
    calibration_path, brier = _calibration_plot(y, score, weight, outdir)
    paths["calibration"] = str(calibration_path)
    paths["score_vs_fz"] = str(_score_vs_fz(arrays, score, outdir))
    paths["strata"] = str(_strata_plot(arrays, score, outdir))

    cut_tpr, tpr_at, fpr_at = _threshold_at_tpr(y, score, weight, 0.9)
    cut_fpr, tpr_low, fpr_low = _threshold_at_fpr(y, score, weight, 0.001)
    paths["confusion"] = str(
        _confusion_plot(
            y,
            score,
            weight,
            {"TPR = 0.9": cut_tpr, "FPR = 0.001": cut_fpr},
            outdir,
        )
    )

    report = {
        "n": int(len(y)),
        "effective_sample_size": float(
            weight.sum() ** 2 / np.square(weight).sum()
        ),
        "positive_rate": float(np.average(y, weights=weight)),
        "auc": float(auc),
        "brier": float(brier),
        "threshold_tpr_0.9": cut_tpr,
        "tpr_at_threshold_tpr_0.9": tpr_at,
        "fpr_at_threshold_tpr_0.9": fpr_at,
        "precision_at_tpr_0.9_pi_2.5e-3": _precision(tpr_at, fpr_at, 2.5e-3),
        "threshold_fpr_0.001": cut_fpr,
        "tpr_at_fpr_0.001": tpr_low,
        "fpr_at_fpr_0.001": fpr_low,
        "precision_at_fpr_0.001_pi_2.5e-3": _precision(tpr_low, fpr_low, 2.5e-3),
        "score_quantiles": {
            "positive": _weighted_quantile(
                score[y == 1], weight[y == 1], (0.05, 0.25, 0.5, 0.75, 0.95)
            ).tolist(),
            "negative": _weighted_quantile(
                score[y == 0], weight[y == 0], (0.05, 0.25, 0.5, 0.75, 0.95)
            ).tolist(),
        },
    }
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2))
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("outdir")
    arguments = parser.parse_args()
    print(json.dumps(make_plots(arguments.dataset, arguments.outdir), indent=2))
