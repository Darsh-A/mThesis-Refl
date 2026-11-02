"""Apply the NPE to the observed Li+22 catalogue and histogram f_pisn.

Trains a 1-D f_pisn NPE on the Li+22-observable feature subset (11 elements ->
55 [X/Y] ratios, from the realistic-noise Takahashi+Limongi18 dataset), then
predicts f_pisn for every Li+22 star.  Sanity check: if the model is behaving,
most very metal-poor stars should have f_pisn near 0 (few PISN-dominated).

Outputs (src/sbi/results/obs/):
    obs_fpisn_hist.png
    obs_fpisn.csv
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.load_data import load_obs_li  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "randomForest", "data")
OUT_DIR = os.path.join(HERE, "results", "obs")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE = os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise_v4_3.npz")

OBS_ELEMENTS = ["C", "Na", "Mg", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Zn"]
ELEMENT_ION = {
    "C": "C", "Na": "NaI", "Mg": "MgI", "Si": "SiI", "Ca": "CaI", "Cr": "CrI",
    "Mn": "MnI", "Fe": "FeI", "Co": "CoI", "Ni": "NiI", "Zn": "ZnI",
}
RATIO_NAMES = [f"[{x}/{y}]" for x, y in combinations(OBS_ELEMENTS, 2)]

N_SAMPLES = 1000


def get_xfe(star, elem):
    if elem == "Fe":
        return 0.0
    rec = star["abundances"].get(ELEMENT_ION[elem])
    if rec is None or rec.get("l_XFe") == "<":
        return np.nan
    xfe = rec.get("XFe")
    return np.nan if xfe is None else float(xfe)


def build_obs_features(stars):
    X = np.full((len(stars), len(RATIO_NAMES)), np.nan)
    feh = np.full(len(stars), np.nan)
    for i, star in enumerate(stars):
        xfe = {e: get_xfe(star, e) for e in OBS_ELEMENTS}
        for j, (x, y) in enumerate(combinations(OBS_ELEMENTS, 2)):
            if not (np.isnan(xfe[x]) or np.isnan(xfe[y])):
                X[i, j] = xfe[x] - xfe[y]
        feh[i] = star.get("params", {}).get("FeH", np.nan)
    return X, feh


def main():
    # ---- training data: 55-feature Li+22 subset ----
    d = np.load(CACHE, allow_pickle=True)
    X_full = d["X"]
    y = d["y"]
    diag = d["diagnostics"].item()
    full_names = list(diag["feature_names"])
    idx = [full_names.index(n) for n in RATIO_NAMES]
    X = X_full[:, idx]
    X = SimpleImputer(strategy="median").fit_transform(X)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))
    inference = NPE(prior=prior, density_estimator="mdn")
    inference.append_simulations(
        torch.from_numpy(y.astype(np.float32).reshape(-1, 1)),
        torch.from_numpy(Xs),
    )
    de = inference.train(max_num_epochs=400, training_batch_size=128,
                         show_train_summary=False)
    posterior = inference.build_posterior(de)

    # ---- observed stars ----
    stars = load_obs_li()
    X_obs, feh = build_obs_features(stars)
    imp = SimpleImputer(strategy="median").fit(X)
    X_obs = imp.transform(X_obs)
    X_obs_s = scaler.transform(X_obs).astype(np.float32)

    n_complete = int(np.sum(np.all(np.isfinite(build_obs_features(stars)[0]), axis=1)))
    print(f"Li+22 stars: {len(stars)}  (with all 11 elements measured: {n_complete})")

    samples = posterior.sample_batched(
        (N_SAMPLES,), x=torch.from_numpy(X_obs_s), reject_outside_prior=False
    ).detach().numpy()
    samples = samples.reshape(N_SAMPLES, len(stars))
    med = np.clip(np.median(samples, axis=0), 0.0, 1.0)

    print(f"\n=== predicted f_pisn (posterior median) over {len(stars)} Li+22 stars ===")
    print(f"  median = {np.median(med):.3f}   mean = {med.mean():.3f}")
    print(f"  frac f_pisn > 0.5 : {(med > 0.5).mean():.3f}  (n={(med > 0.5).sum()})")
    print(f"  frac f_pisn > 0.1 : {(med > 0.1).mean():.3f}  (n={(med > 0.1).sum()})")
    print(f"  frac f_pisn < 0.05: {(med < 0.05).mean():.3f}")
    print(f"  max f_pisn = {med.max():.3f}")

    # ---- histogram ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(med, bins=30, range=(0, 1), color="tab:blue", alpha=0.8)
    ax.axvline(0.5, color="k", ls="--", lw=1, label="f_pisn = 0.5")
    ax.set_xlabel("predicted f_pisn (posterior median)")
    ax.set_ylabel("N stars")
    ax.set_title(f"Li+22: NPE f_pisn (N={len(stars)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "obs_fpisn_hist.png"), dpi=130)
    plt.close(fig)

    # ---- CSV ----
    import csv
    with open(os.path.join(OUT_DIR, "obs_fpisn.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["star", "FeH", "f_pisn_median"])
        for s, m, h in zip(stars, med, feh):
            w.writerow([s["label"], f"{h:.3f}", f"{m:.4f}"])

    print(f"\nsaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
