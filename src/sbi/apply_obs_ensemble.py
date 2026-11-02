"""Ensemble 4-D NPE detection on Li+22 (test v1.6).

Trains N_ENSEMBLE 4-D NPEs (NSF) with DIFFERENT seeds, applies each to the
Li+22 stars, and averages P(f_pisn > 0.5) per star.  The spread across ensemble
members is the honest uncertainty in the PISN-dominated count -- a single NSF
run is unstable (unconverged + out-of-distribution stars), so a point count from
one model is meaningless.

v1.6: only COMPLETE stars (all elements measured, no imputation), and with Zn
and Co DROPPED from the feature set (the yield model underproduces them, so
their ratios are out-of-distribution for observed stars).  Feature set = 9
elements (C, Na, Mg, Si, Ca, Cr, Mn, Fe, Ni) -> 36 ratios.

Run:
    uv run python src/sbi/apply_obs_ensemble.py              # both yield models
    uv run python src/sbi/apply_obs_ensemble.py Nomoto13     # one yield model

Outputs (src/sbi/results/testv1.6/<sn_source>/):
    ensemble_summary.txt
    obs_fpisn_gt05_ensemble.csv
    obs_fpisn_gt05_hist.png
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

CACHES = {
    "Limongi18": "dataset_Limongi_tpop42.npz",
    "Nomoto13": "dataset_Nomoto_tpop42.npz",
}

# Zn and Co dropped: the yield model underproduces them by ~0.5-0.6 dex, so any
# ratio involving them is out-of-distribution for the observed Li+22 stars.
OBS_ELEMENTS = ["C", "Na", "Mg", "Si", "Ca", "Cr", "Mn", "Fe", "Ni"]
ELEMENT_ION = {
    "C": "C", "Na": "NaI", "Mg": "MgI", "Si": "SiI", "Ca": "CaI", "Cr": "CrI",
    "Mn": "MnI", "Fe": "FeI", "Ni": "NiI",
}
RATIO_NAMES = [f"[{x}/{y}]" for x, y in combinations(OBS_ELEMENTS, 2)]

PARAM_NAMES = ["f_pisn", "log10_f_ratio", "log10_tpop2", "M_pisn"]
PRIOR_BOUNDS = {"f_pisn": (0.0, 1.0), "log10_f_ratio": (-4.0, -1.0),
                "log10_tpop2": (6.5, 7.62), "M_pisn": (150.0, 270.0)}

N_ENSEMBLE = 5
MAX_EPOCHS = 400
N_SAMPLES = 1000
CHUNK = 50
THRESH = 0.5


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


def load_training(sn_source):
    cache = os.path.join(DATA_DIR, CACHES[sn_source])
    d = np.load(cache, allow_pickle=True)
    X_full = d["X"]
    y = d["y"]
    diag = d["diagnostics"].item()
    full_names = list(diag["feature_names"])
    idx = [full_names.index(n) for n in RATIO_NAMES]
    X = X_full[:, idx]

    f_pisn = y.astype(float)
    f_ratio = diag["f_ratio"].astype(float)
    tpop2 = diag["tpop2"].astype(float)
    m_pisn = diag["pisn_mass"].astype(float)
    mask = (f_pisn > 0.0) & np.isfinite(m_pisn)
    X = X[mask]
    f_pisn = f_pisn[mask]
    f_ratio = f_ratio[mask]
    tpop2 = tpop2[mask]
    m_pisn = m_pisn[mask]

    theta = np.column_stack([f_pisn, np.log10(f_ratio), np.log10(tpop2), m_pisn]).astype(np.float32)
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    return theta, Xs, scaler, imp


def train_posterior(theta, Xs, seed):
    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    torch.manual_seed(seed)
    np.random.seed(seed)

    low = torch.tensor([PRIOR_BOUNDS[k][0] for k in PARAM_NAMES], dtype=torch.float32)
    high = torch.tensor([PRIOR_BOUNDS[k][1] for k in PARAM_NAMES], dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)

    inference = NPE(prior=prior, density_estimator="nsf")
    inference.append_simulations(torch.from_numpy(theta), torch.from_numpy(Xs))
    de = inference.train(max_num_epochs=MAX_EPOCHS, training_batch_size=128,
                         show_train_summary=False)
    return inference.build_posterior(de)


def predict_gt05(posterior, X_obs_s):
    import torch

    p_gt05 = np.empty(X_obs_s.shape[0])
    for start in range(0, X_obs_s.shape[0], CHUNK):
        end = min(start + CHUNK, X_obs_s.shape[0])
        x_chunk = torch.from_numpy(X_obs_s[start:end])
        s = posterior.sample_batched(
            (N_SAMPLES,), x=x_chunk, reject_outside_prior=False
        ).detach().numpy()
        f_pisn_chunk = s[..., 0]
        p_gt05[start:end] = np.mean(f_pisn_chunk > THRESH, axis=0)
    return p_gt05


def run_ensemble(sn_source):
    import torch

    theta, Xs, scaler, imp = load_training(sn_source)

    stars = load_obs_li()
    X_obs, feh = build_obs_features(stars)
    complete = np.all(np.isfinite(X_obs), axis=1)
    stars = [s for s, c in zip(stars, complete) if c]
    X_obs = X_obs[complete]
    feh = feh[complete]
    X_obs = imp.transform(X_obs)  # no-op for complete stars
    X_obs_s = scaler.transform(X_obs).astype(np.float32)

    n_complete = len(stars)
    print(f"[{sn_source}] complete stars (all 11 elements): {n_complete}")

    p_gt05_members = np.empty((N_ENSEMBLE, len(stars)))
    for m in range(N_ENSEMBLE):
        print(f"[{sn_source}] training ensemble member {m + 1}/{N_ENSEMBLE} "
              f"(seed={m}) ...", flush=True)
        posterior = train_posterior(theta, Xs, seed=m)
        p_gt05_members[m] = predict_gt05(posterior, X_obs_s)
        del posterior

    mean_p = p_gt05_members.mean(axis=0)
    std_p = p_gt05_members.std(axis=0)

    out_dir = os.path.join(HERE, "results", "testv1.6", sn_source)
    os.makedirs(out_dir, exist_ok=True)

    lines = [f"Ensemble 4-D NPE detection, {sn_source} (N_ensemble={N_ENSEMBLE}, "
             f"epochs={MAX_EPOCHS})", ""]
    lines.append(f"Complete Li+22 stars (all 11 elements, no imputation): {n_complete}")
    lines.append("")
    for conf in [0.9, 0.68, 0.5]:
        counts = [(p_gt05_members[m] > conf).sum() for m in range(N_ENSEMBLE)]
        mean_n = (mean_p > conf).sum()
        lines.append(f"P(f_pisn>0.5) > {conf}: mean={mean_n}  "
                     f"member counts={counts}  (mean {np.mean(counts):.1f} +/- {np.std(counts):.1f})")

    order = np.argsort(-mean_p)
    lines.append("")
    lines.append("top-10 by ensemble-mean P(f_pisn>0.5):")
    for i in order[:10]:
        lines.append(f"  {stars[i]['label']:<12} FeH={feh[i]:6.2f}  "
                     f"P={mean_p[i]:.3f} +/- {std_p[i]:.3f}")

    with open(os.path.join(out_dir, "ensemble_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()

    # histogram of ensemble-mean P
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(mean_p, bins=30, range=(0, 1), color="tab:blue", alpha=0.8)
    ax.axvline(0.9, color="k", ls="--", lw=1, label="P = 0.9")
    ax.set_xlabel("ensemble-mean P(f_pisn > 0.5)")
    ax.set_ylabel("N stars")
    ax.set_title(f"{sn_source}: PISN-dominated probability (ensemble, N={len(stars)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "obs_fpisn_gt05_hist.png"), dpi=130)
    plt.close(fig)

    import csv
    with open(os.path.join(out_dir, "obs_fpisn_gt05_ensemble.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["star", "FeH", "P_fpisn_gt_0.5_mean", "P_fpisn_gt_0.5_std"])
        for s, p, sd, h in zip(stars, mean_p, std_p, feh):
            w.writerow([s["label"], f"{h:.3f}", f"{p:.4f}", f"{sd:.4f}"])


def main():
    sources = sys.argv[1:] if len(sys.argv) > 1 else list(CACHES.keys())
    for sn in sources:
        if sn not in CACHES:
            print(f"unknown source {sn!r}; choices: {list(CACHES)}")
            continue
        run_ensemble(sn)


if __name__ == "__main__":
    main()
