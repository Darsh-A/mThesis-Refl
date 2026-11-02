"""4-D NPE applied to Li+22, detection framing (PISN-dominated = f_pisn > 0.5).

Trains a 4-D NPE (f_pisn, log10 f_ratio, log10 tpop2, M_pisn) with a Neural
Spline Flow (bounded prior, so no out-of-bounds leakage) on the Li+22-observable
55-ratio subset, then for each observed star computes P(f_pisn > 0.5) -- the
posterior probability that the star is PISN-dominated.

Outputs (src/sbi/results/obs/):
    obs_fpisn_gt05.csv
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

SN_SOURCE = sys.argv[1] if len(sys.argv) > 1 else "Limongi18"
CACHES = {
    "Limongi18": "dataset_TakaLimon_realisticNoise_v4_3.npz",
    "Nomoto13": "dataset_Nomotov5.npz",
}
CACHE = os.path.join(DATA_DIR, CACHES[SN_SOURCE])

OUT_DIR = os.path.join(HERE, "results", "testv1.4", SN_SOURCE)
os.makedirs(OUT_DIR, exist_ok=True)

OBS_ELEMENTS = ["C", "Na", "Mg", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Zn"]
ELEMENT_ION = {
    "C": "C", "Na": "NaI", "Mg": "MgI", "Si": "SiI", "Ca": "CaI", "Cr": "CrI",
    "Mn": "MnI", "Fe": "FeI", "Co": "CoI", "Ni": "NiI", "Zn": "ZnI",
}
RATIO_NAMES = [f"[{x}/{y}]" for x, y in combinations(OBS_ELEMENTS, 2)]

PARAM_NAMES = ["f_pisn", "log10_f_ratio", "log10_tpop2", "M_pisn"]
PRIOR_BOUNDS = {"f_pisn": (0.0, 1.0), "log10_f_ratio": (-4.0, -1.0),
                "log10_tpop2": (6.5, 7.25), "M_pisn": (150.0, 270.0)}
N_SAMPLES = 1000
CHUNK = 50
THRESH = 0.5
SEED = 0


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
    # ---- training data: 4-D theta + 55-ratio x ----
    d = np.load(CACHE, allow_pickle=True)
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
    X = SimpleImputer(strategy="median").fit_transform(X)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    low = torch.tensor([PRIOR_BOUNDS[k][0] for k in PARAM_NAMES], dtype=torch.float32)
    high = torch.tensor([PRIOR_BOUNDS[k][1] for k in PARAM_NAMES], dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)

    inference = NPE(prior=prior, density_estimator="nsf")
    inference.append_simulations(torch.from_numpy(theta), torch.from_numpy(Xs))
    de = inference.train(max_num_epochs=200, training_batch_size=128,
                         show_train_summary=False)
    posterior = inference.build_posterior(de)

    # ---- observed stars ----
    stars = load_obs_li()
    X_obs, feh = build_obs_features(stars)
    imp = SimpleImputer(strategy="median").fit(X)
    X_obs = imp.transform(X_obs)
    X_obs_s = scaler.transform(X_obs).astype(np.float32)

    n_complete = int(np.sum(np.all(np.isfinite(build_obs_features(stars)[0]), axis=1)))
    print(f"Li+22 stars: {len(stars)}  (with all 11 elements: {n_complete})")

    # sample in chunks to bound memory (NSF sampling of 385 stars at once OOMs)
    p_gt05 = np.empty(len(stars))
    med = np.empty(len(stars))
    for start in range(0, len(stars), CHUNK):
        end = min(start + CHUNK, len(stars))
        x_chunk = torch.from_numpy(X_obs_s[start:end])
        s = posterior.sample_batched(
            (N_SAMPLES,), x=x_chunk, reject_outside_prior=False
        ).detach().numpy()
        f_pisn_chunk = s[..., 0]  # (N, chunk)
        p_gt05[start:end] = np.mean(f_pisn_chunk > THRESH, axis=0)
        med[start:end] = np.median(f_pisn_chunk, axis=0)

    for conf, label in [(0.9, "high-confidence"), (0.68, "1-sigma"), (0.5, "better-than-even")]:
        n = int((p_gt05 > conf).sum())
        print(f"  P(f_pisn>0.5) > {conf}: {n} stars")

    order = np.argsort(-p_gt05)
    print(f"\n  top-10 by P(f_pisn>0.5):")
    for i in order[:10]:
        print(f"    {stars[i]['label']:<12} FeH={feh[i]:6.2f}  "
              f"P(>0.5)={p_gt05[i]:.3f}  median={med[i]:.3f}")

    # ---- histogram of P(f_pisn > 0.5) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(p_gt05, bins=30, range=(0, 1), color="tab:blue", alpha=0.8)
    ax.axvline(0.9, color="k", ls="--", lw=1, label="P = 0.9")
    ax.set_xlabel("P(f_pisn > 0.5)")
    ax.set_ylabel("N stars")
    ax.set_title(f"Li+22: PISN-dominated probability (N={len(stars)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "obs_fpisn_gt05_hist.png"), dpi=130)
    plt.close(fig)

    # ---- CSV ----
    import csv
    with open(os.path.join(OUT_DIR, "obs_fpisn_gt05.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["star", "FeH", "P_fpisn_gt_0.5", "f_pisn_median"])
        for s, m, p, h in zip(stars, med, p_gt05, feh):
            w.writerow([s["label"], f"{h:.3f}", f"{p:.4f}", f"{m:.4f}"])

    print(f"\nsaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
