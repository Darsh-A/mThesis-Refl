"""Cross-yield generalization of the NPE (test v1.2).

Does the learned abundance-ratio -> f_pisn posterior transfer across SNII yield
models?  Trains a 1-D NPE (f_pisn) on one source's synthetic data and evaluates
it on the OTHER source (ground-truth f_pisn still known, but the ratios come
from a different SNII yield grid).

Cells (mirroring the randomForest test_v6 matrix):
    train Limongi18 -> test Limongi18  (within, held-out)
    train Limongi18 -> test Nomoto13   (cross)
    train Nomoto13  -> test Nomoto13   (within, held-out)
    train Nomoto13  -> test Limongi18  (cross)

Datasets (already cached, no regeneration):
    Limongi18 -> dataset_TakaLimon_realisticNoise_v4_3.npz
    Nomoto13  -> dataset_Nomotov5.npz
Both share the identical 91-feature [X/Y] space (14 elements).

Outputs (src/sbi/results/testv1.2/):
    cross_yield_metrics.txt
    cross_yield_scatter.png   (2x2 transfer matrix of pred vs true)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import r2_score, mean_squared_error  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "randomForest", "data")
OUT_DIR = os.path.join(HERE, "results", "testv1.2")
os.makedirs(OUT_DIR, exist_ok=True)

SOURCES = {
    "Limongi18": os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise_v4_3.npz"),
    "Nomoto13": os.path.join(DATA_DIR, "dataset_Nomotov5.npz"),
}

SEED = 0
TEST_SIZE = 0.2
DENSITY_ESTIMATOR = "mdn"
MAX_EPOCHS = 400
BATCH_SIZE = 128
N_POSTERIOR_SAMPLES = 1000


def load_fpisn(cache_path: str):
    """Return (theta, x) for 1-D f_pisn inference, dropping pure-SN rows."""
    d = np.load(cache_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=float)
    f_pisn = np.asarray(d["y"], dtype=float)

    mask = np.isfinite(f_pisn) & (f_pisn > 0.0)
    X = X[mask]
    f_pisn = f_pisn[mask]

    X = SimpleImputer(strategy="median").fit_transform(X)
    return f_pisn.reshape(-1, 1).astype(np.float32), X.astype(np.float32)


def fit_npe(theta: np.ndarray, x: np.ndarray):
    """Fit a 1-D NPE for f_pisn and return (posterior, scaler)."""
    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)

    prior = BoxUniform(
        low=torch.tensor([0.0], dtype=torch.float32),
        high=torch.tensor([1.0], dtype=torch.float32),
    )
    inference = NPE(prior=prior, density_estimator=DENSITY_ESTIMATOR)
    inference.append_simulations(torch.from_numpy(theta), torch.from_numpy(xs))
    density_estimator = inference.train(
        max_num_epochs=MAX_EPOCHS,
        training_batch_size=BATCH_SIZE,
        show_train_summary=False,
    )
    return inference.build_posterior(density_estimator), scaler


def predict_median(posterior, scaler, x: np.ndarray) -> np.ndarray:
    import torch

    xs = scaler.transform(x).astype(np.float32)
    samples = posterior.sample_batched(
        (N_POSTERIOR_SAMPLES,),
        x=torch.from_numpy(xs),
        reject_outside_prior=False,
    ).detach().numpy()
    samples = np.clip(samples.reshape(N_POSTERIOR_SAMPLES, x.shape[0]), 0.0, 1.0)
    return np.median(samples, axis=0)


def evaluate(y_true, y_pred):
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rho, _ = spearmanr(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "rho": rho,
            "bias": float(np.mean(y_pred - y_true))}


def main():
    data = {name: load_fpisn(path) for name, path in SOURCES.items()}

    # held-out split per source (identical to the randomForest pipeline)
    splits = {}
    for name, (theta, x) in data.items():
        x_tr, x_te, th_tr, th_te = train_test_split(
            x, theta, test_size=TEST_SIZE, random_state=SEED
        )
        splits[name] = (x_tr, x_te, th_tr, th_te)

    results = {}
    preds = {}

    # 2 fits: one per source on its 80% train split, reused for within + cross.
    for train_src in SOURCES:
        x_tr, x_te, th_tr, th_te = splits[train_src]
        post, scaler = fit_npe(th_tr, x_tr)

        # within (held-out)
        key = f"train_{train_src}_test_{train_src}"
        y_true = th_te.reshape(-1)
        y_pred = predict_median(post, scaler, x_te)
        results[key] = evaluate(y_true, y_pred)
        preds[key] = (y_true, y_pred)

        # cross -> the other source (full)
        test_src = "Nomoto13" if train_src == "Limongi18" else "Limongi18"
        th_other, x_other = data[test_src]
        key = f"train_{train_src}_test_{test_src}"
        y_true = th_other.reshape(-1)
        y_pred = predict_median(post, scaler, x_other)
        results[key] = evaluate(y_true, y_pred)
        preds[key] = (y_true, y_pred)

    print("=== Cross-yield NPE generalization (test v1.2) ===")
    print(f"{'cell':<30}{'R2':>9}{'RMSE':>9}{'rho':>8}{'bias':>8}")
    for k, m in results.items():
        print(f"{k:<30}{m['r2']:>9.4f}{m['rmse']:>9.4f}"
              f"{m['rho']:>+8.3f}{m['bias']:>+8.3f}")

    # ---- 2x2 transfer scatter ----
    plot_cells = [
        ("train Limongi18 -> test Limongi18\n(within, held-out)",
         "train_Limongi18_test_Limongi18"),
        ("train Limongi18 -> test Nomoto13\n(cross)",
         "train_Limongi18_test_Nomoto13"),
        ("train Nomoto13 -> test Nomoto13\n(within, held-out)",
         "train_Nomoto13_test_Nomoto13"),
        ("train Nomoto13 -> test Limongi18\n(cross)",
         "train_Nomoto13_test_Limongi18"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (title, key) in zip(axes.ravel(), plot_cells):
        y_true, y_pred = preds[key]
        ax.scatter(y_true, y_pred, s=8, alpha=0.35, color="tab:blue")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("true f_pisn")
        ax.set_ylabel("predicted f_pisn")
        ax.set_title(f"{title}\nR2 = {results[key]['r2']:.4f}", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle("Cross-yield NPE: pred vs true f_pisn", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "cross_yield_scatter.png"), dpi=130)
    plt.close(fig)

    lines = ["Cross-yield NPE generalization (test v1.2)", ""]
    for k, m in results.items():
        lines.append(f"{k}: R2={m['r2']:.4f}  RMSE={m['rmse']:.4f}  "
                     f"rho={m['rho']:+.3f}  bias={m['bias']:+.3f}")
    with open(os.path.join(OUT_DIR, "cross_yield_metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\ndone. outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
