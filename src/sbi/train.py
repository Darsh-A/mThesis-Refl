"""Train a Neural Posterior Estimator (NPE) for P(f_pisn | abundance ratios).

test v1.0: infer the single scalar f_pisn from the m=105 pairwise [X/Y]
abundance ratios, using ``sbi`` (Simulation-Based Inference) with a Masked
Autoregressive Flow density estimator conditioned on the observation.

Run:
    uv run python src/sbi/train.py

Outputs (src/sbi/results/testv1.0/):
    npe_posterior.pkl      trained posterior (pickled)
    train_artifacts.npz    scaler, splits, held-out data (for evaluate.py)
    train_summary.txt      configuration + training time
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.sbi.data import load_theta_x  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "testv1.0")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SIZE = 0.2
SEED = 0
DENSITY_ESTIMATOR = "maf"
MAX_EPOCHS = 200
BATCH_SIZE = 128


def main():
    theta, x, feature_names = load_theta_x()
    print(
        f"loaded theta={theta.shape} x={x.shape}  "
        f"f_pisn range=[{theta.min():.3f}, {theta.max():.3f}]"
    )

    x_tr, x_te, th_tr, th_te = train_test_split(
        x, theta, test_size=TEST_SIZE, random_state=SEED
    )

    # Standardize observations on the training split only.
    scaler = StandardScaler().fit(x_tr)
    x_tr = scaler.transform(x_tr).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    th_tr = th_tr.astype(np.float32)
    th_te = th_te.astype(np.float32)

    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    prior = BoxUniform(
        low=torch.tensor([0.0], dtype=torch.float32),
        high=torch.tensor([1.0], dtype=torch.float32),
    )

    inference = NPE(prior=prior, density_estimator=DENSITY_ESTIMATOR)
    inference.append_simulations(
        torch.from_numpy(th_tr), torch.from_numpy(x_tr)
    )

    t0 = time.time()
    density_estimator = inference.train(
        max_num_epochs=MAX_EPOCHS,
        training_batch_size=BATCH_SIZE,
        show_train_summary=True,
    )
    train_time = time.time() - t0

    posterior = inference.build_posterior(density_estimator)

    import joblib

    joblib.dump(posterior, os.path.join(OUT_DIR, "npe_posterior.pkl"))

    np.savez_compressed(
        os.path.join(OUT_DIR, "train_artifacts.npz"),
        x_te=x_te,
        th_te=th_te,
        x_tr=x_tr,
        th_tr=th_tr,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        feature_names=np.array(feature_names, dtype=object),
    )

    with open(os.path.join(OUT_DIR, "train_summary.txt"), "w") as f:
        f.write(f"density_estimator={DENSITY_ESTIMATOR}\n")
        f.write(f"n_train={len(th_tr)} n_test={len(th_te)}\n")
        f.write(f"max_epochs={MAX_EPOCHS} batch_size={BATCH_SIZE}\n")
        f.write(f"train_time_s={train_time:.1f}\n")

    print(f"saved posterior + artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
