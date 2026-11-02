"""Train a multi-parameter Neural Posterior Estimator (test v1.1).

Infers the 4-D parameter vector
    theta = (f_pisn, log10 f_ratio, log10 tpop2, M_pisn)
from the m=105 pairwise [X/Y] abundance ratios, optionally with realistic
observational noise added to the ratios (see ``data.load_theta_x_multi``).

Run:
    uv run python src/sbi/train_multi.py

Outputs (src/sbi/results/testv1.1/):
    npe_posterior.pkl      trained posterior (pickled)
    train_artifacts.npz    scaler, splits, held-out data (for evaluate_multi.py)
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

from src.sbi.data import PARAM_NAMES, PRIOR_BOUNDS, load_theta_x_multi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "testv1.1")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SIZE = 0.2
SEED = 0
ADD_NOISE = True
NOISE_SEED = 42
DENSITY_ESTIMATOR = "nsf"
MAX_EPOCHS = 300
BATCH_SIZE = 128


def main():
    theta, x, feature_names = load_theta_x_multi(
        add_noise=ADD_NOISE, noise_seed=NOISE_SEED
    )
    print(
        f"loaded theta={theta.shape} x={x.shape}  params={PARAM_NAMES}  "
        f"add_noise={ADD_NOISE}"
    )

    x_tr, x_te, th_tr, th_te = train_test_split(
        x, theta, test_size=TEST_SIZE, random_state=SEED
    )

    scaler = StandardScaler().fit(x_tr)
    x_tr = scaler.transform(x_tr).astype(np.float32)
    x_te = scaler.transform(x_te).astype(np.float32)
    th_tr = th_tr.astype(np.float32)
    th_te = th_te.astype(np.float32)

    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    low = torch.tensor([PRIOR_BOUNDS[k][0] for k in PARAM_NAMES], dtype=torch.float32)
    high = torch.tensor([PRIOR_BOUNDS[k][1] for k in PARAM_NAMES], dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)

    inference = NPE(prior=prior, density_estimator=DENSITY_ESTIMATOR)
    inference.append_simulations(torch.from_numpy(th_tr), torch.from_numpy(x_tr))

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
        f.write(f"params={PARAM_NAMES}\n")
        f.write(f"density_estimator={DENSITY_ESTIMATOR}\n")
        f.write(f"add_noise={ADD_NOISE} noise_seed={NOISE_SEED}\n")
        f.write(f"n_train={len(th_tr)} n_test={len(th_te)}\n")
        f.write(f"max_epochs={MAX_EPOCHS} batch_size={BATCH_SIZE}\n")
        f.write(f"train_time_s={train_time:.1f}\n")

    print(f"saved posterior + artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
