"""f_pisn vs Teff for test_v4_3, marking the >0.5 candidates.

Reads obs_li_all.csv (mask-aware QRF predictions), plots f_pisn (mask q50,
with 16-84% bars) against Teff, and highlights stars with f_pisn_mask_q50 > 0.5
(the candidate threshold used by make_yield_plots.py).
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_root(start: str) -> str:
    d = os.path.abspath(start)
    for _ in range(12):
        if os.path.exists(os.path.join(d, "params.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise RuntimeError("Could not locate repo root")


ROOT = _find_root(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v4_3")
CSV = os.path.join(RESULTS_DIR, "obs_li_all.csv")
THRESH = 0.5


def main():
    df = pd.read_csv(CSV, dtype={"star": str})
    df = df[np.isfinite(df["Teff"])].reset_index(drop=True)

    teff = df["Teff"].to_numpy(dtype=float)
    f50 = df["f_pisn_mask_q50"].to_numpy(dtype=float)
    f16 = df["f_pisn_mask_q16"].to_numpy(dtype=float)
    f84 = df["f_pisn_mask_q84"].to_numpy(dtype=float)
    cand = f50 > THRESH

    fig, ax = plt.subplots(figsize=(9, 6))

    # non-candidates
    ax.errorbar(teff[~cand], f50[~cand],
                yerr=[f50[~cand] - f16[~cand], f84[~cand] - f50[~cand]],
                fmt="o", ms=5, alpha=0.5, color="tab:blue",
                ecolor="tab:blue", elinewidth=0.6, capsize=0,
                label=f"f_pisn <= {THRESH} (N={int((~cand).sum())})")

    # candidates
    ax.errorbar(teff[cand], f50[cand],
                yerr=[f50[cand] - f16[cand], f84[cand] - f50[cand]],
                fmt="o", ms=9, color="tab:red", ecolor="tab:red",
                elinewidth=1.2, capsize=3,
                label=f"candidates f_pisn > {THRESH} (N={int(cand.sum())})")

    for x, y, s in zip(teff[cand], f50[cand], df["star"][cand]):
        ax.annotate(s, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="tab:red")

    ax.axhline(THRESH, color="k", ls="--", lw=1, label=f"threshold = {THRESH}")
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("predicted f_pisn (mask-aware q50)")
    ax.set_title("test_v4_3: f_pisn vs Teff — candidates above 0.5")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "fpisn_vs_teff.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")

    cand_rows = df[cand].sort_values("f_pisn_mask_q50", ascending=False)
    print("\ncandidates (f_pisn_mask_q50 > 0.5):")
    print(cand_rows[["star", "FeH", "Teff", "logg",
                     "f_pisn_mask_q50", "f_pisn_std_q50"]].to_string(index=False))


if __name__ == "__main__":
    main()
