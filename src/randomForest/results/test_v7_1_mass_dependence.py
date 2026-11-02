"""Map the PISN mass dependence of the abundance-ratio signal (test_v7.1 helper).

The Takahashi PISN yields change character with initial mass; this maps where
that change happens so the mass-gated review can be split at the *real*
transition instead of an assumed 180 Msun.

For each Takahashi NR grid mass (180..280) the PISN-only [X/Fe] is computed for
the 13 non-Fe elements.  The per-interval change in the full [X/Fe] vector is
then measured; the interval with the largest change locates the dominant jump,
and its midpoint is the recommended split.

Outputs (src/randomForest/results/test_v7_1/):
  * mass_dependence.png  [X/Fe] vs mass (alpha vs iron-peak panels) + jump-magnitude bar
"""

import os
import sys

import numpy as np
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
    raise RuntimeError("Could not locate repo root (params.py not found above cwd)")


ROOT = _find_root(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.load_data import load_takahashi  # noqa: E402
from src.salvadori_funcs import salvadori_yields, salvadori_abundance_ratio  # noqa: E402

OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v7_1")
os.makedirs(OUT_DIR, exist_ok=True)

ELEMENTS = ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
NON_FE = [e for e in ELEMENTS if e != "Fe"]
# grouping for readable panels (very different dynamic ranges)
ALPHA = ["C", "O", "Na", "Mg", "Al", "Si", "Ca"]
IRONPEAK = ["Cr", "Mn", "Co", "Ni", "Cu", "Zn"]


def grid_xfe():
    entries = load_takahashi()["NR"]
    masses = np.array([e["params"]["mass"] for e in entries], float)
    vals = np.full((len(masses), len(NON_FE)), np.nan)
    for i, e in enumerate(entries):
        for k, x in enumerate(NON_FE):
            vals[i, k] = salvadori_abundance_ratio(x, "Fe", e)
    return masses, vals


def find_split(masses, vals):
    """Largest per-interval change in the [X/Fe] vector -> split at its midpoint."""
    dL2 = np.array([np.linalg.norm(vals[i + 1] - vals[i]) for i in range(len(masses) - 1)])
    j = int(np.argmax(dL2))
    split = float((masses[j] + masses[j + 1]) / 2.0)
    return split, j, dL2


def plot(masses, vals, split, j, dL2):
    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)

    for ax, grp in [(axes[0], ALPHA), (axes[1], IRONPEAK)]:
        for k, x in enumerate(NON_FE):
            if x not in grp:
                continue
            ax.plot(masses, vals[:, k], "o-", lw=1.5, label=x)
        ax.set_ylabel("[X/Fe] (PISN only)")
        ax.set_title("PISN [X/Fe] vs initial mass — " +
                     ("alpha / light elements" if grp is ALPHA else "iron-peak / heavy elements"))
        ax.axvspan(200, 220, color="tab:orange", alpha=0.12, label="alpha jump (200-220)")
        ax.axvspan(220, 240, color="tab:red", alpha=0.15, label="iron-peak jump (220-240)")
        ax.axvline(split, color="k", ls="--", lw=1, label=f"split = {split:.0f} Msun")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.25)

    mids = [(masses[i] + masses[i + 1]) / 2 for i in range(len(masses) - 1)]
    ax = axes[2]
    colors = ["tab:red" if i == j else "tab:blue" for i in range(len(dL2))]
    ax.bar(mids, dL2, width=14, color=colors, alpha=0.85)
    ax.set_ylabel("||d[X/Fe]|| between grid masses")
    ax.set_xlabel("PISN initial mass (Msun)")
    ax.set_title("Magnitude of the [X/Fe] change per mass interval")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "mass_dependence.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def main():
    masses, vals = grid_xfe()
    split, j, dL2 = find_split(masses, vals)

    print("Takahashi NR grid masses:", masses.tolist())
    print("per-interval ||d[X/Fe]||:")
    for i in range(len(masses) - 1):
        mark = "  <-- dominant" if i == j else ""
        print(f"  {masses[i]:.0f}->{masses[i+1]:.0f}  {dL2[i]:6.2f}{mark}")
    print(f"\ndetermined split = {split:.0f} Msun "
          f"(midpoint of {masses[j]:.0f}->{masses[j+1]:.0f})")

    # per-element largest jump (which interval)
    print("\nper-element largest jump interval:")
    for k, x in enumerate(NON_FE):
        jumps = np.abs(np.diff(vals[:, k]))
        jj = int(np.argmax(jumps))
        print(f"  {x:<4} {masses[jj]:.0f}->{masses[jj+1]:.0f}  ({jumps[jj]:+.2f} dex)")

    plot(masses, vals, split, j, dL2)


if __name__ == "__main__":
    main()
