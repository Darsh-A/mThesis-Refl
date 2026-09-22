"""Feature-importance analysis for the trained f_pisn Random Forest.

Loads the fitted model (data/rf_model.joblib) and the cached dataset
(data/dataset.npz) and produces:

  1. Per-feature (pairwise [X/Y] ratio) Gini importances, ranked.
  2. Element-level importances (each feature's importance is attributed to
     both of its elements, then summed) so the "driving elements" are clear.
  3. Feature<->target correlation (Pearson) as an independent check of which
     ratios actually separate f_pisn=0 (SN-only) from f_pisn>0.

Plots are written to src/randomForest/diagnostic/.

Run:
    python src/randomForest/feature_importance.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.randomForest.evaluate_rf import load_model, CACHE_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR = os.path.join(HERE, "diagnostic")
os.makedirs(DIAG_DIR, exist_ok=True)


def _parse_pair(name: str) -> tuple[str, str]:
    """'[X/Y]' -> ('X', 'Y')."""
    body = name.strip().strip("[]")
    x, y = body.split("/")
    return x.strip(), y.strip()


def element_importances(feature_names, importances) -> dict[str, float]:
    """Attribute each feature's importance to its two elements and sum."""
    elem_imp: dict[str, float] = {}
    for name, imp in zip(feature_names, importances):
        x, y = _parse_pair(name)
        elem_imp[x] = elem_imp.get(x, 0.0) + imp
        elem_imp[y] = elem_imp.get(y, 0.0) + imp
    return elem_imp


def main():
    bundle = load_model()
    pipe = bundle["model"]
    rf = pipe.named_steps["rf"]
    names = list(bundle["feature_names"])
    importances = np.asarray(rf.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1]

    print(f"=== Feature importances ({len(names)} pairwise ratios) ===")
    print("  top 20:")
    for rank, i in enumerate(order[:20], 1):
        print(f"    {rank:>3}. {names[i]:<10} {importances[i]:.4f}")
    print("  bottom 20:")
    for rank, i in enumerate(order[-20:][::-1], 1):
        print(f"    {rank:>3}. {names[i]:<10} {importances[i]:.4f}")

    # --- Element-level aggregation ---
    elem_imp = element_importances(names, importances)
    elem_sorted = sorted(elem_imp.items(), key=lambda kv: kv[1], reverse=True)
    print("\n=== Element-level importance (summed over all ratios) ===")
    for rank, (el, imp) in enumerate(elem_sorted, 1):
        print(f"    {rank:>3}. {el:<3} {imp:.4f}")

    # --- Plots ---
    # 1. top features bar chart
    top_k = 30
    top_idx = order[:top_k]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(np.arange(top_k)[::-1], importances[top_idx][::-1], color="tab:blue")
    ax.set_yticks(np.arange(top_k)[::-1])
    ax.set_yticklabels([names[i] for i in top_idx][::-1], fontsize=8)
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {top_k} feature importances (f_pisn RF)")
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "feature_importance.png"))
    plt.close(fig)

    # 2. element importance bar chart
    els = [e for e, _ in elem_sorted]
    imps = [i for _, i in elem_sorted]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(np.arange(len(els)), imps, color="tab:green")
    ax.set_xticks(np.arange(len(els)))
    ax.set_xticklabels(els)
    ax.set_ylabel("summed Gini importance")
    ax.set_title("Element-level importance (f_pisn RF)")
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG_DIR, "element_importance.png"))
    plt.close(fig)

    # 3. feature-target correlation
    if os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        corrs = np.zeros(len(names))
        for j in range(len(names)):
            col = X[:, j]
            mask = np.isfinite(col)
            if mask.sum() < 2:
                corrs[j] = 0.0
                continue
            corrs[j] = np.corrcoef(col[mask], y[mask])[0, 1]
        corr_order = np.argsort(np.abs(corrs))[::-1]
        fig, ax = plt.subplots(figsize=(8, 9))
        k = 30
        ax.barh(np.arange(k)[::-1], corrs[corr_order[:k]][::-1], color="tab:orange")
        ax.set_yticks(np.arange(k)[::-1])
        ax.set_yticklabels([names[i] for i in corr_order[:k]][::-1], fontsize=8)
        ax.set_xlabel("Pearson corr(feature, f_pisn)")
        ax.set_title("Top feature-target correlations")
        fig.tight_layout()
        fig.savefig(os.path.join(DIAG_DIR, "feature_target_correlation.png"))
        plt.close(fig)

    print(f"\nsaved plots to {DIAG_DIR}")


if __name__ == "__main__":
    main()
