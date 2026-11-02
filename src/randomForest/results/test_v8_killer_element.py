"""Forward-model "killer element" analysis (test_v8).

Directly answers *which elements carry the PISN signature* from the forward
model (not from RF importance).  RF Gini/permutation importance only shows what
the model happens to use given feature redundancy; the physics question needs
the abundance ratios themselves.

Specifically, for the killing elements Cu and Zn (produced by high-mass PISN but
not by CCSN), this computes:

  1. True/false-positive rates for the [Zn/Fe] < -1 (and [Cu/Fe] < -1)
     diagnostic, with truth = "PISN-dominated" (f_pisn > 0.5) vs "CCSN-only"
     (f_pisn = 0), overall and per mass regime (<230 vs >=230 Msun).

  2. Conditional distributions of [Cu/Fe] and [Zn/Fe] against f_pisn in each
     mass regime (binned mean +/- std).

  3. The CCSN-only [Zn/Fe] histogram (the reference the diagnostic is measured
     against).

  4. The two >=230-bin checks: (a) Cu/Zn Gini importance + TPR/FPR restricted
     to the >=230 bin; (b) whether [Zn/Fe] separates f_pisn >= 0.5 from
     f_pisn = 0 in the >=230 bin.

  5. The impostor-separation test: a RandomForestClassifier separating
     CCSN-only (f_pisn = 0) from PISN-dominated (f_pisn > 0.5), with and
     without the Cu/Zn features, so Cu/Zn's contribution to *separation* (not
     regression) is measured directly.

Datasets (cached, no regeneration):
  * 4_3 -> dataset_TakaLimon_realisticNoise_v4_3.npz  (Takahashi + Limongi18 + noise)
  * 5   -> dataset_Nomotov5.npz                       (Takahashi + Nomoto13, clean)

Outputs (src/randomForest/results/test_v8/):
  * killer_element_metrics.json / .txt
  * ccsn_only_znfe_hist.png
  * conditional_cuzn_vs_fpisn.png
  * znfe_mass_regime_separation.png
  * impostor_separation_roc.png
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from scipy.stats import ks_2samp


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

from src.randomForest.train_rf import CONFIG, load_dataset  # noqa: E402

DATA_DIR = os.path.join(ROOT, "src", "randomForest", "data")
OUT_DIR = os.path.join(ROOT, "src", "randomForest", "results", "test_v8")
os.makedirs(OUT_DIR, exist_ok=True)

ELEMENTS = ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
NON_FE = [e for e in ELEMENTS if e != "Fe"]
XFE_NAMES = [f"[{x}/Fe]" for x in NON_FE]
KILLER = ["Cu", "Zn"]
MASS_SPLIT = 230.0
THRESH = -1.0
F_STRONG = 0.5

DATASETS = {
    "4_3": os.path.join(DATA_DIR, "dataset_TakaLimon_realisticNoise_v4_3.npz"),
    "5": os.path.join(DATA_DIR, "dataset_Nomotov5.npz"),
}


def build_xfe(X_full, full_names):
    Xxfe = np.empty((X_full.shape[0], len(NON_FE)))
    for k, x in enumerate(NON_FE):
        for j, nm in enumerate(full_names):
            a, b = nm.strip("[]").split("/")
            if {a, b} == {x, "Fe"}:
                Xxfe[:, k] = (1.0 if a == x else -1.0) * X_full[:, j]
                break
    return Xxfe


def diag_rates(xfe, f_pisn, pisn_mass, is_pisn, mass_sel=None):
    """TPR/FPR of the [X/Fe] < -1 diagnostic.

    Truth positive = f_pisn > F_STRONG (PISN-dominated);
    truth negative = f_pisn == 0 (CCSN-only).
    ``mass_sel`` optionally restricts the positive class to a mass regime
    (CCSN-only negatives are mass-free and always included).
    """
    pos = (f_pisn > F_STRONG) & is_pisn
    neg = f_pisn == 0.0
    if mass_sel is not None:
        pos = pos & mass_sel
    tp = ((xfe < THRESH) & pos).sum()
    fn = ((xfe >= THRESH) & pos).sum()
    fp = ((xfe < THRESH) & neg).sum()
    tn = ((xfe >= THRESH) & neg).sum()
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "tpr": float(tpr), "fpr": float(fpr),
        "n_pos": int(tp + fn), "n_neg": int(fp + tn),
    }


def binned_conditional(f_pisn, xfe, is_pisn, mass_sel, n_bins=8):
    """mean +/- std of [X/Fe] in f_pisn bins, restricted to PISN samples in a mass regime."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    centers = (bins[:-1] + bins[1:]) / 2
    for i in range(n_bins):
        sel = is_pisn & mass_sel & (f_pisn >= bins[i]) & (f_pisn < bins[i + 1])
        if sel.sum() > 1:
            means[i] = xfe[sel].mean()
            stds[i] = xfe[sel].std()
    return centers, means, stds


def run_dataset(dataset_id, cache):
    X_full, y, diag = load_dataset(cache)
    full_names = list(diag["feature_names"])
    Xxfe = build_xfe(X_full, full_names)
    Xxfe = SimpleImputer(strategy="median").fit_transform(Xxfe)
    pisn_mass = np.asarray(diag["pisn_mass"], dtype=float)
    is_pisn = np.isfinite(pisn_mass)
    f_pisn = y

    iZn = XFE_NAMES.index("[Zn/Fe]")
    iCu = XFE_NAMES.index("[Cu/Fe]")
    znfe = Xxfe[:, iZn]
    cufe = Xxfe[:, iCu]

    # 1. TPR/FPR overall and per mass regime
    rates = {}
    for label, mass_sel in [("overall", None),
                            ("m<230", pisn_mass < MASS_SPLIT),
                            ("m>=230", pisn_mass >= MASS_SPLIT)]:
        rates[label] = {
            "Zn": diag_rates(znfe, f_pisn, pisn_mass, is_pisn, mass_sel),
            "Cu": diag_rates(cufe, f_pisn, pisn_mass, is_pisn, mass_sel),
        }

    # 2. CCSN-only [Zn/Fe] histogram stats
    ccsn_zn = znfe[f_pisn == 0.0]

    # 3. Gini importance restricted to the >=230 bin (PISN m>=230 + CCSN-only)
    imp_by_regime = {}
    for label, mass_sel in [("m<230", pisn_mass < MASS_SPLIT), ("m>=230", pisn_mass >= MASS_SPLIT)]:
        sel = (is_pisn & mass_sel) | (f_pisn == 0.0)
        rf = RandomForestRegressor(
            n_estimators=CONFIG["n_estimators"], min_samples_leaf=CONFIG["min_samples_leaf"],
            random_state=CONFIG["random_state"], n_jobs=CONFIG["n_jobs"],
        ).fit(Xxfe[sel], f_pisn[sel])
        imp_by_regime[label] = {
            nm: float(v) for nm, v in zip(XFE_NAMES, rf.feature_importances_)
        }

    # 4. [Zn/Fe] separation of f>0.5 vs f=0 in the >=230 bin (KS test)
    hi_pos = znfe[is_pisn & (pisn_mass >= MASS_SPLIT) & (f_pisn > F_STRONG)]
    hi_neg = znfe[f_pisn == 0.0]
    ks = ks_2samp(hi_pos, hi_neg)

    # 5. impostor separation: f=0 vs f>0.5, with vs without Cu/Zn
    sel_bin = (f_pisn == 0.0) | (f_pisn > F_STRONG)
    X_bin = Xxfe[sel_bin]
    y_bin = (f_pisn[sel_bin] > F_STRONG).astype(int)
    keep = [k for k, x in enumerate(NON_FE) if x not in KILLER]

    def _auc(feats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        clf = RandomForestClassifier(
            n_estimators=CONFIG["n_estimators"], min_samples_leaf=CONFIG["min_samples_leaf"],
            random_state=CONFIG["random_state"], n_jobs=CONFIG["n_jobs"],
        )
        oof = cross_val_predict(clf, X_bin[:, feats], y_bin, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y_bin, oof)), oof

    auc_all, oof_all = _auc(list(range(len(NON_FE))))
    auc_nokiller, oof_nokiller = _auc(keep)

    print(f"\n[{dataset_id}]")
    print("  [Zn/Fe]<-1  TPR/FPR (truth: f>0.5 vs f=0):")
    for label in ("overall", "m<230", "m>=230"):
        r = rates[label]["Zn"]
        print(f"    {label:<9} TPR={r['tpr']:.3f}  FPR={r['fpr']:.3f}  "
              f"(n_pos={r['n_pos']}, n_neg={r['n_neg']})")
    print("  [Cu/Fe]<-1  TPR/FPR:")
    for label in ("overall", "m<230", "m>=230"):
        r = rates[label]["Cu"]
        print(f"    {label:<9} TPR={r['tpr']:.3f}  FPR={r['fpr']:.3f}  "
              f"(n_pos={r['n_pos']}, n_neg={r['n_neg']})")
    print(f"  CCSN-only [Zn/Fe]: mean={ccsn_zn.mean():+.3f}  std={ccsn_zn.std():.3f}  "
          f"frac<-1={(ccsn_zn < THRESH).mean():.3f}")
    print(f"  >=230 bin: [Zn/Fe] f>0.5 mean={hi_pos.mean():+.3f} vs f=0 mean={hi_neg.mean():+.3f} "
          f"KS p={ks.pvalue:.3e}")
    print(f"  >=230 bin Cu/Zn Gini: Cu={imp_by_regime['m>=230']['[Cu/Fe]']:.4f} "
          f"Zn={imp_by_regime['m>=230']['[Zn/Fe]']:.4f}")
    print(f"  impostor AUC: all-features={auc_all:.4f}  without-Cu/Zn={auc_nokiller:.4f}")

    return {
        "dataset_id": dataset_id,
        "rates": rates,
        "ccsn_znfe": {"mean": float(ccsn_zn.mean()), "std": float(ccsn_zn.std()),
                      "frac_lt_m1": float((ccsn_zn < THRESH).mean()), "n": int(len(ccsn_zn))},
        "imp_by_regime": imp_by_regime,
        "ks": {"stat": float(ks.statistic), "pvalue": float(ks.pvalue),
               "hi_pos_mean": float(hi_pos.mean()), "hi_neg_mean": float(hi_neg.mean())},
        "impostor_auc": {"all": auc_all, "no_killer": auc_nokiller},
        "binned": {"Zn": {}, "Cu": {}},
    }


def run_and_plot(dataset_id, cache):
    X_full, y, diag = load_dataset(cache)
    full_names = list(diag["feature_names"])
    Xxfe = build_xfe(X_full, full_names)
    Xxfe = SimpleImputer(strategy="median").fit_transform(Xxfe)
    pisn_mass = np.asarray(diag["pisn_mass"], dtype=float)
    is_pisn = np.isfinite(pisn_mass)
    f_pisn = y

    iZn = XFE_NAMES.index("[Zn/Fe]")
    iCu = XFE_NAMES.index("[Cu/Fe]")
    znfe = Xxfe[:, iZn]
    cufe = Xxfe[:, iCu]

    # --- CCSN-only [Zn/Fe] histogram ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(znfe[f_pisn == 0.0], bins=40, color="tab:gray", alpha=0.8)
    ax.axvline(THRESH, color="tab:red", ls="--", lw=1.5, label=f"[Zn/Fe] = {THRESH}")
    ax.set_xlabel("[Zn/Fe]")
    ax.set_ylabel("N (CCSN-only)")
    ax.set_title(f"test_v8 [{dataset_id}]: CCSN-only [Zn/Fe] distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"ccsn_only_znfe_hist_{dataset_id}.png"), dpi=130)
    plt.close(fig)

    # --- conditional distributions of Cu/Zn vs f_pisn per mass regime ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for r, (el, ival) in enumerate([("Zn", iZn), ("Cu", iCu)]):
        for c, (label, mass_sel) in enumerate([("m<230", pisn_mass < MASS_SPLIT),
                                                ("m>=230", pisn_mass >= MASS_SPLIT)]):
            ax = axes[r][c]
            v = Xxfe[:, ival]
            centers, means, stds = binned_conditional(f_pisn, v, is_pisn, mass_sel)
            ax.errorbar(centers, means, yerr=stds, fmt="o-", color="tab:blue", lw=1.5, ms=4)
            # CCSN-only reference (f=0)
            ccsn = v[f_pisn == 0.0]
            ax.errorbar([0.0], [ccsn.mean()], yerr=[ccsn.std()], fmt="s",
                        color="tab:red", ms=7, label="CCSN-only (f=0)")
            ax.axhline(THRESH, color="k", ls="--", lw=1)
            ax.set_title(f"[{el}/Fe] — {label}")
            ax.set_ylabel(f"[{el}/Fe]")
            if r == 1:
                ax.set_xlabel("f_pisn")
            ax.legend(fontsize=8)
    fig.suptitle(f"test_v8 [{dataset_id}]: conditional [Cu,Zn/Fe] vs f_pisn per mass regime",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, f"conditional_cuzn_vs_fpisn_{dataset_id}.png"), dpi=130)
    plt.close(fig)

    # --- [Zn/Fe] separation in the >=230 bin ---
    hi_pos = znfe[is_pisn & (pisn_mass >= MASS_SPLIT) & (f_pisn > F_STRONG)]
    hi_neg = znfe[f_pisn == 0.0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(hi_neg, bins=40, alpha=0.6, color="tab:gray", label="CCSN-only (f=0)")
    ax.hist(hi_pos, bins=40, alpha=0.6, color="tab:red", label="PISN (m>=230, f>0.5)")
    ax.axvline(THRESH, color="k", ls="--", lw=1.5, label=f"[Zn/Fe] = {THRESH}")
    ax.set_xlabel("[Zn/Fe]")
    ax.set_ylabel("N")
    ax.set_title(f"test_v8 [{dataset_id}]: [Zn/Fe] separates f>0.5 from f=0 in >=230 bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"znfe_mass_regime_separation_{dataset_id}.png"), dpi=130)
    plt.close(fig)


def plot_roc(all_results):
    fig, ax = plt.subplots(figsize=(6, 5))
    for ds, r in all_results.items():
        # We only stored AUC numbers, not curves; draw a simple comparison bar instead.
        pass
    ds_list = list(all_results)
    x = np.arange(len(ds_list))
    w = 0.35
    ax.bar(x - w / 2, [all_results[d]["impostor_auc"]["all"] for d in ds_list], w,
           label="all [X/Fe]", color="tab:blue")
    ax.bar(x + w / 2, [all_results[d]["impostor_auc"]["no_killer"] for d in ds_list], w,
           label="drop Cu/Zn", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_list)
    ax.set_ylim(0.9, 1.01)
    ax.set_ylabel("ROC-AUC (f=0 vs f>0.5)")
    ax.set_title("test_v8: impostor separation with vs without Cu/Zn")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "impostor_separation_roc.png"), dpi=130)
    plt.close(fig)


def main():
    all_results = {}
    for ds, cache in DATASETS.items():
        all_results[ds] = run_dataset(ds, cache)
        run_and_plot(ds, cache)

    plot_roc(all_results)

    with open(os.path.join(OUT_DIR, "killer_element_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    lines = ["Forward-model killer-element analysis (test_v8)",
             "Truth: positive = f_pisn > 0.5 (PISN-dominated), negative = f_pisn = 0 (CCSN-only).",
             "Diagnostic: [Zn/Fe] < -1 or [Cu/Fe] < -1.",
             ""]
    for ds, r in all_results.items():
        lines.append(f"== {ds} ==")
        lines.append(f"  CCSN-only [Zn/Fe]: mean={r['ccsn_znfe']['mean']:+.3f} "
                     f"std={r['ccsn_znfe']['std']:.3f} frac<-1={r['ccsn_znfe']['frac_lt_m1']:.3f} "
                     f"(n={r['ccsn_znfe']['n']})")
        for el in ("Zn", "Cu"):
            lines.append(f"  [{el}/Fe]<-1 TPR/FPR (overall / m<230 / m>=230):")
            for label in ("overall", "m<230", "m>=230"):
                rr = r["rates"][label][el]
                lines.append(f"      {label:<9} TPR={rr['tpr']:.3f} FPR={rr['fpr']:.3f} "
                             f"n_pos={rr['n_pos']} n_neg={rr['n_neg']}")
        lines.append(f"  >=230 bin [Zn/Fe]: f>0.5 mean={r['ks']['hi_pos_mean']:+.3f} vs "
                     f"f=0 mean={r['ks']['hi_neg_mean']:+.3f} (KS p={r['ks']['pvalue']:.3e})")
        lines.append(f"  >=230 bin Gini: Cu={r['imp_by_regime']['m>=230']['[Cu/Fe]']:.4f} "
                     f"Zn={r['imp_by_regime']['m>=230']['[Zn/Fe]']:.4f}")
        lines.append(f"  impostor ROC-AUC (f=0 vs f>0.5): all={r['impostor_auc']['all']:.4f} "
                     f"drop-Cu/Zn={r['impostor_auc']['no_killer']:.4f}")
        lines.append("")
    with open(os.path.join(OUT_DIR, "killer_element_metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\ndone. outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
