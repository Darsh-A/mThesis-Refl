import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import loguniform
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec

from src.load_data import load_hw2002, load_takahashi, load_ww95
from src.salvadori_funcs import salvadori_combined_abundratio, salvadori_combined_abundratio_WrtH, salvadori_yields
from src.salvadori_funcs import salvadori_yields as salvadori_yields_convert
from src.utils import _combine_elements

from params import ww95_001Z, filter_elements

hw_yields = load_hw2002()
takahashi_yields = load_takahashi()
takahashi_yields = takahashi_yields["NR"]
ww95_yields = load_ww95()

hw_yields = hw_yields[1:]

pisn_yields = hw_yields
sn_yields = ww95_yields

salvadori_pisn_yields = salvadori_yields_convert(pisn_yields)
salvadori_sn_yields = salvadori_yields_convert(sn_yields)

def plot_density_with_marginals(feh_vals, xfe_vals, sn_feh_vals, sn_xfe_vals, elem, f_pisn, save_path=None):
    from matplotlib.patches import Patch
    feh_vals, xfe_vals = np.array(feh_vals), np.array(xfe_vals)
    sn_feh_vals, sn_xfe_vals = np.array(sn_feh_vals), np.array(sn_xfe_vals)
    
    mask = np.isfinite(feh_vals) & np.isfinite(xfe_vals)
    feh_vals, xfe_vals = feh_vals[mask], xfe_vals[mask]
    
    sn_mask = np.isfinite(sn_feh_vals) & np.isfinite(sn_xfe_vals)
    sn_feh_vals, sn_xfe_vals = sn_feh_vals[sn_mask], sn_xfe_vals[sn_mask]
    
    kde = gaussian_kde(np.vstack([feh_vals, xfe_vals]))
    sn_kde = gaussian_kde(np.vstack([sn_feh_vals, sn_xfe_vals]))
    
    xmin, xmax = min(feh_vals.min(), sn_feh_vals.min()), max(feh_vals.max(), sn_feh_vals.max())
    ymin, ymax = min(xfe_vals.min(), sn_xfe_vals.min()), max(xfe_vals.max(), sn_xfe_vals.max())
    xpad, ypad = 0.05 * (xmax - xmin), 0.05 * (ymax - ymin)
    
    Xg, Yg = np.mgrid[xmin-xpad:xmax+xpad:200j, ymin-ypad:ymax+ypad:200j]
    positions = np.vstack([Xg.ravel(), Yg.ravel()])
    Z = np.reshape(kde(positions).T, Xg.shape)
    sn_Z = np.reshape(sn_kde(positions).T, Xg.shape)
    
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    ax_main.contourf(Xg, Yg, Z, levels=12, cmap="inferno", alpha=0.45)
    ax_main.contour(Xg, Yg, Z, levels=12, cmap="inferno", linewidths=1.0, alpha=0.9)
    ax_main.contourf(Xg, Yg, sn_Z, levels=12, cmap="viridis", alpha=0.35)
    ax_main.contour(Xg, Yg, sn_Z, levels=12, cmap="viridis", linewidths=1.0, alpha=0.9)
    
    ax_main.legend(handles=[Patch(facecolor="orange", alpha=0.45, label="PISN + SN"), Patch(facecolor="green", alpha=0.35, label="SN + SN")], loc="upper right", fontsize=8)
    ax_main.set_xlabel("[Fe/H]")
    ax_main.set_ylabel(f"[{elem}/Fe]")
    
    ax_top.hist(feh_vals, bins=40, density=True, color="black", alpha=0.35, label="PISN + SN")
    ax_top.hist(sn_feh_vals, bins=40, density=True, color="blue", alpha=0.35, label="SN only")
    ax_top.tick_params(labelbottom=False, labelleft=False)
    ax_top.set_ylabel("P(x)")
    ax_top.legend(fontsize=7, loc="upper right")
    
    ax_right.hist(xfe_vals, bins=40, density=True, orientation="horizontal", color="black", alpha=0.35)
    ax_right.hist(sn_xfe_vals, bins=40, density=True, orientation="horizontal", color="blue", alpha=0.35)
    ax_right.tick_params(labelbottom=False, labelleft=False)
    ax_right.set_xlabel("P(y)")
    
    ax_top.set_title(f"f_pisn = {f_pisn:.2f}")
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

for elem in filter_elements:
    if elem in ["H", "He", "P", "Fe"]:
        continue

    print(f"Processing element: {elem}")
    X_Fe, Fe_H, sn_X_Fe, sn_Fe_H = [], [], [], []

    for i in range(800):
        pisn_entry = random.choice(pisn_yields)
        sn_entry = random.choice(sn_yields)

        f_pisn = 0.9
        f_ratio = loguniform.rvs(1e-4, 1e-1)
        tpop2_values = [3e6, 6e6, 10e6, 20e6, 30e6]
        tpop2 = random.choice(tpop2_values)


        pisn_details = {
            "mass": pisn_entry["params"]["mass"],
        }

        combined_ratio = salvadori_combined_abundratio(elem,elem,"Fe","Fe", pisn_data=pisn_entry, sn_data=sn_yields, salv_sn_data=salvadori_sn_yields, auto_sn=False, f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2)
        combined_ratio_wrtH = salvadori_combined_abundratio_WrtH("Fe","Fe", pisn_data=pisn_entry, sn_data=sn_yields, salv_sn_data=salvadori_sn_yields, auto_sn=False, f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2)
    
        sn_combined_ratio = salvadori_combined_abundratio(elem,elem,"Fe","Fe", pisn_data=sn_entry, sn_data=sn_yields, salv_sn_data=salvadori_sn_yields, auto_sn=False, f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2)
        sn_combined_ratio_wrtH = salvadori_combined_abundratio_WrtH("Fe","Fe", pisn_data=sn_entry, sn_data=sn_yields, salv_sn_data=salvadori_sn_yields, auto_sn=False, f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2)
    
        if combined_ratio < -3 or combined_ratio_wrtH < -3:
            continue
        if sn_combined_ratio < -4 or sn_combined_ratio > 4 or sn_combined_ratio_wrtH < -4 or sn_combined_ratio_wrtH > 4:
            continue

        X_Fe.append(combined_ratio)
        Fe_H.append(combined_ratio_wrtH)
        sn_X_Fe.append(sn_combined_ratio)
        sn_Fe_H.append(sn_combined_ratio_wrtH)

    plot_density_with_marginals(Fe_H, X_Fe, sn_Fe_H, sn_X_Fe, elem, f_pisn, save_path=f"plots/abundance_scatter_contour_mixed/{elem}_contour.png")

    plt.scatter(Fe_H, X_Fe, alpha=0.5, color="black")
    plt.xlabel("[Fe/H]")
    plt.ylabel(f"[{elem}/Fe]")
    plt.title(f"f_pisn = {f_pisn}, f_ratio = {f_ratio}")
    
    f_ratio_str = f"{f_ratio:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    plt.savefig(f"plots/abundance_scatter/{elem}_scatter.png", dpi=300)
    plt.clf()