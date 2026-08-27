import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import loguniform
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec

from src.load_data import load_hw2002, load_limongi18, load_takahashi, load_ww95
from src.salvadori_funcs import salvadori_H_ratio, salvadori_combined_abundratio, salvadori_combined_abundratio_WrtH, salvadori_yields
from src.salvadori_funcs import salvadori_yields as salvadori_yields_convert
from src.utils import _combine_elements

from params import ww95_001Z

hw_yields = load_hw2002()
takahashi_yields = load_takahashi()
takahashi_yields = takahashi_yields["NR"]
ww95_yields = load_ww95(ww95_001Z)

hw_yields = hw_yields[1:]

limongi18 = load_limongi18()
limongi18 = [e for e in limongi18 if e['params']['velocity'] == 0]

pisn_yields = hw_yields
sn_yields = limongi18

salvadori_pisn_yields = salvadori_yields_convert(pisn_yields)
salvadori_sn_yields = salvadori_yields_convert(sn_yields)


for elem in ["Zn", "Cu"]:
    if elem in ["H", "He", "P", "Fe"]:
        continue

    print(f"Processing element: {elem}")

    X_Fe = []
    Fe_H = []
    for i in range(2000):
        
        pisn_entry = random.choice(salvadori_pisn_yields)

        f_pisn = 0.50
        f_ratio = loguniform.rvs(1e-4, 1e-1)
        tpop2 = loguniform.rvs(3.2e6, 17.4e6)
        # tpop2 = random.choice(tpop2_values)

        pisn_details = {
            "mass": pisn_entry["params"]["mass"],
        }

        combined_ratio = salvadori_combined_abundratio(elem,elem,"Fe","Fe", pisn_data=pisn_entry, sn_data=sn_yields, salv_sn_data=salvadori_sn_yields, auto_sn=False, single_sn=False, sn_input="Limongi18", f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2)
        combined_ratio_wrtH = salvadori_combined_abundratio_WrtH("Fe", "Fe", pisn_data=pisn_entry, sn_data=sn_yields, salv_sn_data=salvadori_sn_yields, auto_sn=False, single_sn=False, sn_input="Limongi18", f_pisn=f_pisn, f_ratio=f_ratio, tpop2=tpop2)

        if combined_ratio < -4 or combined_ratio_wrtH < -5:
            print("Very low ratio detected")
            print("Elem", elem,"\n", "PISN", pisn_details, "\n", "SN", "_NA_", "\n", "Combined Ratio", combined_ratio, "\n", "Combined Ratio w.r.t H", combined_ratio_wrtH)
            print("-------------------------------")
            #continue # do not add to the list if the ratio is very low, but print details for debugging

        X_Fe.append(combined_ratio)
        Fe_H.append(combined_ratio_wrtH)

    def plot_density_with_marginals(feh_vals, xfe_vals, elem, f_pisn, save_path=None):
        feh_vals = np.array(feh_vals)
        xfe_vals = np.array(xfe_vals)

        mask = np.isfinite(feh_vals) & np.isfinite(xfe_vals)
        feh_vals, xfe_vals = feh_vals[mask], xfe_vals[mask]

        xy = np.vstack([feh_vals, xfe_vals])
        kde = gaussian_kde(xy)

        xmin, xmax = feh_vals.min(), feh_vals.max()
        ymin, ymax = xfe_vals.min(), xfe_vals.max()
        xpad = 0.05 * (xmax - xmin)
        ypad = 0.05 * (ymax - ymin)

        Xg, Yg = np.mgrid[xmin-xpad:xmax+xpad:200j, ymin-ypad:ymax+ypad:200j]
        positions = np.vstack([Xg.ravel(), Yg.ravel()])
        Z = np.reshape(kde(positions).T, Xg.shape)

        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                                wspace=0.05, hspace=0.05)

        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        cs = ax_main.contourf(Xg, Yg, Z, levels=20, cmap='inferno')
        ax_main.set_xlabel("[Fe/H]")
        ax_main.set_ylabel(f"[{elem}/Fe]")

        feh_kde = gaussian_kde(feh_vals)
        feh_x = np.linspace(feh_vals.min(), feh_vals.max(), 200)
        ax_top.plot(feh_x, feh_kde(feh_x), color='gray')
        ax_top.tick_params(labelbottom=False, labelleft=False)
        ax_top.set_ylabel("P(x)")

        xfe_kde = gaussian_kde(xfe_vals)
        xfe_y = np.linspace(xfe_vals.min(), xfe_vals.max(), 200)
        ax_right.plot(xfe_kde(xfe_y), xfe_y, color='gray')
        ax_right.tick_params(labelbottom=False, labelleft=False)
        ax_right.set_xlabel("P(y)")

        ax_top.set_title(f"f_pisn = {f_pisn:.2f}")

        if save_path:
            fig.savefig(save_path, dpi=300)
        plt.close(fig)

    plot_density_with_marginals(Fe_H, X_Fe, elem, f_pisn,
                                save_path=f"plots/abundance_scatter_contour/{elem}_{f_pisn:.2f}_contour.png")
        
    plt.scatter(Fe_H, X_Fe, alpha=0.5, color='black')
    plt.xlabel("[Fe/H]")
    plt.ylabel("[" + elem + "/Fe]")
    plt.title("f_pisn = " + str(f_pisn) + ", f_ratio = " + str(f_ratio))
    f_ratio_str = f"{f_ratio:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    plt.savefig(f"plots/abundance_scatter/{elem}_{f_pisn:.2f}_scatter.png", dpi=300)
    plt.clf()
        