import matplotlib.pyplot as plt
import numpy as np

from src.load_data import load_hw2002, load_takahashi, load_ww95
from src.salvadori_funcs import salvadori_combined_abundratio_WrtH
from src.salvadori_funcs import salvadori_yields as salvadori_yields_convert
from src.utils import _combine_elements

from params import ww95_001Z

hw_yields = load_hw2002()
takahashi_yields = load_takahashi()
takahashi_yields = takahashi_yields["NR"]
ww95_yields = load_ww95(ww95_001Z)

hw_yields = hw_yields[1:]

pisn_yields = hw_yields
sn_yields = ww95_yields

salvadori_pisn_yields = salvadori_yields_convert(pisn_yields)
salvadori_sn_yields = salvadori_yields_convert(sn_yields)

# Style keyed to match Salvadori+2019 Fig. 4 (label -> (marker, color))
STYLE = {
    3:  ("<", "#e6555a"),   # red left-triangle
    6:  ("p", "#f2a13c"),   # orange pentagon
    10: ("s", "#c9c93f"),   # olive/yellow square
    20: ("o", "#4caf6a"),   # green circle
    30: ("^", "#b09cd9"),   # purple triangle
}

tpop2_list = [3e6, 6e6, 10e6, 20e6, 30e6]  # in years
f_pisn = 0.5

fig, ax = plt.subplots(figsize=(6, 5))

for tpop2 in tpop2_list:
    F_He = []
    mPISN = []

    for pisn in salvadori_pisn_yields:
        f_ratio = 1e-3
        Fe_H = salvadori_combined_abundratio_WrtH(
            "Fe", "Fe",
            pisn_data=pisn,
            f_pisn=f_pisn,
            f_ratio=f_ratio,
            tpop2=tpop2,
        )
        mPISN.append(pisn["params"]["mass"])
        F_He.append(Fe_H)

    t_myr = int(tpop2 / 1e6)
    marker, color = STYLE[t_myr]
    ax.plot(
        mPISN, F_He,
        marker=marker, color=color, linewidth=1.2, markersize=7,
        markeredgecolor="black", markeredgewidth=0.4,
        label=rf"$t_\mathrm{{popII}} = {t_myr}\ \mathrm{{Myr}}$",
    )

ax.set_xlabel(r"$M_\mathrm{PISN}\ (M_\odot)$")
ax.set_ylabel(r"$[\mathrm{Fe}/\mathrm{H}]$")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.minorticks_on()
plt.title(r"Abundance ratio $[\mathrm{Fe}/\mathrm{H}]$ vs. mPISN for different $t_\mathrm{popII}$", fontsize=10)
plt.savefig("plots/salv_figures/fig4.png", dpi=300, bbox_inches="tight")
plt.show()