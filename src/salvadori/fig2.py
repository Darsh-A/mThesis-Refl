import matplotlib.pyplot as plt
import numpy as np

from src.load_data import load_ishigaki, load_hw2002, load_ishigaki_selected, load_takahashi, load_ww95

from src.salvadori_funcs import salvadori_H_ratio
from src.salvadori_funcs import salvadori_yields as salvadori_yields_convert

hw_yields = load_hw2002()
takahashi_yields = load_takahashi()
takahashi_yields = takahashi_yields["NR"] # NOTE: only use non-rotating models for now
ww95_yields = load_ww95()

hw_yields = hw_yields[1:]

sal_tk_yields = salvadori_yields_convert(takahashi_yields)
sal_hw_yields = salvadori_yields_convert(hw_yields)

fig2_values_tk = []
fig2_values_hw = []

for f_ratio in [1e-4, 1e-3, 1e-2, 1e-1]:
    Fe_H_to_mPISN = []

    for y in sal_tk_yields:
        mass = y["params"]["mass"]
        Fe_H = salvadori_H_ratio("Fe", f_ratio, y)
        Fe_H_to_mPISN.append((mass, Fe_H))

    fig2_values_tk.append((f_ratio, Fe_H_to_mPISN))

for f_ratio in [1e-4, 1e-3, 1e-2, 1e-1]:
    Fe_H_to_mPISN = []

    for y in sal_hw_yields:
        mass = y["params"]["mass"]
        Fe_H = salvadori_H_ratio("Fe", f_ratio, y)
        Fe_H_to_mPISN.append((mass, Fe_H))

    fig2_values_hw.append((f_ratio, Fe_H_to_mPISN))

fig, ax = plt.subplots(figsize=(6, 5))

for f_ratio, Fe_H_to_mPISN in fig2_values_tk:
    masses = np.array([m for m, _ in Fe_H_to_mPISN])
    fe_h = np.array([r for _, r in Fe_H_to_mPISN])
    ax.plot(masses, fe_h, marker="o", linewidth=1.2,
            label=rf"$f_\star/f_\mathrm{{dil}} = {f_ratio:g}$ (Takahashi)")
    

for f_ratio, Fe_H_to_mPISN in fig2_values_hw:
    masses = np.array([m for m, _ in Fe_H_to_mPISN])
    fe_h = np.array([r for _, r in Fe_H_to_mPISN])
    ax.plot(masses, fe_h, marker="s", linewidth=1.2,
            label=rf"$f_\star/f_\mathrm{{dil}} = {f_ratio:g}$ (Heger)")

ax.set_xlabel(r"$M_\mathrm{PISN}\ (M_\odot)$")
ax.set_ylabel(r"$[\mathrm{Fe}/\mathrm{H}]$")
ax.legend(fontsize=8)
fig.set_size_inches(6, 5)
ax.grid(True, alpha=0.3)
plt.minorticks_on()
plt.title(r"Fig2. Abundance ratio $[\mathrm{Fe}/\mathrm{H}]$ vs. mPISN for different $f_\star/f_\mathrm{dil}$", fontsize=10)
plt.savefig("plots/salv_figures/fig2.png", dpi=300, bbox_inches="tight")
plt.show()
