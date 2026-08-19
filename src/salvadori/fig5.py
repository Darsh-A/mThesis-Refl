import matplotlib.pyplot as plt
import numpy as np

from src.load_data import load_ww95
from src.salvadori_funcs import salvadori_Y_X_II, salvadori_Y_Z_II
from src.utils import _combine_elements
from src.formulas import raiteri_lifetime
from params import ww95_001Z, Z_SUN

# Fig. 5 is computed at a single, fixed stellar metallicity: Z* = 0.01 Zsun,
# "typical of an environment enriched by a single PISN" (Salvadori+19, p.8).
Z_STAR = 0.01 * Z_SUN
MODEL = "A"

ww_data = load_ww95(ww95_001Z)

# Element groups, colors, and per-element line styles keyed to match
# Salvadori+2019 Fig. 5 (top-to-bottom in each panel = legend order).
GROUPS = [
    {
        "color": "#3355cc",
        "members": [
            ("Z",  {"marker": "None", "ls": ":",  "lw": 1.6}),
            ("O",  {"marker": "o",    "ls": ":",  "mfc": "none"}),
            ("C",  {"marker": "o",    "ls": "-",  "mfc": "full"}),
            ("N",  {"marker": "s",    "ls": "--", "mfc": "none"}),
            ("F",  {"marker": "s",    "ls": "-",  "mfc": "full"}),
        ],
    },
    {
        "color": "#33aa55",
        "members": [
            ("Ne", {"marker": "o", "ls": ":",  "mfc": "none"}),
            ("Na", {"marker": "*", "ls": ":",  "mfc": "full"}),
            ("Mg", {"marker": "o", "ls": "-",  "mfc": "full"}),
            ("Si", {"marker": "s", "ls": "-.", "mfc": "none"}),
            ("Al", {"marker": "^", "ls": "--", "mfc": "none"}),
            ("S",  {"marker": "s", "ls": "-",  "mfc": "full"}),
            ("P",  {"marker": "*", "ls": "-",  "mfc": "full"}),
        ],
    },
    {
        "color": "#ccaa22",
        "members": [
            ("Ar", {"marker": "o", "ls": ":",  "mfc": "none"}),
            ("Ca", {"marker": "s", "ls": ":",  "mfc": "none"}),
            ("Ti", {"marker": "o", "ls": "-.", "mfc": "none"}),
            ("Cl", {"marker": "*", "ls": "-.", "mfc": "full"}),
            ("K",  {"marker": "s", "ls": "-.", "mfc": "full"}),
            ("V",  {"marker": "*", "ls": "--", "mfc": "full"}),
            ("Sc", {"marker": "^", "ls": "--", "mfc": "none"}),
        ],
    },
    {
        "color": "#cc3333",
        "members": [
            ("Fe", {"marker": "o", "ls": "-",  "mfc": "none"}),
            ("Ni", {"marker": "*", "ls": ":",  "mfc": "full"}),
            ("Cr", {"marker": "s", "ls": ":",  "mfc": "none"}),
            ("Mn", {"marker": "o", "ls": "--", "mfc": "full"}),
            ("Co", {"marker": "s", "ls": "--", "mfc": "full"}),
            ("Zn", {"marker": "^", "ls": "--", "mfc": "full"}),
            ("Cu", {"marker": "^", "ls": "-",  "mfc": "full"}),
        ],
    },
]


def masses_for_model(data, model=MODEL):
    ms = sorted(set(e["params"]["mass"] for e in data if e["params"]["model"] == model))
    return np.asarray(ms, dtype=float)


masses = masses_for_model(ww_data)
lifetimes = np.array([raiteri_lifetime(m, Z_STAR) for m in masses])

# --- Left panel: raw per-star yield mass, m_X^II(m), vs that star's lifetime ---
def per_star_yield(data, elem, model=MODEL):
    """Raw ejecta mass of `elem` for each tabulated mass, in the same mass
    order as `masses_for_model`. 0.0 where the element isn't tabulated for
    that mass/table at all."""
    by_mass = {}
    for entry in data:
        if entry["params"]["model"] != model:
            continue
        yv = _combine_elements(entry["yields"])
        if elem == "Z":
            val = sum(v for el, v in yv.items() if el not in ("H", "He"))
        else:
            val = yv.get(elem, 0.0)
        by_mass[entry["params"]["mass"]] = val
    return np.array([by_mass.get(m, 0.0) for m in masses])


# --- Right panel: IMF-integrated, time-cumulative yield, Y_X^II(t) ---
def cumulative_yield(elem):
    """Y_X^II evaluated at m_popII = each tabulated mass (i.e. at t = that
    star's lifetime), using the existing salvadori_Y_X_II / Y_Z_II."""
    vals = []
    for m in masses:
        if elem == "Z":
            vals.append(salvadori_Y_Z_II(ww_data, m_popII=m, model=MODEL, m_max=100.0))
        else:
            vals.append(salvadori_Y_X_II(ww_data, elem=elem, m_popII=m, model=MODEL, m_max=100.0))
    return np.array(vals)


fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True)

for row, group in enumerate(GROUPS):
    ax_left, ax_right = axes[row, 0], axes[row, 1]
    color = group["color"]

    for elem, style in group["members"]:
        y_left = per_star_yield(ww_data, elem)
        y_right = cumulative_yield(elem)

        mfc = color if style.get("mfc", "full") == "full" else "none"
        marker = style["marker"]
        ls = style["ls"]

        mask_l = y_left > 0
        mask_r = y_right > 0

        ax_left.plot(
            lifetimes[mask_l], np.log10(y_left[mask_l]),
            marker=marker if marker != "None" else None,
            linestyle=ls, color=color, markerfacecolor=mfc,
            markeredgecolor=color, markersize=5, linewidth=1.0,
        )
        ax_right.plot(
            lifetimes[mask_r], np.log10(y_right[mask_r]),
            marker=marker if marker != "None" else None,
            linestyle=ls, color=color, markerfacecolor=mfc,
            markeredgecolor=color, markersize=5, linewidth=1.0,
            label=elem,
        )

    ax_left.set_xscale("log")
    ax_right.set_xscale("log")
    ax_left.grid(alpha=0.2)
    ax_right.grid(alpha=0.2)
    ax_right.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
                     frameon=False, handlelength=2.2)

axes[0, 0].set_title(r"$\log(m_X^{II}/M_\odot)$ vs lifetime  (single star)", fontsize=10)
axes[0, 1].set_title(r"$\log(Y_X^{II}/M_\odot)$ vs $t_\mathrm{popII}$  (IMF-integrated)", fontsize=10)

for col in range(2):
    axes[-1, col].set_xlabel("timescale [yr]")
axes[0, 0].set_ylabel(r"$\log(m_X^{II}/M_\odot)$")
axes[0, 1].set_ylabel(r"$\log(Y_X^{II}/M_\odot)$")

# Top axis showing mpopII mass labels, matching the paper's twin axis.
def add_mass_axis(ax):
    top = ax.secondary_xaxis("top")
    tick_masses = [35, 25, 15, 10]
    tick_masses = [m for m in tick_masses if lifetimes.min() <= raiteri_lifetime(m, Z_STAR) <= lifetimes.max()]
    top.set_xticks([raiteri_lifetime(m, Z_STAR) for m in tick_masses])
    top.set_xticklabels([f"{m:.0f}" for m in tick_masses])
    top.set_xlabel(r"$m_\mathrm{popII}\ (M_\odot)$", fontsize=9)

add_mass_axis(axes[0, 0])
add_mass_axis(axes[0, 1])

fig.suptitle(r"Fig. 5 replication: $Z_* = 0.01\,Z_\odot$ Pop II yields (WW95)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("plots/salv_figures/fig5.png", dpi=200, bbox_inches="tight")
plt.show()