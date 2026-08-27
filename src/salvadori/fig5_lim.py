import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from src.load_data import load_limongi18
from src.salvadori_funcs import (
    salvadori_Y_X_II,
    salvadori_Y_Z_II,
)
from src.utils import limongi_lifetime, salvadori_select_limongi_feh
from params import Z_SUN

# Fig. 5 is computed at a single, fixed stellar metallicity: Z* = 0.01 Zsun,
# "typical of an environment enriched by a single PISN" (Salvadori+19, p.8).
# 0.01 Zsun -> [Fe/H] = -2 exactly (log10(0.01) = -2), so this lands
# precisely on a Limongi18 grid point -- no interpolation ambiguity here,
# but derived via the selector rather than hardcoded in case Z_STAR changes.

Z_STAR = 0.01 * Z_SUN
FEH = salvadori_select_limongi_feh(Z_STAR / Z_SUN)
VELOCITY = 0

limongi18 = load_limongi18()
limongi18 = [
    e for e in limongi18
    if e["params"]["velocity"] == VELOCITY and e["params"]["feh"] == FEH
]

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

masses = np.asarray(sorted({e["params"]["mass"] for e in limongi18}), dtype=float)

# Lifetime is a direct table lookup (Limongi18's own PSN lifetimes) --
# no decay/nucleosynthesis calculation involved here, just retrieval.
lifetimes = np.array([
    limongi_lifetime(VELOCITY, FEH, "PSN", int(m))["lifetime_yr"]
    for m in masses
])

mass_grid = np.linspace(masses.min(), masses.max(), 500)

lifetime_interp = interp1d(
    masses, lifetimes, kind="linear", bounds_error=False, fill_value="extrapolate",
)
time_grid = lifetime_interp(mass_grid)


def per_star_yield(data, elem):
    """Raw ejecta mass of `elem` per tabulated mass. Limongi18 yields are
    already element-level with decay applied (.dec, ele), so this is a
    direct lookup -- no decay/isotope-combining step needed."""
    by_mass = {}
    for entry in data:
        yv = entry["yields"]
        if elem == "Z":
            val = sum(v for el, v in yv.items() if el not in ("H", "He"))
        else:
            val = yv.get(elem, 0.0)
        by_mass[entry["params"]["mass"]] = val
    return np.array([by_mass.get(m, 0.0) for m in masses])


def cumulative_yield(elem, m_grid):
    """Y_X^II evaluated over a dense m_popII grid.

    "Z" (total metals) is routed through salvadori_Y_Z_II rather than
    salvadori_Y_X_II, since "Z" is never a literal key in element_yields --
    salvadori_Y_X_II would silently return 0.0 for it otherwise.
    """
    vals = []
    for m in m_grid:
        if elem == "Z":
            vals.append(
                salvadori_Y_Z_II(limongi18, m_popII=m, m_max=120.0, sn_input="Limongi18")
            )
        else:
            vals.append(
                salvadori_Y_X_II(limongi18, elem=elem, m_popII=m, m_max=120.0, sn_input="Limongi18")
            )
    return np.array(vals)


fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True)

for row, group in enumerate(GROUPS):
    ax_left, ax_right = axes[row, 0], axes[row, 1]
    color = group["color"]

    for elem, style in group["members"]:
        y_left = per_star_yield(limongi18, elem)
        mask_l = y_left > 0

        if np.count_nonzero(mask_l) >= 2:
            yield_interp = interp1d(
                masses[mask_l], y_left[mask_l], kind="linear",
                bounds_error=False, fill_value="extrapolate",
            )
            y_left_dense = yield_interp(mass_grid)
            mask_dense_l = y_left_dense > 0
        else:
            y_left_dense = np.full_like(mass_grid, np.nan)
            mask_dense_l = np.zeros_like(mass_grid, dtype=bool)

        y_right_dense = cumulative_yield(elem, mass_grid)
        mask_dense_r = y_right_dense > 0

        mask_points_l = y_left > 0
        y_right_points = cumulative_yield(elem, masses)
        mask_points_r = y_right_points > 0

        mfc = color if style.get("mfc", "full") == "full" else "none"
        marker = style["marker"]
        ls = style["ls"]

        ax_left.plot(
            time_grid[mask_dense_l], np.log10(y_left_dense[mask_dense_l]),
            linestyle=ls, color=color, linewidth=1.0,
        )
        if marker != "None":
            ax_left.plot(
                lifetimes[mask_points_l], np.log10(y_left[mask_points_l]),
                linestyle="None", marker=marker, color=color,
                markerfacecolor=mfc, markeredgecolor=color, markersize=5,
            )

        ax_right.plot(
            time_grid[mask_dense_r], np.log10(y_right_dense[mask_dense_r]),
            linestyle=ls, color=color, linewidth=1.0, label=elem,
        )
        if marker != "None":
            ax_right.plot(
                lifetimes[mask_points_r], np.log10(y_right_points[mask_points_r]),
                linestyle="None", marker=marker, color=color,
                markerfacecolor=mfc, markeredgecolor=color, markersize=5,
            )

    ax_left.set_xscale("log")
    ax_right.set_xscale("log")
    ax_left.grid(alpha=0.2)
    ax_right.grid(alpha=0.2)
    leg = ax_right.legend(
        fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, handlelength=2.2,
    )
    for handle, (elem, style) in zip(leg.legend_handles, group["members"]):
        marker = style["marker"]
        mfc = color if style.get("mfc", "full") == "full" else "none"
        if marker != "None":
            handle.set_marker(marker)
            handle.set_markerfacecolor(mfc)
            handle.set_markeredgecolor(color)
            handle.set_markersize(5)

axes[0, 0].set_title(r"$\log(m_X^{II}/M_\odot)$ vs lifetime  (single star)", fontsize=10)
axes[0, 1].set_title(r"$\log(Y_X^{II}/M_\odot)$ vs $t_\mathrm{popII}$  (IMF-integrated)", fontsize=10)

for col in range(2):
    axes[-1, col].set_xlabel("timescale [yr]")
axes[0, 0].set_ylabel(r"$\log(m_X^{II}/M_\odot)$")
axes[0, 1].set_ylabel(r"$\log(Y_X^{II}/M_\odot)$")


def add_mass_axis(ax):
    top = ax.secondary_xaxis("top")
    tick_masses = [13, 15, 20, 25, 30, 40, 60, 80, 120]
    top.set_xticks([lifetime_interp(m) for m in tick_masses])
    top.set_xticklabels([f"{m:.0f}" for m in tick_masses])
    top.set_xlabel(r"$m_{\rm popII}\;(M_\odot)$", fontsize=9)


add_mass_axis(axes[0, 0])
add_mass_axis(axes[0, 1])

fig.suptitle(
    rf"Fig. 5 replication: $Z_* = 0.01\,Z_\odot$ Pop II yields (Limongi18 [Fe/H]={FEH})",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("plots/salv_figures/fig5_limongi.png", dpi=200, bbox_inches="tight")
plt.show()