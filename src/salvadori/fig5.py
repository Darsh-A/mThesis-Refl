import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from src.load_data import load_limongi18
from src.salvadori_funcs import salvadori_Y_X_II, salvadori_Y_Z_II
from src.formulas import raiteri_lifetime
from params import Z_SUN

# Fig. 5 is computed at a single, fixed stellar metallicity: Z* = 0.01 Zsun,
# "typical of an environment enriched by a single PISN" (Salvadori+19, p.8).
# Limongi18 tabulates [Fe/H] = {0, -1, -2, -3}; since Z/Zsun = 10^[Fe/H],
# [Fe/H] = -2 is the 0.01 Zsun analog.  We use the non-rotating (v = 0) grid.
Z_STAR = 0.01 * Z_SUN
FEH = -3
VELOCITY = 0

limongi18 = load_limongi18()
limongi18 = [
    e for e in limongi18
    if e["params"]["velocity"] == VELOCITY and e["params"]["feh"] == FEH
]

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


masses = np.asarray(sorted({e["params"]["mass"] for e in limongi18}), dtype=float)
lifetimes = np.array([raiteri_lifetime(m, Z_STAR) for m in masses])

# Dense mass grid used for the interpolated curves.
# Limongi18 tabulated masses are still used for the plotted markers.
mass_grid = np.linspace(masses.min(), masses.max(), 500)
time_grid = np.array([
    raiteri_lifetime(m, Z_STAR) for m in mass_grid
])

# --- Left panel: raw per-star yield mass, m_X^II(m), vs that star's lifetime ---
def per_star_yield(data, elem):
    """Raw ejecta mass of `elem` for each tabulated mass, in the same mass
    order as `masses`. 0.0 where the element isn't tabulated for that mass.
    Limongi18 yields are already element-level with radioactive decay built
    into the `.dec` tables, so no decay/isotope-combining step is needed."""
    by_mass = {}
    for entry in data:
        yv = entry["yields"]
        if elem == "Z":
            val = sum(v for el, v in yv.items() if el not in ("H", "He"))
        else:
            val = yv.get(elem, 0.0)

        by_mass[entry["params"]["mass"]] = val
    return np.array([by_mass.get(m, 0.0) for m in masses])


# --- Right panel: IMF-integrated, time-cumulative yield, Y_X^II(t) ---
def cumulative_yield(elem, m_grid):
    """Y_X^II evaluated over a dense m_popII grid."""
    vals = []

    for m in m_grid:
        if elem == "Z":
            vals.append(
                salvadori_Y_Z_II(
                    limongi18,
                    m_popII=m,
                    m_max=100.0,
                    sn_input="Limongi18",
                )
            )
        else:
            vals.append(
                salvadori_Y_X_II(
                    limongi18,
                    elem=elem,
                    m_popII=m,
                    m_max=100.0,
                    sn_input="Limongi18",
                )
            )

    return np.array(vals)


fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex=True)

for row, group in enumerate(GROUPS):
    ax_left, ax_right = axes[row, 0], axes[row, 1]
    color = group["color"]

    for elem, style in group["members"]:
        # Limongi18 points
        y_left = per_star_yield(limongi18, elem)

        # Dense interpolated single-star curve
        mask_l = y_left > 0

        if np.count_nonzero(mask_l) >= 2:
            yield_interp = interp1d(
                masses[mask_l],
                y_left[mask_l],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            y_left_dense = yield_interp(mass_grid)

            # Do not allow interpolation to produce non-positive values
            mask_dense_l = y_left_dense > 0
        else:
            y_left_dense = np.full_like(mass_grid, np.nan)
            mask_dense_l = np.zeros_like(mass_grid, dtype=bool)

        # Dense IMF-integrated curve
        y_right_dense = cumulative_yield(elem, mass_grid)
        mask_dense_r = y_right_dense > 0

        # Limongi18 points for markers
        mask_points_l = y_left > 0

        # Right-panel values at the Limongi18 masses
        y_right_points = cumulative_yield(elem, masses)
        mask_points_r = y_right_points > 0

        mfc = color if style.get("mfc", "full") == "full" else "none"
        marker = style["marker"]
        ls = style["ls"]

        # ---------------------------------------------------------
        # LEFT: smooth interpolated line
        # ---------------------------------------------------------
        ax_left.plot(
            time_grid[mask_dense_l],
            np.log10(y_left_dense[mask_dense_l]),
            linestyle=ls,
            color=color,
            linewidth=1.0,
        )

        # Limongi18 data points
        if marker != "None":
            ax_left.plot(
                lifetimes[mask_points_l],
                np.log10(y_left[mask_points_l]),
                linestyle="None",
                marker=marker,
                color=color,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markersize=5,
            )

        # ---------------------------------------------------------
        # RIGHT: smooth IMF-integrated curve
        # ---------------------------------------------------------
        ax_right.plot(
            time_grid[mask_dense_r],
            np.log10(y_right_dense[mask_dense_r]),
            linestyle=ls,
            color=color,
            linewidth=1.0,
            label=elem,
        )

        # Limongi18 mass locations
        if marker != "None":
            ax_right.plot(
                lifetimes[mask_points_r],
                np.log10(y_right_points[mask_points_r]),
                linestyle="None",
                marker=marker,
                color=color,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markersize=5,
            )

    ax_left.set_xscale("log")
    ax_right.set_xscale("log")
    ax_left.grid(alpha=0.2)
    ax_right.grid(alpha=0.2)
    leg = ax_right.legend(
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        handlelength=2.2,
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

fig.suptitle(r"Fig. 5 replication: $Z_* = 0.01\,Z_\odot$ Pop II yields (Limongi18 [Fe/H]=-2)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("plots/salv_figures/fig5.png", dpi=200, bbox_inches="tight")
plt.show()
