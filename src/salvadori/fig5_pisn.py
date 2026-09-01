import matplotlib.pyplot as plt
import numpy as np

from src.load_data import load_hw2002
from src.salvadori_funcs import salvadori_yields
from src.utils import build_pisn_interpolator

# Element groups, colors, and per-element line styles keyed to match
# Salvadori+2019 Fig. 5 (top-to-bottom in each panel = legend order).
# Reused verbatim from fig5.py so the two figures are directly comparable.
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

# HW2002 PISN yields (13 models, 150-270 Msun). Salvatori et al. use the
# mass-normalized yields, and build_pisn_interpolator works on those same
# mass fractions, so the continuous-mass curves below are directly
# comparable to the fig7 abundance-ratio machinery.
hw_yields = load_hw2002()[1:]
pisn_yields = salvadori_yields(hw_yields)
pisn_interp = build_pisn_interpolator(pisn_yields)

masses = np.asarray(sorted({e["params"]["mass"] for e in pisn_yields}), float)

# Dense PISN mass grid used for the interpolated curves.
# Tabulated HW2002 masses are still used for the plotted markers.
mass_grid = np.linspace(masses.min(), masses.max(), 500)


def per_star_yield(elem, m_grid):
    """Raw (continuous-mass) PISN ejecta mass of `elem` for each mass in
    `m_grid`, in solar masses. Interpolated over the HW2002 grid; no
    radioactive-decay correction and no Fe*0.5 rescaling (the PISN yields
    are used as-is). 0.0 where the element is not tabulated."""
    vals = []
    for m in m_grid:
        yv = pisn_interp(float(m))["yields"]
        if elem == "Z":
            frac = sum(v for el, v in yv.items() if el not in ("H", "He"))
        else:
            frac = yv.get(elem, 0.0)
        vals.append(frac * float(m))
    return np.asarray(vals)


fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

for row, group in enumerate(GROUPS):
    ax = axes[row]
    color = group["color"]

    for elem, style in group["members"]:
        # Dense interpolated single-star curve
        y_dense = per_star_yield(elem, mass_grid)
        mask_dense = y_dense > 0

        # Original HW2002 points for markers
        y_points = per_star_yield(elem, masses)
        mask_points = y_points > 0

        mfc = color if style.get("mfc", "full") == "full" else "none"
        marker = style["marker"]
        ls = style["ls"]

        ax.plot(
            mass_grid[mask_dense],
            np.log10(y_dense[mask_dense]),
            linestyle=ls,
            color=color,
            linewidth=1.0,
            label=elem,
        )

        if marker != "None":
            ax.plot(
                masses[mask_points],
                np.log10(y_points[mask_points]),
                linestyle="None",
                marker=marker,
                color=color,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markersize=5,
            )

    ax.grid(alpha=0.2)
    ax.set_ylabel(r"$\log(m_X^{\rm PISN}/M_\odot)$")

    leg = ax.legend(
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

axes[0].set_title(
    r"PISN per-star yields: $\log(m_X^{\rm PISN}/M_\odot)$ vs progenitor mass",
    fontsize=10,
)
axes[-1].set_xlabel(r"$M_{\rm PISN}\ (M_\odot)$")

fig.suptitle(r"Fig. 5 (PISN analog): HW2002 PISN yields, continuous mass", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("plots/salv_figures/fig5_pisn.png", dpi=200, bbox_inches="tight")
plt.show()
