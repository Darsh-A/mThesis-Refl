import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

from .utils import _combine_elements
from .formulas import abundance_ratio

def plot_yields(
    data: list[dict[str, float]],
    title: str = "Yields",
    wrt: str = "Fe",
    combine_elements: bool = True,
    filter_elements: list[str] = None
) -> None:

    masses = [e["params"]["mass"] for e in data]
    # lfejs = [e["params"]["lfej"] for e in data]
    norm = plt.Normalize(min(masses), max(masses))
    # norm = plt.Normalize(min(lfejs), max(lfejs))
    cmap = plt.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(8, 5)) 

    for entry in data:
        mass = entry["params"]["mass"]
        # lfej = entry["params"]["lfej"]
        yields = entry["yields"]
        if combine_elements:
            yields = _combine_elements(yields)

        isotopes = list(yields.keys())
        if filter_elements:
            isotopes = [iso for iso in isotopes if iso in filter_elements]

        values = [
            abundance_ratio(iso, wrt, yields[iso], yields.get(wrt, 0.0))
            for iso in isotopes
        ]
        for e, v in zip(isotopes, values):
            if v > 9:
                print(f"Warning: Abundance ratio for {e} is very low ({v:.2f}) for params {entry['params']}")

        ax.plot(isotopes, values,color=cmap(norm(mass)),marker="o",markersize=5,linewidth=1.2,markeredgecolor="white",markeredgewidth=0.5,alpha=0.85)

    ax.set_xticks(range(len(isotopes)))
    ax.set_xticklabels([rf"$\mathrm{{{iso}}}$" for iso in isotopes])
    
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    ax.axhline(0, ls="--", color="black", lw=0.8, alpha=0.8, zorder=0)
    ax.grid(visible=True, which="major", color="gray", alpha=0.15, linestyle="-")

    ax.set_title(title, pad=15)
    ax.set_xlabel("Element")
    ax.set_ylabel(rf"[$\mathrm{{X}} / \mathrm{{{wrt}}}$]")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Mass ($M_\odot$)")

    plt.tight_layout()
    plt.show()

