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
    filter_elements: list[str] = None,
    ax: plt.Axes = None,
) -> plt.Axes:

    masses = [e["params"]["mass"] for e in data]
    unique_masses = sorted(set(masses))
    n_mass = len(unique_masses)

    cmap = plt.get_cmap("rainbow").reversed()
    color_of = {
        mass: cmap(i / max(n_mass - 1, 1))
        for i, mass in enumerate(unique_masses)
    }

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for entry in data:
        mass = entry["params"]["mass"]
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

        ax.plot(isotopes, values, color=color_of[mass], marker="o", markersize=5,
                 linewidth=1.2, markeredgecolor="white", markeredgewidth=0.5, alpha=0.85,
                 label=rf"${mass:g}\,M_\odot$")

    ax.set_xticks(range(len(isotopes)))
    ax.set_xticklabels([rf"$\mathrm{{{iso}}}$" for iso in isotopes])
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.axhline(0, ls="--", color="black", lw=0.8, alpha=0.8, zorder=0)
    ax.grid(visible=True, which="major", color="gray", alpha=0.15, linestyle="-")
    ax.set_title(title, pad=15)
    ax.set_xlabel("Element")
    ax.set_ylabel(rf"[$\mathrm{{X}} / \mathrm{{{wrt}}}$]")

    ax.legend(
        title=r"Mass ($M_\odot$)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    if standalone:
        plt.tight_layout(rect=(0, 0, 0.80, 1))
        plt.show()

    return ax

def plot_one_yield(
    entry: dict[str, float],
    title: str = "Yield",
    wrt: str = "Fe",
    filter_elements: list[str] = None
) -> None:
    """
    Plots the general yield (X/Fe vs elements) for a mass single entry.

    Args:
        entry: A single dict - 
            {
                "C": 0.1, # NOTE: This is the abundance ratio [X/Fe] for each element, not the raw yield.
                "N": 0.2,
            }
        title: Title of the plot.
        wrt: Element to normalize the yields against (default is "Fe").
        filter_elements: Optional list of elements to include in the plot.
    """
    
    yields = entry

    # if filter_elements:
    #     yields = {iso: val for iso, val in yields.items() if iso in filter_elements}
    
    elements = list(yields.keys())
    X_Fe = list(yields.values())

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(elements, X_Fe, color="blue", marker="o", markersize=5, linewidth=1.2, markeredgecolor="white", markeredgewidth=0.5, alpha=0.85)

    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels([rf"$\mathrm{{{iso}}}$" for iso in elements])
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.axhline(0, ls="--", color="black", lw=0.8, alpha=0.8, zorder=0)
    ax.grid(visible=True, which="major", color="gray", alpha=0.15, linestyle="-")
    ax.set_title(title, pad=15)
    ax.set_xlabel("Element")
    ax.set_ylabel(rf"[$\mathrm{{X}} / \mathrm{{{wrt}}}$]")

    plt.tight_layout()
    plt.show()


