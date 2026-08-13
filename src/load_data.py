import glob
import os
import re

from .utils import _isotope_label
from params import heger_woosley_2002_yields, ishigaki_2018_yields, ishigaki18_selected_yields

def load_hw2002(filepath: str=heger_woosley_2002_yields) -> list[dict]:
    """Load Heger & Woosley 2002 yields.

    Fixed-width format:
      header:  isotope <mass_he_1> <mass_he_2> ... <mass_x>
      rows:    <isotope> <yield_at_mass_he_1> ...

    The column header values are He-core masses.

    Returns:
    [
        {
            "label": "M=12",
            "params": {"mass": 12, "mass_he": 6},
            "yields": {"H1": 0.1, "He4": 0.2, ...},
        },
        ...
    ]
    """
    with open(filepath) as f:
        lines = f.readlines()

    header_parts = lines[0].split()
    mass_he_values = [float(v) for v in header_parts[1:]]
    mass_he_to_total = lambda mass_he: mass_he * 2 + 10

    entries = []
    for ei, mass_he in enumerate(mass_he_values):
        mass = mass_he_to_total(mass_he)
        yld = {}
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < ei + 2:
                continue
            label = _isotope_label(parts[0])
            val = float(parts[ei + 1])
            yld[label] = val

        entries.append({
            "label": f"M={mass:.0f}",
            "params": {"mass": mass, "mass_he": mass_he},
            "yields": yld,
        })

    return entries


def load_ishigaki(data_dir: str=ishigaki_2018_yields) -> list[dict]:
    """Load Ishigaki+18 *fort.10.dat files (skip Natoms_*).

    Tab-separated, header row with isotope names, one data row per model.
    First 6 columns after the index are parameters; remaining columns are
    per-isotope mass yields.

    Mass values: [11, 13, 15, 25, 40, 100]

    Returns:
    [
        {
            "label": "M=12 E=1.00",
            "params": {"mass": 12, "energy": 1.0, "Mcut": 1.0, "MCO": 1.0, "Mout": 1.0, "lfej": 1.0},
            "yields": {"H1": 0.1, "He4": 0.2, ...}
        },
        ...
    ]
    """
    pattern = os.path.join(data_dir, "M*_fort.10.dat")
    files = sorted(glob.glob(pattern))

    entries = []
    for filepath in files:
        with open(filepath) as f:
            lines = f.readlines()

        if len(lines) < 2:
            continue

        header_parts = lines[0].strip().split('\t')
        iso_names = header_parts[6:]   # skip mass,energy,Mcut,MCO,Mout,lfej
        iso_labels = [_isotope_label(n) for n in iso_names]

        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 7 + len(iso_names):
                continue

            param_vals = [float(p) for p in parts[1:7]]
            yield_vals = [float(v) for v in parts[7:]]
            mass, energy, mcut, mco, mout, lfej = param_vals

            yld = {}
            for iso, val in zip(iso_labels, yield_vals):
                yld[iso] = val

            entries.append({
                "label": f"M={mass:.0f}",
                "params": {"mass": mass, "energy": energy, "Mcut": mcut,
                           "MCO": mco, "Mout": mout, "lfej": lfej},
                "yields": yld,
            })

    return entries

def load_ishigaki_selected(data_dir: str = ishigaki18_selected_yields,) -> list[dict]:
    """Load Ishigaki+18 selected yields from a TSV file (from hartwig chi2 analysis)"""

    entries = []

    with open(data_dir) as f:
        lines = f.readlines()

    if len(lines) < 2:
        return entries

    header = lines[0].strip().split("\t")

    # First 6 columns are metadata; everything after is an element.
    element_names = header[6:]

    for line in lines[1:]:
        parts = line.strip().split("\t")

        if len(parts) < 6 + len(element_names):
            continue

        mass = float(parts[0])
        energy = float(parts[1])
        mcut = float(parts[2])
        mco = float(parts[3])
        mout = float(parts[4])
        lfej = float(parts[5])

        # Elements start at column 6.
        yield_vals = [float(v) for v in parts[6:]]

        yields = {
            element: value
            for element, value in zip(element_names, yield_vals)
        }

        entries.append({
            "label": f"M={mass:.0f}",
            "params": {
                "mass": mass,
                "energy": energy,
                "Mcut": mcut,
                "MCO": mco,
                "Mout": mout,
                "lfej": lfej,
            },
            "yields": yields,
        })

    return entries