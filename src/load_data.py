import glob
import os
import re

from .utils import _isotope_label, _isotope_to_element
from params import heger_woosley_2002_yields, ishigaki_2018_yields, ishigaki18_selected_yields, ww95_1Z, limongi18_yields, nomoto_ck13_yields

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

def load_takahashi(data_dir="data/raw/Takahashi_PISN"):
    """Load Takahashi+19 PISN yield files from the raw CDS/MRT format.

    Each .txt file is one model.  Model type and initial mass are parsed
    from the filename (e.g. nr280.txt -> NR, 280).

    Returns dict: model_type -> list[YieldEntry]
        {"NR": [...], "NM": [...], "MR": [...]}
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    by_model = {}

    for filepath in files:
        with open(filepath) as f:
            lines = f.readlines()

        basename = os.path.splitext(os.path.basename(filepath))[0]
        model_type = basename[:2].upper()
        mass = int(basename[2:])

        yld = {}
        for line in lines:
            if not (line.startswith('NR ') or line.startswith('NM ') or
                    line.startswith('MR ')):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            elem = _isotope_to_element(parts[2])
            try:
                val = float(parts[3])
            except ValueError:
                continue
            yld[elem] = yld.get(elem, 0.0) + val

        if yld:
            entry = {
                "label": f"{model_type} M={mass}",
                "params": {"mass": mass, "model_type": model_type},
                "yields": yld,
            }
            by_model.setdefault(model_type, []).append(entry)

    return by_model


def load_ww95(filepath: str = ww95_1Z) -> list[dict]:
    """Load Woosley & Weaver 1995 yields (Tables 5A+5B).

    TSV with a header row of model labels (S11A ... S40C) and one row per
    isotope. Model labels encode initial mass and model series (A/B/C):
    S30A and S30B share the same mass but differ by series.

    Returns:
    [
        {
            "label": "30A",
            "params": {"mass": 30, "model": "A"},
            "yields": {"H1": 0.1, "He4": 0.2, ...},
        },
        ...
    ]
    """
    with open(filepath) as f:
        lines = f.readlines()

    model_cols = lines[0].strip().split('\t')[3:]

    entries = []
    for ei, model in enumerate(model_cols):
        label = model[1:]
        mass = int(label[:-1])
        model_type = label[-1]

        yld = {}
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 4 + ei:
                continue
            # Row type column (parts[1]) is "yield" in Tables 5A/5B but
            # "isotope" in Tables 12A/12B; the isotope name in parts[2] is
            # the reliable discriminator for yield rows in both files.
            m = re.match(r'(\d+)([A-Za-z]+)', parts[2])
            if not m:
                continue
            iso = m.group(2).capitalize() + m.group(1)
            yld[iso] = float(parts[3 + ei])

        entries.append({
            "label": label,
            "params": {"mass": mass, "model": model_type},
            "yields": yld,
        })

    return entries


def load_limongi18(filepath: str = limongi18_yields) -> list[dict]:
    """Load Limongi & Chieffi 2018 total elemental yields (Set R).

    Explosive yields with unstable nuclei fully decayed (`.dec`), summed
    over isotopes into elements (`ele`).  The file is split into blocks of
    54 lines: one header row naming the nine mass columns, followed by 53
    element rows:

        <element> <Z> <A> <initial> <13M> <15M> ... <120M>

    Each mass column is named `xxxmyyy` where xxx is the initial mass
    (013..120), m encodes [Fe/H] (a=0, b=-1, c=-2, d=-3) and yyy is the
    rotational velocity (000/150/300).  Only the element name and the nine
    yield values are used; the Z/A/initial columns are ignored.

    Models are the ([Fe/H], velocity, mass) combinations: 4 metallicities
    x 3 velocities x 9 initial masses [13, 15, 20, 25, 30, 40, 60, 80, 120].

    Returns:
    [
        {
            "label": "M=13 v=0 [Fe/H]=0",
            "params": {"mass": 13, "velocity": 0, "feh": 0},
            "yields": {"H": 6.16, "He": 4.63, ..., "Bi": 0.0},
        },
        ...
    ]
    """
    masses = [13, 15, 20, 25, 30, 40, 60, 80, 120]
    feh_map = {"a": 0, "b": -1, "c": -2, "d": -3}

    with open(filepath) as f:
        lines = f.readlines()

    # Each block starts with a header row naming the mass columns; the name
    # encodes the model's [Fe/H] and rotational velocity.  Group the element
    # rows that follow by that model so each entry can be built column-wise.
    current_model = None
    rows_by_model = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue

        if parts[0] == "ele":
            col = parts[4]  # e.g. "013a000"
            feh = feh_map[col[3]]
            velocity = int(col[4:])
            current_model = (feh, velocity)
            rows_by_model[current_model] = []
            continue

        name = _isotope_label(parts[0])  # element names pass through as-is
        values = [float(v) for v in parts[4:4 + len(masses)]]
        rows_by_model[current_model].append((name, values))

    entries = []
    for (feh, velocity), rows in rows_by_model.items():
        for mi, mass in enumerate(masses):
            yields = {name: vals[mi] for name, vals in rows}
            entries.append({
                "label": f"M={mass} v={velocity} [Fe/H]={feh}",
                "params": {"mass": mass, "velocity": velocity, "feh": feh},
                "yields": yields,
            })

    return entries


def load_nomoto13(filepath: str = nomoto_ck13_yields) -> list[dict]:
    """Load Nomoto, Kobayashi & Tominaga (2013) yields (YIELD_CK13.dat).

    The file is a sequence of fixed-width blocks, one per metallicity.  Each
    block is:

        Z=     <Z>                  # metallicity (mass fraction)
        M      <m1> ... <mN>        # initial mass (Msun), one per model
        E      <e1> ... <eN>        # explosion energy (10^51 erg)
        Mrem   <r1> ... <rN>        # remnant mass (Msun)
        <elem> <A>  <y1> ... <yN>   # per-isotope ejected mass (Msun)

        Z	      M (M☉) with E=1	            Mrem (M☉)
        0.0010	  13, 15, 18, 20, 25, 30, 40	1.65, 1.53, 1.70, 1.85, 1.91, 2.06, 3.17
        0.0040	  13, 15, 18, 20, 25, 30, 40	1.61, 1.50, 2.143, 1.76, 1.68, 2.56, 2.81
        0.0080	  13, 15, 18, 20, 25, 30, 40	1.606, 1.50, 1.901, 1.67, 1.734, 2.362, 2.552
        0.0200	  13, 15, 18, 20, 25, 30, 40	1.60, 1.50, 1.58, 1.55, 1.804, 2.10, 2.21
        0.0500	  13, 15, 18, 20, 25, 30, 40	1.54, 1.62, 1.46, 1.63, 1.88, 2.32, 2.28

    Returns:
    [
        {
            "label": "Z=0.02 M=20 E=1",
            "params": {"Z": 0.02, "mass": 20.0, "energy": 1.0, "Mrem": 1.66},
            "yields": {"H1": 0.1, "He4": 0.2, ...},
        },
        ...
    ]
    """
    with open(filepath) as f:
        lines = f.readlines()

    entries = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        parts = lines[i].split()
        if not parts or not parts[0].startswith("Z="):
            i += 1
            continue

        z = float(lines[i].split("=")[1])
        masses = [float(v) for v in lines[i + 1].split()[1:]]
        energies = [float(v) for v in lines[i + 2].split()[1:]]
        mrems = [float(v) for v in lines[i + 3].split()[1:]]

        # Collect isotope rows until the next block (or EOF).
        iso_yields = {}  # isotope label -> list of per-model values
        j = i + 4
        while j < n_lines:
            row = lines[j].split()
            if not row:
                j += 1
                continue
            if row[0].startswith("Z="):
                break
            if len(row) < 2 + len(masses):
                j += 1
                continue
            label = _isotope_label(row[0] + row[1])
            iso_yields[label] = [float(v) for v in row[2:]]

            j += 1

        for k, mass in enumerate(masses):
            entries.append({
                "label": f"Z={z:g} M={mass:g} E={energies[k]:g}",
                "params": {
                    "Z": z,
                    "mass": mass,
                    "energy": energies[k],
                    "Mrem": mrems[k],
                },
                "yields": {iso: vals[k] for iso, vals in iso_yields.items()},
            })

        i = j

    return entries