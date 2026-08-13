import re

def _get_element(data: dict, isotope: str) -> float:
    """Get the element mass value from data

    Args:
        data: A dictionary containing yield data.
        isotope: The isotope name (e.g., "C12", "He4").
    NOTE: This is meant to use after yield summation, so it will return the mass of the element, not the isotope.
    it may return wrong mass if summation is not done!!!

    Returns:
        The mass of the element
    """
    # Normalize isotope name to element symbol
    element = _isotope_to_element(isotope)

    # Check if the element is in the data
    if element in data["yields"]:
        return data["yields"][element]

    raise ValueError(f"Element {element} not found in data.")

def _isotope_label(raw: str) -> str:
    """Normalize an isotope name like 'h1' or 'he4' to 'H1', 'He4'."""
    name = raw.strip().lower()
    m = re.match(r'([a-z]+)(\d+)', name)
    if not m:
        return raw
    return m.group(1).capitalize() + m.group(2)


def _isotope_to_element(name:str) -> str:
    """Extract element symbol from an isotope name.

    "c12"      -> "C"
    "$^{12}$C" -> "C"   (LaTeX)
    "$^{}$p"   -> "H"   (proton)
    "$^{}$d"   -> "H"   (deuteron)
    """
    name = name.strip()
    name = name.replace('$', '').replace('{', '').replace('}', '').replace('^', '')
    name = name.lstrip('0123456789').lower()
    m = re.match(r'([a-z]+)', name)
    if not m:
        return name
    symbol = m.group(1)
    if symbol in ('p', 'd'):
        return 'H'
    return symbol.capitalize()


def _combine_elements(yields: dict[str, float]) -> dict[str, float]:
    """Combine isotopes into elements, summing their yields.

    For example, if yields contains "C12" and "C13", the result will contain
    a single entry for "C" with the sum of the two yields.
    """
    combined = {}
    for iso, val in yields.items():
        elem = _isotope_to_element(iso)
        combined[elem] = combined.get(elem, 0.0) + val
    return combined
