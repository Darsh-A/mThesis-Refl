"""Common yield-vector helpers and PISN interpolation."""

from __future__ import annotations

import numpy as np

from src.utils import _combine_elements

from ..constants import ELEMENTS, ELEMENT_INDEX


def element_vector(payload: dict) -> np.ndarray:
    """Return a 32-element mass vector in ``ELEMENTS`` order.

    ``payload`` may be an entry with a ``yields`` mapping or a raw mapping.
    Isotopes are folded into elements; masses are in solar masses.
    """
    yields = payload["yields"] if "yields" in payload else payload
    combined = _combine_elements(yields)
    vector = np.zeros(len(ELEMENTS), dtype=float)
    for element, mass in combined.items():
        index = ELEMENT_INDEX.get(element)
        if index is not None:
            vector[index] = float(mass)
    return vector


class PISNTable:
    """A one-dimensional grid of PISN ejecta vectors."""

    def __init__(self, name: str, masses: np.ndarray, vectors: np.ndarray) -> None:
        masses = np.asarray(masses, dtype=float)
        vectors = np.asarray(vectors, dtype=float)
        if masses.ndim != 1 or vectors.ndim != 2:
            raise ValueError("masses must be 1-D and vectors 2-D")
        if vectors.shape != (masses.size, len(ELEMENTS)):
            raise ValueError("vectors must have shape (n_mass, n_element)")
        if masses.size < 2:
            raise ValueError("at least two PISN masses are needed")

        order = np.argsort(masses)
        self.name = name
        self.masses = masses[order]
        self.vectors = vectors[order]
        if np.any(np.diff(self.masses) <= 0.0):
            raise ValueError("PISN masses must be unique")
        if np.any(self.vectors < 0.0):
            raise ValueError("PISN yields must be non-negative")

        positive = np.all(self.vectors > 0.0, axis=0)
        dynamic_range = np.ones(self.vectors.shape[1])
        safe = positive.copy()
        dynamic_range[safe] = (
            self.vectors[:, safe].max(axis=0) / self.vectors[:, safe].min(axis=0)
        )
        self.log_interp = positive & (dynamic_range > 1.0e3)

    @classmethod
    def from_entries(cls, entries: list[dict], name: str) -> "PISNTable":
        """Build a table from loader entries."""
        masses = []
        vectors = []
        for entry in entries:
            params = entry.get("params", {})
            mass = params.get("mass")
            if mass is None:
                label = str(entry.get("label", ""))
                mass = float(label.split("=")[-1])
            masses.append(float(mass))
            vectors.append(element_vector(entry))
        return cls(name, np.asarray(masses), np.asarray(vectors))

    @property
    def m_min(self) -> float:
        return float(self.masses[0])

    @property
    def m_max(self) -> float:
        return float(self.masses[-1])

    def at_masses(self, masses: np.ndarray | float) -> np.ndarray:
        """Interpolate ejecta vectors at one or more PISN masses."""
        query = np.atleast_1d(np.asarray(masses, dtype=float))
        if np.any(query < self.m_min) or np.any(query > self.m_max):
            raise ValueError(
                f"PISN mass outside table range [{self.m_min}, {self.m_max}]"
            )

        out = np.empty((query.size, len(ELEMENTS)), dtype=float)
        for index in range(len(ELEMENTS)):
            values = self.vectors[:, index]
            if self.log_interp[index]:
                out[:, index] = np.exp(
                    np.interp(query, self.masses, np.log(values))
                )
            else:
                out[:, index] = np.interp(query, self.masses, values)
        return out[0] if np.ndim(masses) == 0 else out

    def at_mass(self, mass: float) -> np.ndarray:
        """Interpolate the ejecta vector at a single PISN mass."""
        return self.at_masses(mass)

    def mass_closure(self) -> np.ndarray:
        """Return total ejecta mass divided by progenitor mass."""
        return self.vectors.sum(axis=1) / self.masses

    def check_mass_closure(self, max_fraction: float = 1.05) -> np.ndarray:
        """Raise if any entry ejects more mass than its progenitor."""
        fractions = self.mass_closure()
        if np.any(fractions > max_fraction):
            raise ValueError(
                f"{self.name} ejecta exceed progenitor mass: {fractions.max():.3f}"
            )
        return fractions

    def __len__(self) -> int:
        return int(self.masses.size)
