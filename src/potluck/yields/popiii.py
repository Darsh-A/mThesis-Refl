"""Pop III core-collapse yield table from the Ishigaki et al. grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from params import ishigaki_2018_yields
from src.utils import _isotope_to_element

from ..constants import ELEMENTS, ELEMENT_INDEX

_PARAM_NAMES = ("mass", "energy", "Mcut", "MCO", "Mout", "lfej")


class PopIIITable:
    """Event-level Pop III ejecta from the raw Ishigaki files."""

    def __init__(
        self,
        vectors: np.ndarray,
        params: dict[str, np.ndarray],
        isotope_names: tuple[str, ...],
    ) -> None:
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != len(ELEMENTS):
            raise ValueError("vectors must have shape (n_event, n_element)")
        if vectors.shape[0] == 0:
            raise ValueError("the Pop III table is empty")
        if np.any(vectors < 0.0):
            raise ValueError("Pop III yields must be non-negative")

        self.vectors = vectors
        self.params = {key: np.asarray(value) for key, value in params.items()}
        self.isotope_names = isotope_names
        self.masses = self.params["mass"]
        self.energies = self.params["energy"]

        keys = sorted(set(zip(self.masses.tolist(), self.energies.tolist())))
        self.combo_keys = keys
        self.combo_indices = [
            np.flatnonzero((self.masses == mass) & (self.energies == energy))
            for mass, energy in keys
        ]
        self.combo_counts = np.asarray([len(idx) for idx in self.combo_indices])

    @classmethod
    def from_raw(cls, directory: str | Path | None = None) -> "PopIIITable":
        """Parse every ``M*_fort.10.dat`` file in ``directory``."""
        root = Path(directory or ishigaki_2018_yields)
        files = sorted(root.glob("M*_fort.10.dat"))
        if not files:
            raise FileNotFoundError(
                f"No Ishigaki yield files found under {root}; "
                "the raw grid is not tracked by git."
            )

        all_params = []
        all_vectors = []
        isotope_names: tuple[str, ...] | None = None

        for path in files:
            with open(path) as handle:
                header = handle.readline().rstrip("\n").split("\t")
            offset = 1 if header and not header[0].strip() else 0
            names = tuple(header[offset + 6 :])
            if isotope_names is None:
                isotope_names = names

            data = np.loadtxt(path, delimiter="\t", skiprows=1)
            if data.ndim == 1:
                data = data[None, :]
            if data.shape[1] != 7 + len(names):
                raise ValueError(f"Unexpected column count in {path}")

            all_params.append(data[:, 1:7])
            all_vectors.append(_fold_isotopes(data[:, 7:], names))

        params_array = np.vstack(all_params)
        vectors = np.vstack(all_vectors)
        params = {
            name: params_array[:, index]
            for index, name in enumerate(_PARAM_NAMES)
        }
        return cls(vectors, params, isotope_names or ())

    @property
    def n_events(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def n_combos(self) -> int:
        return len(self.combo_keys)

    def sample_indices(
        self,
        rng: np.random.Generator,
        n: int,
        mode: str = "combo",
        weights: tuple[float, ...] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw event rows, either uniformly per combo or uniformly per row."""
        if mode == "row":
            if weights is not None:
                raise ValueError("weights are only supported with mode='combo'")
            return rng.integers(0, self.n_events, size=n, dtype=np.int64)

        if mode != "combo":
            raise ValueError("mode must be 'combo' or 'row'")

        if weights is None:
            probabilities = np.full(self.n_combos, 1.0 / self.n_combos)
        else:
            probabilities = np.asarray(weights, dtype=float)
            if probabilities.size != self.n_combos:
                raise ValueError("weights must have one entry per combo")
            if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
                raise ValueError("weights must be non-negative and non-zero")
            probabilities = probabilities / probabilities.sum()

        chosen = rng.choice(self.n_combos, size=n, p=probabilities)
        indices = np.empty(n, dtype=np.int64)
        for combo in range(self.n_combos):
            mask = chosen == combo
            count = int(mask.sum())
            if count:
                indices[mask] = rng.choice(
                    self.combo_indices[combo], size=count, replace=True
                )
        return indices

    def vectors_at(self, indices: np.ndarray) -> np.ndarray:
        """Return ejecta vectors for event indices."""
        return self.vectors[np.asarray(indices, dtype=np.int64)]

    def row_log_probabilities(
        self,
        weights: tuple[float, ...] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Log probability of each row under the configured sampling mode."""
        if weights is None:
            probabilities = np.full(self.n_combos, 1.0 / self.n_combos)
        else:
            probabilities = np.asarray(weights, dtype=float)
            if probabilities.size != self.n_combos:
                raise ValueError("weights must have one entry per combo")
            if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
                raise ValueError("weights must be non-negative and non-zero")
            probabilities = probabilities / probabilities.sum()

        log_probabilities = np.empty(self.n_events, dtype=float)
        for combo, indices in enumerate(self.combo_indices):
            log_probabilities[indices] = np.log(
                probabilities[combo] / indices.size
            )
        return log_probabilities

    @property
    def group_masses(self) -> np.ndarray:
        """Progenitor mass of each (mass, energy) group."""
        return np.asarray([key[0] for key in self.combo_keys], dtype=float)

    def group_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-group mean and covariance of the element vectors."""
        mean = np.zeros((self.n_combos, len(ELEMENTS)), dtype=float)
        covariance = np.zeros(
            (self.n_combos, len(ELEMENTS), len(ELEMENTS)), dtype=float
        )
        for combo, indices in enumerate(self.combo_indices):
            values = self.vectors[indices]
            mean[combo] = values.mean(axis=0)
            covariance[combo] = np.cov(values, rowvar=False, bias=True)
        return mean, covariance


    def mass_closure(self) -> np.ndarray:
        """Return total ejecta mass divided by progenitor mass."""
        return self.vectors.sum(axis=1) / self.masses

    def check_mass_closure(self, max_fraction: float = 1.05) -> np.ndarray:
        """Raise if any event ejects more mass than its progenitor."""
        fractions = self.mass_closure()
        if np.any(fractions > max_fraction):
            raise ValueError(
                f"Pop III ejecta exceed progenitor mass: {fractions.max():.3f}"
            )
        return fractions

    def combo_summary(self) -> list[dict[str, float | int]]:
        """Return one row per (mass, energy) group."""
        return [
            {
                "mass": float(mass),
                "energy": float(energy),
                "n_rows": int(len(indices)),
            }
            for (mass, energy), indices in zip(self.combo_keys, self.combo_indices)
        ]

    def __len__(self) -> int:
        return self.n_events


def _fold_isotopes(yields: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Sum isotope columns into the common element vector."""
    elements = [_isotope_to_element(name) for name in names]
    indices = np.asarray(
        [ELEMENT_INDEX.get(element, -1) for element in elements], dtype=int
    )
    valid = indices >= 0
    mapping = np.zeros((len(indices), len(ELEMENTS)), dtype=float)
    mapping[np.flatnonzero(valid), indices[valid]] = 1.0
    return yields[:, valid] @ mapping[valid]
