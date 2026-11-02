"""Pop II yields: IMF-averaged grid and binned compound-Poisson sampler."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.formulas import A_LARSON
from src.salvadori_funcs import salvadori_Y_X_II
from src.yield_sources import get_source

from ..chronology import popii_turnoff
from ..constants import ELEMENTS, ELEMENT_INDEX, METALS
from .base import element_vector

_CACHE_VERSION = 5
_IMF_MIN = 0.1


class PopIITable:
    """Pop II ejecta per unit stellar mass formed.

    The deterministic mode stores the IMF integral
    ``Y_X(t, Z) = integral m_X(m) phi(m) dm`` over the surviving mass range.
    The stochastic mode replaces that integral with the exact compound
    Poisson sum over a discretised IMF,

        n_k ~ Poisson(M_star * nu_k),   M_X = sum_k n_k y_X,k,

    where ``nu_k`` is the number of stars per unit stellar mass in mass bin
    ``k`` and ``y_X,k`` is the yield of one star at the bin representative
    mass.  The bins include every tabulated progenitor mass, so yield kinks
    are not crossed inside a bin.
    """

    def __init__(
        self,
        source_name: str = "Limongi18",
        model: str = "A",
        m_max: float | None = None,
        fe_scale_ww95: float = 0.5,
        z_grid_size: int = 24,
        t_grid_size: int = 25,
        z_min: float = 1.0e-6,
        z_max: float = 5.0e-2,
        z_grid: Sequence[float] | None = None,
        t_min_myr: float = 1.0,
        t_max_myr: float = 100.0,
        z_floor: float = 7.0e-5,
        lower_limit_mode: str = "table",
        popii_stochastic: bool = True,
        cache_dir: str | Path | None = "data/potluck/cache",
        overwrite_cache: bool = False,
    ) -> None:
        if lower_limit_mode not in {"table", "salvadori"}:
            raise ValueError("lower_limit_mode must be 'table' or 'salvadori'")
        if not (0.0 < t_min_myr < t_max_myr):
            raise ValueError("t_min_myr must be below t_max_myr")

        self.source = get_source(source_name)
        self.model = model
        self.m_max = float(self.source.m_max if m_max is None else m_max)
        if self.m_max > self.source.m_max:
            raise ValueError(
                f"m_max={self.m_max} exceeds {source_name} table maximum "
                f"{self.source.m_max}"
            )
        self.fe_scale_ww95 = float(fe_scale_ww95)
        self._custom_z_nodes = z_grid is not None
        if z_grid is None:
            if not (0.0 < z_min < z_max):
                raise ValueError("z_min must be below z_max")
            self.z_grid = np.logspace(
                np.log10(z_min), np.log10(z_max), z_grid_size
            )
        else:
            nodes = np.asarray(z_grid, dtype=float)
            if nodes.ndim != 1 or nodes.size < 2:
                raise ValueError("z_grid must be a 1-D sequence of at least two nodes")
            if np.any(nodes <= 0.0) or np.any(np.diff(nodes) <= 0.0):
                raise ValueError("z_grid must be positive and strictly increasing")
            self.z_grid = nodes
        self.t_grid = np.logspace(
            np.log10(t_min_myr), np.log10(t_max_myr), t_grid_size
        )
        self.lower_limit_mode = lower_limit_mode
        self.popii_stochastic = bool(popii_stochastic)
        self.z_floor = float(z_floor)
        self.table_min_mass = self._find_table_min_mass()
        self.mass_edges = self._make_bin_edges()
        self.bin_masses = np.sqrt(self.mass_edges[:-1] * self.mass_edges[1:])
        self._imf_scale = A_LARSON * 0.35**2.35
        self._gauss_nodes, self._gauss_weights = np.polynomial.legendre.leggauss(8)

        self.cache_path = self._cache_path(cache_dir)
        cached = None
        if self.cache_path is not None and not overwrite_cache:
            cached = self._read_cache(self.cache_path)
        if cached is None:
            cached = self._build_grid()
            if self.cache_path is not None:
                self._write_cache(self.cache_path, cached)

        self.values, self.bin_yields, self.turnoff_grid = cached
        self._interpolator = RegularGridInterpolator(
            (np.log10(self.z_grid), np.log10(self.t_grid)),
            self.values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        self._bin_interpolator = RegularGridInterpolator(
            (np.log10(self.z_grid),),
            self.bin_yields.reshape(self.z_grid.size, -1),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        self._turnoff_interpolator = RegularGridInterpolator(
            (np.log10(self.z_grid), np.log10(self.t_grid)),
            self.turnoff_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def _find_table_min_mass(self) -> float:
        probe = self.source.load_for_metallicity(
            float(np.sqrt(self.z_grid[0] * self.z_grid[-1]))
        )
        feh = probe[0]["params"].get("feh") if probe else None
        masses = [
            entry["params"]["mass"]
            for entry in probe
            if self.source.select(entry, {"model": self.model, "feh": feh})
        ]
        if not masses:
            raise ValueError(f"No usable entries in {self.source.name}")
        return float(min(masses))

    def _make_bin_edges(self) -> np.ndarray:
        """Build mass-bin edges including every tabulated progenitor mass."""
        probe = self.source.load_for_metallicity(
            float(np.sqrt(self.z_grid[0] * self.z_grid[-1]))
        )
        feh = probe[0]["params"].get("feh") if probe else None
        masses = [
            float(entry["params"]["mass"])
            for entry in probe
            if self.source.select(entry, {"model": self.model, "feh": feh})
        ]
        edges = [self.table_min_mass, self.m_max]
        edges.extend(
            mass
            for mass in masses
            if self.table_min_mass <= mass <= self.m_max
        )
        # Fine geometric grid gives a midpoint quadrature with sub-percent
        # accuracy while keeping K ~ 200 and the sampler O(K) per sample.
        edges.extend(
            np.geomspace(self.table_min_mass, self.m_max, 201).tolist()
        )
        return np.unique(np.asarray(edges, dtype=float))

    def _mass_yields(self, data: list[dict], feh: float | None) -> np.ndarray:
        """Return tabulated per-star element yields at one metallicity."""
        masses: list[float] = []
        vectors: list[np.ndarray] = []
        for entry in data:
            if not self.source.select(entry, {"model": self.model, "feh": feh}):
                continue
            masses.append(float(entry["params"]["mass"]))
            vectors.append(element_vector(entry))
        if not masses:
            raise ValueError(f"No usable entries in {self.source.name}")
        order = np.argsort(masses)
        return np.asarray(masses, dtype=float)[order], np.asarray(vectors)[order]

    def _bin_yields_at(
        self, masses: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """Return the IMF-weighted mean yield in every mass bin.

        The bin mean, rather than the yield at the geometric centre, is what
        makes ``sum_k nu_k y_k`` reproduce the deterministic IMF integral
        even when a tabulated yield changes sharply inside a bin.
        """
        left = self.mass_edges[:-1]
        right = self.mass_edges[1:]
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        nodes = mid[:, None] + half[:, None] * self._gauss_nodes[None, :]
        weights = half[:, None] * self._gauss_weights[None, :]
        phi = A_LARSON * (nodes / 0.35) ** (-2.35)
        weighted_phi = weights * phi

        result = np.zeros((left.size, len(ELEMENTS)), dtype=float)
        for index in range(len(ELEMENTS)):
            values = np.interp(
                nodes,
                masses,
                vectors[:, index],
                left=vectors[0, index],
                right=0.0,
            )
            result[:, index] = (weighted_phi * values).sum(axis=1)
        result /= weighted_phi.sum(axis=1)[:, None]
        if self.source.name == "WW95" and self.fe_scale_ww95 != 0.5:
            result[:, ELEMENT_INDEX["Fe"]] *= self.fe_scale_ww95 / 0.5
        return result

    def _build_grid(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.zeros((self.z_grid.size, self.t_grid.size, len(ELEMENTS)))
        bin_yields = np.zeros(
            (self.z_grid.size, self.bin_masses.size, len(ELEMENTS))
        )
        turnoff_grid = np.zeros((self.z_grid.size, self.t_grid.size))
        for iz, metallicity in enumerate(self.z_grid):
            data = self.source.load_for_metallicity(float(metallicity))
            if not data:
                raise ValueError(
                    f"Empty yield table at Z={metallicity:g} for {self.source.name}"
                )
            feh = data[0]["params"].get("feh")
            masses, vectors = self._mass_yields(data, feh)
            bin_yields[iz] = self._bin_yields_at(masses, vectors)
            for it, t_myr in enumerate(self.t_grid):
                turnoff = popii_turnoff(
                    float(t_myr),
                    float(metallicity),
                    self.m_max,
                    z_floor=self.z_floor,
                )
                turnoff_grid[iz, it] = turnoff
                if self.lower_limit_mode == "table":
                    lower = max(turnoff, self.table_min_mass)
                else:
                    lower = turnoff
                if lower >= self.m_max:
                    continue
                if self.lower_limit_mode == "table":
                    lower_edges = np.maximum(self.mass_edges[:-1], lower)
                    upper_edges = self.mass_edges[1:]
                    nu = self._imf_cumulative(upper_edges) - self._imf_cumulative(
                        lower_edges
                    )
                    nu = np.where(
                        upper_edges > lower_edges, np.maximum(nu, 0.0), 0.0
                    )
                    values[iz, it] = nu @ bin_yields[iz]
                else:
                    for ie, element in enumerate(ELEMENTS):
                        values[iz, it, ie] = salvadori_Y_X_II(
                            data,
                            element,
                            lower,
                            model=self.model,
                            m_max=self.m_max,
                            sn_input=self.source,
                            feh=feh,
                        )

        if self.source.name == "WW95" and self.fe_scale_ww95 != 0.5:
            index = ELEMENT_INDEX["Fe"]
            values[..., index] *= self.fe_scale_ww95 / 0.5
        return values, bin_yields, turnoff_grid

    def _cache_path(self, cache_dir: str | Path | None) -> Path | None:
        if cache_dir is None:
            return None
        payload = {
            "version": _CACHE_VERSION,
            "source": self.source.name,
            "model": self.model,
            "m_max": self.m_max,
            "nz": int(self.z_grid.size),
            "nt": int(self.t_grid.size),
            "z_min": float(self.z_grid[0]),
            "z_max": float(self.z_grid[-1]),
            "t_min": float(self.t_grid[0]),
            "t_max": float(self.t_grid[-1]),
            "lower_limit_mode": self.lower_limit_mode,
            "fe_scale_ww95": self.fe_scale_ww95,
        }
        if self._custom_z_nodes:
            payload["z_nodes"] = [float(z) for z in self.z_grid]
        digest = hashlib.md5(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:12]
        return Path(cache_dir) / f"popii_{self.source.name}_{digest}.npz"

    def _read_cache(
        self, path: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                values = np.asarray(data["values"], dtype=float)
                bin_yields = np.asarray(data["bin_yields"], dtype=float)
                turnoff_grid = np.asarray(data["turnoff_grid"], dtype=float)
                z_grid = np.asarray(data["z_grid"], dtype=float)
                t_grid = np.asarray(data["t_grid"], dtype=float)
        except (OSError, ValueError, KeyError):
            return None
        if not (
            np.allclose(z_grid, self.z_grid)
            and np.allclose(t_grid, self.t_grid)
        ):
            return None
        if values.shape != (self.z_grid.size, self.t_grid.size, len(ELEMENTS)):
            return None
        if bin_yields.shape != (
            self.z_grid.size,
            self.bin_masses.size,
            len(ELEMENTS),
        ):
            return None
        if turnoff_grid.shape != (self.z_grid.size, self.t_grid.size):
            return None
        return values, bin_yields, turnoff_grid

    def _write_cache(
        self,
        path: Path,
        cached: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        values, bin_yields, turnoff_grid = cached
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            z_grid=self.z_grid,
            t_grid=self.t_grid,
            values=values,
            bin_yields=bin_yields,
            turnoff_grid=turnoff_grid,
        )

    def integrated(
        self,
        metallicity: np.ndarray | float,
        t_myr: np.ndarray | float,
    ) -> np.ndarray:
        """Return the IMF-averaged ``Y_X(t, Z)`` vectors."""
        z_arr, t_arr = np.broadcast_arrays(
            np.asarray(metallicity, dtype=float),
            np.asarray(t_myr, dtype=float),
        )
        shape = z_arr.shape
        points = np.column_stack(
            [
                np.log10(np.clip(z_arr, self.z_grid[0], self.z_grid[-1])).ravel(),
                np.log10(np.clip(t_arr, self.t_grid[0], self.t_grid[-1])).ravel(),
            ]
        )
        return self._interpolator(points).reshape(shape + (len(ELEMENTS),))

    def _imf_cumulative(self, mass: np.ndarray) -> np.ndarray:
        """Return the number of stars above ``_IMF_MIN`` up to ``mass``."""
        mass = np.maximum(np.asarray(mass, dtype=float), _IMF_MIN)
        return self._imf_scale * (
            mass ** (-1.35) - _IMF_MIN ** (-1.35)
        ) / (-1.35)

    def _binned_components(
        self,
        metallicity: np.ndarray,
        t_myr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(nu, bin_yields, m_lo)`` for a batch of samples."""
        z_arr, t_arr = np.broadcast_arrays(
            np.asarray(metallicity, dtype=float),
            np.asarray(t_myr, dtype=float),
        )
        shape = z_arr.shape
        z_flat = z_arr.ravel()
        t_flat = t_arr.ravel()
        points = np.column_stack(
            [
                np.log10(np.clip(z_flat, self.z_grid[0], self.z_grid[-1])),
                np.log10(np.clip(t_flat, self.t_grid[0], self.t_grid[-1])),
            ]
        )
        # The turnoff is cheap enough to evaluate exactly; interpolating it
        # over the (Z, t) grid biases the partial first bin.
        m_lo = np.asarray(
            [
                popii_turnoff(
                    float(ti),
                    float(zi),
                    self.m_max,
                    z_floor=self.z_floor,
                )
                for zi, ti in zip(z_flat, t_flat)
            ],
            dtype=float,
        )
        # Stochastic sampling is defined on the tabulated mass range; the
        # legacy "salvadori" option remains deterministic only.
        m_lo = np.maximum(m_lo, self.table_min_mass)

        lower = np.maximum(self.mass_edges[:-1][None, :], m_lo[:, None])
        upper = self.mass_edges[1:][None, :]
        nu = self._imf_cumulative(upper) - self._imf_cumulative(lower)
        nu = np.where(upper > lower, np.maximum(nu, 0.0), 0.0)

        z_points = np.log10(
            np.clip(z_flat, self.z_grid[0], self.z_grid[-1])
        )[:, None]
        bin_y = self._bin_interpolator(z_points).reshape(
            z_flat.size, self.bin_masses.size, len(ELEMENTS)
        )
        nu = nu.reshape(shape + (self.bin_masses.size,))
        bin_y = bin_y.reshape(
            shape + (self.bin_masses.size, len(ELEMENTS))
        )

        # The binned mean and the interpolated deterministic grid differ by a
        # few per cent because bin means and grid integrals do not interpolate
        # identically in Z.  Normalise each element so the compound-Poisson
        # mean is exactly the deterministic IMF integral; the correction is
        # small and preserves the event-level yield ratios.
        raw = np.einsum("...k,...kd->...d", nu, bin_y)
        target = self.integrated(z_arr, t_arr)
        factor = np.divide(
            target,
            raw,
            out=np.ones_like(target),
            where=raw > 0.0,
        )
        factor = np.clip(factor, 0.5, 2.0)
        bin_y = bin_y * factor[..., None, :]
        return nu, bin_y, m_lo.reshape(shape)

    def binned_mean(
        self,
        metallicity: np.ndarray | float,
        t_myr: np.ndarray | float,
        mstar_ii: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """Return the exact mean of the binned compound-Poisson sampler."""
        nu, bin_y, _ = self._binned_components(metallicity, t_myr)
        mstar = np.asarray(mstar_ii, dtype=float)
        return np.einsum("...k,...kd->...d", nu * mstar[..., None], bin_y)

    def sample(
        self,
        metallicity: np.ndarray | float,
        t_myr: np.ndarray | float,
        mstar_ii: np.ndarray | float,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw one stochastic Pop II contribution per sample.

        Returns ``(M_II, n_total)`` where ``M_II`` has shape
        ``broadcast(...) + (32,)`` and ``n_total`` is the number of exploded
        CCSN in that sample.  With ``popii_stochastic=False`` the exact
        deterministic mean is returned and ``n_total`` is zero.
        """
        z_arr, t_arr, m_arr = np.broadcast_arrays(
            np.asarray(metallicity, dtype=float),
            np.asarray(t_myr, dtype=float),
            np.asarray(mstar_ii, dtype=float),
        )
        shape = z_arr.shape
        if not self.popii_stochastic:
            return (
                self.integrated(z_arr, t_arr) * m_arr[..., None],
                np.zeros(shape, dtype=np.int64),
            )

        if rng is None:
            rng = np.random.default_rng(0)
        nu, bin_y, _ = self._binned_components(z_arr, t_arr)
        nu_flat = nu.reshape(-1, self.bin_masses.size)
        y_flat = bin_y.reshape(-1, self.bin_masses.size, len(ELEMENTS))
        m_flat = m_arr.ravel()
        result = np.zeros((m_flat.size, len(ELEMENTS)), dtype=float)
        n_total = np.zeros(m_flat.size, dtype=np.int64)
        for start in range(0, m_flat.size, 1024):
            stop = min(start + 1024, m_flat.size)
            lam = m_flat[start:stop, None] * nu_flat[start:stop]
            counts = rng.poisson(lam)
            result[start:stop] = np.einsum(
                "nk,nkd->nd", counts.astype(float), y_flat[start:stop]
            )
            n_total[start:stop] = counts.sum(axis=1)
        return result.reshape(shape + (len(ELEMENTS),)), n_total.reshape(shape)

    def metal_fraction(
        self,
        metallicity: np.ndarray | float,
        t_myr: np.ndarray | float,
    ) -> np.ndarray | float:
        """Return the total metal yield per unit stellar mass formed."""
        indices = [ELEMENT_INDEX[element] for element in METALS]
        result = self.integrated(metallicity, t_myr)[..., indices].sum(axis=-1)
        return float(result) if np.ndim(result) == 0 else result

    def __repr__(self) -> str:
        return (
            f"PopIITable({self.source.name!r}, m_max={self.m_max:g}, "
            f"z=[{self.z_grid[0]:g}, {self.z_grid[-1]:g}], "
            f"t=[{self.t_grid[0]:g}, {self.t_grid[-1]:g}] Myr, "
            f"bins={self.bin_masses.size})"
        )
