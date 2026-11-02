"""Tunable model configuration.

Every numerical bound used by the generator lives here.  The defaults are
choices of this model, not constraints inherited from Salvadori et al.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_FEATURE_ELEMENTS,
    LI_FEATURE_ELEMENTS,
    METALS,
    REFERENCE,
)


@dataclass(frozen=True)
class PriorConfig:
    """Priors for event counts, masses, clocks, and dilution."""

    n_pisn_probs: tuple[float, ...] = (0.5, 0.3, 0.15, 0.05)
    lambda_iii_range: tuple[float, float] = (0.1, 1000.0)
    n_iii_max: int = 4096
    pisn_mass_range: tuple[float, float] = (140.0, 270.0)
    delta_range_myr: tuple[float, float] = (1.0, 100.0)
    t_range_myr: tuple[float, float] = (3.0, 40.0)
    mdil_range: tuple[float, float] = (2.0e5, 1.0e9)
    mstar_ii_range: tuple[float, float] = (1.0e3, 1.0e7)
    mstar_ii_max_ratio: float | None = None
    mstar_total_max_ratio: float | None = 0.1
    z_max: float | None = None
    z_cr: float | None = 1.0e-6
    feh_range: tuple[float, float] | None = (-3.8, -1.7)
    popiii_sampling: str = "combo"
    popiii_mass_energy_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        probs = self.n_pisn_probs
        if len(probs) < 1:
            raise ValueError("n_pisn_probs must not be empty")
        if any(p < 0.0 for p in probs):
            raise ValueError("n_pisn_probs must be non-negative")
        if not math.isclose(sum(probs), 1.0, abs_tol=1e-6):
            raise ValueError("n_pisn_probs must sum to one")
        for name, bounds in (
            ("lambda_iii_range", self.lambda_iii_range),
            ("pisn_mass_range", self.pisn_mass_range),
            ("delta_range_myr", self.delta_range_myr),
            ("t_range_myr", self.t_range_myr),
            ("mdil_range", self.mdil_range),
            ("mstar_ii_range", self.mstar_ii_range),
        ):
            lo, hi = bounds
            if not (0.0 < lo < hi):
                raise ValueError(f"{name} must satisfy 0 < min < max")
        if self.n_iii_max < 1:
            raise ValueError("n_iii_max must be at least one")
        if self.mstar_ii_max_ratio is not None and self.mstar_ii_max_ratio <= 0.0:
            raise ValueError("mstar_ii_max_ratio must be positive or None")
        if self.mstar_total_max_ratio is not None and self.mstar_total_max_ratio <= 0.0:
            raise ValueError("mstar_total_max_ratio must be positive or None")
        if self.z_max is not None and self.z_max <= 0.0:
            raise ValueError("z_max must be positive or None")
        if self.z_cr is not None and self.z_cr < 0.0:
            raise ValueError("z_cr must be non-negative or None")
        if self.feh_range is not None:
            lo, hi = self.feh_range
            if not (lo < hi):
                raise ValueError("feh_range must satisfy min < max")
        if self.popiii_sampling not in {"combo", "row"}:
            raise ValueError("popiii_sampling must be 'combo' or 'row'")
        if self.popiii_mass_energy_weights is not None:
            weights = self.popiii_mass_energy_weights
            if any(w < 0.0 for w in weights) or sum(weights) <= 0.0:
                raise ValueError("popiii_mass_energy_weights must be non-negative")


@dataclass(frozen=True)
class YieldConfig:
    """Yield-table choices."""

    pisn_table: str = "HW2002"
    popiii_table: str = "Ishigaki18"
    popii_table: str = "Limongi18"
    popii_model: str = "A"
    fe_scale_ww95: float = 0.5
    popii_m_max: float | None = None
    popii_z_grid_size: int = 4
    popii_t_grid_size: int = 25
    popii_z_min: float = 1.0e-6
    popii_z_max: float = 5.0e-2
    popii_z_nodes: tuple[float, ...] | None = (
        3.236e-5,
        3.236e-4,
        3.236e-3,
        1.345e-2,
    )
    popii_t_min_myr: float = 3.0
    popii_t_max_myr: float = 40.0
    raiteri_z_floor: float = 7.0e-5
    popii_cache: bool = True
    popii_stochastic: bool = True

    def __post_init__(self) -> None:
        if self.popii_z_grid_size < 2 or self.popii_t_grid_size < 2:
            raise ValueError("Pop II grid sizes must be at least two")
        if not (0.0 < self.popii_z_min < self.popii_z_max):
            raise ValueError("popii_z_min must be below popii_z_max")
        if self.popii_z_nodes is not None:
            nodes = self.popii_z_nodes
            if len(nodes) < 2 or any(z <= 0.0 for z in nodes):
                raise ValueError("popii_z_nodes must hold at least two positive values")
            if any(b <= a for a, b in zip(nodes[:-1], nodes[1:])):
                raise ValueError("popii_z_nodes must be strictly increasing")
        if not (0.0 < self.popii_t_min_myr < self.popii_t_max_myr):
            raise ValueError("popii_t_min_myr must be below popii_t_max_myr")
        if self.raiteri_z_floor <= 0.0:
            raise ValueError("raiteri_z_floor must be positive")
        if self.popii_m_max is not None and self.popii_m_max <= 0.0:
            raise ValueError("popii_m_max must be positive or None")


@dataclass(frozen=True)
class DetectConfig:
    """Placeholder noise model for the counterfactual detection label."""

    sigma_dex: float = 0.2
    sigma_fe_dex: float = 0.1
    k: float = 3.0
    min_detected: int = 4
    inject_noise: bool = False

    def __post_init__(self) -> None:
        if self.sigma_dex <= 0.0 or self.sigma_fe_dex <= 0.0:
            raise ValueError("noise scales must be positive")
        if self.k <= 0.0:
            raise ValueError("k must be positive")
        if self.min_detected < 1:
            raise ValueError("min_detected must be at least one")


@dataclass(frozen=True)
class ObservationConfig:
    """Li+ observation model settings."""

    catalogue: str = "Li"
    feature_elements: tuple[str, ...] = LI_FEATURE_ELEMENTS
    exclude_elements: tuple[str, ...] = ()
    feh_kernel_dex: float = 0.2
    c_mode: str = "inflate"
    c_giant_sigma_dex: float = 0.3
    default_sigma_dex: float = 0.2
    non_detection_mode: str = "threshold"
    non_detection_elements: tuple[str, ...] = ("Si", "Mn", "Co", "Zn")
    non_detection_quantile: float = 0.0
    non_detection_snr_bins: int = 6
    weight_by_feh: bool = True
    weight_clip: float = 20.0

    @property
    def active_elements(self) -> tuple[str, ...]:
        """Observation elements that are synthesised and modelled."""
        return tuple(
            element
            for element in self.feature_elements
            if element not in self.exclude_elements
        )

    def __post_init__(self) -> None:
        if self.catalogue not in {"Li"}:
            raise ValueError("only the Li+ catalogue is implemented")
        if not self.feature_elements:
            raise ValueError("observation feature_elements must not be empty")
        if set(self.feature_elements) - set(LI_FEATURE_ELEMENTS):
            raise ValueError("observation elements must be a subset of Li+ elements")
        if REFERENCE not in self.feature_elements:
            raise ValueError("observation features must include Fe")
        if set(self.exclude_elements) - set(self.feature_elements):
            raise ValueError("excluded elements must be observation features")
        if REFERENCE in self.exclude_elements:
            raise ValueError("the reference element cannot be excluded")
        if self.feh_kernel_dex <= 0.0:
            raise ValueError("feh_kernel_dex must be positive")
        if self.c_mode not in {"inflate", "drop", "keep"}:
            raise ValueError("c_mode must be 'inflate', 'drop', or 'keep'")
        if self.c_giant_sigma_dex < 0.0:
            raise ValueError("c_giant_sigma_dex must be non-negative")
        if self.default_sigma_dex <= 0.0:
            raise ValueError("default_sigma_dex must be positive")
        if self.non_detection_mode not in {"copy", "threshold"}:
            raise ValueError(
                "non_detection_mode must be 'copy' or 'threshold'"
            )
        if set(self.non_detection_elements) - set(self.feature_elements):
            raise ValueError(
                "non_detection_elements must be observation features"
            )
        if not (0.0 <= self.non_detection_quantile < 1.0):
            raise ValueError("non_detection_quantile must lie in (0, 1)")
        if self.non_detection_snr_bins < 2:
            raise ValueError("non_detection_snr_bins must be at least two")
        if self.weight_clip <= 1.0:
            raise ValueError("weight_clip must exceed one")


@dataclass(frozen=True)
class RuntimeConfig:
    """Sampling and storage settings."""

    seed: int = 42
    n_samples: int = 10000
    chunk_size: int = 5000
    max_draw_factor: int = 20
    cache_dir: str = "data/potluck/cache"
    output_dir: str = "data/potluck"
    overwrite_cache: bool = False

    def __post_init__(self) -> None:
        if self.n_samples < 1 or self.chunk_size < 1:
            raise ValueError("n_samples and chunk_size must be positive")
        if self.max_draw_factor < 1:
            raise ValueError("max_draw_factor must be at least one")


@dataclass(frozen=True)
class ModelConfig:
    """Top-level configuration for the generator."""

    feature_elements: tuple[str, ...] = DEFAULT_FEATURE_ELEMENTS
    reference: str = REFERENCE
    min_valid_features: int = 3
    min_element_fraction: float = 1e-10
    min_fe_fraction: float = 1e-10
    prior: PriorConfig = field(default_factory=PriorConfig)
    yields: YieldConfig = field(default_factory=YieldConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        if not self.feature_elements:
            raise ValueError("feature_elements must not be empty")
        missing = set(self.feature_elements) - set(METALS)
        if missing:
            raise ValueError(f"Non-metal or unknown feature elements: {sorted(missing)}")
        if len(set(self.feature_elements)) != len(self.feature_elements):
            raise ValueError("feature_elements contains duplicates")
        if self.reference not in self.feature_elements:
            raise ValueError("reference must be one of the feature elements")
        if not (1 <= self.min_valid_features <= len(self.feature_elements)):
            raise ValueError("min_valid_features is outside the feature set")
        if self.min_element_fraction <= 0.0 or self.min_fe_fraction <= 0.0:
            raise ValueError("mass fractions must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Build a configuration from a dictionary of overrides."""
        data = dict(data)
        data["feature_elements"] = tuple(data.get("feature_elements", DEFAULT_FEATURE_ELEMENTS))

        prior_data = dict(data.get("prior", {}))
        for key in (
            "n_pisn_probs",
            "n_iii_probs",
            "pisn_mass_range",
            "delta_range_myr",
            "t_range_myr",
            "mdil_range",
            "mstar_ii_range",
        ):
            if key in prior_data:
                prior_data[key] = tuple(prior_data[key])
        if prior_data.get("popiii_mass_energy_weights") is not None:
            prior_data["popiii_mass_energy_weights"] = tuple(
                prior_data["popiii_mass_energy_weights"]
            )
        data["prior"] = PriorConfig(**prior_data)
        data["yields"] = YieldConfig(**data.get("yields", {}))
        data["detect"] = DetectConfig(**data.get("detect", {}))
        observation_data = dict(data.get("observation", {}))
        for key in ("feature_elements", "exclude_elements"):
            if key in observation_data:
                observation_data[key] = tuple(observation_data[key])
        data["observation"] = ObservationConfig(**observation_data)
        data["runtime"] = RuntimeConfig(**data.get("runtime", {}))
        return cls(**data)


def load_config(path: str) -> ModelConfig:
    """Load a configuration from JSON, filling unspecified fields with defaults."""
    with open(path) as handle:
        return ModelConfig.from_dict(json.load(handle))


def save_config(path: str, config: ModelConfig) -> None:
    """Write a configuration to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(config.to_dict(), handle, indent=2)
