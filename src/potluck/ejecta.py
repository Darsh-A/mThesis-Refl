"""Build per-source ejecta matrices and apply the selection cuts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chronology import popiii_lifetime
from .config import ModelConfig
from .constants import ELEMENT_INDEX, METALS, SOLAR_MASS_FRACTION
from .events import Events
from .yields.base import PISNTable
from .yields.popii import PopIITable
from .yields.popiii import PopIIITable

SOURCE_NAMES: tuple[str, ...] = ("PISN", "POPIII", "POPII", "SNIA")
PISN, POPIII, POPII, SNIA = range(4)


@dataclass
class Ejecta:
    """Ejecta of every source plus the cuts applied to one sample."""

    events: Events
    matrix: np.ndarray
    M_tot: np.ndarray
    M_pre: np.ndarray
    M_z_tot: np.ndarray
    M_z_pre: np.ndarray
    z_star: np.ndarray
    z_final: np.ndarray
    mdil: np.ndarray
    mstar_ii: np.ndarray
    popii_n: np.ndarray
    feh: np.ndarray
    M_fe: np.ndarray
    valid_features: np.ndarray
    n_valid_features: np.ndarray
    source_metal: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    r3: np.ndarray
    r4: np.ndarray
    accepted: np.ndarray
    stage: np.ndarray

    def __len__(self) -> int:
        return int(self.M_tot.shape[0])

    def subset(self, mask: np.ndarray) -> "Ejecta":
        """Return the samples selected by a boolean mask or integer indices."""
        mask = np.asarray(mask)
        return Ejecta(
            events=self.events.subset(mask),
            matrix=self.matrix[mask],
            M_tot=self.M_tot[mask],
            M_pre=self.M_pre[mask],
            M_z_tot=self.M_z_tot[mask],
            M_z_pre=self.M_z_pre[mask],
            z_star=self.z_star[mask],
            z_final=self.z_final[mask],
            mdil=self.mdil[mask],
            mstar_ii=self.mstar_ii[mask],
            popii_n=self.popii_n[mask],
            feh=self.feh[mask],
            M_fe=self.M_fe[mask],
            valid_features=self.valid_features[mask],
            n_valid_features=self.n_valid_features[mask],
            source_metal=self.source_metal[mask],
            r1=self.r1[mask],
            r2=self.r2[mask],
            r3=self.r3[mask],
            r4=self.r4[mask],
            accepted=self.accepted[mask],
            stage=self.stage[mask],
        )


def build_ejecta(
    events: Events,
    config: ModelConfig,
    pisn_table: PISNTable,
    popiii_table: PopIIITable,
    popii_table: PopIITable,
    rng: np.random.Generator | None = None,
) -> Ejecta:
    """Sum the event ejecta, gate them by the two clocks, and cut."""
    n = len(events)
    n_element = len(SOLAR_MASS_FRACTION)
    matrix = np.zeros((n, len(SOURCE_NAMES), n_element), dtype=float)

    pisn_vectors, tau_pisn = _pisn_events(events, pisn_table)
    delta = events.delta_myr
    horizon = delta + events.t_myr
    pre_pisn = (tau_pisn <= delta[:, None]) & (events.pisn_masses > 0.0)
    final_pisn = (tau_pisn <= horizon[:, None]) & (events.pisn_masses > 0.0)

    M_pre = (pisn_vectors * pre_pisn[..., None]).sum(axis=1)
    matrix[:, PISN] = (pisn_vectors * final_pisn[..., None]).sum(axis=1)

    iii_final, iii_pre = _popiii_sums(events, popiii_table, delta, horizon)
    M_pre += iii_pre
    matrix[:, POPIII] = iii_final

    metal_indices = [ELEMENT_INDEX[element] for element in METALS]
    M_z_pre = M_pre[:, metal_indices].sum(axis=1)

    r1 = (events.n_pisn + events.n_iii) > 0
    z_star = np.zeros(n, dtype=float)
    np.divide(M_z_pre, events.mdil, out=z_star, where=M_z_pre > 0.0)

    r2 = r1 & (M_z_pre > 0.0) & np.isfinite(z_star)
    if config.prior.z_cr is not None:
        r2 &= z_star >= config.prior.z_cr
    if config.prior.z_max is not None:
        r2 &= z_star <= config.prior.z_max

    popii_mask = r1 & (M_z_pre > 0.0)
    popii_n = np.zeros(n, dtype=np.int64)
    if np.any(popii_mask):
        if rng is None:
            rng = np.random.default_rng(0)
        popii_ejecta, popii_n[popii_mask] = popii_table.sample(
            z_star[popii_mask],
            events.t_myr[popii_mask],
            events.mstar_ii[popii_mask],
            rng,
        )
        matrix[popii_mask, POPII] = popii_ejecta

    M_tot = matrix.sum(axis=1)
    M_z_tot = M_tot[:, metal_indices].sum(axis=1)

    ratio = np.zeros(n, dtype=float)
    np.divide(M_z_tot, M_z_pre, out=ratio, where=M_z_pre > 0.0)
    z_final = z_star * ratio

    M_fe = M_tot[:, ELEMENT_INDEX["Fe"]].copy()
    feh = np.full(n, np.nan, dtype=float)
    positive_fe = M_fe > 0.0
    feh[positive_fe] = np.log10(
        M_fe[positive_fe]
        / events.mdil[positive_fe]
        / SOLAR_MASS_FRACTION["Fe"]
    )

    feature_indices = np.asarray(
        [ELEMENT_INDEX[element] for element in config.feature_elements]
    )
    feature_fraction = np.zeros((n, feature_indices.size), dtype=float)
    np.divide(
        M_tot[:, feature_indices],
        M_z_tot[:, None],
        out=feature_fraction,
        where=M_z_tot[:, None] > 0.0,
    )
    valid_features = (
        np.isfinite(feature_fraction)
        & (feature_fraction >= config.min_element_fraction)
    )
    n_valid = valid_features.sum(axis=1)

    fe_fraction = np.zeros(n, dtype=float)
    np.divide(M_fe, M_z_tot, out=fe_fraction, where=M_z_tot > 0.0)
    fe_valid = np.isfinite(fe_fraction) & (
        fe_fraction >= config.min_fe_fraction
    )

    r3 = r2 & fe_valid & (n_valid >= config.min_valid_features)
    r4 = r3.copy()

    stage = np.full(n, 3, dtype=np.int8)
    stage[~r1] = 1
    stage[r1 & ~r2] = 2
    stage[r2 & ~r3] = 3
    stage[r3] = 0

    return Ejecta(
        events=events,
        matrix=matrix,
        M_tot=M_tot,
        M_pre=M_pre,
        M_z_tot=M_z_tot,
        M_z_pre=M_z_pre,
        z_star=z_star,
        z_final=z_final,
        mdil=events.mdil.copy(),
        mstar_ii=events.mstar_ii.copy(),
        popii_n=popii_n,
        feh=feh,
        M_fe=M_fe,
        valid_features=valid_features,
        n_valid_features=n_valid,
        source_metal=matrix[:, :, metal_indices].sum(axis=2),
        r1=r1,
        r2=r2,
        r3=r3,
        r4=r4,
        accepted=r3,
        stage=stage,
    )


def _pisn_events(
    events: Events, table: PISNTable
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded PISN vectors and their lifetimes."""
    masses = events.pisn_masses
    n, max_events = masses.shape
    vectors = np.zeros((n, max_events, len(SOLAR_MASS_FRACTION)), dtype=float)
    valid = masses > 0.0
    if np.any(valid):
        vectors[valid] = table.at_masses(masses[valid])
    tau = np.asarray(popiii_lifetime(masses), dtype=float)
    return vectors, tau


def _popiii_sums(
    events: Events,
    table: PopIIITable,
    delta: np.ndarray,
    horizon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return final and pre-enrichment Pop III sums from all events."""
    n = len(events)
    n_element = len(SOLAR_MASS_FRACTION)
    final = np.zeros((n, n_element), dtype=float)
    pre = np.zeros((n, n_element), dtype=float)
    counts = events.iii_group_counts
    group_tau = np.asarray(popiii_lifetime(table.group_masses), dtype=float)

    for group in range(table.n_combos):
        start = int(events.iii_group_offsets[group])
        stop = int(events.iii_group_offsets[group + 1])
        if start == stop:
            continue
        sample_index = events.iii_sample_index[start:stop]
        row_index = events.iii_row_index[start:stop]
        group_sum = np.zeros((n, n_element), dtype=float)
        np.add.at(group_sum, sample_index, table.vectors[row_index])

        pre_active = (counts[:, group] > 0) & (group_tau[group] <= delta)
        final_active = (counts[:, group] > 0) & (
            group_tau[group] <= horizon
        )
        final[final_active] += group_sum[final_active]
        pre[pre_active] += group_sum[pre_active]
    return final, pre
