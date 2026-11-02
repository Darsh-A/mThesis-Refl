"""Draw the enrichment history of one gas cloud."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.special import gammaln

from .config import ModelConfig, PriorConfig
from .yields.base import PISNTable
from .yields.popiii import PopIIITable


@dataclass
class Events:
    """Per-sample event counts, masses, clocks, and dilution mass."""

    n_pisn: np.ndarray
    pisn_masses: np.ndarray
    lambda_iii: np.ndarray
    n_iii: np.ndarray
    iii_group_counts: np.ndarray
    iii_group_offsets: np.ndarray
    iii_sample_index: np.ndarray
    iii_group_index: np.ndarray
    iii_row_index: np.ndarray
    mstar_iii: np.ndarray
    delta_myr: np.ndarray
    t_myr: np.ndarray
    mdil: np.ndarray
    mstar_ii: np.ndarray
    n_ia: np.ndarray
    log_prior: np.ndarray

    def __len__(self) -> int:
        return int(self.n_pisn.size)

    def subset(self, mask: np.ndarray) -> "Events":
        """Return the samples selected by a boolean mask or integer indices."""
        mask = np.asarray(mask)
        if mask.dtype == bool:
            index = np.flatnonzero(mask)
        else:
            index = mask.astype(np.int64)

        mapping = np.full(len(self), -1, dtype=np.int64)
        mapping[index] = np.arange(index.size, dtype=np.int64)
        keep = mapping[self.iii_sample_index] >= 0
        new_sample = mapping[self.iii_sample_index[keep]]
        new_group = self.iii_group_index[keep]
        new_row = self.iii_row_index[keep]

        order = np.argsort(new_group, kind="stable")
        new_sample = new_sample[order]
        new_group = new_group[order]
        new_row = new_row[order]

        group_counts = self.iii_group_counts[index]
        offsets = np.concatenate(
            [[0], np.cumsum(group_counts.sum(axis=0), dtype=np.int64)]
        )
        return Events(
            **{
                field.name: getattr(self, field.name)[index]
                for field in fields(self)
                if field.name
                not in {
                    "iii_group_counts",
                    "iii_group_offsets",
                    "iii_sample_index",
                    "iii_group_index",
                    "iii_row_index",
                }
            },
            iii_group_counts=group_counts,
            iii_group_offsets=offsets,
            iii_sample_index=new_sample,
            iii_group_index=new_group,
            iii_row_index=new_row,
        )


def draw_events(
    n: int,
    config: ModelConfig,
    rng: np.random.Generator,
    pisn_table: PISNTable,
    popiii_table: PopIIITable,
) -> Events:
    """Draw ``n`` enrichment histories from the configured prior."""
    if n < 1:
        raise ValueError("n must be positive")

    prior = config.prior
    n_pisn = _draw_counts(rng, n, prior.n_pisn_probs)

    max_pisn = len(prior.n_pisn_probs) - 1
    mass_lo = max(prior.pisn_mass_range[0], pisn_table.m_min)
    mass_hi = min(prior.pisn_mass_range[1], pisn_table.m_max)
    if mass_lo >= mass_hi:
        raise ValueError("PISN mass range does not overlap the yield table")
    pisn_masses = mass_lo + rng.random((n, max_pisn)) * (mass_hi - mass_lo)
    pisn_masses[np.arange(max_pisn)[None, :] >= n_pisn[:, None]] = 0.0

    lambda_iii = _log_uniform(rng, n, *prior.lambda_iii_range)
    n_iii = rng.poisson(lambda_iii).astype(np.int64)
    if np.any(n_iii > prior.n_iii_max):
        raise ValueError(
            "Pop III count exceeded n_iii_max; increase the bound or lower lambda"
        )

    group_probabilities = _popiii_group_probabilities(popiii_table, prior)
    n_groups = popiii_table.n_combos
    group_counts = np.zeros((n, n_groups), dtype=np.int64)
    for index, count in enumerate(n_iii):
        if count:
            group_counts[index] = rng.multinomial(
                int(count), group_probabilities
            )

    sample_parts = []
    group_parts = []
    row_parts = []
    offsets = [0]
    log_explicit = np.zeros(n, dtype=float)
    for group, rows in enumerate(popiii_table.combo_indices):
        counts = group_counts[:, group]
        total = int(counts.sum())
        if total:
            sample_parts.append(
                np.repeat(np.arange(n, dtype=np.int32), counts)
            )
            group_parts.append(np.full(total, group, dtype=np.int8))
            row_parts.append(
                rng.choice(rows, size=total, replace=True).astype(np.int32)
            )
            log_explicit -= total * np.log(rows.size)
        offsets.append(offsets[-1] + total)

    if sample_parts:
        iii_sample_index = np.concatenate(sample_parts)
        iii_group_index = np.concatenate(group_parts)
        iii_row_index = np.concatenate(row_parts)
    else:
        iii_sample_index = np.empty(0, dtype=np.int32)
        iii_group_index = np.empty(0, dtype=np.int8)
        iii_row_index = np.empty(0, dtype=np.int32)

    mstar_iii = group_counts @ popiii_table.group_masses

    delta_myr = _log_uniform(rng, n, *prior.delta_range_myr)
    t_myr = _log_uniform(rng, n, *prior.t_range_myr)

    mdil_lo = np.full(n, prior.mdil_range[0], dtype=float)
    if prior.mstar_total_max_ratio is not None:
        mdil_lo = np.maximum(
            mdil_lo,
            (mstar_iii + prior.mstar_ii_range[0])
            / prior.mstar_total_max_ratio,
        )
    if np.any(mdil_lo >= prior.mdil_range[1]):
        raise ValueError(
            "dilution-mass range cannot host the sampled Pop III budget"
        )
    mdil = _log_uniform(rng, n, mdil_lo, prior.mdil_range[1])

    mstar_lo = prior.mstar_ii_range[0]
    mstar_hi = np.full(n, prior.mstar_ii_range[1], dtype=float)
    if prior.mstar_total_max_ratio is not None:
        mstar_hi = np.minimum(
            mstar_hi,
            prior.mstar_total_max_ratio * mdil - mstar_iii,
        )
    if prior.mstar_ii_max_ratio is not None:
        mstar_hi = np.minimum(mstar_hi, prior.mstar_ii_max_ratio * mdil)
    if np.any(mstar_hi < mstar_lo):
        raise ValueError("M_starII range is empty for the drawn dilution masses")
    mstar_ii = _log_uniform(rng, n, mstar_lo, mstar_hi)

    log_prior = (
        np.log(np.asarray(prior.n_pisn_probs)[n_pisn])
        + n_pisn * np.log(1.0 / (mass_hi - mass_lo))
        + _log_uniform_logpdf(prior.lambda_iii_range, n)
        + _poisson_logpmf(n_iii, lambda_iii)
        + _multinomial_logpmf(group_counts, group_probabilities)
        + log_explicit
        + _log_uniform_logpdf(prior.delta_range_myr, n)
        + _log_uniform_logpdf(prior.t_range_myr, n)
        + _log_uniform_logpdf((mdil_lo, prior.mdil_range[1]), n)
        + _log_uniform_logpdf((mstar_lo, mstar_hi), n)
    )

    return Events(
        n_pisn=n_pisn,
        pisn_masses=pisn_masses,
        lambda_iii=lambda_iii,
        n_iii=n_iii,
        iii_group_counts=group_counts,
        iii_group_offsets=np.asarray(offsets, dtype=np.int64),
        iii_sample_index=iii_sample_index,
        iii_group_index=iii_group_index,
        iii_row_index=iii_row_index,
        mstar_iii=mstar_iii,
        delta_myr=delta_myr,
        t_myr=t_myr,
        mdil=mdil,
        mstar_ii=mstar_ii,
        n_ia=np.zeros(n, dtype=np.int64),
        log_prior=log_prior,
    )


def _draw_counts(
    rng: np.random.Generator, n: int, probabilities: tuple[float, ...]
) -> np.ndarray:
    """Draw counts in ``0..len(probabilities)-1``."""
    return rng.choice(len(probabilities), size=n, p=np.asarray(probabilities))


def _popiii_group_probabilities(
    table: PopIIITable, prior: PriorConfig
) -> np.ndarray:
    """Return the probability of each (mass, energy) group."""
    if prior.popiii_sampling == "row":
        probabilities = table.combo_counts.astype(float)
    elif prior.popiii_mass_energy_weights is None:
        probabilities = np.ones(table.n_combos, dtype=float)
    else:
        probabilities = np.asarray(
            prior.popiii_mass_energy_weights, dtype=float
        )
        if probabilities.size != table.n_combos:
            raise ValueError("weights must have one entry per combo")
    probabilities = np.asarray(probabilities, dtype=float)
    if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
        raise ValueError("Pop III group weights must be non-negative")
    return probabilities / probabilities.sum()


def _poisson_logpmf(counts: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Natural-log Poisson probability mass function."""
    return counts * np.log(lam) - lam - gammaln(counts + 1.0)


def _multinomial_logpmf(
    counts: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Natural-log multinomial probability mass function."""
    total = counts.sum(axis=1)
    terms = np.where(
        counts > 0,
        counts * np.log(np.where(counts > 0, probabilities, 1.0)),
        0.0,
    )
    return (
        gammaln(total + 1.0)
        - gammaln(counts + 1.0).sum(axis=1)
        + terms.sum(axis=1)
    )


def _log_uniform(
    rng: np.random.Generator,
    n: int,
    low: float | np.ndarray,
    high: float | np.ndarray,
) -> np.ndarray:
    """Draw from a log-uniform distribution on ``[low, high]``."""
    low_arr = np.asarray(low, dtype=float)
    high_arr = np.asarray(high, dtype=float)
    if np.any(low_arr <= 0.0) or np.any(high_arr <= 0.0):
        raise ValueError("log-uniform bounds must be positive")
    if np.any(high_arr < low_arr):
        raise ValueError("log-uniform upper bound is below the lower bound")
    log_low = np.log10(low_arr)
    log_high = np.log10(high_arr)
    return 10.0 ** (log_low + rng.random(n) * (log_high - log_low))


def _log_uniform_logpdf(
    bounds: tuple[float | np.ndarray, float | np.ndarray],
    n: int,
) -> np.ndarray:
    """Natural-log density of a log-uniform distribution."""
    low = np.asarray(bounds[0], dtype=float)
    high = np.asarray(bounds[1], dtype=float)
    ratio = high / low
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(ratio > 1.0, -np.log(np.log(ratio)), 0.0)
    return np.broadcast_to(density, (n,)).copy()
