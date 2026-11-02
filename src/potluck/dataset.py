"""Rejection sampling and assembly of the model dataset."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .config import ModelConfig
from .constants import FE_INDEPENDENT_RATIOS
from .detect import Detection, detect_labels
from .ejecta import Ejecta, build_ejecta
from .events import draw_events
from .features import Features, build_features
from .io import build_meta
from .observation import (
    Observation,
    sample_observation,
    within_feh_range,
)
from .yields.base import PISNTable
from .yields.pisn import load_pisn_table
from .yields.popii import PopIITable
from .yields.popiii import PopIIITable


def build_tables(config: ModelConfig) -> tuple[PISNTable, PopIIITable, PopIITable]:
    """Load the PISN, Pop III, and Pop II yield tables."""
    pisn = load_pisn_table(config.yields.pisn_table)
    popiii = PopIIITable.from_raw()
    popii = PopIITable(
        source_name=config.yields.popii_table,
        model=config.yields.popii_model,
        m_max=config.yields.popii_m_max,
        fe_scale_ww95=config.yields.fe_scale_ww95,
        z_grid_size=config.yields.popii_z_grid_size,
        t_grid_size=config.yields.popii_t_grid_size,
        z_min=config.yields.popii_z_min,
        z_max=config.yields.popii_z_max,
        z_grid=config.yields.popii_z_nodes,
        t_min_myr=config.yields.popii_t_min_myr,
        t_max_myr=config.yields.popii_t_max_myr,
        z_floor=config.yields.raiteri_z_floor,
        popii_stochastic=config.yields.popii_stochastic,
        cache_dir=config.runtime.cache_dir,
        overwrite_cache=config.runtime.overwrite_cache,
    )
    return pisn, popiii, popii


def generate_dataset(
    config: ModelConfig,
    n_samples: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Draw accepted samples until the requested size is reached."""
    if n_samples is not None:
        config = replace(config, runtime=replace(config.runtime, n_samples=n_samples))
    if seed is not None:
        config = replace(config, runtime=replace(config.runtime, seed=seed))

    n_target = config.runtime.n_samples
    rng = np.random.default_rng(config.runtime.seed)
    pisn, popiii, popii = build_tables(config)

    ledger = {
        "drawn": 0,
        "r1": 0,
        "r2": 0,
        "r3": 0,
        "r4": 0,
        "accepted": 0,
        "discarded": 0,
        "r2_no_pre": 0,
        "r2_above_z_max": 0,
        "r2_below_z_cr": 0,
        "r2_other": 0,
        "pisn_drawn": 0,
        "no_pisn_drawn": 0,
        "pisn_r1": 0,
        "no_pisn_r1": 0,
        "pisn_r2": 0,
        "no_pisn_r2": 0,
        "pisn_r3": 0,
        "no_pisn_r3": 0,
        "pisn_r4": 0,
        "no_pisn_r4": 0,
        "pisn_accepted": 0,
        "no_pisn_accepted": 0,
        "pisn_discarded": 0,
        "no_pisn_discarded": 0,
    }
    parts: list[dict[str, np.ndarray]] = []
    max_draw = n_target * config.runtime.max_draw_factor

    while ledger["accepted"] < n_target and ledger["drawn"] < max_draw:
        batch = min(config.runtime.chunk_size, max_draw - ledger["drawn"])
        events = draw_events(batch, config, rng, pisn, popiii)
        ejecta = build_ejecta(events, config, pisn, popiii, popii, rng=rng)
        pisn_present = events.n_pisn > 0
        ledger["drawn"] += batch
        ledger["pisn_drawn"] += int(np.count_nonzero(pisn_present))
        ledger["no_pisn_drawn"] += int(np.count_nonzero(~pisn_present))
        for stage in (1, 2, 3):
            failing = ejecta.stage == stage
            ledger[f"r{stage}"] += int(np.count_nonzero(failing))
            ledger[f"pisn_r{stage}"] += int(np.count_nonzero(failing & pisn_present))
            ledger[f"no_pisn_r{stage}"] += int(
                np.count_nonzero(failing & ~pisn_present)
            )
        r2_mask = ejecta.stage == 2
        no_pre = r2_mask & (ejecta.M_z_pre <= 0.0)
        above = r2_mask & ~no_pre
        if config.prior.z_max is not None:
            above = above & (ejecta.z_star > config.prior.z_max)
        else:
            above = np.zeros_like(r2_mask)
        below = r2_mask & ~no_pre
        if config.prior.z_cr is not None:
            below = below & (ejecta.z_star < config.prior.z_cr)
        else:
            below = np.zeros_like(r2_mask)
        ledger["r2_no_pre"] += int(np.count_nonzero(no_pre))
        ledger["r2_above_z_max"] += int(np.count_nonzero(above))
        ledger["r2_below_z_cr"] += int(np.count_nonzero(below))
        ledger["r2_other"] += int(
            np.count_nonzero(r2_mask & ~(no_pre | above | below))
        )

        accepted = ejecta.accepted
        if np.any(accepted):
            sub = ejecta.subset(accepted)
            sub_present = sub.events.n_pisn > 0
            features = build_features(sub, config)
            observation = sample_observation(features, sub, config, rng)
            r4_ok = within_feh_range(observation, config)
            r4_fail = ~r4_ok
            ledger["r4"] += int(np.count_nonzero(r4_fail))
            ledger["pisn_r4"] += int(np.count_nonzero(r4_fail & sub_present))
            ledger["no_pisn_r4"] += int(
                np.count_nonzero(r4_fail & ~sub_present)
            )

            if np.any(r4_ok):
                sub = sub.subset(r4_ok)
                sub_present = sub_present[r4_ok]
                features = _subset_features(features, np.flatnonzero(r4_ok))
                observation = observation.subset(np.flatnonzero(r4_ok))
                detection = detect_labels(
                    features, sub, config, observation=observation
                )
                remaining = n_target - ledger["accepted"]
                n_keep = len(sub)
                if len(sub) > remaining:
                    n_keep = remaining
                    ledger["discarded"] += len(sub) - remaining
                    ledger["pisn_discarded"] += int(
                        np.count_nonzero(sub_present[n_keep:])
                    )
                    ledger["no_pisn_discarded"] += int(
                        np.count_nonzero(~sub_present[n_keep:])
                    )
                    keep = np.arange(remaining)
                    sub = sub.subset(keep)
                    features = _subset_features(features, keep)
                    observation = observation.subset(keep)
                    detection = _subset_detection(detection, keep)
                ledger["pisn_accepted"] += int(
                    np.count_nonzero(sub_present[:n_keep])
                )
                ledger["no_pisn_accepted"] += int(
                    np.count_nonzero(~sub_present[:n_keep])
                )
                parts.append(
                    _arrays_from(sub, features, detection, observation)
                )
                ledger["accepted"] += len(sub)

        if verbose:
            rate = ledger["accepted"] / ledger["drawn"] if ledger["drawn"] else 0.0
            print(
                f"drawn={ledger['drawn']} accepted={ledger['accepted']} "
                f"acceptance={rate:.2f}",
                flush=True,
            )

    if ledger["accepted"] < n_target:
        raise RuntimeError(
            f"only {ledger['accepted']} of {n_target} samples accepted after "
            f"{ledger['drawn']} draws; ledger={ledger}"
        )

    arrays = {
        name: np.concatenate([part[name] for part in parts], axis=0)
        for name in parts[0]
    }
    feature_names = tuple(
        f"{element}/{config.reference}"
        for element in config.feature_elements
        if element != config.reference
    )
    meta = build_meta(
        config,
        ledger,
        n_drawn=ledger["drawn"],
        n_accepted=ledger["accepted"],
        feature_names=feature_names,
        extra={
            "n_accepted_drawn": ledger["accepted"] + ledger["discarded"],
            "independent_feature_names": tuple(
                f"{numerator}/{denominator}"
                for numerator, denominator in FE_INDEPENDENT_RATIOS
            ),
            "observation_feature_names": tuple(
                f"{element}/{config.reference}"
                for element in config.observation.active_elements
                if element != config.reference
            ),
            "observation_excluded_elements": config.observation.exclude_elements,
            "observation_independent_names": tuple(
                f"{numerator}/{denominator}"
                for numerator, denominator in FE_INDEPENDENT_RATIOS
                if numerator in config.observation.feature_elements
                and denominator in config.observation.feature_elements
            ),
        },
    )
    return arrays, meta


def _arrays_from(
    ejecta: Ejecta,
    features: Features,
    detection: Detection,
    observation: Observation,
) -> dict[str, np.ndarray]:
    """Flatten one accepted batch into storage arrays."""
    events = ejecta.events
    return {
        "features": features.ratios.astype(np.float32),
        "ratio_mask": features.ratio_mask,
        "independent_features": features.independent_ratios.astype(np.float32),
        "independent_mask": features.independent_mask,
        "obs_features": observation.ratios.astype(np.float32),
        "obs_flags": observation.flags.astype(np.int8),
        "obs_sigma": observation.sigma.astype(np.float32),
        "obs_mask": observation.mask.astype(bool),
        "obs_independent_features": observation.independent_ratios.astype(np.float32),
        "obs_independent_mask": observation.independent_mask.astype(bool),
        "obs_independent_sigma": observation.independent_sigma.astype(np.float32),
        "feh_obs": observation.feh_obs.astype(np.float32),
        "feh_err": observation.feh_err.astype(np.float32),
        "logg": observation.logg.astype(np.float32),
        "is_giant": observation.is_giant.astype(bool),
        "record_index": observation.record_index.astype(np.int32),
        "importance_weight": observation.weight.astype(np.float32),
        "log_abundance": features.log_abundance.astype(np.float32),
        "f_zp": features.f_zp.astype(np.float32),
        "f_fep": features.f_fep.astype(np.float32),
        "f_xp": features.f_xp.astype(np.float32),
        "z_pisn": (events.n_pisn > 0).astype(bool),
        "y_f_05": (features.f_zp >= 0.5).astype(bool),
        "y_f_08": (features.f_zp >= 0.8).astype(bool),
        "y": detection.y.astype(bool),
        "D": detection.D.astype(np.float32),
        "D2": detection.D2.astype(np.float32),
        "detect_reason": detection.reason.astype(np.int8),
        "n_detected": detection.n_used.astype(np.int16),
        "matrix": ejecta.matrix.astype(np.float32),
        "M_tot": ejecta.M_tot.astype(np.float32),
        "M_pre": ejecta.M_pre.astype(np.float32),
        "M_z_tot": ejecta.M_z_tot.astype(np.float32),
        "M_z_pre": ejecta.M_z_pre.astype(np.float32),
        "source_metal": ejecta.source_metal.astype(np.float32),
        "valid_features": ejecta.valid_features.astype(bool),
        "z_star": ejecta.z_star.astype(np.float32),
        "z_final": ejecta.z_final.astype(np.float32),
        "feh": ejecta.feh.astype(np.float32),
        "mdil": ejecta.mdil.astype(np.float32),
        "mstar_ii": ejecta.mstar_ii.astype(np.float32),
        "popii_n": ejecta.popii_n.astype(np.int32),
        "delta_myr": events.delta_myr.astype(np.float32),
        "t_myr": events.t_myr.astype(np.float32),
        "n_pisn": events.n_pisn.astype(np.int16),
        "n_iii": events.n_iii.astype(np.int16),
        "lambda_iii": events.lambda_iii.astype(np.float32),
        "mstar_iii": events.mstar_iii.astype(np.float32),
        "iii_group_counts": events.iii_group_counts.astype(np.int16),
        "n_ia": events.n_ia.astype(np.int16),
        "pisn_masses": events.pisn_masses.astype(np.float32),
        "log_prior": events.log_prior.astype(np.float32),
    }


def _subset_features(features: Features, index: np.ndarray) -> Features:
    """Slice a feature bundle by integer index."""
    return Features(
        log_abundance=features.log_abundance[index],
        ratios=features.ratios[index],
        ratio_mask=features.ratio_mask[index],
        independent_ratios=features.independent_ratios[index],
        independent_mask=features.independent_mask[index],
        f_zp=features.f_zp[index],
        f_fep=features.f_fep[index],
        f_xp=features.f_xp[index],
        ratio_names=features.ratio_names,
        independent_names=features.independent_names,
    )


def _subset_detection(detection: Detection, index: np.ndarray) -> Detection:
    """Slice a detection bundle by integer index."""
    return Detection(
        D=detection.D[index],
        D2=detection.D2[index],
        y=detection.y[index],
        reason=detection.reason[index],
        n_used=detection.n_used[index],
    )
