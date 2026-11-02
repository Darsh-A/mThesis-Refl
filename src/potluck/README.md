# Potluck

`potluck` is an inverse-problem generator for metal-poor stellar abundance
patterns. It draws a mixed enrichment history, builds the ejecta of each
source, applies physical cuts, and stores noiseless features, labels, and a
counterfactual detectability label.

The generator is deliberately separate from the Salvadori forward model. The
shared physics is linear mixing in one cloud; the sampling, chronology, and
labels are different. No Salvadori `f_PISN`, `f_ratio`, or star-formation cap
is imposed.

## Model

Three sources are implemented:

| Source | Table | Sampling |
| --- | --- | --- |
| PISN | HW2002 (default), Takahashi MR/NM/NR | count and individual masses |
| Pop III CCSN | raw Ishigaki18 grid | count and individual (mass, energy) events |
| Pop II CCSN | Nomoto13 (default), Limongi18, WW95 | burst mass `M_*II` |

The chronology uses two clocks. Pop III forms at `T=0`, Pop II at `T=Delta`,
and the observed star at `T=Delta+t`. A Pop III event pre-enriches the Pop II
gas only if its lifetime is shorter than `Delta`; it reaches the final mix if
its lifetime is shorter than `Delta+t`. Pop II stars contribute if their
lifetime is shorter than `t`.

The ejecta matrix has one row per source and one column per element. The total
metal mass fixes `Z_* = M_Zpre / M_dil` and `Z_final = Z_* M_Ztot / M_Zpre`.
Features are `[X/Fe]`; `[Fe/H]` is stored for diagnostics but is not a feature.

The detectability label removes the PISN row, recomputes the pattern, and asks
whether the shift exceeds the measurement noise:

```text
D^2 = delta^T C^-1 delta,    y = 1[D > k]
C = diag(sigma_X^2) + sigma_Fe^2 11^T
```

`sigma_X`, `sigma_Fe`, and `k` are the noiseless reference settings in
`DetectConfig`. The dataset's default `y` is computed after the Li+ observation
layer, using only detected ratios and their catalogue errors.

The observation layer samples a real Li+ record matched in `[Fe/H]`, adds
ratio-level noise from that record's `sigma_Total`, copies coverage masks,
censors C upper limits, and forces non-detections below a lower-envelope
`[X/H]` threshold for Si, Mn, Co, and Zn. It reweights the mocks to the Li+
`[Fe/H]` distribution within giants and dwarfs. The observed `[Fe/H]` cut is
applied after this layer.

Ti is excluded by default (`ObservationConfig.exclude_elements=("Ti",)`). Every
tested Pop II table underproduces Ti by 0.6–0.7 dex relative to Li+, and the offset
is not explained by noise, the solar scale, NLTE, metallicity, or S/N. Excluded
elements stay in the stored arrays as missing, so the column layout is unchanged;
the classifier and the support gate use only populated elements. Set
`exclude_elements=()` to restore Ti for the Ti diagnostic.

## Quick start

```bash
# Write a tunable configuration
python -m src.potluck config my_config.json

# Generate a dataset (default: 10,000 accepted samples)
python -m src.potluck generate --config my_config.json --n 10000

# Inspect and validate
python -m src.potluck diagnose data/potluck/dataset.npz
python -m src.potluck validate data/potluck/dataset.npz

# Diagnostics figures, noiseless ceiling, and prior sweeps
python -m src.potluck plots data/potluck/dataset.npz
python -m src.potluck ceiling data/potluck/dataset.npz
python -m src.potluck noise data/potluck/dataset.npz
python -m src.potluck ablation --config my_config.json --n 1000

# Cross-check the PISN + Pop II slice against Salvadori Eq. 13
python -m src.potluck eq13 --config my_config.json
```

The command-line module is `src/potluck/cli.py`; `python -m src.potluck` is
equivalent.

## Configuration

Every numerical bound and prior choice is in `src/potluck/config.py`. The
config is a frozen dataclass and can be written to or loaded from JSON. Use
`dataclasses.replace` for programmatic changes.

Defaults worth knowing:

- `PriorConfig`: PISN count probabilities `(0.5, 0.3, 0.15, 0.05)`, PISN mass
  range 140–270 `Msun`, `Delta` log-uniform over 1–100 Myr, `t` over 3–40 Myr,
  `lambda_III` log-uniform over 0.1–1000, `M_dil` over `2e5–1e9 Msun`, and
  `M_*III + M_*II <= 0.1 M_dil`.
- `YieldConfig`: HW2002, raw Ishigaki18, Nomoto13. The Pop II grid is cached
  under `data/potluck/cache`; Pop II is a binned compound-Poisson sampler.
- `DetectConfig`: `sigma_X=0.2` dex, `sigma_Fe=0.1` dex, `k=3`,
  `min_detected=4` (noiseless reference only).
- `ObservationConfig`: Li+ catalogue, 12 elements, ratio errors from
  `sigma_Total`, C limits, and non-detection thresholds for Si/Mn/Co/Zn.
- `ModelConfig`: 32 stored elements, 15 noiseless feature elements, reference
  `Fe`, relative validity floor `1e-10`, at least 3 valid features.

`z_cr` defaults to `1e-6`; `z_max` defaults to `None`. The final cut is on
observed `[Fe/H]` in `[-3.8, -1.7]`. Set `mstar_total_max_ratio=None` to disable
the joint stellar cap.

## Output

`save_dataset` writes a compressed `.npz` plus `.meta.json` with the config
hash, git commit, element and feature names, cut definitions, and rejection
ledger. Arrays include:

- `features` (14 noiseless ratios), `ratio_mask`, `log_abundance`,
  `independent_features`, `independent_mask`
- `f_zp`, `f_fep`, `f_xp`, `y`, `y_f_05`, `y_f_08`, `D`, `D2`,
  `detect_reason`, `n_detected`
- `obs_features`, `obs_flags`, `obs_sigma`, `obs_mask`,
  `obs_independent_features`, `obs_independent_mask`,
  `obs_independent_sigma`, `feh_obs`, `feh_err`, `logg`, `is_giant`,
  `record_index`, `importance_weight`
- `matrix`, `M_tot`, `M_pre`, `M_z_tot`, `M_z_pre`, `source_metal`
- `z_star`, `z_final`, `feh`, `mdil`, `mstar_ii`, `delta_myr`, `t_myr`
- `n_pisn`, `n_iii`, `n_ia`, `pisn_masses`, `lambda_iii`, `mstar_iii`,
  `iii_group_counts`, `popii_n`, `log_prior`

## Layout

```text
src/potluck/
  config.py       bounds and priors
  constants.py    element list and solar mass fractions
  yield tables    yields/{base,pisn,popiii,popii}.py
  chronology.py   lifetimes and the two-clock gates
  events.py       prior draws
  ejecta.py       source matrix, cuts, Z_*
  features.py     log abundances, ratios, labels
  detect.py       counterfactual D and y
  observation.py  Li+ masks, errors, limits, weights
  dataset.py      rejection sampling and array schema
  io.py           npz + JSON provenance
  diagnostics/    ledger, prior report, plots, noiseless ceiling
  validation.py   Eq. 13, internal checks, ablation
  cli.py          command-line interface
tests/potluck/    unit and integration tests
```

## Limitations

- No SNIa yet. The observation model uses Li+ only; non-C non-detections are
  modelled with a hard lower-envelope threshold, not a probabilistic detection
  model. Only C has catalogue upper limits, so the threshold fit is C-anchored.
- The 100 M⊙ Ishigaki group is a PPI-bracketed CCSN (normal and hypernova
  energies), not a plain CCSN; its weight is the largest prior uncertainty.
- The Pop II table is interpolated in metallicity and time; below the table
  floor it is clamped, and a warning is not raised.
- Structural zeros and tiny yields are handled by a relative validity floor,
  not by rejecting every sample with a zero element. Invalid elements are
  stored as `NaN` with a `False` mask.
- The raw Ishigaki18 grid is gitignored and must exist locally; `PopIIITable`
  raises a clear error if it is missing.
- The model is a simulator, not a fit. Training, calibration, and real-star
  support checks are outside this package.
