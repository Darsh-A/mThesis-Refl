"""Checks for the binned compound-Poisson Pop II sampler."""

import numpy as np
from scipy.integrate import quad
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.formulas import larson_imf
from src.potluck.yields.popii import PopIITable


def _table(stochastic: bool = True) -> PopIITable:
    return PopIITable(
        cache_dir=None,
        z_grid_size=6,
        t_grid_size=6,
        popii_stochastic=stochastic,
    )


def test_binned_mean_matches_deterministic_grid() -> None:
    table = _table()
    for metallicity, t_myr in (
        (1.0e-3, 10.0),
        (1.0e-3, 20.0),
        (5.0e-3, 10.0),
        (1.0e-4, 30.0),
    ):
        mean = table.binned_mean(metallicity, t_myr, 1.0)
        target = table.integrated(metallicity, t_myr)
        assert np.allclose(mean, target, rtol=1e-9, atol=1e-30)


def test_binned_mean_matches_high_resolution_integral() -> None:
    table = _table()
    metallicity = float(table.z_grid[3])
    t_myr = float(table.t_grid[3])
    data = table.source.load_for_metallicity(metallicity)
    feh = data[0]["params"].get("feh")
    masses, vectors = table._mass_yields(data, feh)
    lower = np.maximum(
        table.mass_edges[:-1], table.table_min_mass
    )
    upper = table.mass_edges[1:]
    nu = table._imf_cumulative(upper) - table._imf_cumulative(lower)
    nu = np.where(upper > lower, np.maximum(nu, 0.0), 0.0)
    raw = nu @ table._bin_yields_at(masses, vectors)

    breakpoints = masses[(masses > table.table_min_mass) & (masses < table.m_max)]
    for index, reference in enumerate(raw):
        if reference <= 0.0:
            continue
        values = np.interp(
            masses,
            masses,
            vectors[:, index],
            left=vectors[0, index],
            right=0.0,
        )
        integral = quad(
            lambda mass: float(np.interp(mass, masses, values)) * larson_imf(mass),
            table.table_min_mass,
            table.m_max,
            points=breakpoints,
            limit=200,
        )[0]
        assert abs(reference - integral) / reference < 0.01


def test_compound_poisson_mean_and_covariance() -> None:
    table = _table()
    metallicity, t_myr, mstar = 1.0e-3, 10.0, 3000.0
    nu, bin_y, _ = table._binned_components(
        np.array([metallicity]), np.array([t_myr])
    )
    lam = mstar * nu[0]
    analytic_mean = np.einsum("k,kd->d", lam, bin_y[0])
    analytic_cov = np.einsum("k,kd,ke->de", lam, bin_y[0], bin_y[0])

    draws, _ = table.sample(
        np.full(4000, metallicity),
        np.full(4000, t_myr),
        np.full(4000, mstar),
        np.random.default_rng(5),
    )
    sample_mean = draws.mean(axis=0)
    sample_cov = np.cov(draws, rowvar=False)
    assert np.allclose(sample_mean, analytic_mean, rtol=0.1, atol=1e-8)
    scale = np.sqrt(np.diag(analytic_cov))
    relative = np.abs(np.diag(sample_cov) - np.diag(analytic_cov)) / np.maximum(
        scale**2, 1e-30
    )
    assert relative.max() < 0.2


def test_distribution_matches_naive_event_loop() -> None:
    table = _table()
    metallicity, t_myr = 1.0e-3, 10.0
    nu, bin_y, _ = table._binned_components(
        np.array([metallicity]), np.array([t_myr])
    )
    nu = nu[0]
    bin_y = bin_y[0]
    total_rate = nu.sum()
    mstar = 5.0 / total_rate
    n_sample = 5000
    rng = np.random.default_rng(17)

    binned, _ = table.sample(
        np.full(n_sample, metallicity),
        np.full(n_sample, t_myr),
        np.full(n_sample, mstar),
        rng,
    )

    probabilities = nu / total_rate
    naive = np.zeros_like(binned)
    for index in range(n_sample):
        n_events = rng.poisson(mstar * total_rate)
        bins = rng.choice(nu.size, size=n_events, p=probabilities)
        naive[index] = bin_y[bins].sum(axis=0)

    features = np.vstack([binned / mstar, naive / mstar])
    labels = np.r_[np.zeros(n_sample), np.ones(n_sample)]
    splits = StratifiedKFold(3, shuffle=True, random_state=3)
    probability = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=120, max_depth=3),
        features,
        labels,
        cv=splits,
        method="predict_proba",
    )[:, 1]
    auc = roc_auc_score(labels, probability)
    assert abs(auc - 0.5) <= 0.02, auc


def test_deterministic_mode_matches_integrated() -> None:
    table = _table(stochastic=False)
    z = np.array([1.0e-3, 5.0e-3])
    t = np.array([10.0, 20.0])
    mstar = np.array([1.0e3, 5.0e3])
    ejecta, n_total = table.sample(z, t, mstar, np.random.default_rng(1))
    assert np.allclose(ejecta, table.integrated(z, t) * mstar[:, None])
    assert np.all(n_total == 0)


def test_zero_popii_fraction() -> None:
    table = _table()
    z = np.full(200, 1.0e-3)
    early, n_early = table.sample(
        z, np.full(200, 1.0), np.full(200, 1.0e4), np.random.default_rng(2)
    )
    late, n_late = table.sample(
        z, np.full(200, 10.0), np.full(200, 1.0e4), np.random.default_rng(2)
    )
    assert np.all(early == 0.0)
    assert np.all(n_early == 0)
    assert np.any(late > 0.0)
    assert np.any(n_late > 0)
