"""Tests for lifetimes and two-clock gates."""

from __future__ import annotations

import numpy as np
import pytest

from src.potluck.chronology import (
    popiii_lifetime,
    popiii_pre_enriches,
    popiii_reaches_final,
    popii_lifetime,
    popii_turnoff,
)


def test_popiii_lifetime_table_points():
    masses = np.array([5.0, 15.0, 40.0, 120.0, 200.0])
    expected = np.array([61.90, 10.40, 3.864, 2.521, 2.204])
    np.testing.assert_allclose(popiii_lifetime(masses), expected, rtol=1.0e-3)


def test_popiii_lifetime_is_monotonic():
    masses = np.logspace(np.log10(5.0), np.log10(300.0), 50)
    lifetimes = popiii_lifetime(masses)
    assert np.all(np.diff(lifetimes) < 0.0)


def test_popii_lifetime_matches_raiteri():
    lifetime = popii_lifetime(20.0, 1.0e-4)
    assert 7.0 < lifetime < 9.0


def test_popii_turnoff_behaviour():
    assert popii_turnoff(1.0, 1.0e-4, 120.0) == 120.0
    turnoff = popii_turnoff(10.0, 1.0e-4, 120.0)
    assert 10.0 < turnoff < 30.0
    assert popii_turnoff(100.0, 1.0e-4, 120.0) < 10.0


def test_two_clock_gates():
    assert bool(popiii_pre_enriches(40.0, 2.0)) is False
    assert bool(popiii_pre_enriches(40.0, 5.0)) is True
    assert bool(popiii_reaches_final(40.0, 1.0, 3.0)) is True
    assert bool(popiii_reaches_final(40.0, 1.0, 1.0)) is False


def test_zero_metallicity_is_floored():
    assert np.isfinite(popii_lifetime(20.0, 0.0))
    assert np.isfinite(popii_turnoff(10.0, 0.0, 120.0))
