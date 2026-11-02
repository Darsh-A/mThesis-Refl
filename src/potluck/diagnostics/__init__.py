"""Diagnostics for generated datasets."""

from .ceiling import ceiling_report, noise_sweep, observed_matrix
from .ledger import format_ledger, ledger_fractions
from .plots import (
    event_mass_summary,
    plot_all,
    plot_ceiling,
    plot_noise_sweep,
    plot_reliability,
    plot_stratified,
    plot_transfer,
)
from .priors import diagnose, format_report

__all__ = [
    "ceiling_report",
    "diagnose",
    "event_mass_summary",
    "format_ledger",
    "format_report",
    "ledger_fractions",
    "noise_sweep",
    "observed_matrix",
    "plot_all",
    "plot_ceiling",
    "plot_noise_sweep",
    "plot_reliability",
    "plot_stratified",
    "plot_transfer",
]
