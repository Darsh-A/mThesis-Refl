"""Simulation-Based Inference (SBI) for the PISN f_pisn problem.

The Salvadori+2019 forward model (PISN + SNII abundance ratios) acts as the
simulator: given parameters (f_pisn, f_ratio, tpop2, M_pisn) it produces
abundance-ratio observables.  We train a neural density estimator to learn the
posterior P(theta | observed ratios) without an analytic likelihood.

test v1.0 = 1-D inference of f_pisn only (see results/testv1.0/).
"""
