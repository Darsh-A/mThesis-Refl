# Classifier diagnostics v1

Versioned results, not unit tests. Generated from
`data/potluck/analysis/v8_threshold_100k.npz` with the Li+ observation model,
100,000 accepted mocks, and 3-fold out-of-fold predictions.

- Positive label: `f_Z >= 0.5` (PISN contributes at least half the metals).
- Model: `HistGradientBoostingClassifier`, 150 iterations, depth 3, learning
  rate 0.05.
- Weights: Li+ `[Fe/H]` importance weights; effective sample size 79,290.
- AUC: 0.9736.
- Weighted Brier score: 0.0510.
- At TPR = 0.9: FPR = 7.08%, precision = 3.09% at prevalence 2.5e-3.
- At FPR = 0.1%: TPR = 36.1%, precision = 47.5% at prevalence 2.5e-3.

| File | Content |
| --- | --- |
| `01_score_histogram.png` | Score distribution by true class. |
| `02_roc.png` | Weighted ROC, full and low-FPR zoom. |
| `03_precision_recall.png` | Precision vs recall at realistic prevalence. |
| `04_calibration.png` | Reliability curve and weighted Brier score. |
| `05_score_vs_fz.png` | Score vs true PISN metal fraction. |
| `06_strata.png` | Score vs `n_III`, observed `[Fe/H]`, giant/dwarf. |
| `07_confusion.png` | Weighted confusion matrices at TPR 0.9 and FPR 0.1%. |
| `metrics.json` | Machine-readable summary. |

Regenerate with:

```bash
python -m src.potluck.diagnostics.classifier_plots \
  data/potluck/analysis/v8_threshold_100k.npz src/potluck/tests/v1
```
