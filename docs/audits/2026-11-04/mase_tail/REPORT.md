# Why mean holdout MASE is much larger than its median

The saved run contains 3,171 tables. Recomputing weighted MASE from all 826,948
finite, valid-scale scored points reproduces every reported model mean to
floating-point precision. The aggregation is not accidentally including null
future actuals or unweighted later-token errors.

Three tables (0.095% of the sample) dominate the mean. They contribute **95.19%
of Basin's scaled-error numerator** and **94.62% of Timestral's**. These are the
same three largest contributors for all four models. Their removal below is
only a sensitivity calculation, not a revised benchmark or admission rule.

| Model | Published weighted MASE | MASE excluding the same three tables |
| --- | ---: | ---: |
| Timestral | 51.2993 | 2.7641 |
| Chronos | 51.4389 | 2.9029 |
| Naive | 52.4557 | 3.9189 |
| Basin | 62.2763 | 2.9965 |

## Sources and contributions

All three sources are under `data/01_generated/2026-10-31/`; each directory
contains `data.parquet`.

| Table directory | Basin per-table MASE | Contribution to Basin pooled MASE | Contribution to Timestral pooled MASE |
| --- | ---: | ---: | ---: |
| `table_285516885900_019943` | 149,536.68 | 47.44485 | 36.69995 |
| `table_285516885900_149295` | 22,692.34 | 7.19980 | 7.19976 |
| `table_285516885900_072778` | 14,618.12 | 4.63802 | 4.63806 |

The single worst table contributes 76.18% of Basin's numerator and 71.54% of
Timestral's. Small differences between per-table and pooled weighting do not
explain this issue: both versions exhibit the same tail.

## What happens at the individual forecast origins

MASE divides absolute forecast error by mean absolute historical one-step
change. A tiny positive denominator remains a legitimate positive scale under
the current policy; only exactly zero/invalid scales are excluded.

For table `019943`, the context at origin 144 is nearly flat: its target ranges
from approximately -0.000029 to -0.000003. The historical MASE denominator is
**0.000000582284**. Actual values within the next 64 points rise to **2.5074**.
Every model misses this transition enough to create huge scaled errors:

| Model | Weighted MAE at origin 144 | Contribution to overall pooled MASE |
| --- | ---: | ---: |
| Naive | 0.3367567 | approximately 36.699 |
| Chronos | 0.3367583 | 36.69905 |
| Timestral | 0.3367537 | 36.69854 |
| Basin | 0.3556581 | 38.75870 |

Thus fairly similar raw errors add approximately the same 36.7 to the first
three headline means. This common additive penalty makes naive appear close
to the foundation models even though its ordinary-table median is much worse.

The other two large contributors have the same structural problem:

- `149295`, origin 96: historical scale 0.0000112338; a near-constant level
  around 349.1888 precedes future values reaching 373.9482. All models have
  weighted MAE approximately 1.2745 and origin MASE approximately 113,454.
- `072778`, origin 32: scale 0.0000474881; historical values near -0.0004
  precede future values reaching 8.8945. All models have weighted MAE
  approximately 3.4701 and origin MASE approximately 73,073.

These observations demonstrate the transitions and metric amplification; they
do not establish that the transitions are impossible to anticipate using all
historical covariates, nor identify their generator mechanism without further
analysis of generation records.

## Basin also has a real additional error

At origin 32 of `019943`, the actual future stays tiny (approximately -0.000029
to -0.000008). The MASE denominator is 0.000000487501. Weighted MAE is:

- Basin: **0.0667274**; origin MASE **136,876.24**.
- Chronos: **0.0000104223**.
- Timestral: **0.0000083290**.
- Naive: **0.0000124997**.

That one origin alone adds **8.68559** to Basin's overall MASE, compared with
0.00136 for Chronos and 0.00108 for Timestral. Basin's worse mean is therefore
not solely a shared metric pathology. After removing the same three tables,
Basin beats naive but still trails Chronos and Timestral slightly.

A concrete preprocessing concern exists in `src/04_creating_embeddings/embed.py`:
`tokenize_next` replaces any standard deviation at or below `1e-5` with **1.0**.
The target context standard deviations at origins 32 and 144 of `019943` are
approximately 0.00000231 and 0.00000677, respectively, so these contexts enter
the unit-scale fallback. This discards their natural tiny amplitude as an
output scale. A normalized forecast deviation of a few hundredths then becomes
a raw deviation of a few hundredths, instead of a few millionths. This is a
plausible contributor to the observed flat-segment error. Establishing causality
and a safe replacement requires controlled normalization experiments and
consideration of training/inference consistency; no such policy was changed here.

## Interpretation and next steps

Keep the existing median/p10/p90 and KDE alongside the arithmetic mean: the
tail is real but unrepresentative of typical-table performance. Preserve the
unmodified benchmark rather than selectively dropping these tables or silently
adding an epsilon to MASE denominators. A changed denominator is a changed
metric and should be labeled and compared across every model consistently.

The next engineering investigation should target the absolute `1e-5` to
unit-scale normalization fallback and constant/near-constant histories. A
robust or scale-stratified supplementary score could improve interpretation,
but it must not hide genuine large raw forecast errors. Report shared-exclusion
sensitivity only as an audit, never as the headline selection criterion.

## Evidence and reproduction

- [Exact evidence and origin breakdown](evidence.json)
- [All per-table contributions](table_contributions.csv)
- [Interactive actuals and saved forecasts for the worst three tables](worst_tables.html)
- Script: `uv run python scripts/diagnose_holdout_mase_tail.py`

The script reads saved predictions and raw tables; it does not load model
weights, rerun inference, change screening, or overwrite production holdout
results. Plots retain the complete 64-point forecasts, including unobserved
future points. All analysis artifacts are saved under this audit directory.
