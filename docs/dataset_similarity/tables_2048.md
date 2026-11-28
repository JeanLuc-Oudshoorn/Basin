# Tables for windows of 2048 points

## Series per group

| Group | Collected | With 2048 points | Without a shape | Analysed |
|---|---|---|---|---|
| Synthetic | 25,000 | 12,722 | 3 | 12,719 |
| FakeBench-TS | 5,000 | 2,500 | 0 | 2,500 |
| Operational benchmarks | 5,112 | 2,536 | 37 | 2,499 |
| IND-Eval | 19,022 | 15,417 | 26 | 15,391 |

## Length of the collected series (points)

| Group | 25% | Median | 75% | Share with 512+ | Share with 2,048+ |
|---|---|---|---|---|---|
| Synthetic | 1,139 | 2,056 | 2,056 | 92% | 51% |
| FakeBench-TS | 512 | 1,280 | 2,048 | 100% | 50% |
| Operational benchmarks | 512 | 512 | 2,048 | 100% | 50% |
| IND-Eval | 2,048 | 2,048 | 2,048 | 99% | 81% |

## Coverage

| Group | Dataset | Series | Close | Covered | Median distance ratio |
|---|---|---|---|---|---|
| FakeBench-TS | normal | 100 | 94% | 100% | 0.53 |
| FakeBench-TS | logistic | 75 | 91% | 100% | 0.54 |
| FakeBench-TS | uniform | 75 | 93% | 100% | 0.58 |
| FakeBench-TS | arcsine | 75 | 85% | 100% | 0.59 |
| FakeBench-TS | rademacher | 75 | 17% | 100% | 1.10 |
| FakeBench-TS | lazy_grid | 75 | 52% | 100% | 0.99 |
| FakeBench-TS | skellam | 75 | 71% | 100% | 0.88 |
| FakeBench-TS | bimodal | 75 | 75% | 100% | 0.60 |
| FakeBench-TS | laplace | 100 | 93% | 100% | 0.59 |
| FakeBench-TS | jump_mixture | 125 | 74% | 100% | 0.93 |
| FakeBench-TS | stretched_exponential | 125 | 82% | 100% | 0.87 |
| FakeBench-TS | sign_lognormal | 125 | 84% | 100% | 0.80 |
| FakeBench-TS | student_t | 150 | 87% | 100% | 0.64 |
| FakeBench-TS | skew_gamma | 250 | 98% | 100% | 0.63 |
| FakeBench-TS | skew_two_point | 200 | 16% | 100% | 1.15 |
| FakeBench-TS | skew_poisson | 200 | 86% | 100% | 0.61 |
| FakeBench-TS | skew_negative_binomial | 200 | 96% | 100% | 0.69 |
| FakeBench-TS | skew_inverse_gaussian | 200 | 96% | 100% | 0.75 |
| FakeBench-TS | skew_jump | 200 | 91% | 100% | 0.76 |
| Operational benchmarks | caster | 1,056 | 33% | 98% | 1.15 |
| Operational benchmarks | cloud | 203 | 45% | 97% | 1.03 |
| Operational benchmarks | ucr | 840 | 0% | 94% | 1.50 |
| Operational benchmarks | cats | 400 | 51% | 100% | 0.99 |
| IND-Eval | 3w | 3,000 | 11% | 96% | 1.34 |
| IND-Eval | electricity | 2,730 | 1% | 95% | 1.70 |
| IND-Eval | mds | 3,000 | 12% | 85% | 1.30 |
| IND-Eval | metropt3 | 800 | 0% | 65% | 1.80 |
| IND-Eval | neats | 2,974 | 12% | 68% | 1.66 |
| IND-Eval | pronto | 48 | 4% | 65% | 1.61 |
| IND-Eval | solar | 2,839 | 0% | 1% | 2.32 |
| Synthetic | synthetic | 10,081 | 50% | 95% | 1.00 |
| Synthetic, set aside | synthetic | 2,638 | 49% | 95% | 1.02 |
| FakeBench-TS | all walks | 2,500 | 80% | 100% | 0.76 |
| Operational benchmarks | all | 2,499 | 26% | 97% | 1.29 |
| IND-Eval | all | 15,391 | 7% | 69% | 1.67 |
| Synthetic | all | 10,081 | 50% | 95% | 1.00 |
| Synthetic, set aside | all | 2,638 | 49% | 95% | 1.02 |

## Coverage under other settings (neighbours k, quantile q)

| Setting | FakeBench-TS | Operational benchmarks | IND-Eval | Synthetic | Synthetic, set aside |
|---|---|---|---|---|---|
| k1_q50 | 70% | 21% | 7% | 50% | 50% |
| k1_q90 | 99% | 78% | 49% | 90% | 89% |
| k1_q95 | 100% | 95% | 64% | 95% | 95% |
| k1_q99 | 100% | 100% | 93% | 99% | 99% |
| k5_q50 | 80% | 26% | 7% | 50% | 49% |
| k5_q90 | 100% | 85% | 54% | 90% | 89% |
| k5_q95 | 100% | 97% | 69% | 95% | 95% |
| k5_q99 | 100% | 100% | 95% | 99% | 99% |
| k20_q50 | 88% | 33% | 7% | 50% | 50% |
| k20_q90 | 100% | 91% | 61% | 90% | 90% |
| k20_q95 | 100% | 99% | 71% | 95% | 95% |
| k20_q99 | 100% | 100% | 95% | 99% | 99% |

## Share of synthetic series with benchmark series nearby

| Benchmark | Share of synthetic series |
|---|---|
| fakebench | 49% |
| operational | 65% |
| ind_eval | 86% |
| operational or ind_eval | 87% |

## Classifier: telling a group from the synthetic series (AUC, 0.5 = cannot)

| Group | Series | All descriptors | Without sensor behaviour | Synthetic series that resemble the group | Group series that resemble synthetic | Synthetic series that count after reweighting |
|---|---|---|---|---|---|---|
| FakeBench-TS | 2,500 | 0.999 | 0.999 | 0.6% | 1.4% | 0.2% |
| Operational benchmarks | 2,499 | 0.969 | 0.937 | 0.8% | 21.4% | 0.1% |
| IND-Eval | 15,391 | 0.992 | 0.983 | 5.5% | 3.1% | 1.0% |
| Synthetic, set aside | 2,638 | 0.491 | 0.497 | 46.8% | 55.2% | 97.2% |
| Operational and IND-Eval together | - | 0.986 | - | 5.3% | 5.4% | 1.2% |

## Descriptors: FakeBench-TS against synthetic (10% / median / 90%)

| Descriptor | Synthetic | FakeBench-TS | Chance higher | Above synthetic 99% | Below synthetic 1% |
|---|---|---|---|---|---|
| power_slow | 0.06 / 0.82 / 0.96 | 0.91 / 0.96 / 0.98 | 0.83 | 1% | 0% |
| volatility_clustering | 0.00 / 0.18 / 0.76 | -0.03 / 0.00 / 0.25 | 0.18 | 0% | 0% |
| acf_4 | 0.41 / 0.94 / 0.99 | 0.97 / 0.99 / 0.99 | 0.80 | 0% | 0% |
| mode_share | 0.00 / 0.00 / 0.24 | 0.00 / 0.00 / 0.00 | 0.20 | 0% | 0% |
| distinct_share | 0.14 / 0.99 / 1.00 | 0.99 / 1.00 / 1.00 | 0.79 | 0% | 0% |
| acf_16 | 0.03 / 0.81 / 0.97 | 0.90 / 0.95 / 0.98 | 0.79 | 2% | 0% |
| power_mid | 0.03 / 0.12 / 0.65 | 0.02 / 0.04 / 0.08 | 0.22 | 0% | 0% |
| spectral_entropy | 0.21 / 0.36 / 0.65 | 0.14 / 0.24 / 0.34 | 0.22 | 0% | 4% |
| spectral_slope | 0.61 / 1.62 / 2.02 | 1.81 / 1.94 / 2.04 | 0.78 | 0% | 0% |
| power_fast | 0.00 / 0.02 / 0.24 | 0.00 / 0.00 / 0.01 | 0.22 | 0% | 0% |
| acf_1 | 0.84 / 0.99 / 1.00 | 0.99 / 1.00 / 1.00 | 0.76 | 0% | 0% |
| long_memory | 0.71 / 1.26 / 1.68 | 1.42 / 1.49 / 1.56 | 0.76 | 0% | 0% |
| acf_64 | -0.13 / 0.58 / 0.89 | 0.64 / 0.82 / 0.90 | 0.73 | 3% | 0% |
| stability | 0.03 / 0.70 / 0.98 | 0.79 / 0.92 / 0.97 | 0.73 | 0% | 0% |
| trend_strength | 0.01 / 0.68 / 0.98 | 0.77 / 0.91 / 0.97 | 0.71 | 0% | 0% |
| diff_acf_1 | -0.36 / 0.17 / 0.83 | -0.03 / -0.00 / 0.03 | 0.29 | 0% | 0% |
| power_jitter | 0.00 / 0.00 / 0.05 | 0.00 / 0.00 / 0.00 | 0.30 | 0% | 0% |
| lumpiness | 0.00 / 0.03 / 0.23 | 0.00 / 0.00 / 0.04 | 0.32 | 0% | 0% |
| flat_share | 0.00 / 0.00 / 0.45 | 0.00 / 0.00 / 0.00 | 0.33 | 0% | 0% |
| variance_shift | 0.31 / 1.04 / 3.85 | 1.09 / 1.40 / 2.01 | 0.66 | 0% | 0% |
| season_acf | 0.00 / 0.00 / 0.78 | 0.00 / 0.00 / 0.00 | 0.34 | 0% | 0% |
| seasonal_strength | 0.00 / 0.00 / 0.50 | 0.00 / 0.00 / 0.00 | 0.34 | 0% | 0% |
| season_amplitude_change | 0.00 / 0.00 / 0.38 | 0.00 / 0.00 / 0.00 | 0.34 | 0% | 0% |
| season_harmonics | 0.00 / 0.00 / 0.27 | 0.00 / 0.00 / 0.00 | 0.35 | 0% | 0% |
| permutation_entropy | 0.41 / 0.85 / 0.99 | 0.66 / 0.95 / 0.97 | 0.64 | 0% | 0% |
| noise_share | 0.00 / 0.05 / 0.32 | 0.02 / 0.03 / 0.06 | 0.39 | 0% | 0% |
| level_shift | 0.38 / 1.09 / 2.24 | 0.83 / 1.35 / 2.07 | 0.61 | 0% | 0% |
| linear_trend | 0.00 / 0.17 / 0.93 | 0.02 / 0.44 / 0.84 | 0.59 | 0% | 0% |
| at_extreme_share | 0.00 / 0.00 / 0.13 | 0.00 / 0.00 / 0.00 | 0.41 | 0% | 0% |
| kurtosis | -0.80 / -0.41 / 0.89 | -0.83 / -0.54 / 0.22 | 0.41 | 0% | 0% |
| missing_share | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.45 | 0% | 0% |
| skewness | -0.56 / 0.04 / 0.61 | -0.49 / 0.00 / 0.50 | 0.47 | 0% | 0% |
| bimodality | 0.34 / 0.53 / 0.80 | 0.40 / 0.52 / 0.68 | 0.52 | 0% | 0% |
| diff_skewness | -1.59 / 0.00 / 1.37 | -1.22 / 0.00 / 1.20 | 0.51 | 0% | 0% |
| jump_share | 0.07 / 0.13 / 0.79 | 0.03 / 0.16 / 0.46 | 0.50 | 0% | 7% |
| diff_kurtosis | -0.12 / 1.17 / 4.98 | -0.77 / 1.48 / 3.56 | 0.50 | 0% | 5% |

## Descriptors: Operational benchmarks against synthetic (10% / median / 90%)

| Descriptor | Synthetic | Operational benchmarks | Chance higher | Above synthetic 99% | Below synthetic 1% |
|---|---|---|---|---|---|
| diff_skewness | -1.59 / 0.00 / 1.37 | -0.21 / 0.46 / 1.16 | 0.66 | 0% | 0% |
| diff_acf_1 | -0.36 / 0.17 / 0.83 | -0.15 / 0.53 / 0.84 | 0.63 | 0% | 0% |
| volatility_clustering | 0.00 / 0.18 / 0.76 | 0.00 / 0.41 / 0.80 | 0.63 | 0% | 0% |
| acf_64 | -0.13 / 0.58 / 0.89 | -0.32 / 0.19 / 0.89 | 0.40 | 2% | 0% |
| distinct_share | 0.14 / 0.99 / 1.00 | 0.06 / 0.51 / 1.00 | 0.40 | 0% | 2% |
| bimodality | 0.34 / 0.53 / 0.80 | 0.37 / 0.56 / 0.86 | 0.59 | 0% | 0% |
| flat_share | 0.00 / 0.00 / 0.45 | 0.00 / 0.03 / 0.26 | 0.59 | 2% | 0% |
| linear_trend | 0.00 / 0.17 / 0.93 | 0.00 / 0.04 / 0.77 | 0.41 | 0% | 0% |
| power_mid | 0.03 / 0.12 / 0.65 | 0.03 / 0.23 / 0.96 | 0.59 | 7% | 1% |
| acf_16 | 0.03 / 0.81 / 0.97 | -0.05 / 0.58 / 0.97 | 0.42 | 2% | 0% |
| power_jitter | 0.00 / 0.00 / 0.05 | 0.00 / 0.00 / 0.02 | 0.42 | 0% | 1% |
| kurtosis | -0.80 / -0.41 / 0.89 | -0.82 / -0.14 / 0.99 | 0.57 | 1% | 0% |
| stability | 0.03 / 0.70 / 0.98 | 0.03 / 0.40 / 0.98 | 0.44 | 0% | 0% |
| level_shift | 0.38 / 1.09 / 2.24 | 0.57 / 1.28 / 2.22 | 0.56 | 2% | 0% |
| power_slow | 0.06 / 0.82 / 0.96 | 0.02 / 0.60 / 0.97 | 0.44 | 2% | 0% |
| missing_share | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.45 | 0% | 0% |
| seasonal_strength | 0.00 / 0.00 / 0.50 | 0.00 / 0.00 / 0.89 | 0.55 | 3% | 0% |
| mode_share | 0.00 / 0.00 / 0.24 | 0.00 / 0.01 / 0.03 | 0.54 | 0% | 0% |
| season_acf | 0.00 / 0.00 / 0.78 | 0.00 / 0.00 / 0.92 | 0.54 | 0% | 0% |
| trend_strength | 0.01 / 0.68 / 0.98 | 0.01 / 0.39 / 0.98 | 0.46 | 0% | 0% |
| skewness | -0.56 / 0.04 / 0.61 | -0.90 / 0.13 / 0.69 | 0.53 | 0% | 1% |
| noise_share | 0.00 / 0.05 / 0.32 | 0.00 / 0.05 / 0.28 | 0.47 | 0% | 0% |
| lumpiness | 0.00 / 0.03 / 0.23 | 0.00 / 0.04 / 0.16 | 0.53 | 1% | 0% |
| acf_4 | 0.41 / 0.94 / 0.99 | 0.42 / 0.89 / 0.99 | 0.47 | 1% | 2% |
| permutation_entropy | 0.41 / 0.85 / 0.99 | 0.57 / 0.88 / 0.99 | 0.53 | 0% | 1% |
| season_harmonics | 0.00 / 0.00 / 0.27 | 0.00 / 0.00 / 0.38 | 0.52 | 0% | 0% |
| long_memory | 0.71 / 1.26 / 1.68 | 0.86 / 1.34 / 1.67 | 0.52 | 0% | 0% |
| acf_1 | 0.84 / 0.99 / 1.00 | 0.90 / 0.99 / 1.00 | 0.52 | 1% | 0% |
| at_extreme_share | 0.00 / 0.00 / 0.13 | 0.00 / 0.00 / 0.00 | 0.48 | 0% | 0% |
| jump_share | 0.07 / 0.13 / 0.79 | 0.08 / 0.14 / 0.48 | 0.51 | 2% | 0% |
| spectral_slope | 0.61 / 1.62 / 2.02 | 0.74 / 1.47 / 2.07 | 0.49 | 0% | 0% |
| power_fast | 0.00 / 0.02 / 0.24 | 0.00 / 0.01 / 0.22 | 0.49 | 1% | 0% |
| variance_shift | 0.31 / 1.04 / 3.85 | 0.31 / 1.01 / 2.90 | 0.50 | 0% | 0% |
| spectral_entropy | 0.21 / 0.36 / 0.65 | 0.21 / 0.38 / 0.67 | 0.50 | 0% | 1% |
| diff_kurtosis | -0.12 / 1.17 / 4.98 | 0.08 / 1.15 / 3.66 | 0.50 | 0% | 0% |
| season_amplitude_change | 0.00 / 0.00 / 0.38 | 0.00 / 0.00 / 0.35 | 0.50 | 0% | 0% |

## Descriptors: IND-Eval against synthetic (10% / median / 90%)

| Descriptor | Synthetic | IND-Eval | Chance higher | Above synthetic 99% | Below synthetic 1% |
|---|---|---|---|---|---|
| distinct_share | 0.14 / 0.99 / 1.00 | 0.00 / 0.17 / 0.90 | 0.15 | 0% | 14% |
| mode_share | 0.00 / 0.00 / 0.24 | 0.00 / 0.08 / 0.63 | 0.82 | 2% | 0% |
| flat_share | 0.00 / 0.00 / 0.45 | 0.00 / 0.32 / 0.84 | 0.81 | 3% | 0% |
| at_extreme_share | 0.00 / 0.00 / 0.13 | 0.00 / 0.00 / 0.63 | 0.71 | 2% | 0% |
| acf_64 | -0.13 / 0.58 / 0.89 | -0.38 / 0.05 / 0.88 | 0.33 | 3% | 0% |
| bimodality | 0.34 / 0.53 / 0.80 | 0.38 / 0.66 / 0.90 | 0.67 | 2% | 1% |
| volatility_clustering | 0.00 / 0.18 / 0.76 | 0.05 / 0.37 / 0.87 | 0.65 | 4% | 1% |
| permutation_entropy | 0.41 / 0.85 / 0.99 | 0.30 / 0.59 / 0.98 | 0.36 | 0% | 1% |
| stability | 0.03 / 0.70 / 0.98 | 0.02 / 0.22 / 0.97 | 0.37 | 1% | 0% |
| linear_trend | 0.00 / 0.17 / 0.93 | 0.00 / 0.02 / 0.61 | 0.37 | 0% | 1% |
| seasonal_strength | 0.00 / 0.00 / 0.50 | 0.00 / 0.00 / 0.89 | 0.62 | 6% | 0% |
| lumpiness | 0.00 / 0.03 / 0.23 | 0.00 / 0.06 / 0.39 | 0.62 | 3% | 0% |
| trend_strength | 0.01 / 0.68 / 0.98 | 0.02 / 0.24 / 0.97 | 0.40 | 0% | 0% |
| acf_16 | 0.03 / 0.81 / 0.97 | -0.32 / 0.55 / 0.97 | 0.40 | 4% | 0% |
| season_harmonics | 0.00 / 0.00 / 0.27 | 0.00 / 0.00 / 0.32 | 0.59 | 0% | 0% |
| season_acf | 0.00 / 0.00 / 0.78 | 0.00 / 0.00 / 0.86 | 0.59 | 1% | 0% |
| spectral_slope | 0.61 / 1.62 / 2.02 | 0.50 / 1.31 / 2.19 | 0.41 | 2% | 0% |
| power_mid | 0.03 / 0.12 / 0.65 | 0.02 / 0.25 / 0.82 | 0.59 | 0% | 4% |
| power_slow | 0.06 / 0.82 / 0.96 | 0.03 / 0.62 / 0.97 | 0.41 | 5% | 0% |
| diff_kurtosis | -0.12 / 1.17 / 4.98 | 0.18 / 1.97 / 4.19 | 0.58 | 0% | 0% |
| skewness | -0.56 / 0.04 / 0.61 | -0.60 / 0.16 / 1.00 | 0.58 | 5% | 2% |
| variance_shift | 0.31 / 1.04 / 3.85 | 0.33 / 1.52 / 4.31 | 0.57 | 3% | 0% |
| noise_share | 0.00 / 0.05 / 0.32 | 0.00 / 0.09 / 0.44 | 0.57 | 4% | 0% |
| season_amplitude_change | 0.00 / 0.00 / 0.38 | 0.00 / 0.00 / 0.40 | 0.56 | 1% | 0% |
| acf_1 | 0.84 / 0.99 / 1.00 | 0.77 / 0.97 / 1.00 | 0.44 | 5% | 4% |
| kurtosis | -0.80 / -0.41 / 0.89 | -0.86 / -0.19 / 1.68 | 0.56 | 4% | 1% |
| acf_4 | 0.41 / 0.94 / 0.99 | 0.26 / 0.90 / 0.99 | 0.44 | 5% | 0% |
| power_jitter | 0.00 / 0.00 / 0.05 | 0.00 / 0.01 / 0.08 | 0.55 | 4% | 2% |
| long_memory | 0.71 / 1.26 / 1.68 | 0.71 / 1.34 / 1.87 | 0.55 | 1% | 0% |
| jump_share | 0.07 / 0.13 / 0.79 | 0.07 / 0.18 / 0.64 | 0.55 | 1% | 0% |
| diff_skewness | -1.59 / 0.00 / 1.37 | -0.84 / 0.06 / 0.93 | 0.55 | 0% | 0% |
| power_fast | 0.00 / 0.02 / 0.24 | 0.00 / 0.03 / 0.29 | 0.54 | 0% | 3% |
| missing_share | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.46 | 0% | 0% |
| spectral_entropy | 0.21 / 0.36 / 0.65 | 0.20 / 0.40 / 0.72 | 0.54 | 3% | 3% |
| level_shift | 0.38 / 1.09 / 2.24 | 0.46 / 0.99 / 2.40 | 0.52 | 2% | 0% |
| diff_acf_1 | -0.36 / 0.17 / 0.83 | -0.40 / 0.16 / 0.88 | 0.50 | 3% | 4% |

## Cycles

| Group | Has a cycle | Strong cycle (strength above 0.6) | Cycle length 25% | Median | 75% |
|---|---|---|---|---|---|
| Synthetic | 33% | 8% | 19 | 49 | 143 |
| FakeBench-TS | 3% | 0% | 215 | 240 | 413 |
| Operational benchmarks | 35% | 27% | 24 | 24 | 98 |
| IND-Eval | 47% | 38% | 24 | 96 | 144 |

### Cycles per dataset

| Group | Dataset | Series | Has a cycle | Strong cycle | Median cycle length |
|---|---|---|---|---|---|
| FakeBench-TS | normal | 100 | 2% | 1% | 422 |
| FakeBench-TS | logistic | 75 | 0% | 0% | - |
| FakeBench-TS | uniform | 75 | 3% | 1% | 438 |
| FakeBench-TS | arcsine | 75 | 4% | 0% | 240 |
| FakeBench-TS | rademacher | 75 | 1% | 1% | 632 |
| FakeBench-TS | lazy_grid | 75 | 1% | 0% | 240 |
| FakeBench-TS | skellam | 75 | 1% | 0% | 240 |
| FakeBench-TS | bimodal | 75 | 4% | 0% | 682 |
| FakeBench-TS | laplace | 100 | 1% | 0% | 614 |
| FakeBench-TS | jump_mixture | 125 | 2% | 1% | 273 |
| FakeBench-TS | stretched_exponential | 125 | 2% | 1% | 413 |
| FakeBench-TS | sign_lognormal | 125 | 6% | 1% | 215 |
| FakeBench-TS | student_t | 150 | 4% | 0% | 244 |
| FakeBench-TS | skew_gamma | 250 | 2% | 0% | 218 |
| FakeBench-TS | skew_two_point | 200 | 2% | 0% | 273 |
| FakeBench-TS | skew_poisson | 200 | 4% | 1% | 278 |
| FakeBench-TS | skew_negative_binomial | 200 | 2% | 0% | 240 |
| FakeBench-TS | skew_inverse_gaussian | 200 | 3% | 0% | 315 |
| FakeBench-TS | skew_jump | 200 | 2% | 0% | 258 |
| Operational benchmarks | caster | 1,056 | 3% | 0% | 16 |
| Operational benchmarks | cloud | 203 | 0% | 0% | 195 |
| Operational benchmarks | ucr | 840 | 100% | 79% | 24 |
| Operational benchmarks | cats | 400 | 3% | 0% | 328 |
| IND-Eval | 3w | 3,000 | 19% | 14% | 412 |
| IND-Eval | electricity | 2,730 | 95% | 84% | 96 |
| IND-Eval | mds | 3,000 | 9% | 4% | 288 |
| IND-Eval | metropt3 | 800 | 67% | 21% | 131 |
| IND-Eval | neats | 2,974 | 16% | 2% | 131 |
| IND-Eval | pronto | 48 | 15% | 0% | 64 |
| IND-Eval | solar | 2,839 | 100% | 99% | 143 |

## Basin's error against the other models (stage 11 forecast cases)

2,498 cases; Basin's error over the mean error of the three other models, median 1.03. Rank correlation with the distance to the synthetic series: 0.18 overall, 0.11 within datasets.

| Distance quarter | Distance ratio | Cases | Basin over others | Basin worse |
|---|---|---|---|---|
| 1 | 0.49 to 0.99 | 625 | 1.01 | 55% |
| 2 | 0.99 to 1.29 | 624 | 1.01 | 55% |
| 3 | 1.29 to 1.51 | 624 | 1.04 | 62% |
| 4 | 1.51 to 3.78 | 625 | 1.10 | 69% |

| Dataset | Cases | Basin over others | Median distance ratio |
|---|---|---|---|
| caster | 1,056 | 1.00 | 1.15 |
| cats | 400 | 1.01 | 0.99 |
| cloud | 202 | 1.24 | 1.03 |
| ucr | 840 | 1.14 | 1.50 |

### Descriptors that go with Basin losing (rank correlation)

| Descriptor | Overall | Within datasets |
|---|---|---|
| season_amplitude_change | +0.15 | -0.21 |
| season_harmonics | +0.25 | +0.21 |
| bimodality | +0.16 | +0.16 |
| lumpiness | -0.06 | -0.14 |
| flat_share | +0.03 | +0.13 |
| level_shift | -0.17 | -0.13 |
| noise_share | -0.03 | -0.13 |
| long_memory | +0.00 | +0.12 |
| spectral_entropy | -0.11 | -0.11 |
| power_jitter | -0.05 | -0.11 |
| acf_1 | -0.01 | +0.11 |
| distinct_share | +0.05 | -0.11 |
| acf_16 | -0.07 | +0.11 |
| skewness | -0.09 | -0.11 |
| seasonal_strength | +0.22 | +0.10 |
| acf_4 | -0.03 | +0.10 |
| mode_share | -0.01 | +0.10 |
| diff_kurtosis | +0.12 | +0.10 |
| power_fast | +0.02 | -0.09 |
| trend_strength | -0.11 | -0.08 |
| spectral_slope | -0.13 | -0.07 |
| jump_share | +0.09 | +0.07 |
| power_slow | -0.12 | -0.06 |
| permutation_entropy | +0.01 | +0.06 |
| variance_shift | -0.15 | -0.05 |
| kurtosis | -0.07 | -0.05 |
| at_extreme_share | +0.01 | +0.04 |
| stability | -0.12 | -0.04 |
| diff_acf_1 | +0.07 | +0.04 |
| linear_trend | -0.15 | -0.03 |
| diff_skewness | +0.09 | +0.03 |
| volatility_clustering | +0.09 | +0.02 |
| acf_64 | -0.14 | -0.01 |
| power_mid | +0.11 | -0.00 |
| season_acf | +0.20 | -0.00 |
| missing_share | +0.00 | +0.00 |

## Benchmark series without synthetic neighbours, by kind

| Kind | Series | Mostly | Differs from the typical synthetic series |
|---|---|---|---|
| A | 1,740 | solar 1,685, mds 46, neats 7, electricity 2 | at_extreme_share higher (+3.6); mode_share higher (+3.5); season_amplitude_change higher (+2.7); flat_share higher (+2.5); seasonal_strength higher (+2.3); season_acf higher (+2.1) |
| B | 1,168 | solar 1,139, mds 13, electricity 9, neats 7 | at_extreme_share higher (+3.2); mode_share higher (+3.1); acf_16 lower (-2.9); seasonal_strength higher (+2.5); acf_4 lower (-2.4); season_acf higher (+2.4) |
| C | 477 | metropt3 269, neats 85, electricity 67, ucr 49 | season_harmonics higher (+2.5); jump_share higher (+2.5); bimodality higher (+2.4); power_mid higher (+2.1); kurtosis higher (+2.1); distinct_share lower (-2.0) |
| D | 423 | mds 189, neats 114, 3w 101, electricity 8 | flat_share higher (+2.9); variance_shift higher (+2.7); distinct_share lower (-2.3); permutation_entropy lower (-1.8); level_shift higher (+1.8); mode_share higher (+1.7) |
| E | 353 | neats 303, electricity 33, mds 14, caster 2 | kurtosis higher (+4.0); power_fast higher (+3.2); noise_share higher (+3.0); acf_4 lower (-2.8); mode_share higher (+2.7); skewness higher (+2.7) |
| F | 318 | neats 212, mds 64, pronto 13, caster 13 | lumpiness higher (+7.6); kurtosis higher (+4.7); variance_shift higher (+3.5); jump_share higher (+3.4); skewness higher (+3.0); diff_kurtosis higher (+2.4) |
| G | 226 | neats 158, mds 46, electricity 20, caster 2 | power_jitter higher (+6.7); acf_1 lower (-6.2); noise_share higher (+5.6); mode_share higher (+4.0); spectral_entropy higher (+3.1); acf_4 lower (-3.0) |
| H | 169 | mds 86, neats 76, cloud 5, electricity 2 | lumpiness higher (+7.0); power_jitter higher (+6.9); acf_1 lower (-6.5); at_extreme_share higher (+5.7); kurtosis higher (+5.7); noise_share higher (+5.5) |

## Synthetic series without benchmark neighbours, by kind

| Kind | Series | Mostly | Differs from the typical synthetic series |
|---|---|---|---|
| A | 371 | synthetic 371 | season_harmonics higher (+2.6); season_acf higher (+2.1); season_amplitude_change higher (+2.0); linear_trend higher (+2.0); long_memory lower (-1.0); level_shift lower (-0.9) |
| B | 294 | synthetic 294 | jump_share higher (+2.0); trend_strength lower (-1.7); diff_kurtosis higher (+1.6); power_slow lower (-1.6); stability lower (-1.6); power_mid higher (+1.6) |
| C | 190 | synthetic 190 | missing_share higher (+7.0); linear_trend higher (+1.5); diff_kurtosis higher (+0.8); variance_shift higher (+0.7); permutation_entropy lower (-0.7); trend_strength higher (+0.6) |
| D | 169 | synthetic 169 | missing_share higher (+8.8); trend_strength lower (-1.6); stability lower (-1.4); power_slow lower (-1.3); acf_64 lower (-1.2); acf_16 lower (-1.2) |
| E | 155 | synthetic 155 | at_extreme_share higher (+4.2); mode_share higher (+3.5); variance_shift higher (+3.3); jump_share higher (+3.2); flat_share higher (+3.2); kurtosis higher (+2.9) |
| F | 139 | synthetic 139 | power_fast higher (+5.2); acf_4 lower (-4.3); spectral_slope lower (-2.5); acf_1 lower (-2.4); long_memory lower (-2.4); power_slow lower (-2.3) |

## Principal components

| Component | Share of variation | Moves most with |
|---|---|---|
| 1 | 29% | +trend_strength, +linear_trend, −noise_share, +stability, +spectral_slope |
| 2 | 15% | +permutation_entropy, +noise_share, −at_extreme_share, −mode_share, −flat_share |
| 3 | 12% | +variance_shift, −season_acf, +noise_share, +flat_share, +mode_share |
