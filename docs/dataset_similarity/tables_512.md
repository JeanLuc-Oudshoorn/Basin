# Tables for windows of 512 points

## Series per group

| Group | Collected | With 512 points | Without a shape | Analysed |
|---|---|---|---|---|
| Synthetic | 25,000 | 22,895 | 396 | 22,499 |
| FakeBench-TS | 5,000 | 5,000 | 0 | 5,000 |
| Operational benchmarks | 5,112 | 2,576 | 40 | 2,536 |
| IND-Eval | 19,022 | 18,771 | 193 | 18,578 |

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
| FakeBench-TS | normal | 200 | 90% | 100% | 0.56 |
| FakeBench-TS | logistic | 150 | 90% | 100% | 0.59 |
| FakeBench-TS | uniform | 150 | 88% | 100% | 0.59 |
| FakeBench-TS | arcsine | 150 | 84% | 100% | 0.62 |
| FakeBench-TS | rademacher | 150 | 4% | 99% | 1.21 |
| FakeBench-TS | lazy_grid | 150 | 55% | 99% | 0.99 |
| FakeBench-TS | skellam | 150 | 71% | 99% | 0.89 |
| FakeBench-TS | bimodal | 150 | 75% | 100% | 0.65 |
| FakeBench-TS | laplace | 200 | 88% | 100% | 0.67 |
| FakeBench-TS | jump_mixture | 250 | 60% | 100% | 0.95 |
| FakeBench-TS | stretched_exponential | 250 | 68% | 100% | 0.91 |
| FakeBench-TS | sign_lognormal | 250 | 76% | 99% | 0.84 |
| FakeBench-TS | student_t | 300 | 87% | 100% | 0.63 |
| FakeBench-TS | skew_gamma | 500 | 95% | 100% | 0.67 |
| FakeBench-TS | skew_two_point | 400 | 8% | 100% | 1.20 |
| FakeBench-TS | skew_poisson | 400 | 84% | 100% | 0.66 |
| FakeBench-TS | skew_negative_binomial | 400 | 94% | 100% | 0.71 |
| FakeBench-TS | skew_inverse_gaussian | 400 | 92% | 100% | 0.76 |
| FakeBench-TS | skew_jump | 400 | 92% | 100% | 0.77 |
| Operational benchmarks | caster | 1,054 | 45% | 97% | 1.04 |
| Operational benchmarks | cloud | 242 | 40% | 96% | 1.09 |
| Operational benchmarks | ucr | 840 | 0% | 85% | 1.39 |
| Operational benchmarks | cats | 400 | 37% | 98% | 1.07 |
| IND-Eval | 3w | 2,996 | 36% | 96% | 1.12 |
| IND-Eval | electricity | 2,905 | 3% | 95% | 1.54 |
| IND-Eval | mds | 2,885 | 19% | 87% | 1.32 |
| IND-Eval | metropt3 | 950 | 1% | 56% | 1.92 |
| IND-Eval | neats | 2,933 | 19% | 76% | 1.49 |
| IND-Eval | pronto | 72 | 21% | 79% | 1.43 |
| IND-Eval | solar | 2,839 | 0% | 12% | 2.44 |
| IND-Eval | tablet_compression | 2,998 | 26% | 82% | 1.32 |
| Synthetic | synthetic | 17,870 | 50% | 95% | 1.00 |
| Synthetic, set aside | synthetic | 4,629 | 51% | 95% | 0.99 |
| FakeBench-TS | all walks | 5,000 | 76% | 100% | 0.77 |
| Operational benchmarks | all | 2,536 | 29% | 93% | 1.21 |
| IND-Eval | all | 18,578 | 17% | 74% | 1.51 |
| Synthetic | all | 17,870 | 50% | 95% | 1.00 |
| Synthetic, set aside | all | 4,629 | 51% | 95% | 0.99 |

## Coverage under other settings (neighbours k, quantile q)

| Setting | FakeBench-TS | Operational benchmarks | IND-Eval | Synthetic | Synthetic, set aside |
|---|---|---|---|---|---|
| k1_q50 | 68% | 23% | 18% | 50% | 50% |
| k1_q90 | 99% | 81% | 59% | 90% | 90% |
| k1_q95 | 100% | 87% | 72% | 95% | 95% |
| k1_q99 | 100% | 100% | 88% | 99% | 99% |
| k5_q50 | 76% | 29% | 17% | 50% | 51% |
| k5_q90 | 100% | 83% | 64% | 90% | 90% |
| k5_q95 | 100% | 93% | 74% | 95% | 95% |
| k5_q99 | 100% | 100% | 95% | 99% | 99% |
| k20_q50 | 83% | 33% | 16% | 50% | 50% |
| k20_q90 | 100% | 86% | 66% | 90% | 90% |
| k20_q95 | 100% | 98% | 75% | 95% | 95% |
| k20_q99 | 100% | 100% | 97% | 99% | 99% |

## Share of synthetic series with benchmark series nearby

| Benchmark | Share of synthetic series |
|---|---|
| fakebench | 50% |
| operational | 67% |
| ind_eval | 83% |
| operational or ind_eval | 86% |

## Classifier: telling a group from the synthetic series (AUC, 0.5 = cannot)

| Group | Series | All descriptors | Without sensor behaviour | Synthetic series that resemble the group | Group series that resemble synthetic | Synthetic series that count after reweighting |
|---|---|---|---|---|---|---|
| FakeBench-TS | 5,000 | 0.997 | 0.996 | 2.0% | 2.5% | 0.4% |
| Operational benchmarks | 2,536 | 0.955 | 0.914 | 2.0% | 23.2% | 0.6% |
| IND-Eval | 18,578 | 0.985 | 0.967 | 7.0% | 5.3% | 1.9% |
| Synthetic, set aside | 4,629 | 0.497 | 0.501 | 47.8% | 53.1% | 98.2% |
| Operational and IND-Eval together | - | 0.977 | - | 7.4% | 8.0% | 3.3% |

## Descriptors: FakeBench-TS against synthetic (10% / median / 90%)

| Descriptor | Synthetic | FakeBench-TS | Chance higher | Above synthetic 99% | Below synthetic 1% |
|---|---|---|---|---|---|
| power_slow | 0.03 / 0.52 / 0.84 | 0.59 / 0.81 / 0.91 | 0.79 | 2% | 0% |
| volatility_clustering | -0.01 / 0.16 / 0.75 | -0.05 / 0.01 / 0.21 | 0.22 | 0% | 0% |
| spectral_entropy | 0.26 / 0.41 / 0.69 | 0.18 / 0.30 / 0.42 | 0.24 | 0% | 3% |
| acf_4 | 0.20 / 0.86 / 0.97 | 0.90 / 0.95 / 0.98 | 0.76 | 0% | 0% |
| stability | 0.12 / 0.71 / 0.97 | 0.79 / 0.92 / 0.97 | 0.75 | 0% | 0% |
| acf_16 | -0.05 / 0.57 / 0.89 | 0.64 / 0.82 / 0.90 | 0.75 | 2% | 0% |
| trend_strength | 0.04 / 0.68 / 0.98 | 0.77 / 0.91 / 0.97 | 0.74 | 0% | 0% |
| power_fast | 0.01 / 0.04 / 0.36 | 0.01 / 0.02 / 0.04 | 0.27 | 0% | 0% |
| lumpiness | 0.00 / 0.04 / 0.31 | 0.00 / 0.01 / 0.04 | 0.27 | 0% | 0% |
| long_memory | 0.70 / 1.25 / 1.72 | 1.33 / 1.47 / 1.60 | 0.73 | 0% | 0% |
| spectral_slope | 0.63 / 1.65 / 2.07 | 1.68 / 1.89 / 2.03 | 0.70 | 0% | 0% |
| acf_1 | 0.72 / 0.97 / 0.99 | 0.97 / 0.99 / 0.99 | 0.70 | 0% | 0% |
| mode_share | 0.00 / 0.00 / 0.15 | 0.00 / 0.00 / 0.00 | 0.30 | 0% | 0% |
| power_mid | 0.11 / 0.28 / 0.76 | 0.08 / 0.17 / 0.37 | 0.31 | 0% | 0% |
| diff_acf_1 | -0.38 / 0.17 / 0.84 | -0.06 / -0.00 / 0.06 | 0.31 | 0% | 0% |
| distinct_share | 0.24 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 0.69 | 0% | 0% |
| acf_64 | -0.23 / 0.15 / 0.61 | 0.00 / 0.37 / 0.61 | 0.65 | 2% | 0% |
| season_acf | 0.00 / 0.00 / 0.69 | 0.00 / 0.00 / 0.00 | 0.36 | 0% | 0% |
| seasonal_strength | 0.00 / 0.00 / 0.67 | 0.00 / 0.00 / 0.00 | 0.36 | 0% | 0% |
| season_amplitude_change | 0.00 / 0.00 / 0.33 | 0.00 / 0.00 / 0.00 | 0.36 | 0% | 0% |
| season_harmonics | 0.00 / 0.00 / 0.24 | 0.00 / 0.00 / 0.00 | 0.36 | 0% | 0% |
| permutation_entropy | 0.40 / 0.85 / 0.99 | 0.66 / 0.95 / 0.97 | 0.63 | 0% | 0% |
| linear_trend | 0.00 / 0.14 / 0.91 | 0.03 / 0.45 / 0.84 | 0.63 | 0% | 0% |
| flat_share | 0.00 / 0.00 / 0.36 | 0.00 / 0.00 / 0.00 | 0.37 | 0% | 0% |
| power_jitter | 0.00 / 0.01 / 0.10 | 0.00 / 0.00 / 0.01 | 0.38 | 0% | 0% |
| kurtosis | -0.79 / -0.39 / 0.91 | -0.84 / -0.53 / 0.28 | 0.42 | 0% | 0% |
| at_extreme_share | 0.00 / 0.00 / 0.06 | 0.00 / 0.00 / 0.00 | 0.43 | 0% | 0% |
| bimodality | 0.35 / 0.50 / 0.77 | 0.39 / 0.53 / 0.69 | 0.56 | 0% | 0% |
| noise_share | 0.00 / 0.08 / 0.46 | 0.03 / 0.06 / 0.11 | 0.45 | 0% | 0% |
| missing_share | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.46 | 0% | 0% |
| variance_shift | 0.53 / 1.41 / 4.04 | 1.11 / 1.46 / 2.31 | 0.53 | 0% | 0% |
| level_shift | 0.51 / 1.52 / 2.53 | 0.83 / 1.38 / 2.15 | 0.47 | 0% | 0% |
| diff_kurtosis | -0.26 / 0.78 / 4.25 | -0.76 / 1.37 / 3.35 | 0.53 | 0% | 5% |
| jump_share | 0.06 / 0.11 / 0.68 | 0.03 / 0.15 / 0.45 | 0.53 | 0% | 6% |
| skewness | -0.54 / 0.03 / 0.58 | -0.50 / 0.01 / 0.50 | 0.48 | 0% | 0% |
| diff_skewness | -1.44 / 0.00 / 1.35 | -1.24 / 0.00 / 1.21 | 0.50 | 0% | 0% |

## Descriptors: Operational benchmarks against synthetic (10% / median / 90%)

| Descriptor | Synthetic | Operational benchmarks | Chance higher | Above synthetic 99% | Below synthetic 1% |
|---|---|---|---|---|---|
| diff_skewness | -1.44 / 0.00 / 1.35 | -0.24 / 0.33 / 1.02 | 0.64 | 0% | 0% |
| distinct_share | 0.24 / 1.00 / 1.00 | 0.13 / 0.67 / 1.00 | 0.36 | 0% | 2% |
| volatility_clustering | -0.01 / 0.16 / 0.75 | -0.01 / 0.38 / 0.80 | 0.63 | 0% | 0% |
| flat_share | 0.00 / 0.00 / 0.36 | 0.00 / 0.02 / 0.25 | 0.62 | 2% | 0% |
| acf_64 | -0.23 / 0.15 / 0.61 | -0.32 / -0.00 / 0.55 | 0.38 | 1% | 0% |
| diff_acf_1 | -0.38 / 0.17 / 0.84 | -0.22 / 0.52 / 0.84 | 0.62 | 0% | 0% |
| acf_16 | -0.05 / 0.57 / 0.89 | -0.14 / 0.25 / 0.88 | 0.40 | 1% | 0% |
| bimodality | 0.35 / 0.50 / 0.77 | 0.36 / 0.56 / 0.86 | 0.60 | 1% | 0% |
| mode_share | 0.00 / 0.00 / 0.15 | 0.00 / 0.01 / 0.05 | 0.60 | 0% | 0% |
| power_mid | 0.11 / 0.28 / 0.76 | 0.11 / 0.47 / 0.94 | 0.59 | 5% | 0% |
| seasonal_strength | 0.00 / 0.00 / 0.67 | 0.00 / 0.00 / 0.96 | 0.58 | 10% | 0% |
| power_jitter | 0.00 / 0.01 / 0.10 | 0.00 / 0.00 / 0.07 | 0.42 | 0% | 1% |
| season_acf | 0.00 / 0.00 / 0.69 | 0.00 / 0.00 / 0.80 | 0.57 | 0% | 0% |
| linear_trend | 0.00 / 0.14 / 0.91 | 0.00 / 0.06 / 0.75 | 0.43 | 0% | 1% |
| lumpiness | 0.00 / 0.04 / 0.31 | 0.00 / 0.08 / 0.31 | 0.57 | 0% | 0% |
| stability | 0.12 / 0.71 / 0.97 | 0.19 / 0.52 / 0.96 | 0.43 | 0% | 0% |
| season_harmonics | 0.00 / 0.00 / 0.24 | 0.00 / 0.00 / 0.43 | 0.56 | 0% | 0% |
| power_slow | 0.03 / 0.52 / 0.84 | 0.02 / 0.30 / 0.86 | 0.44 | 2% | 0% |
| season_amplitude_change | 0.00 / 0.00 / 0.33 | 0.00 / 0.00 / 0.33 | 0.54 | 1% | 0% |
| missing_share | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.46 | 0% | 0% |
| trend_strength | 0.04 / 0.68 / 0.98 | 0.01 / 0.61 / 0.97 | 0.47 | 0% | 0% |
| kurtosis | -0.79 / -0.39 / 0.91 | -0.84 / -0.27 / 0.76 | 0.53 | 0% | 0% |
| variance_shift | 0.53 / 1.41 / 4.04 | 0.68 / 1.51 / 3.89 | 0.53 | 0% | 0% |
| power_fast | 0.01 / 0.04 / 0.36 | 0.01 / 0.04 / 0.28 | 0.47 | 0% | 0% |
| acf_4 | 0.20 / 0.86 / 0.97 | 0.35 / 0.85 / 0.97 | 0.47 | 1% | 1% |
| level_shift | 0.51 / 1.52 / 2.53 | 0.81 / 1.56 / 2.34 | 0.53 | 1% | 0% |
| acf_1 | 0.72 / 0.97 / 0.99 | 0.77 / 0.97 / 1.00 | 0.53 | 2% | 0% |
| permutation_entropy | 0.40 / 0.85 / 0.99 | 0.55 / 0.88 / 0.99 | 0.53 | 0% | 0% |
| skewness | -0.54 / 0.03 / 0.58 | -0.88 / 0.08 / 0.66 | 0.52 | 1% | 0% |
| noise_share | 0.00 / 0.08 / 0.46 | 0.01 / 0.06 / 0.42 | 0.48 | 1% | 0% |
| jump_share | 0.06 / 0.11 / 0.68 | 0.07 / 0.12 / 0.40 | 0.52 | 2% | 0% |
| long_memory | 0.70 / 1.25 / 1.72 | 0.81 / 1.22 / 1.67 | 0.48 | 0% | 0% |
| diff_kurtosis | -0.26 / 0.78 / 4.25 | -0.10 / 0.97 / 3.12 | 0.51 | 0% | 0% |
| spectral_slope | 0.63 / 1.65 / 2.07 | 0.76 / 1.55 / 2.20 | 0.49 | 1% | 0% |
| at_extreme_share | 0.00 / 0.00 / 0.06 | 0.00 / 0.00 / 0.01 | 0.50 | 1% | 0% |
| spectral_entropy | 0.26 / 0.41 / 0.69 | 0.25 / 0.41 / 0.72 | 0.50 | 0% | 1% |

## Descriptors: IND-Eval against synthetic (10% / median / 90%)

| Descriptor | Synthetic | IND-Eval | Chance higher | Above synthetic 99% | Below synthetic 1% |
|---|---|---|---|---|---|
| distinct_share | 0.24 / 1.00 / 1.00 | 0.01 / 0.25 / 0.94 | 0.14 | 0% | 14% |
| mode_share | 0.00 / 0.00 / 0.15 | 0.01 / 0.13 / 0.62 | 0.85 | 2% | 0% |
| flat_share | 0.00 / 0.00 / 0.36 | 0.00 / 0.24 / 0.81 | 0.82 | 2% | 0% |
| at_extreme_share | 0.00 / 0.00 / 0.06 | 0.00 / 0.01 / 0.61 | 0.71 | 2% | 0% |
| acf_64 | -0.23 / 0.15 / 0.61 | -0.36 / -0.01 / 0.53 | 0.36 | 1% | 0% |
| linear_trend | 0.00 / 0.14 / 0.91 | 0.00 / 0.03 / 0.68 | 0.38 | 1% | 1% |
| bimodality | 0.35 / 0.50 / 0.77 | 0.33 / 0.60 / 0.90 | 0.62 | 2% | 3% |
| volatility_clustering | -0.01 / 0.16 / 0.75 | 0.02 / 0.29 / 0.82 | 0.62 | 3% | 1% |
| trend_strength | 0.04 / 0.68 / 0.98 | 0.01 / 0.32 / 0.98 | 0.38 | 1% | 0% |
| spectral_slope | 0.63 / 1.65 / 2.07 | 0.20 / 1.20 / 2.18 | 0.39 | 2% | 2% |
| power_slow | 0.03 / 0.52 / 0.84 | 0.01 / 0.26 / 0.85 | 0.39 | 3% | 0% |
| power_jitter | 0.00 / 0.01 / 0.10 | 0.00 / 0.02 / 0.40 | 0.61 | 9% | 2% |
| noise_share | 0.00 / 0.08 / 0.46 | 0.00 / 0.14 / 0.86 | 0.60 | 8% | 0% |
| acf_1 | 0.72 / 0.97 / 0.99 | 0.17 / 0.92 / 1.00 | 0.40 | 3% | 8% |
| stability | 0.12 / 0.71 / 0.97 | 0.05 / 0.58 / 0.97 | 0.40 | 1% | 1% |
| acf_16 | -0.05 / 0.57 / 0.89 | -0.28 / 0.39 / 0.89 | 0.41 | 2% | 0% |
| lumpiness | 0.00 / 0.04 / 0.31 | 0.00 / 0.08 / 0.46 | 0.59 | 2% | 1% |
| permutation_entropy | 0.40 / 0.85 / 0.99 | 0.29 / 0.73 / 0.99 | 0.41 | 0% | 0% |
| spectral_entropy | 0.26 / 0.41 / 0.69 | 0.25 / 0.46 / 0.89 | 0.58 | 4% | 2% |
| variance_shift | 0.53 / 1.41 / 4.04 | 0.50 / 1.90 / 4.86 | 0.58 | 3% | 1% |
| acf_4 | 0.20 / 0.86 / 0.97 | 0.03 / 0.77 / 0.98 | 0.42 | 3% | 0% |
| diff_acf_1 | -0.38 / 0.17 / 0.84 | -0.49 / 0.02 / 0.85 | 0.43 | 3% | 7% |
| power_fast | 0.01 / 0.04 / 0.36 | 0.01 / 0.08 / 0.39 | 0.57 | 0% | 2% |
| kurtosis | -0.79 / -0.39 / 0.91 | -0.85 / -0.19 / 1.89 | 0.56 | 3% | 1% |
| skewness | -0.54 / 0.03 / 0.58 | -0.64 / 0.11 / 1.01 | 0.56 | 3% | 2% |
| diff_kurtosis | -0.26 / 0.78 / 4.25 | -0.15 / 1.41 / 3.69 | 0.56 | 0% | 0% |
| seasonal_strength | 0.00 / 0.00 / 0.67 | 0.00 / 0.00 / 0.90 | 0.55 | 3% | 0% |
| season_harmonics | 0.00 / 0.00 / 0.24 | 0.00 / 0.00 / 0.34 | 0.55 | 0% | 0% |
| diff_skewness | -1.44 / 0.00 / 1.35 | -0.70 / 0.05 / 0.94 | 0.54 | 0% | 0% |
| missing_share | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.46 | 0% | 0% |
| season_acf | 0.00 / 0.00 / 0.69 | 0.00 / 0.00 / 0.78 | 0.54 | 1% | 0% |
| long_memory | 0.70 / 1.25 / 1.72 | 0.56 / 1.18 / 1.81 | 0.46 | 1% | 0% |
| season_amplitude_change | 0.00 / 0.00 / 0.33 | 0.00 / 0.00 / 0.34 | 0.52 | 1% | 0% |
| jump_share | 0.06 / 0.11 / 0.68 | 0.06 / 0.13 / 0.50 | 0.52 | 1% | 1% |
| level_shift | 0.51 / 1.52 / 2.53 | 0.58 / 1.45 / 2.67 | 0.51 | 2% | 1% |
| power_mid | 0.11 / 0.28 / 0.76 | 0.08 / 0.29 / 0.81 | 0.50 | 0% | 1% |

## Cycles

| Group | Has a cycle | Strong cycle (strength above 0.6) | Cycle length 25% | Median | 75% |
|---|---|---|---|---|---|
| Synthetic | 31% | 12% | 15 | 36 | 86 |
| FakeBench-TS | 3% | 1% | 60 | 68 | 130 |
| Operational benchmarks | 41% | 34% | 24 | 24 | 98 |
| IND-Eval | 36% | 29% | 24 | 96 | 139 |

### Cycles per dataset

| Group | Dataset | Series | Has a cycle | Strong cycle | Median cycle length |
|---|---|---|---|---|---|
| FakeBench-TS | normal | 200 | 4% | 2% | 78 |
| FakeBench-TS | logistic | 150 | 3% | 0% | 56 |
| FakeBench-TS | uniform | 150 | 3% | 1% | 80 |
| FakeBench-TS | arcsine | 150 | 3% | 0% | 60 |
| FakeBench-TS | rademacher | 150 | 2% | 0% | 143 |
| FakeBench-TS | lazy_grid | 150 | 3% | 1% | 68 |
| FakeBench-TS | skellam | 150 | 2% | 0% | 68 |
| FakeBench-TS | bimodal | 150 | 3% | 0% | 103 |
| FakeBench-TS | laplace | 200 | 4% | 1% | 68 |
| FakeBench-TS | jump_mixture | 250 | 4% | 0% | 60 |
| FakeBench-TS | stretched_exponential | 250 | 2% | 0% | 151 |
| FakeBench-TS | sign_lognormal | 250 | 3% | 0% | 68 |
| FakeBench-TS | student_t | 300 | 1% | 0% | 68 |
| FakeBench-TS | skew_gamma | 500 | 4% | 0% | 68 |
| FakeBench-TS | skew_two_point | 400 | 3% | 1% | 73 |
| FakeBench-TS | skew_poisson | 400 | 3% | 1% | 68 |
| FakeBench-TS | skew_negative_binomial | 400 | 4% | 0% | 68 |
| FakeBench-TS | skew_inverse_gaussian | 400 | 3% | 1% | 117 |
| FakeBench-TS | skew_jump | 400 | 2% | 1% | 68 |
| Operational benchmarks | caster | 1,054 | 9% | 1% | 40 |
| Operational benchmarks | cloud | 242 | 30% | 5% | 24 |
| Operational benchmarks | ucr | 840 | 100% | 99% | 24 |
| Operational benchmarks | cats | 400 | 6% | 2% | 66 |
| IND-Eval | 3w | 2,996 | 8% | 2% | 48 |
| IND-Eval | electricity | 2,905 | 84% | 79% | 96 |
| IND-Eval | mds | 2,885 | 5% | 1% | 47 |
| IND-Eval | metropt3 | 950 | 64% | 50% | 125 |
| IND-Eval | neats | 2,933 | 16% | 2% | 14 |
| IND-Eval | pronto | 72 | 14% | 4% | 48 |
| IND-Eval | solar | 2,839 | 89% | 87% | 141 |
| IND-Eval | tablet_compression | 2,998 | 10% | 1% | 12 |

## Basin's error against the other models (stage 11 forecast cases)

2,535 cases; Basin's error over the mean error of the three other models, median 1.05. Rank correlation with the distance to the synthetic series: 0.27 overall, 0.13 within datasets.

| Distance quarter | Distance ratio | Cases | Basin over others | Basin worse |
|---|---|---|---|---|
| 1 | 0.32 to 0.95 | 634 | 1.00 | 50% |
| 2 | 0.95 to 1.21 | 634 | 1.01 | 56% |
| 3 | 1.21 to 1.46 | 635 | 1.13 | 70% |
| 4 | 1.46 to 3.30 | 632 | 1.16 | 73% |

| Dataset | Cases | Basin over others | Median distance ratio |
|---|---|---|---|
| caster | 1,054 | 1.00 | 1.04 |
| cats | 400 | 1.00 | 1.07 |
| cloud | 241 | 1.01 | 1.09 |
| ucr | 840 | 1.22 | 1.39 |

### Descriptors that go with Basin losing (rank correlation)

| Descriptor | Overall | Within datasets |
|---|---|---|
| season_harmonics | +0.38 | +0.24 |
| bimodality | +0.28 | +0.20 |
| season_amplitude_change | +0.27 | -0.18 |
| skewness | -0.13 | -0.14 |
| diff_kurtosis | +0.16 | +0.14 |
| long_memory | +0.02 | +0.12 |
| jump_share | +0.12 | +0.11 |
| permutation_entropy | -0.03 | +0.11 |
| stability | -0.06 | +0.11 |
| acf_1 | +0.09 | +0.11 |
| variance_shift | +0.08 | +0.10 |
| acf_4 | -0.01 | +0.10 |
| noise_share | -0.15 | -0.09 |
| flat_share | +0.03 | +0.09 |
| acf_16 | -0.15 | +0.09 |
| diff_skewness | +0.22 | +0.08 |
| seasonal_strength | +0.37 | +0.08 |
| power_jitter | -0.14 | -0.07 |
| power_fast | -0.06 | -0.07 |
| season_acf | +0.33 | -0.06 |
| kurtosis | -0.07 | -0.06 |
| spectral_entropy | -0.11 | -0.06 |
| linear_trend | -0.10 | +0.05 |
| diff_acf_1 | +0.18 | +0.05 |
| acf_64 | -0.23 | +0.05 |
| mode_share | -0.02 | +0.05 |
| at_extreme_share | +0.02 | +0.05 |
| distinct_share | +0.07 | -0.04 |
| volatility_clustering | +0.18 | +0.04 |
| trend_strength | -0.09 | -0.04 |
| power_mid | +0.21 | -0.03 |
| level_shift | +0.03 | +0.03 |
| lumpiness | +0.12 | -0.03 |
| spectral_slope | -0.02 | +0.01 |
| power_slow | -0.16 | +0.01 |
| missing_share | +0.00 | +0.00 |

## Benchmark series without synthetic neighbours, by kind

| Kind | Series | Mostly | Differs from the typical synthetic series |
|---|---|---|---|
| A | 1,395 | solar 1,366, mds 13, tablet_compression 8, 3w 3 | at_extreme_share higher (+3.6); mode_share higher (+3.4); variance_shift higher (+2.7); flat_share higher (+2.5); seasonal_strength higher (+2.4); bimodality higher (+2.1) |
| B | 1,202 | solar 1,143, neats 26, mds 23, electricity 10 | at_extreme_share higher (+3.4); mode_share higher (+3.2); season_acf higher (+2.6); season_amplitude_change higher (+2.5); seasonal_strength higher (+2.5); acf_16 lower (-2.4) |
| C | 511 | metropt3 272, ucr 129, electricity 62, neats 31 | season_harmonics higher (+2.7); seasonal_strength higher (+2.6); power_mid higher (+2.2); bimodality higher (+2.2); season_acf higher (+2.1); jump_share higher (+2.1) |
| D | 498 | tablet_compression 412, neats 33, electricity 27, caster 15 | kurtosis higher (+4.4); power_jitter higher (+4.1); acf_1 lower (-4.1); lumpiness higher (+3.4); noise_share higher (+2.9); acf_4 lower (-2.5) |
| E | 417 | neats 203, mds 159, tablet_compression 39, electricity 15 | at_extreme_share higher (+5.7); mode_share higher (+4.1); power_jitter higher (+4.0); noise_share higher (+3.9); acf_1 lower (-3.9); flat_share higher (+3.0) |
| F | 352 | metropt3 140, neats 115, mds 39, 3w 17 | lumpiness higher (+4.2); kurtosis higher (+3.2); jump_share higher (+3.0); variance_shift higher (+2.8); bimodality higher (+2.5); diff_kurtosis higher (+2.3) |
| G | 318 | mds 110, 3w 97, neats 49, tablet_compression 22 | flat_share higher (+3.2); distinct_share lower (-2.5); variance_shift higher (+2.2); permutation_entropy lower (-2.2); mode_share higher (+1.8); diff_kurtosis higher (+1.3) |
| H | 306 | neats 240, tablet_compression 26, mds 17, electricity 17 | lumpiness higher (+4.8); kurtosis higher (+4.4); skewness higher (+3.1); mode_share higher (+3.0); variance_shift higher (+2.7); power_fast higher (+2.7) |

## Synthetic series without benchmark neighbours, by kind

| Kind | Series | Mostly | Differs from the typical synthetic series |
|---|---|---|---|
| A | 679 | synthetic 679 | missing_share higher (+10.0); volatility_clustering higher (+0.3); variance_shift higher (+0.2); power_mid higher (+0.2); level_shift higher (+0.1); bimodality lower (-0.1) |
| B | 609 | synthetic 609 | power_mid higher (+1.9); trend_strength lower (-1.8); season_acf higher (+1.6); acf_16 lower (-1.5); power_slow lower (-1.5); stability lower (-1.4) |
| C | 428 | synthetic 428 | jump_share higher (+3.7); diff_kurtosis higher (+2.7); permutation_entropy lower (-2.2); variance_shift higher (+2.0); level_shift higher (+1.3); bimodality higher (+1.1) |
| D | 303 | synthetic 303 | power_fast higher (+4.2); acf_4 lower (-3.5); season_acf higher (+2.6); long_memory lower (-2.1); stability lower (-2.1); spectral_slope lower (-2.0) |
| E | 283 | synthetic 283 | lumpiness higher (+4.3); kurtosis higher (+3.9); jump_share higher (+3.3); diff_kurtosis higher (+2.4); variance_shift higher (+2.4); skewness lower (-2.1) |
| F | 268 | synthetic 268 | season_harmonics higher (+2.4); season_amplitude_change higher (+2.3); season_acf higher (+2.0); linear_trend higher (+1.8); level_shift lower (-1.1); acf_64 higher (+1.1) |

## Second opinion: Chronos-2 embeddings

| Group | Dataset | Series | Close | Covered | Median distance ratio |
|---|---|---|---|---|---|
| FakeBench-TS | normal | 200 | 72% | 100% | 0.88 |
| FakeBench-TS | logistic | 150 | 67% | 98% | 0.87 |
| FakeBench-TS | uniform | 150 | 71% | 99% | 0.91 |
| FakeBench-TS | arcsine | 150 | 64% | 99% | 0.91 |
| FakeBench-TS | rademacher | 150 | 65% | 99% | 0.90 |
| FakeBench-TS | lazy_grid | 150 | 70% | 97% | 0.88 |
| FakeBench-TS | skellam | 150 | 69% | 99% | 0.91 |
| FakeBench-TS | bimodal | 150 | 61% | 99% | 0.94 |
| FakeBench-TS | laplace | 200 | 70% | 99% | 0.88 |
| FakeBench-TS | jump_mixture | 250 | 17% | 98% | 1.22 |
| FakeBench-TS | stretched_exponential | 250 | 26% | 97% | 1.15 |
| FakeBench-TS | sign_lognormal | 250 | 43% | 97% | 1.05 |
| FakeBench-TS | student_t | 300 | 60% | 99% | 0.91 |
| FakeBench-TS | skew_gamma | 500 | 78% | 100% | 0.87 |
| FakeBench-TS | skew_two_point | 400 | 45% | 98% | 1.04 |
| FakeBench-TS | skew_poisson | 400 | 82% | 100% | 0.84 |
| FakeBench-TS | skew_negative_binomial | 400 | 71% | 100% | 0.89 |
| FakeBench-TS | skew_inverse_gaussian | 400 | 44% | 100% | 1.04 |
| FakeBench-TS | skew_jump | 400 | 26% | 99% | 1.16 |
| Operational benchmarks | caster | 1,054 | 49% | 98% | 1.01 |
| Operational benchmarks | cloud | 242 | 30% | 98% | 1.10 |
| Operational benchmarks | ucr | 840 | 30% | 100% | 1.24 |
| Operational benchmarks | cats | 400 | 30% | 86% | 1.36 |
| IND-Eval | 3w | 791 | 44% | 94% | 1.04 |
| IND-Eval | electricity | 754 | 2% | 92% | 1.32 |
| IND-Eval | mds | 771 | 28% | 85% | 1.27 |
| IND-Eval | metropt3 | 269 | 48% | 99% | 1.03 |
| IND-Eval | neats | 825 | 25% | 85% | 1.31 |
| IND-Eval | pronto | 20 | 20% | 70% | 1.27 |
| IND-Eval | solar | 732 | 1% | 64% | 1.30 |
| IND-Eval | tablet_compression | 798 | 29% | 86% | 1.20 |
| Synthetic | synthetic | 6,270 | 50% | 95% | 1.00 |
| Synthetic, set aside | synthetic | 1,587 | 49% | 95% | 1.00 |
| FakeBench-TS | all | 5,000 | 57% | 99% | 0.95 |
| Operational benchmarks | all | 2,536 | 38% | 96% | 1.11 |
| IND-Eval | all | 4,960 | 23% | 85% | 1.24 |
| Synthetic | all | 6,270 | 50% | 95% | 1.00 |
| Synthetic, set aside | all | 1,587 | 49% | 95% | 1.00 |

| Group | Classifier AUC | Rank correlation of the two distances | Same verdict on covered |
|---|---|---|---|
| FakeBench-TS | 0.974 | 0.47 | 99% |
| Operational benchmarks | 0.916 | 0.30 | 90% |
| IND-Eval | 0.937 | 0.43 | 75% |
| Synthetic, set aside | 0.510 | 0.48 | 92% |

## Principal components

| Component | Share of variation | Moves most with |
|---|---|---|
| 1 | 31% | +trend_strength, −noise_share, +linear_trend, +stability, +power_slow |
| 2 | 16% | +permutation_entropy, +noise_share, −variance_shift, −seasonal_strength, −diff_acf_1 |
| 3 | 11% | +flat_share, −permutation_entropy, +mode_share, −season_acf, −seasonal_strength |
