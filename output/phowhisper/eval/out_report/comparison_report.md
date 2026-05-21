# COMPREHENSIVE COMPARISON REPORT (BASE vs STAGE 1 vs STAGE 2)

## 1. Global Summary

| Metric | Base Model | Best Stage 1 | Best Stage 2 | Delta (B vs S2) |
| :--- | :---: | :---: | :---: | :---: |
| **WER (%)** | **18.72%** | **13.87%** | **13.01%** | 5.72 |
| Substitutions (S) | 3691 | 2743 | 2803 | - |
| Deletions (D) | 1936 | 2185 | 2110 | - |
| Insertions (I) | 3589 | 1897 | 1490 | - |

## 2. Dataset/Group Breakdown

| Dataset | Base WER | S1 WER (S/D/I) | S2 WER (S/D/I) | Δ (B-S2) |
| :--- | :---: | :---: | :---: | :---: |
| chuyen_ho_chuyen_minh | 19.22% | 18.03% (6.40/6.12/5.51) | 16.93% (6.65/5.85/4.43) |  2.29 |
| coi_mo | 24.73% | 19.99% (9.77/6.86/3.36) | 19.96% (10.04/6.56/3.36) |  4.77 |
| conan | 11.86% | 7.75% (3.86/2.79/1.10) | 7.38% (3.83/2.73/0.82) |  4.48 |
| dustin_on_go | 20.00% | 16.12% (5.87/5.09/5.16) | 13.62% (5.88/4.87/2.87) |  6.38 |
| vif | 14.97% | 6.23% (2.38/1.43/2.42) | 6.48% (2.46/1.49/2.53) |  8.49 |

## 3. Category/Bucket Breakdown

| Bucket | Base WER | S1 WER | S2 WER | Δ (B-S2) |
| :--- | :---: | :---: | :---: | :---: |
| gold_real | 14.97% | 6.23% | 6.48% | 8.49 |
| hard_real | 24.73% | 19.99% | 19.96% | 4.77 |
| silver_real | 18.63% | 15.71% | 14.01% | 4.62 |

## 4. Processing Time

| Model | Total Time (s) | Average (s/sample) |
| :--- | :---: | :---: |
| Base Model | 437.68s | 0.2645s |
| Best Stage 1 | 441.32s | 0.2667s |
| Best Stage 2 | 428.12s | 0.2587s |
