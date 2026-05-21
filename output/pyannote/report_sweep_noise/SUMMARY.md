# Vivo Selected Cases Low-Threshold Sweep Summary

This report keeps the original diarization logic unchanged and sweeps only the low-threshold hyperparameter.

Checkpoints: baseline, best_model

Thresholds: 0.05, 0.25

## Summary by sample, checkpoint, and low-threshold

| checkpoint_label   | noise_group   | sample_label              | case_id   |   noise_pct | overlap_bucket   |   low_threshold |      der |     miss |       fa |      conf |   ov_precision |   ov_recall |      ov_f1 |   pred_segments |   ref_segments |
|:-------------------|:--------------|:--------------------------|:----------|------------:|:-----------------|----------------:|---------:|---------:|---------:|----------:|---------------:|------------:|-----------:|----------------:|---------------:|
| baseline           | 0% noise      | data11 | ov10 | 0% noise  | data11    |           0 | ov10             |            0.05 | 59.0738  | 17.6005  | 26.7528  | 14.7205   |    0.135689    |   0.199618  | 0.161559   |             232 |            285 |
| baseline           | 0% noise      | data11 | ov10 | 0% noise  | data11    |           0 | ov10             |            0.25 | 57.2443  | 19.7223  | 22.001   | 15.521    |    0.175484    |   0.179307  | 0.177374   |             187 |            285 |
| baseline           | 0% noise      | data2 | ov0 | 0% noise    | data2     |           0 | ov0              |            0.05 | 56.3102  | 13.4489  | 21.9275  | 20.9337   |    0.000671411 |   0.0641026 | 0.0013289  |             175 |            257 |
| baseline           | 0% noise      | data2 | ov0 | 0% noise    | data2     |           0 | ov0              |            0.25 | 55.9744  | 17.1024  | 17.2378  | 21.6343   |    0.000825083 |   0.025641  | 0.00159872 |             173 |            257 |
| baseline           | 0% noise      | data56 | ov5 | 0% noise   | data56    |           0 | ov5              |            0.05 | 55.7789  | 14.1967  | 25.2447  | 16.3375   |    0.0626076   |   0.133753  | 0.0852916  |             183 |            257 |
| baseline           | 0% noise      | data56 | ov5 | 0% noise   | data56    |           0 | ov5              |            0.25 | 51.8109  | 16.6686  | 18.7269  | 16.4154   |    0.0681206   |   0.0659862 | 0.0670364  |             159 |            257 |
| baseline           | with noise    | data14 | ov10 | 17% noise | data14    |          17 | ov10             |            0.05 | 65.3949  | 16.289   | 28.0035  | 21.1025   |    0.203635    |   0.286576  | 0.238089   |             267 |            324 |
| baseline           | with noise    | data14 | ov10 | 17% noise | data14    |          17 | ov10             |            0.25 | 63.9517  | 23.4305  | 18.7587  | 21.7625   |    0.190289    |   0.129023  | 0.153778   |             210 |            324 |
| baseline           | with noise    | data5 | ov0 | 23% noise   | data5     |          23 | ov0              |            0.05 | 62.5244  | 11.8347  | 23.2056  | 27.4841   |    0.00098464  |   0.16129   | 0.00195733 |             207 |            237 |
| baseline           | with noise    | data5 | ov0 | 23% noise   | data5     |          23 | ov0              |            0.25 | 58.493   | 17.31    | 16.0008  | 25.1823   |    0.00311915  |   0.0806452 | 0.00600601 |             174 |            237 |
| baseline           | with noise    | data62 | ov5 | 23% noise  | data62    |          23 | ov5              |            0.05 | 68.6824  | 17.7504  | 24.4128  | 26.5191   |    0.121838    |   0.192293  | 0.149165   |             227 |            294 |
| baseline           | with noise    | data62 | ov5 | 23% noise  | data62    |          23 | ov5              |            0.25 | 66.4056  | 23.5831  | 18.0587  | 24.7638   |    0.184356    |   0.135791  | 0.15639    |             203 |            294 |
| best_model         | 0% noise      | data11 | ov10 | 0% noise  | data11    |           0 | ov10             |            0.05 |  3.55182 |  1.6974  |  1.59382 |  0.260595 |    0.977522    |   0.904668  | 0.939685   |             402 |            285 |
| best_model         | 0% noise      | data11 | ov10 | 0% noise  | data11    |           0 | ov10             |            0.25 |  3.55182 |  1.6974  |  1.59382 |  0.260595 |    0.977522    |   0.904668  | 0.939685   |             402 |            285 |
| best_model         | 0% noise      | data2 | ov0 | 0% noise    | data2     |           0 | ov0              |            0.05 | 16.1207  |  3.61065 |  1.23619 | 11.2738   |    0.0151515   |   0.0128205 | 0.0138889  |             352 |            257 |
| best_model         | 0% noise      | data2 | ov0 | 0% noise    | data2     |           0 | ov0              |            0.25 | 16.1207  |  3.61065 |  1.23619 | 11.2738   |    0.0151515   |   0.0128205 | 0.0138889  |             352 |            257 |
| best_model         | 0% noise      | data56 | ov5 | 0% noise   | data56    |           0 | ov5              |            0.05 | 21.3847  |  2.74128 |  1.39085 | 17.2525   |    0.964031    |   0.81747   | 0.884722   |             348 |            257 |
| best_model         | 0% noise      | data56 | ov5 | 0% noise   | data56    |           0 | ov5              |            0.25 | 21.3847  |  2.74128 |  1.39085 | 17.2525   |    0.964031    |   0.81747   | 0.884722   |             348 |            257 |
| best_model         | with noise    | data14 | ov10 | 17% noise | data14    |          17 | ov10             |            0.05 | 23.1714  |  4.46491 |  1.93734 | 16.7692   |    0.972041    |   0.753571  | 0.848977   |             460 |            324 |
| best_model         | with noise    | data14 | ov10 | 17% noise | data14    |          17 | ov10             |            0.25 | 23.1714  |  4.46491 |  1.93734 | 16.7692   |    0.972041    |   0.753571  | 0.848977   |             460 |            324 |
| best_model         | with noise    | data5 | ov0 | 23% noise   | data5     |          23 | ov0              |            0.05 | 10.4864  |  2.40946 |  2.89784 |  5.17914  |    0.0151515   |   0.0322581 | 0.0206186  |             328 |            237 |
| best_model         | with noise    | data5 | ov0 | 23% noise   | data5     |          23 | ov0              |            0.25 | 10.4864  |  2.40946 |  2.89784 |  5.17914  |    0.0151515   |   0.0322581 | 0.0206186  |             328 |            237 |
| best_model         | with noise    | data62 | ov5 | 23% noise  | data62    |          23 | ov5              |            0.05 | 18.3844  |  5.57582 |  1.75546 | 11.0531   |    0.962347    |   0.65339   | 0.778329   |             407 |            294 |
| best_model         | with noise    | data62 | ov5 | 23% noise  | data62    |          23 | ov5              |            0.25 | 18.3844  |  5.57582 |  1.75546 | 11.0531   |    0.962347    |   0.65339   | 0.778329   |             407 |            294 |

## Run counts

- Total successful runs: 24
- Unique checkpoints: 2
- Unique selected cases: 6
- Unique thresholds: 2
