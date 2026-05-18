# Audit Finalisasi TVT v02 H4 FLF-LSTM

- Final selected: `lstm_tvt_s5_w10_h256_actrelu_lr0.0009_e60_lam0.8_sig0.1_b96`
- Final mean `MAE avg`: `13.038886` pips
- Final selected_from: `carry_forward`

## Parameter Final

- `window` = `10`
- `units` = `256`
- `activation` = `relu`
- `lr` = `0.0009`
- `epochs` = `60`
- `lambda_coef` = `0.8`
- `sigma_coef` = `0.1`
- `batch` = `96`

## Ringkasan Stage

| Stage | Stage best MAE | Selected after stage | Source |
| --- | ---: | ---: | --- |
| 1 | 13.235534 | 13.235534 | stage_best |
| 2 | 13.413150 | 13.235534 | carry_forward |
| 3 | 13.179811 | 13.179811 | stage_best |
| 4 | 13.358678 | 13.179811 | carry_forward |
| 5 | 13.038886 | 13.038886 | stage_best |
| 6 | 13.074972 | 13.038886 | carry_forward |
