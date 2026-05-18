# Audit Finalisasi TVT v02 D1 FLF-LSTM

- Final selected: `lstm_d1_s7_w4_h128_actrelu_lr0.0009_e60_lam0.8_sig0.1_b96`
- Final mean `MAE avg`: `38.184672` pips
- Final selected_from: `stage_best`

## Parameter Final

- `window` = `4`
- `units` = `128`
- `activation` = `relu`
- `lr` = `0.0009`
- `epochs` = `60`
- `lambda_coef` = `0.8`
- `sigma_coef` = `0.1`
- `batch` = `96`
- `dropout` = `0.0`
- `recurrent_dropout` = `0.0`
- `l2_reg` = `1e-05`

## Ringkasan Stage

| Stage | Stage best MAE | Selected after stage | Source |
| --- | ---: | ---: | --- |
| 1 | 46.114800 | 46.114800 | stage_best |
| 2 | 41.656460 | 41.656460 | stage_best |
| 3 | 41.656476 | 41.656460 | carry_forward |
| 4 | 41.669189 | 41.656460 | carry_forward |
| 5 | 38.984583 | 38.984583 | stage_best |
| 6 | 38.985411 | 38.984583 | carry_forward |
| 7 | 38.184672 | 38.184672 | stage_best |
