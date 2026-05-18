# Audit Finalisasi TVT v02 H4 FLF-BiLSTM

- Final selected: `bilstm_tvt_s6_w12_h256_acttanh_lr0.0003_e50_lam0.8_sig0.15_b128`
- Final mean `MAE avg`: `13.665059` pips
- Final selected_from: `stage_best`

## Parameter Final

- `window` = `12`
- `units` = `256`
- `activation` = `tanh`
- `lr` = `0.0003`
- `epochs` = `50`
- `lambda_coef` = `0.8`
- `sigma_coef` = `0.15`
- `batch` = `128`

## Ringkasan Stage

| Stage | Stage best MAE | Selected after stage | Source |
| --- | ---: | ---: | --- |
| 1 | 13.806040 | 13.806040 | stage_best |
| 2 | 14.574125 | 13.806040 | carry_forward |
| 3 | 13.665085 | 13.665085 | stage_best |
| 4 | 13.665122 | 13.665085 | carry_forward |
| 5 | 13.665075 | 13.665075 | stage_best |
| 6 | 13.665059 | 13.665059 | stage_best |
