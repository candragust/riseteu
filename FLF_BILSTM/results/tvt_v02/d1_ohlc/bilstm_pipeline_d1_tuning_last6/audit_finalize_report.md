# Audit Finalisasi TVT v02 D1 FLF-BiLSTM

- Final selected: `bilstm_d1_s7_w4_h256_actrelu_lr0.0007_e50_lam1_sig0.05_b32`
- Final mean `MAE avg`: `30.685042` pips
- Final selected_from: `stage_best`

## Parameter Final

- `window` = `4`
- `units` = `256`
- `activation` = `relu`
- `lr` = `0.0007`
- `epochs` = `50`
- `lambda_coef` = `1.0`
- `sigma_coef` = `0.05`
- `batch` = `32`

## Ringkasan Stage

| Stage | Stage best MAE | Selected after stage | Source |
| --- | ---: | ---: | --- |
| 1 | 32.544371 | 32.544371 | stage_best |
| 2 | 36.475282 | 32.544371 | carry_forward |
| 3 | 31.948052 | 31.948052 | stage_best |
| 4 | 32.100897 | 31.948052 | carry_forward |
| 5 | 30.708755 | 30.708755 | stage_best |
| 6 | 30.929748 | 30.708755 | carry_forward |
| 7 | 30.685042 | 30.685042 | stage_best |
