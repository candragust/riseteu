# FLF-LSTM EURUSD

Folder ini berisi implementasi `FLF-LSTM` untuk EURUSD H4 sebagai pembanding langsung terhadap pipeline `FLF-BiLSTM`.

Prinsip eksperimennya sengaja dibuat sepadan:

- data source sama
- preprocessing dan standardisasi sama
- target tetap OHLC satu langkah ke depan
- loss tetap memakai Forex Loss Function (FLF)
- runner tuning tetap bertahap per stage
- format output dibuat kompatibel dengan ringkasan pipeline yang sudah ada

Perbedaan arsitektur inti:

- `FLF-BiLSTM`: `Bidirectional(LSTM(...))`
- `FLF-LSTM`: `LSTM(...)`

File utama:

- `lstm_flf_experiment.py`
- `lstm_flf_config.json`
- `lstm_flf_config_wf72_test1_latest.json`
- `eurusd_lstm_pipeline_runner.py`
- `rolling_fixed_lstm_runner.py`

Tujuan folder ini adalah menghasilkan eksperimen pembanding yang bersih antara LSTM unidirectional dan BiLSTM, tanpa mencampur artefak hasil eksperimen lama.

Default yang sekarang diprioritaskan untuk eksperimen adalah horizon pendek:

- train `72 bulan`
- test `1 bulan`
- dataset tuning awal memakai fold terbaru dari skema `WF 72m/1m`

Grid tuning yang sekarang dipersempit di sekitar baseline terbaik BiLSTM:

- Stage 1: `window = [10, 12, 14]`
- Stage 2: `units = [224, 256]`
- Stage 3: `relu + lr = [5e-4, 7e-4, 9e-4]`
- Stage 4: `(lambda, sigma) = [(0.8, 0.15), (0.9, 0.10), (0.8, 0.10)]`
- Stage 5: `batch = [96, 128]`
- Stage 6: `epochs = [40, 50, 60]`
