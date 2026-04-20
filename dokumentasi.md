## Konsep Riset BiLSTM + FLF – Versi 4

**Problem Statement**  
Memprediksi OHLC candle berikutnya pada pasangan EURUSD interval 4 jam sambil meminimalkan error open/high/low/close menggunakan Forex Loss Function (FLF), sehingga akurasi lebih baik daripada loss generik seperti MSE pada model LSTM standar.

**Dataset & Input**  
- Dataset utama: `EURUSD_H4_25Oct17.csv` (10.078 candle H4, Jun 2015–Sep 2018).  
- Sliding window membentuk pasangan supervised: setiap sampel memuat `window` candle berturut-turut sebagai input dan candle berikutnya sebagai target.  
- Validasi memakai holdout berbasis waktu (earliest 70% untuk training, 30% terbaru untuk validasi/pengujian).  
- Normalisasi z-score diestimasi dari subset train dan diterapkan konsisten pada validation/test sebelum denormalisasi output.

**Metodologi / Model**  
- Arsitektur: satu layer BiDirectional LSTM (units per arah = parameter `units`; total aktivasi dua kali lipat) + Dense 4 output.  
- Optimizer: Nadam (`lr=1e-4`, `beta1=0.9`, `beta2=0.999`, `schedule_decay=0.004` default).  
- Loss: FLF (λ=0.9, σ=0.1) yang menyesuaikan penalti berdasarkan rata-rata body/shadow candle.  
- Evaluasi: MAE per komponen OHLC + perbandingan baseline (Single-/Multi-/OHLC-LSTM, ARIMA, FBProphet dari paper).

**Solusi Teknis**  
- Skrip `bilstm_flf_experiment.py` memuat CLI untuk memilih dataset, delimiter, window, units, aktivasi, konstanta FLF, serta jalur output.  
- Pipeline otomatis: deteksi kolom OHLC → sliding window → training BiLSTM dengan loss FLF → output CSV prediksi & history (opsional).  
- Nilai default mengikuti konfigurasi studi terdahulu (window 20, split 0.7, units 200, epochs 75, batch 128, aktivasi tanh/linear).

**Rencana Eksperimen (Grid Draft)**  
1. **Baseline check** – window 20, units 200 (→ 200 unit forward + 200 backward), tanh, lr 1e-4, epochs 75 memastikan pipeline berjalan dan memantau indikasi under/overfitting lewat loss train vs val.  
2. **Window sweep** – window ∈ {16, 24, 32, 48} dengan parameter lain tetap untuk melihat panjang konteks terbaik.  
3. **Units sweep** – units ∈ {128, 200, 256} pada window unggulan; cek memori vs performa.  
4. **Activation & LR** – bandingkan tanh vs relu (opsi elu) serta lr {1e-4, 5e-4} demi keseimbangan stabilitas dan kecepatan konvergensi.  
5. **Epoch tuning** – 60 vs 90 epoch untuk konfigurasi favorit guna mencegah under/overfit.  
6. **Loss comparison** – jalankan MSE vs FLF pada konfigurasi terbaik untuk menegaskan kontribusi loss domain.  

_Catatan: Studi terdahulu (FLF-LSTM, Applied Soft Computing 2020) menggunakan dataset dan pendekatan sliding window yang sama; pipeline baru memperluasnya dengan BiLSTM satu layer configurable dan validasi holdout berbasis waktu._

**Template Rekap Hyperparameter (isi setelah run tersedia)**
| Step            | Param disweeping | Setting tetap (baseline)                                 | Nilai dicoba (input)                                   | Metric (MAE avg OHLC) | Best |
|-----------------|------------------|-----------------------------------------------------------|--------------------------------------------------------|-----------------------|------|
| 1. Window sweep | window           | units=200, act=tanh, lr=1e-4, epochs=75, batch=128       | 16, 24, 32, 48                                         |                       |      |
| 2. Units sweep  | units            | window=WIN_BEST, act=tanh, lr=1e-4, epochs=75, batch=128 | 128, 200, 256                                          |                       |      |
| 3. Act + LR     | activation, lr   | window=WIN_BEST, units=UNITS_BEST, epochs=75, batch=128  | (tanh,1e-4),(relu,1e-4),(relu,3e-4),(relu,5e-4),(tanh,3e-4) |                       |      |
| 4. Epoch tune   | epochs           | window=WIN_BEST, units=UNITS_BEST, act=ACT_BEST, lr=LR_BEST, batch=128 | 60, 75, 90                                             |                       |      |
| 5. Loss compare | loss             | config terbaik (hasil langkah 1–4)                       | FLF, MSE                                               |                       |      |
