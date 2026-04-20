# Handover Chat: Optimasi Hyperparameter EURUSD H4

## Tujuan

File ini dipakai sebagai ringkasan konteks ketika diskusi dipindahkan ke chat baru, khususnya untuk topik **optimasi hyperparameter** pada penelitian tesis EURUSD H4.

## Topik Utama

- Objek riset: **EURUSD H4**
- Model utama:
  - **FLF-BiLSTM**
  - **FLF-LSTM**
- Fokus diskusi: **metodologi optimasi hyperparameter dan regularisasi**

## Kesimpulan Metodologis yang Sudah Disepakati

Metode yang dipakai **bukan**:

- Bayesian optimization
- metaheuristic optimization
- full exhaustive grid search
- nama algoritma baku tunggal yang secara eksplisit dikenal sebagai `stage-wise sequential grid search`

Metode yang **paling tepat** untuk menjelaskan prosedur penelitian adalah:

> **prosedur tuning hyperparameter berbasis grid search bertahap pada model deep learning dengan evaluasi temporal pada data time series**

Jika ingin memakai istilah Inggris, versi yang masih aman adalah:

> **a stage-wise sequential grid-search procedure for hyperparameter tuning**

Catatan penting:

- istilah `stage-wise sequential grid search` dipakai sebagai **label deskriptif internal**
- istilah itu **tidak boleh** diklaim sebagai satu algoritma baku tunggal yang punya satu paper pencetus spesifik

## Posisi Metode dalam Literatur

Secara **substantif**, penelitian ini berada pada kategori:

- **optimasi hyperparameter arsitektur dan pelatihan deep learning**

Secara **prosedural**, penelitian ini berada pada kategori:

- **simple search-based tuning**
- lebih spesifik: **grid search bertahap**

## Paper Pendukung yang Sudah Diidentifikasi

Paper yang relevan untuk mendukung penjelasan metodologi:

1. **Cohen and Aiche (2023)**
   - mendukung penggunaan **cross-validation dan grid search**

2. **Dakalbab et al. (2025)**
   - mendukung ide **modular / independent tuning**

3. **Abolmakarem et al. (2024)**
   - mendukung ide **multi-stage approach**

4. **Zhang and Pinsky (2025)**
   - mendukung evaluasi **sistematis** beberapa hyperparameter LSTM

5. **Enkhbayar and Ślepaczuk (2025)**
   - mendukung tuning pada **predefined hyperparameter space** dalam **rolling walk-forward window**

6. **Chen and Huang (2021)**
   - berguna untuk membedakan metode ini dari **Bayesian optimization**

7. Pembanding lain:
   - **Abu-Doush (2023)**
   - **Perla et al. (2023)**
   - sebagai contoh **metaheuristic / optimization-algorithm-based tuning**

## Status Dokumen yang Sudah Direvisi

### Dokumen literature/methodology

- `/home/hduser/jupyter/gust/RisetEU/bukuThesis/bahan/Ref OPTIMASI HYPERPARAMETER.md`
- `/home/hduser/jupyter/gust/RisetEU/bukuThesis/bahan/Ref OPTIMASI HYPERPARAMETER.html`
- `/home/hduser/jupyter/gust/RisetEU/bukuThesis/bahan/Ref OPTIMASI HYPERPARAMETER.docx`

### Dokumen implementasi eksperimen

- `/home/hduser/jupyter/gust/RisetEU/bukuThesis/optimasi_hyperparameter_flf_bilstm_lstm_eurusd.md`
- `/home/hduser/jupyter/gust/RisetEU/bukuThesis/optimasi_hyperparameter_flf_bilstm_lstm_eurusd.html`
- `/home/hduser/jupyter/gust/RisetEU/bukuThesis/optimasi_hyperparameter_flf_bilstm_lstm_eurusd.docx`

## Inti Revisi yang Sudah Dilakukan

1. klaim nama metode dibuat lebih aman secara akademik
2. istilah `stage-wise sequential grid search` diposisikan sebagai label deskriptif
3. dasar referensi dari paper diperjelas
4. pembedaan dengan:
   - grid search biasa
   - Bayesian optimization
   - metaheuristic optimization
   - feature-selection-based optimization
   sudah dijelaskan

## Prompt Siap Pakai untuk Chat Baru

```text
Kita lanjutkan diskusi optimasi hyperparameter tesis EURUSD H4.

Tolong gunakan file handover ini sebagai konteks utama:
/home/hduser/jupyter/gust/RisetEU/bukuThesis/handover_hyperparameter_chat.md

File acuan utama:
/home/hduser/jupyter/gust/RisetEU/bukuThesis/bahan/Ref OPTIMASI HYPERPARAMETER.md
/home/hduser/jupyter/gust/RisetEU/bukuThesis/optimasi_hyperparameter_flf_bilstm_lstm_eurusd.md

Konteks penting:
- model: FLF-BiLSTM dan FLF-LSTM
- metode yang dipakai: prosedur tuning hyperparameter berbasis grid search bertahap
- stage-wise sequential grid search hanya label deskriptif, bukan nama algoritma baku tunggal
- lanjutkan dari konteks ini, jangan mulai dari nol
```

## Catatan

Jika diskusi dilanjutkan pada chat baru, fokus berikut yang paling masuk akal adalah:

- menyelaraskan istilah ini ke bab metodologi tesis utama
- menambahkan sitasi formal ke paragraf-paragraf yang sekarang masih berupa argumen sintesis
- merapikan gaya sitasi agar sesuai format kampus/jurnal yang dipakai
