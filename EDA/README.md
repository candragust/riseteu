# EDA Folder

Tujuan: menyiapkan data untuk forecasting dengan mengecek kualitas data, pola dasar, dan potensi isu yang bisa merusak model.

## Isi EDA yang direkomendasikan
- Ringkasan dataset: jumlah baris/kolom, kolom target, rentang waktu, frekuensi data.
- Kualitas data: missing values, duplikasi timestamp, gap/irregular interval, outlier ekstrem.
- Konsistensi OHLC: cek High >= Open/Close/Low dan Low <= Open/Close/High.
- Statistik deskriptif: mean, std, quantile, min/max untuk OHLC dan fitur turunan.
- Distribusi: histogram/boxplot return, range, body, shadow.
- Korelasi: antar OHLC, serta autokorelasi sederhana untuk target.
- Visual time series: plot Close, rolling mean/volatility.
- Split time-based: rencana train/val/test berbasis waktu.
- Catatan transformasi: normalisasi/standarisasi, log return, atau scaling yang dipakai.

## File yang sudah ada
- `eda_eurusd_h4.md`: laporan EDA awal untuk `EURUSD_H4_25Oct17.csv`.
- `assets/`: gambar pendukung dari laporan EDA.

## Template file (opsional)
- `eda_<dataset>.md`: laporan EDA per dataset.
- `data_profile_<dataset>.csv`: ringkasan statistik deskriptif.
