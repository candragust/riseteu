# Laporan Progres Implementasi Pipeline FLF-BiLSTM untuk EURUSD H4

Tanggal: 17 April 2026

## 1. Tujuan dokumen

Dokumen ini menyajikan uraian teknis dan hasil sementara dari pipeline **FLF-BiLSTM** yang telah diimplementasikan untuk peramalan satu langkah ke depan (`1-step ahead`) pada data **EURUSD timeframe 4H**. Dokumen ini disusun sebagai bahan laporan progres tesis dan difokuskan pada komponen yang telah tervalidasi secara eksperimental di dalam folder kerja `RisetEU`.

Secara khusus, dokumen ini membahas:

- rancangan pipeline data-to-model untuk prediksi OHLC,
- implementasi `Bidirectional LSTM` dengan `Forex Loss Function (FLF)`,
- prosedur tuning hyperparameter bertahap,
- evaluasi berbasis holdout dan walk-forward,
- ringkasan kinerja model untuk horizon uji pendek,
- artefak eksperimen yang telah dihasilkan.

Catatan pembaruan: pada tahap terbaru penelitian, baseline `ARIMA` dan jalur `FLF-LSTM` sudah berhasil dioperasionalkan pada skema komparatif `walk-forward 72 bulan train / 1 bulan test` untuk lima fold terakhir (`17-21`). Namun, dokumen ini tetap difokuskan pada jalur implementasi **FLF-BiLSTM** sebagai cabang eksperimen yang paling matang dari sisi pipeline dan artefak visual.

## 2. Ringkasan status implementasi

Pada tahap ini, implementasi yang paling matang di folder `RisetEU` adalah jalur **FLF-BiLSTM untuk EURUSD H4**. Kematangan tersebut ditunjukkan oleh ketersediaan komponen-komponen berikut:

- model BiLSTM utama yang dapat dikonfigurasi,
- loss khusus berbasis struktur candlestick (FLF),
- pipeline tuning hyperparameter 6 tahap,
- validasi temporal pada beberapa skema walk-forward,
- artefak visual untuk analisis error dan perilaku prediksi,
- model terlatih, file prediksi, dan riwayat training.

Dari sudut pandang progres tesis, cabang ini sudah layak diposisikan sebagai **hasil implementasi utama** yang dapat dipresentasikan, karena tidak lagi berhenti pada tahap desain metodologi, tetapi sudah menghasilkan keluaran numerik dan visual yang dapat dianalisis secara ilmiah.

## 3. Peta file utama implementasi

File inti yang membentuk pipeline ini adalah:

- `data_prep_eurusd.py`
- `bilstm_flf_experiment.py`
- `eurusd_pipeline_runner.py`
- `rolling_fixed_runner.py`
- `pipeline_summary_generator.py`
- `mae_atr_report.py`
- `ohlc_dot_report.py`
- `plot_loss_curves.py`

Artefak hasil eksperimen utama berada di:

- `results/eurusd_pipeline/`
- `results/validation_report_wf72_test1_last5_bilstm.html`
- `results/rolling_train72_test1_last5/`
- `results/comparison/comparison_models_wf72_test1_last5.html`
- `results/mae_atr_wf72_test1_fold21_full.html`
- `results/ohlc_dot_wf72_test1_last5_all_folds.html`
- `results/loss_wf72_test1_last5_bilstm.html`
- `results/loss_gradient_wf72_test1_last5_bilstm.html`

Artefak final model dan prediksi juga tersedia di:

- `results/finalweightmodel.h5`
- `results/final_preds.csv`
- `results/final_history.csv`

## 4. Alur pipeline FLF-BiLSTM

### 4.1. Definisi masalah dan data masukan

Masalah yang diselesaikan dalam pipeline ini adalah **multivariate sequence forecasting** untuk memprediksi `Open`, `High`, `Low`, dan `Close` candle berikutnya berdasarkan sejumlah candle historis sebelumnya. Dataset utama yang dipakai adalah:

- `EURUSD_H4_25Oct17.csv`
- hasil pembersihan: `results/EURUSD_H4_clean.csv`

Dengan demikian, setiap observasi diperlakukan sebagai vektor empat dimensi `[open, high, low, close]`, dan target model adalah vektor OHLC satu langkah ke depan.

### 4.2. Akuisisi, pembersihan, dan standardisasi data

Tahap prapemrosesan dilakukan melalui `data_prep_eurusd.py` dan alur pembacaan data di `bilstm_flf_experiment.py`. Secara teknis, tahapan ini mencakup:

- deteksi dan pemetaan kolom OHLC,
- pemaksaan tipe data numerik,
- penghapusan observasi tidak valid (`dropna`),
- penataan urutan kronologis data,
- pembentukan subset train dan test berbasis waktu.

Setelah pemisahan temporal dilakukan, fitur dan target distandardisasi menggunakan **mean** dan **standard deviation** yang diestimasi hanya dari subset train. Strategi ini penting untuk menjaga disiplin evaluasi time-series dan menghindari kebocoran informasi (`data leakage`) dari periode uji ke periode latih.

### 4.3. Konstruksi dataset supervised dengan sliding window

Pipeline membentuk pasangan input-target menggunakan mekanisme `sliding window`. Jika panjang jendela adalah `W`, maka satu sampel input terdiri atas `W` candle historis berturut-turut, sedangkan targetnya adalah candle ke-`W+1`.

Sebagai contoh, pada konfigurasi terbaik dengan `window = 12`, model menerima dua belas candle historis EURUSD H4 untuk memprediksi satu candle OHLC berikutnya. Pendekatan ini konsisten dengan literatur FLF-LSTM dan sesuai dengan rancangan eksperimen yang dijelaskan dalam proposal tesis.

### 4.4. Arsitektur FLF-BiLSTM

Model utama didefinisikan pada `bilstm_flf_experiment.py` dengan struktur konseptual sebagai berikut:

1. `InputLayer` untuk tensor berukuran `(window, feature_dim)`,
2. satu layer `Bidirectional(LSTM)` dengan jumlah unit per arah yang dapat dikonfigurasi,
3. satu layer `Dense(4)` untuk menghasilkan prediksi `open`, `high`, `low`, dan `close`.

Secara metodologis, penggunaan BiLSTM dipilih untuk menangkap representasi temporal yang lebih kaya dibanding LSTM satu arah, terutama ketika pola harga jangka pendek dan menengah saling mempengaruhi dalam satu jendela observasi.

### 4.5. Mekanisme optimisasi dan regularisasi

Training model dilakukan menggunakan optimizer `Nadam`. Pipeline juga sudah mendukung komponen regularisasi berikut:

- `dropout`,
- `recurrent_dropout`,
- `L2 regularization` (`l2_reg`),
- `EarlyStopping` berbasis `val_loss` dengan `restore_best_weights=True`.

Poin ini penting secara ilmiah karena menunjukkan bahwa pipeline tidak hanya mengejar error minimum, tetapi juga mengontrol kestabilan training dan risiko overfitting.

### 4.6. Forex Loss Function (FLF)

Komponen pembeda utama penelitian ini adalah penggunaan **Forex Loss Function (FLF)** yang diimplementasikan pada fungsi `make_flf_loss(...)`. Berbeda dari loss generik seperti `MSE`, FLF dirancang agar penalti tidak hanya didasarkan pada selisih titik harga, tetapi juga mempertimbangkan struktur candlestick secara lebih eksplisit.

Dua parameter utama pada FLF adalah:

- `lambda_coef`, yang mengontrol bobot kesalahan harga langsung,
- `sigma_coef`, yang mengontrol penalti terhadap deviasi rata-rata struktur candle.

Secara praktis, FLF diarahkan untuk menekan kesalahan prediksi sambil tetap menjaga konsistensi bentuk candlestick yang dihasilkan model. Dengan demikian, model tidak hanya dinilai “dekat” secara numerik, tetapi juga “masuk akal” secara struktural untuk konteks data forex.

### 4.7. Reproducibility

Pipeline juga telah mengakomodasi `seed` tetap untuk `Python`, `NumPy`, dan `TensorFlow`, sehingga eksperimen dapat diulang dengan konsistensi yang lebih baik. Fitur ini penting untuk kebutuhan tesis karena hasil yang dilaporkan harus dapat direproduksi secara teknis.

## 5. Pipeline tuning hyperparameter

### 5.1. Rasionalitas tuning bertahap

Tuning dilakukan oleh `eurusd_pipeline_runner.py` menggunakan **prosedur tuning hyperparameter berbasis grid search bertahap dengan evaluasi temporal pada data time series**. Strategi ini dipilih karena ruang pencarian parameter cukup besar; oleh sebab itu, pencarian dilakukan secara bertahap agar biaya eksperimen tetap terkontrol dan interpretasi hasil per tahap tetap jelas.

Dalam konteks dokumentasi internal, istilah `stage-wise sequential tuning` dapat tetap dipakai sebagai label deskriptif untuk alur eksperimen. Namun, secara metodologis istilah tersebut tidak diperlakukan sebagai nama algoritma baku tunggal. Posisi metode yang lebih tepat adalah **simple search-based tuning**, khususnya **grid search bertahap**, sehingga berbeda dari **Bayesian optimization**, **metaheuristic optimization**, maupun **full exhaustive grid search**.

Enam stage yang telah diimplementasikan adalah:

1. `window sweep`,
2. `units sweep`,
3. `activation + learning rate`,
4. `lambda / sigma sweep`,
5. `batch size sweep`,
6. `epoch sweep`.

Konfigurasi terbaik dari satu stage digunakan sebagai baseline pada stage berikutnya. Dengan pendekatan ini, pengaruh masing-masing kelompok parameter dapat diamati secara lebih terstruktur.

### 5.2. Hasil numerik tuning per stage

Berikut adalah ringkasan hasil terbaik per stage berdasarkan file `results/eurusd_pipeline/stage*_summary.csv`.

| Stage | Best config | MAE avg (pips) | MAE Open | MAE High | MAE Low | MAE Close |
|---|---|---:|---:|---:|---:|---:|
| Stage 1 | `w12 h256 relu lr5e-4 e50 lam1.0 sig0.05 b128` | 9.3712 | 2.7112 | 9.9461 | 10.5209 | 14.3065 |
| Stage 2 | `w12 h256 relu lr5e-4 e50 lam1.0 sig0.05 b128` | 9.3378 | 2.6964 | 10.0403 | 10.3747 | 14.2399 |
| Stage 3 | `w12 h256 relu lr7e-4 e50 lam1.0 sig0.05 b128` | 9.2032 | 2.6013 | 9.7416 | 10.2836 | 14.1863 |
| Stage 4 | `w12 h256 relu lr7e-4 e50 lam0.8 sig0.15 b128` | 9.0975 | 2.3853 | 9.7353 | 10.1396 | 14.1296 |
| Stage 5 | `w12 h256 relu lr7e-4 e50 lam0.8 sig0.15 b128` | 9.0889 | 2.3843 | 9.7352 | 10.1144 | 14.1217 |
| Stage 6 | `w12 h256 relu lr7e-4 e50 lam0.8 sig0.15 b128` | 9.1196 | 2.4512 | 9.7274 | 10.1843 | 14.1154 |

### 5.3. Interpretasi hasil tuning

Beberapa temuan penting dari proses tuning adalah sebagai berikut:

- `window = 12` memberikan keseimbangan terbaik antara konteks historis dan stabilitas generalisasi; window yang terlalu panjang (`16` dan `20`) justru menaikkan error.
- `units = 256` konsisten lebih baik daripada `128` dan `200`, menunjukkan bahwa kapasitas representasi yang lebih besar masih bermanfaat pada data EURUSD H4.
- kombinasi `activation = relu` dan `learning rate = 0.0007` memberikan perbaikan dibanding konfigurasi `tanh` dan learning rate yang lebih rendah.
- optimasi parameter FLF menunjukkan bahwa `lambda = 0.8` dan `sigma = 0.15` lebih efektif daripada kombinasi yang lebih dekat ke baseline awal.
- `batch = 128` lebih stabil dibanding `batch = 64`.
- pada stage akhir, `epoch = 50` tetap menjadi pilihan terbaik dibanding `30` dan `39`.

### 5.4. Konfigurasi akhir hasil pipeline

Konfigurasi terbaik hasil tuning holdout dapat dirangkum sebagai berikut:

- `window = 12`
- `units = 256`
- `activation = relu`
- `lr = 0.0007`
- `epochs = 50`
- `lambda_coef = 0.8`
- `sigma_coef = 0.15`
- `batch = 128`

Konfigurasi ini tercatat sebagai:

- `eur_s6_w12_h256_actrelu_lr0.0007_e50_lam0.8_sig0.15_b128`

## 6. Validasi model

### 6.1. Holdout sebagai tahap optimasi awal

Tahap tuning awal dijalankan menggunakan holdout berbasis waktu, yaitu 70% data awal untuk train dan 30% data akhir untuk validasi. Pada skema ini, best run menghasilkan:

- `MAE rata-rata = 9.1196 pips`
- `MAE open = 2.4512 pips`
- `MAE high = 9.7274 pips`
- `MAE low = 10.1843 pips`
- `MAE close = 14.1154 pips`

Secara metodologis, hasil holdout digunakan terutama sebagai **tahap optimasi hyperparameter**, bukan sebagai dasar tunggal untuk klaim performa final.

### 6.2. Walk-forward validation sebagai evaluasi utama

Setelah konfigurasi terbaik diperoleh, model dievaluasi kembali menggunakan validasi time-series yang lebih ketat melalui:

- `rolling_fixed_runner.py`

Pada tahap akhir penelitian, skema evaluasi yang dijadikan acuan utama adalah **walk-forward fixed 72 bulan data latih dan 1 bulan data uji**. Fokus ini dipilih agar narasi tesis tetap konsisten dengan protokol komparatif utama yang juga dipakai untuk `FLF-LSTM` dan baseline `ARIMA`.

### 6.3. Fokus pada horizon uji pendek

Untuk kebutuhan operasional prediksi forex, horizon uji yang lebih pendek lebih relevan karena model diharapkan akurat pada data terbaru, bukan semata-mata pada akumulasi performa rata-rata jangka panjang. Dalam dokumen progres ini, horizon uji yang digunakan sebagai fokus utama adalah **1 bulan**, karena horizon tersebut memberikan keseimbangan yang lebih baik antara relevansi operasional dan kecukupan jumlah sampel evaluasi pada timeframe `4H`.

### 6.4. Ringkasan error untuk 72 bulan train dengan horizon uji pendek

| Skema validasi | Fold yang dianalisis | Test samples per fold | MAE avg (pips) | Median (pips) | Best fold | Worst fold | Tail-20 MAE avg |
|---|---|---:|---:|---:|---|---|---:|
| Train 72 bln, Test 1 bln, Step 1 bln | 17-21 | 127-138 | 12.40 | 12.04 | #21 = 10.70 | #20 = 15.21 | 13.50 |

### 6.5. Interpretasi ilmiah terhadap horizon uji pendek

Pada lima fold terakhir skema `72 bulan/1 bulan`, `FLF-BiLSTM` menghasilkan rata-rata `MAE(pips) = 12.3993` dengan median `12.0381` pips. Fold terbaik adalah fold `21` dengan `10.6951` pips, sedangkan fold terburuk adalah fold `20` dengan `15.2108` pips. Variasi ini menunjukkan bahwa model masih cukup sensitif terhadap perubahan kondisi pasar terbaru, tetapi tetap mempertahankan performa yang kompetitif untuk tugas prediksi OHLC satu langkah ke depan.

Nilai `tail-20 MAE avg` gabungan sebesar sekitar `13.50` pips menunjukkan bahwa pada bagian akhir tiap segmen uji, error tidak mengalami degradasi ekstrem secara sistematis. Dengan kata lain, walaupun terdapat perbedaan performa antar fold, model masih mempertahankan kestabilan relatif pada bagian akhir horizon evaluasi 1 bulan.

### 6.6. Implikasi untuk arah eksperimen tesis

Jika orientasi tesis diarahkan pada **relevansi operasional jangka pendek**, maka skema `72 bulan train` dengan `test 1 bulan` paling tepat dijadikan fokus utama pembahasan.

Secara metodologis, pilihan ini dapat dibenarkan karena:

- data uji lebih dekat dengan penggunaan riil model,
- evaluasi lebih sensitif terhadap stabilitas model pada periode terbaru,
- jumlah sampel uji masih cukup untuk membangun perbandingan yang fair,
- hasil prediksi menjadi lebih relevan untuk konteks trading dan analisis teknikal jangka pendek.

Dengan demikian, untuk narasi tesis, horizon `1 bulan` diposisikan sebagai **skenario evaluasi utama**, sedangkan eksperimen horizon lain yang pernah dijalankan tidak lagi dijadikan pusat pembahasan hasil.

## 7. Artefak visual yang sudah tersedia

Pipeline ini sudah menghasilkan berbagai jenis visual report yang bisa langsung dipakai sebagai bahan pembahasan di laporan tesis.

### 7.1. Ringkasan tuning pipeline

File:

- `results/eurusd_pipeline/eurusd_pipeline_summary.html`

Isi:

- tabel hasil tiap stage,
- konfigurasi terbaik pada setiap tahap,
- MAE per komponen OHLC,
- durasi training.

### 7.2. Validation report

File:

- `results/validation_report_wf72_test1_last5_bilstm.html`

Isi:

- ringkasan evaluasi `walk-forward 72 bulan / 1 bulan`,
- fold schedule `17-21`,
- MAE rata-rata, median, best fold, dan worst fold.

### 7.3. MAE vs ATR report

Contoh file:

- `results/mae_atr_wf72_test1_fold21_full.html`

Isi:

- perbandingan error prediksi dengan volatilitas candle,
- analisis apakah error model masih proporsional terhadap range pasar pada fold evaluasi utama.

### 7.4. OHLC plots dan dot plots

Contoh file:

- `results/ohlc_dot_wf72_test1_last5_all_folds.html`
- `results/ohlc_dot_wf72_test1_fold21_full.html`

Isi:

- visualisasi actual vs predicted untuk OHLC pada skenario `72 bulan/1 bulan`,
- inspeksi detail perilaku model pada seluruh candle test dan gabungan lima fold utama.

### 7.5. Loss curve

File:

- `results/loss_wf72_test1_last5_bilstm.html`
- `results/loss_gradient_wf72_test1_last5_bilstm.html`

Isi:

- kurva training loss,
- kurva validation loss,
- bahan analisis stabilitas training dan indikasi overfitting/underfitting pada skenario evaluasi utama.

## 8. Penyimpanan model dan hasil prediksi

Pipeline ini juga sudah menghasilkan artefak model yang dapat dipakai kembali.

File penting:

- `results/finalweightmodel.h5`
- `results/final_preds.csv`
- `results/final_history.csv`

Makna artefak:

- model tersimpan dapat digunakan kembali untuk inferensi,
- file prediksi dapat dipakai untuk analisis lanjutan,
- file history dapat dipakai untuk evaluasi proses training.

Dengan adanya artefak ini, alur menuju inferensi candle berikutnya secara teknis sudah terbuka, karena model terlatih dan window historis terakhir dapat dipakai untuk menghasilkan prediksi 1-step berikutnya.

## 9. Keterkaitan dengan proposal tesis

Pipeline yang sudah ada ini sudah memenuhi sebagian besar porsi implementasi utama pada proposal, khususnya untuk jalur:

- `FLF-BiLSTM`
- `FLF-LSTM` pembanding pada skema komparatif utama `72 bulan/1 bulan`
- `EURUSD`
- `timeframe 4H`
- `prediksi 1 candle OHLC ahead`
- `tuning hyperparameter`
- `walk-forward validation`
- `evaluasi berbasis MAE, MAE(pips), serta awal integrasi Squared Correlation dan Directional Accuracy`

Secara substansi, bagian tesis yang sudah dapat dilaporkan berdasarkan implementasi ini adalah:

1. desain pipeline forecasting OHLC berbasis BiLSTM,
2. implementasi custom Forex Loss Function,
3. mekanisme tuning hyperparameter bertahap,
4. evaluasi pada skema holdout dan walk-forward,
5. penyusunan visual report untuk pembahasan hasil.

## 10. Batasan implementasi saat ini

Walaupun jalur FLF-BiLSTM sudah kuat, masih ada beberapa bagian proposal yang belum sepenuhnya selesai sebagai paket penelitian final:

- baseline `ARIMA` dan `FLF-LSTM` sudah operasional pada skema komparatif utama `72 bulan/1 bulan`, tetapi belum seluruhnya diperluas ke semua horizon, timeframe, dan variasi report seperti jalur BiLSTM,
- metrik `Squared Correlation (Pearson corr²)` dan `Directional Accuracy` sudah diintegrasikan pada report komparatif utama, tetapi belum menjadi evaluator yang sepenuhnya seragam di seluruh report lama,
- komparasi akhir `4H vs 1D` belum tersusun sebagai bab hasil yang final,
- fitur teknikal tambahan (`RSI`, `MACD`, `ATR`) belum menjadi cabang final utama pada pipeline EURUSD inti,
- sinkronisasi seluruh dokumen laporan/proposal terhadap hasil komparatif tiga model masih perlu dijaga agar narasinya tidak tertinggal dari implementasi terbaru.

Poin ini penting untuk disampaikan dalam laporan progres agar posisi capaian penelitian tetap jujur dan sistematis.

## 11. Kesimpulan progres

Berdasarkan implementasi di folder `RisetEU`, dapat disimpulkan bahwa penelitian sudah berhasil membangun **pipeline FLF-BiLSTM untuk EURUSD H4** yang lengkap dari sisi engineering eksperimen. Pipeline ini sudah mencakup persiapan data, sequence construction, pelatihan model BiLSTM dengan Forex Loss Function, tuning hyperparameter bertahap, validasi walk-forward, penyimpanan artefak model, serta pelaporan visual hasil eksperimen.

Pada tahap terbaru, pipeline tersebut juga sudah berhasil ditempatkan dalam pembandingan yang fair terhadap `FLF-LSTM` dan baseline statistik `ARIMA` pada skenario utama `walk-forward 72 bulan train / 1 bulan test` untuk lima fold terakhir. Hasil komparatif menunjukkan bahwa `FLF-BiLSTM` tetap lebih baik daripada `ARIMA`, walaupun pada skenario tersebut `FLF-LSTM` memberikan performa agregat terbaik. Dengan demikian, posisi `FLF-BiLSTM` dalam penelitian ini tetap kuat sebagai model pembanding arsitektural yang matang dan sahih secara eksperimental.

## 12. Rekomendasi tahap berikutnya

Urutan kerja yang paling rasional setelah capaian ini adalah:

1. merapikan satu baseline eksperimen final tiga model yang dijadikan acuan laporan tesis,
2. memperluas evaluasi ke timeframe `1D` dengan protokol yang setara,
3. menyeragamkan evaluator `Directional Accuracy` dan `Squared Correlation (Pearson corr²)` pada seluruh report utama,
4. menutup komparasi `4H vs 1D`,
5. baru setelah itu memperluas eksperimen dengan indikator teknikal tambahan.

Urutan ini akan membuat implementasi repo semakin selaras dengan struktur bab metodologi dan hasil dalam tesis.
