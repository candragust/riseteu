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

Dokumen ini belum membahas baseline `ARIMA` dan `FLF-LSTM` murni secara lengkap karena kedua jalur tersebut belum memiliki runner, validasi, dan report yang setara dengan jalur BiLSTM utama.

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
- `rolling_fixed_runner_days.py`
- `pipeline_summary_generator.py`
- `mae_atr_report.py`
- `ohlc_dot_report.py`
- `plot_loss_curves.py`

Artefak hasil eksperimen utama berada di:

- `results/eurusd_pipeline/`
- `results/validation_report.html`
- `results/rolling_fixed/`
- `results/rolling_train24_test3/`
- `results/rolling_train48_test3/`
- `results/rolling_train60_test3/`
- `results/rolling_train72_test3/`
- `results/rolling_train72_test1/`
- `results/rolling_train72_test14d_days/`

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
- `rolling_fixed_runner_days.py`

Skema validasi yang sudah dijalankan meliputi:

- Walk-Forward Fixed (Train 12 bulan, Test 6 bulan, Step 6 bulan)
- Walk-Forward Fixed (Train 24 bulan, Test 3 bulan, Step 3 bulan)
- Walk-Forward Fixed (Train 48 bulan, Test 3 bulan, Step 3 bulan)
- Walk-Forward Fixed (Train 60 bulan, Test 3 bulan, Step 3 bulan)
- Walk-Forward Fixed (Train 72 bulan, Test 3 bulan, Step 3 bulan)
- Walk-Forward Fixed (Train 72 bulan, Test 1 bulan, Step 1 bulan)
- Walk-Forward Fixed (Train 72 bulan, Test 14 hari, Step 14 hari)
- Rolling-Origin Expanding pada beberapa variasi train awal dan step

Ragam skema ini penting karena menunjukkan bahwa model telah diuji dalam konteks evaluasi temporal yang lebih realistis daripada holdout tunggal.

### 6.3. Fokus pada horizon uji pendek

Untuk kebutuhan operasional prediksi forex, horizon uji yang lebih pendek lebih relevan karena model diharapkan akurat pada data terbaru, bukan semata-mata pada akumulasi performa rata-rata jangka panjang. Dalam repo saat ini, terdapat tiga horizon uji pendek yang paling penting:

- `14 hari`
- `1 bulan`
- `3 bulan`

Semua eksperimen berikut menggunakan panjang train `72 bulan`, sehingga perbandingan antarskenario tetap berada pada basis historis yang sama.

### 6.4. Ringkasan error untuk 72 bulan train dengan horizon uji pendek

| Skema validasi | Fold yang dianalisis | Test samples per fold | MAE avg (pips) | Median (pips) | Best fold | Worst fold | Tail-20 MAE avg |
|---|---|---:|---:|---:|---|---|---:|
| Train 72 bln, Test 3 bln, Step 3 bln | 1-7 | 384-396 | 10.15 | 10.34 | #01 = 6.84 | #06 = 14.12 | 9.81 |
| Train 72 bln, Test 1 bln, Step 1 bln | 19-21 | 126-138 | 12.68 | 12.52 | #21 = 10.70 | #19 = 14.81 | 13.56 |
| Train 72 bln, Test 14 hari, Step 14 hari | 42-46 | 60-61 | 14.01 | 10.15 | #43 = 9.77 | #46 = 20.87 | 13.64 |

### 6.5. Interpretasi ilmiah terhadap horizon uji pendek

Terdapat tiga temuan penting dari perbandingan ini.

Pertama, **secara rata-rata agregat**, horizon uji `3 bulan` memberikan error terendah (`MAE avg = 10.15 pips`). Ini menunjukkan bahwa konfigurasi FLF-BiLSTM yang telah dituning masih mampu menjaga kestabilan generalisasi ketika dievaluasi pada blok data yang relatif lebih panjang.

Kedua, ketika horizon uji dipersingkat menjadi `1 bulan`, error rata-rata meningkat menjadi `12.68 pips`. Kenaikan ini mengindikasikan bahwa model menjadi lebih sensitif terhadap dinamika lokal yang lebih baru, walaupun performanya masih berada pada rentang yang kompetitif untuk prediksi OHLC.

Ketiga, pada horizon uji `14 hari`, error rata-rata agregat meningkat lagi menjadi `14.01 pips`, dan dispersi antarfold juga lebih besar, terlihat dari selisih antara fold terbaik (`9.77 pips`) dan fold terburuk (`20.87 pips`). Hal ini konsisten dengan karakter evaluasi jangka sangat pendek, di mana perubahan rezim mikro, volatilitas sesaat, dan noise lokal lebih dominan.

### 6.6. Implikasi untuk arah eksperimen tesis

Jika orientasi tesis diarahkan pada **relevansi operasional jangka pendek**, maka skema `72 bulan train` dengan `test 14 hari` atau `test 1 bulan` lebih tepat dijadikan fokus utama pembahasan, walaupun nilainya sedikit lebih buruk dibanding skema `3 bulan`.

Secara metodologis, pilihan ini dapat dibenarkan karena:

- data uji lebih dekat dengan penggunaan riil model,
- evaluasi lebih sensitif terhadap stabilitas model pada periode terbaru,
- hasil prediksi menjadi lebih relevan untuk konteks trading dan analisis teknikal jangka pendek.

Dengan demikian, untuk narasi tesis, horizon `3 bulan` dapat dipakai sebagai **benchmark stabilitas**, sedangkan horizon `14 hari` dan `1 bulan` dapat diposisikan sebagai **benchmark operasional**.

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

- `results/validation_report.html`

Isi:

- ringkasan holdout,
- ringkasan berbagai skema walk-forward dan rolling-origin,
- fold schedule,
- MAE rata-rata, median, best fold, dan worst fold.

### 7.3. MAE vs ATR report

Contoh file:

- `results/mae_atr_report.html`
- `results/mae_atr_holdout_tail30.html`
- `results/mae_atr_wf72_test14d_fold46_tail30.html`

Isi:

- perbandingan error prediksi dengan volatilitas candle,
- analisis apakah error model masih proporsional terhadap range pasar.

### 7.4. OHLC plots dan dot plots

Contoh file:

- `results/ohlc_plots.html`
- `results/ohlc_dot_holdout_full.html`
- `results/ohlc_dot_wf72_test14d_all_folds.html`

Isi:

- visualisasi actual vs predicted untuk OHLC,
- inspeksi detail perilaku model pada seluruh candle test maupun tail candle.

### 7.5. Loss curve

File:

- `results/loss_curves.html`

Isi:

- kurva training loss,
- bahan analisis stabilitas training dan indikasi overfitting/underfitting.

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
- `FLF-LSTM` pembanding untuk skema validasi horizon pendek tertentu
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

Walaupun jalur FLF-BiLSTM sudah kuat, ada beberapa bagian proposal yang belum sepenuhnya selesai dalam cabang implementasi ini:

- baseline `ARIMA` belum menjadi pipeline operasional utama,
- `FLF-LSTM` sudah tersedia untuk pembanding pada skema tertentu, tetapi belum seluruhnya diparalelkan untuk semua horizon, timeframe, dan report seperti jalur BiLSTM utama,
- metrik `Squared Correlation (Pearson corr²)` dan `Directional Accuracy` sudah mulai diintegrasikan pada report komparatif, tetapi belum menjadi evaluator utama yang konsisten di seluruh report,
- komparasi akhir `4H vs 1D` belum tersusun sebagai bab hasil yang final,
- fitur teknikal tambahan (`RSI`, `MACD`, `ATR`) belum menjadi cabang final utama pada pipeline EURUSD inti.

Poin ini penting untuk disampaikan dalam laporan progres agar posisi capaian penelitian tetap jujur dan sistematis.

## 11. Kesimpulan progres

Berdasarkan implementasi di folder `RisetEU`, dapat disimpulkan bahwa penelitian sudah berhasil membangun **pipeline FLF-BiLSTM untuk EURUSD H4** yang lengkap dari sisi engineering eksperimen. Pipeline ini sudah mencakup persiapan data, sequence construction, pelatihan model BiLSTM dengan Forex Loss Function, tuning hyperparameter bertahap, validasi walk-forward, penyimpanan artefak model, serta pelaporan visual hasil eksperimen.

Dengan demikian, untuk kebutuhan laporan progres tesis, cabang ini sudah cukup kuat untuk dipresentasikan sebagai **hasil implementasi utama yang telah berjalan**. Tahap penelitian berikutnya secara logis adalah melengkapi baseline pembanding dan evaluator tambahan agar keseluruhan desain eksperimen dalam proposal dapat terpenuhi sepenuhnya.

## 12. Rekomendasi tahap berikutnya

Urutan kerja yang paling rasional setelah capaian ini adalah:

1. merapikan satu baseline eksperimen final FLF-BiLSTM yang dijadikan acuan,
2. memperluas baseline `FLF-LSTM` agar setara pada lebih banyak skema validasi dan report,
3. menambahkan baseline `ARIMA`,
4. menyebarkan evaluator `Directional Accuracy` dan `Squared Correlation (Pearson corr²)` ke seluruh report utama,
5. menutup komparasi `4H vs 1D`,
6. baru setelah itu memperluas eksperimen dengan indikator teknikal tambahan.

Urutan ini akan membuat implementasi repo semakin selaras dengan struktur bab metodologi dan hasil dalam tesis.
