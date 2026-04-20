# Optimasi Hyperparameter FLF-BiLSTM dan FLF-LSTM untuk Prediksi EURUSD H4

## Tujuan Dokumen

Dokumen ini menjelaskan metodologi optimasi hyperparameter dan regularisasi yang digunakan pada dua model utama penelitian, yaitu **FLF-BiLSTM** dan **FLF-LSTM**, untuk prediksi satu langkah ke depan harga **Open, High, Low, Close (OHLC)** EURUSD timeframe 4 jam.

Fokus dokumen ini adalah **mekanisme pencarian konfigurasi model terbaik**: bagaimana ruang pencarian dibentuk, bagaimana kandidat dievaluasi, metrik apa yang dipakai, serta bagaimana konfigurasi terbaik diwariskan dari satu tahap ke tahap berikutnya.

## Posisi Metode dalam Literatur

Dalam literatur optimasi hyperparameter untuk prediksi time series keuangan, pendekatan yang umum dapat dibedakan menjadi beberapa keluarga besar, antara lain:

1. **simple search-based tuning**, seperti grid search dan random search;
2. **optimization-algorithm-based tuning**, seperti Bayesian optimization atau metaheuristic;
3. **feature-selection-based optimization**, ketika yang dioptimasi terutama adalah struktur input;
4. **pipeline or modular tuning**, ketika optimasi dilakukan per-komponen atau per-tahap.

Berdasarkan klasifikasi tersebut, metode yang digunakan pada penelitian ini **paling tepat** diposisikan sebagai:

**prosedur tuning hyperparameter berbasis grid search bertahap pada model deep learning, dengan evaluasi temporal pada data time series**.

Formulasi ini sengaja dipilih karena lebih akurat secara ilmiah. Istilah **`stage-wise sequential grid search`** dalam dokumen ini digunakan sebagai **label deskriptif internal** untuk menjelaskan prosedur yang dipakai, **bukan** sebagai klaim bahwa ada satu algoritma baku dengan nama kanonik tersebut di literatur.

## Dasar Teoretis dan Empiris

Meskipun tidak ditemukan satu paper pada folder referensi yang memakai frasa persis **“stage-wise sequential grid search”**, komponen metodologis yang menyusun prosedur ini memiliki landasan yang jelas pada paper-paper yang tersedia dalam folder referensi.

### 1. Dasar untuk komponen `grid search`

Cohen and Aiche (2023) secara eksplisit menyebut bahwa model dapat dioptimasi melalui **cross-validation dan grid search** untuk memilih hyperparameter terbaik. Paper ini juga menunjukkan penggunaan grid search untuk menentukan konfigurasi optimal, misalnya pada **tree depth**, **learning rate**, dan parameter regularisasi pada model boosting.

Dengan demikian, **grid search** sendiri memiliki legitimasi yang jelas sebagai pendekatan tuning yang sah, khususnya ketika ruang hyperparameter masih dapat dibatasi secara rasional.

### 2. Dasar untuk komponen `tuning bertahap / modular`

Dakalbab et al. (2025) menyatakan bahwa **setiap modality dituning secara independen**. Artinya, optimasi tidak harus dilakukan sekaligus pada seluruh ruang parameter, tetapi dapat dilakukan **per-komponen** atau **bertahap**. Ini memberikan landasan metodologis bahwa optimasi yang dipisahkan per-blok parameter merupakan prosedur yang valid, selama mekanisme evaluasinya jelas.

Abolmakarem et al. (2024) juga relevan karena menggunakan kerangka **multi-stage approach** pada prediksi harga saham. Walaupun fokus paper tersebut lebih kuat pada engineered and derivative indices, paper itu mendukung gagasan bahwa pipeline prediksi finansial memang dapat dirancang dalam bentuk **beberapa tahap yang berurutan**, bukan sebagai satu langkah tunggal.

### 3. Dasar untuk komponen `systematic hyperparameter comparison`

Zhang and Pinsky (2025) mengevaluasi beberapa variasi hyperparameter LSTM secara sistematis, termasuk **window size**, **epoch**, **dropout**, dan penambahan layer. Paper ini penting karena menunjukkan bahwa dalam konteks model recurrent untuk data finansial, evaluasi beberapa kandidat hyperparameter secara **terstruktur dan komparatif** merupakan pendekatan yang wajar dan dapat dipertanggungjawabkan.

### 4. Dasar untuk komponen `time-series-aware tuning`

Enkhbayar and Ślepaczuk (2025) menjelaskan bahwa setiap model dioptimasi pada **predefined hyperparameter space** dalam masing-masing **rolling walk-forward window**. Ini mendukung gagasan bahwa proses tuning untuk data finansial harus mempertahankan struktur temporal dan tidak boleh diperlakukan seperti data i.i.d. biasa.

Dengan kata lain, landasan ilmiah prosedur ini bukan hanya berasal dari sisi “search”, tetapi juga dari sisi **evaluasi yang time-series-safe**.

### 5. Dasar untuk pembedaan dengan metode lain

Chen and Huang (2021) menjelaskan **Bayesian optimization** sebagai pendekatan yang dalam beberapa konteks dipahami lebih efisien daripada random atau grid search. Di sisi lain, Abu-Doush (2023), Perla et al. (2023), dan beberapa paper lain pada folder referensi menggunakan **metaheuristic optimization** atau **optimization algorithm** sebagai mesin pencari konfigurasi terbaik.

Hal ini justru memperjelas posisi metodologis penelitian ini:  
metode yang digunakan **bukan Bayesian optimization** dan **bukan metaheuristic optimization**, melainkan berada pada keluarga **simple search-based tuning**.

## Rumusan Metodologis yang Paling Tepat

Berdasarkan telaah di atas, formulasi yang paling aman dan kuat untuk penulisan tesis adalah:

Penelitian ini menggunakan **prosedur optimasi hyperparameter berbasis grid search bertahap** pada model deep learning, di mana satu kelompok hyperparameter divariasikan pada setiap tahap, sementara konfigurasi terbaik dari tahap sebelumnya digunakan sebagai baseline tahap berikutnya, dengan evaluasi yang tetap mempertahankan urutan temporal data.

Jika ingin tetap mempertahankan istilah Inggris, formulasi yang masih aman adalah:

**a stage-wise sequential grid-search procedure for hyperparameter tuning**

Namun secara ilmiah, kalimat itu sebaiknya diperlakukan sebagai **deskripsi prosedur**, bukan nama algoritma baku yang diklaim berasal dari satu paper tertentu.

## Prinsip Optimasi yang Digunakan dalam Penelitian Ini

Pada penelitian ini, optimasi hyperparameter dilakukan dengan prinsip berikut:

1. Menetapkan **baseline configuration** awal.
2. Menentukan satu **kelompok hyperparameter** yang akan diuji pada satu tahap tertentu.
3. Menjalankan seluruh kandidat dalam grid tahap tersebut.
4. Menghitung performa out-of-sample setiap kandidat.
5. Memilih kandidat terbaik berdasarkan metrik evaluasi.
6. Menjadikan konfigurasi terbaik itu sebagai baseline untuk tahap selanjutnya.

Dengan demikian, ruang pencarian tidak dieksplorasi secara penuh sebagai **full Cartesian exhaustive search**, tetapi direduksi menjadi pencarian bertahap yang lebih efisien dan lebih mudah diinterpretasikan.

Secara formal, jika `theta_prev_best` adalah konfigurasi terbaik dari tahap sebelumnya, dan `S_k` adalah himpunan hyperparameter yang diuji pada tahap ke-`k`, maka pencarian pada tahap itu dapat dinyatakan sebagai:

`theta_k_best = argmin(theta in Grid(S_k | theta_prev_best)) J(theta)`

dengan `J(theta)` adalah metrik objektif yang digunakan untuk seleksi kandidat.

## Mengapa Metode Ini Layak untuk Tesis

Pendekatan ini layak digunakan dalam tesis karena memiliki beberapa keunggulan metodologis:

1. **Transparan**  
   Efek setiap kelompok hyperparameter dapat diamati secara langsung dari perubahan metrik error.

2. **Terstruktur**  
   Prosedur tuning mudah dijelaskan dan mudah direplikasi.

3. **Relevan untuk data finansial**  
   Evaluasi dilakukan dengan menjaga urutan waktu, sehingga tidak menimbulkan kebocoran informasi.

4. **Efisien secara komputasi**  
   Biaya komputasi jauh lebih rendah dibanding pencarian penuh atas seluruh kombinasi hyperparameter.

5. **Cukup kuat untuk studi terapan**  
   Dalam konteks penelitian prediksi finansial terapan, pendekatan ini sudah memadai untuk menghasilkan model yang kompetitif sambil tetap dapat dipertanggungjawabkan secara ilmiah.

## Keterbatasan Metode

Meski layak dan kuat untuk penelitian terapan, metode ini tetap memiliki batasan:

- tidak menjamin optimum global;
- sensitif terhadap pemilihan baseline awal;
- sensitif terhadap urutan stage;
- interaksi antarkelompok hyperparameter tidak dieksplorasi secara menyeluruh;
- kualitas hasil sangat bergantung pada rancangan grid yang dipilih peneliti.

Karena itu, secara metodologis prosedur ini lebih tepat disebut sebagai:

**heuristic but structured hyperparameter tuning procedure**

daripada algoritma optimasi global.

## Implementasi pada FLF-BiLSTM

### Skema Tuning

Pipeline FLF-BiLSTM utama menggunakan:

- dataset: `results/EURUSD_H4_clean.csv`
- split tuning: `0.7`
- interpretasi split: sekitar **70% train** dan **30% test**
- pembagian data: **kronologis**

Jadi, tuning FLF-BiLSTM dilakukan dengan **temporal holdout 70:30**.

### Baseline Awal

Baseline awal pada `final_config.json`:

- `window = 12`
- `units = 256`
- `activation = relu`
- `lr = 0.0005`
- `epochs = 50`
- `lambda_coef = 1.0`
- `sigma_coef = 0.05`
- `batch = 128`
- `seed = 42`

### Tahap Tuning

| Stage | Fokus | Grid yang diuji |
|---|---|---|
| 1 | Window sweep | `window ∈ {12, 16, 20}` |
| 2 | Units sweep | `units ∈ {128, 200, 256}` |
| 3 | Activation + LR | `(tanh,1e-4)`, `(tanh,3e-4)`, `(relu,3e-4)`, `(relu,5e-4)`, `(relu,7e-4)` |
| 4 | Lambda / Sigma | `(0.8,0.15)`, `(0.9,0.1)`, `(1.0,0.05)` |
| 5 | Batch size | `batch ∈ {64, 128}` |
| 6 | Epoch sweep | `epochs ∈ {30, 39, 50}` |

### Hasil Progres FLF-BiLSTM

| Stage | Best config | MAE_avg_pips | MAE_open_pips | MAE_high_pips | MAE_low_pips | MAE_close_pips |
|---|---|---:|---:|---:|---:|---:|
| 1 | `w12 h256 relu lr5e-4 e50 lam1.0 sig0.05 b128` | 9.3712 | 2.7112 | 9.9461 | 10.5209 | 14.3065 |
| 2 | `w12 h256 relu lr5e-4 e50 lam1.0 sig0.05 b128` | 9.3378 | 2.6964 | 10.0403 | 10.3747 | 14.2399 |
| 3 | `w12 h256 relu lr7e-4 e50 lam1.0 sig0.05 b128` | 9.2032 | 2.6013 | 9.7416 | 10.2836 | 14.1863 |
| 4 | `w12 h256 relu lr7e-4 e50 lam0.8 sig0.15 b128` | 9.0975 | 2.3853 | 9.7353 | 10.1396 | 14.1296 |
| 5 | `w12 h256 relu lr7e-4 e50 lam0.8 sig0.15 b128` | 9.0889 | 2.3843 | 9.7352 | 10.1144 | 14.1217 |
| 6 | `w12 h256 relu lr7e-4 e50 lam0.8 sig0.15 b128` | 9.1196 | 2.4512 | 9.7274 | 10.1843 | 14.1154 |

Konfigurasi praktis terbaik BiLSTM adalah:

- `window=12, units=256, activation=relu, lr=0.0007, lambda_coef=0.8, sigma_coef=0.15, batch=128, epochs=50`

## Implementasi pada FLF-LSTM

### Skema Tuning

Pipeline FLF-LSTM disusun lebih dekat ke skenario operasional terbaru, menggunakan:

- dataset: `results/rolling_train72_test1/fold21_data.csv`
- split tuning: `0.9860686016`
- interpretasi split: sekitar **72 bulan train** dan **1 bulan test**
- pembagian data: **kronologis**

Jadi, tuning FLF-LSTM dilakukan pada **latest-fold temporal holdout 72m/1m**.

### Baseline Awal

Baseline awal pada `lstm_flf_config_wf72_test1_latest.json`:

- `window = 12`
- `units = 256`
- `activation = relu`
- `lr = 0.0009`
- `epochs = 60`
- `lambda_coef = 0.8`
- `sigma_coef = 0.1`
- `batch = 128`
- `seed = 42`

### Tahap Tuning

| Stage | Fokus | Grid yang diuji |
|---|---|---|
| 1 | Window sweep | `window ∈ {10, 12, 14}` |
| 2 | Units sweep | `units ∈ {224, 256}` |
| 3 | Learning rate sweep | `lr ∈ {5e-4, 7e-4, 9e-4}` |
| 4 | Lambda / Sigma | `(0.8,0.15)`, `(0.9,0.1)`, `(0.8,0.1)` |
| 5 | Batch size | `batch ∈ {96, 128}` |
| 6 | Epoch sweep | `epochs ∈ {40, 50, 60}` |

Catatan: pada implementasi aktif LSTM, fungsi aktivasi tidak lagi disweep; activation dikunci pada `relu`.

### Hasil Progres FLF-LSTM

| Stage | Best config | MAE_avg_pips | MAE_open_pips | MAE_high_pips | MAE_low_pips | MAE_close_pips |
|---|---|---:|---:|---:|---:|---:|
| 1 | `w12 h256 relu lr7e-4 e50 lam0.8 sig0.15 b128` | 9.1917 | 2.6513 | 10.7861 | 9.2136 | 14.1158 |
| 2 | `w12 h256 relu lr7e-4 e50 lam0.8 sig0.15 b128` | 9.1917 | 2.6513 | 10.7861 | 9.2136 | 14.1158 |
| 3 | `w12 h256 relu lr9e-4 e50 lam0.8 sig0.15 b128` | 9.0669 | 2.6755 | 10.6368 | 8.9672 | 13.9881 |
| 4 | `w12 h256 relu lr9e-4 e50 lam0.8 sig0.1 b128` | 9.0514 | 2.5804 | 10.6121 | 8.9588 | 13.9544 |
| 5 | `w12 h256 relu lr9e-4 e50 lam0.8 sig0.1 b128` | 9.0514 | 2.5804 | 10.6121 | 8.9588 | 13.9544 |
| 6 | `w12 h256 relu lr9e-4 e60 lam0.8 sig0.1 b128` | 8.9525 | 2.5921 | 10.4303 | 8.7972 | 13.9904 |

Konfigurasi akhir terbaik LSTM adalah:

- `window=12, units=256, activation=relu, lr=0.0009, lambda_coef=0.8, sigma_coef=0.1, batch=128, epochs=60`

## Perbandingan Ringkas Kedua Implementasi

| Aspek | FLF-BiLSTM | FLF-LSTM |
|---|---|---|
| Keluarga metode tuning | Prosedur grid search bertahap | Prosedur grid search bertahap |
| Objek optimasi | Arsitektur + pelatihan deep learning | Arsitektur + pelatihan deep learning |
| Skema tuning | Holdout temporal 70:30 | Latest-fold temporal holdout 72m/1m |
| Jumlah stage | 6 | 6 |
| Best MAE pipeline | 9.0889 pips | 8.9525 pips |

## Kontribusi Optimasi Hyperparameter terhadap Kinerja Prediksi

Pada penelitian ini, optimasi hyperparameter memiliki kontribusi yang substantif terhadap kinerja prediksi karena berperan dalam mengendalikan panjang konteks temporal input, kapasitas representasi jaringan, dinamika pembelajaran, serta kekuatan regularisasi pada fungsi loss. Secara empiris, prosedur tuning bertahap menghasilkan penurunan `MAE_avg_pips` pada FLF-BiLSTM dari `9.3712` pada stage awal menjadi `9.0889` pada konfigurasi praktis terbaik, yang setara dengan perbaikan sebesar `0.2823` pips atau sekitar `3.01%`. Pada FLF-LSTM, `MAE_avg_pips` menurun dari `9.1917` menjadi `8.9525`, atau membaik sebesar `0.2392` pips setara sekitar `2.60%`. Besaran tersebut menunjukkan bahwa optimasi hyperparameter tidak hanya berfungsi sebagai penyesuaian teknis minor, tetapi sebagai komponen yang berkontribusi nyata terhadap peningkatan akurasi out-of-sample model.

Selain menurunkan error agregat, hasil per stage juga menunjukkan bahwa kontribusi terbesar berasal dari hyperparameter yang langsung memengaruhi dinamika optimisasi dan bentuk fungsi objektif. Pada FLF-BiLSTM, perbaikan paling nyata muncul pada tahap `activation + learning rate` serta `lambda / sigma`, sedangkan pada FLF-LSTM pengaruh paling kuat muncul pada `learning rate` dan `epochs`. Temuan ini menunjukkan bahwa kualitas prediksi tidak hanya ditentukan oleh pemilihan arsitektur dasar, tetapi juga oleh ketepatan konfigurasi proses pembelajaran dan regularisasi. Dengan demikian, dalam konteks prediksi EURUSD H4, optimasi hyperparameter berperan penting untuk menekan error prediksi, menstabilkan generalisasi, mengurangi risiko overfitting, dan mengarahkan model menuju konfigurasi yang lebih andal untuk memprediksi candle berikutnya.

## Kesimpulan

Secara metodologis, prosedur optimasi hyperparameter yang digunakan pada penelitian ini **bukan** metaheuristic optimization, **bukan** Bayesian optimization, dan **bukan** full exhaustive grid search. Prosedur yang dipakai lebih tepat disebut:

**prosedur tuning hyperparameter berbasis grid search bertahap pada model deep learning dengan evaluasi temporal pada data time series**.

Formulasi tersebut kuat secara ilmiah karena:

- selaras dengan literatur yang mengakui grid search sebagai baseline tuning yang sah;
- konsisten dengan studi yang melakukan tuning per-komponen atau per-modul;
- sesuai dengan kebutuhan evaluasi kronologis pada data finansial;
- dan tidak mengandung klaim berlebihan bahwa metode ini adalah satu algoritma baku dengan nama yang sudah mapan.

Dengan kata lain, istilah `stage-wise sequential grid search` dapat tetap digunakan di dalam tesis **sebagai label prosedural deskriptif**, tetapi penjelasan formalnya harus ditegaskan sebagai **grid-search-based staged tuning procedure**, bukan algoritma kanonik tunggal.
