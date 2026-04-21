# I. IMPLEMENTASI DAN PERCOBAAN / HASIL DAN PEMBAHASAN

## 1. Spesifikasi / Lingkungan Penelitian

Implementasi penelitian dilakukan pada lingkungan komputasi berbasis Python dengan fokus pada prediksi satu langkah ke depan (`1-step ahead`) untuk nilai `Open`, `High`, `Low`, dan `Close` (OHLC) pasangan mata uang EURUSD timeframe 4 jam. Pada tahap ini, eksperimen utama masih menggunakan **baseline OHLC-only**, sehingga masukan model dibatasi pada informasi candlestick historis tanpa penambahan feature engineering RSI, MACD, dan ATR. Pembatasan tersebut dilakukan agar kemampuan dasar masing-masing model dapat diamati terlebih dahulu sebelum tahap pengayaan fitur dilakukan.

Secara perangkat lunak, implementasi model deep learning menggunakan `TensorFlow/Keras`, sedangkan baseline statistik menggunakan `statsmodels`. Pengolahan data dan pelaporan hasil dibantu oleh `pandas` dan `numpy`. Jalur implementasi `FLF-BiLSTM` dikelola melalui pipeline utama pada `eurusd_pipeline_runner.py` dan `rolling_fixed_runner.py`, jalur `FLF-LSTM` dikelola melalui runner di folder `FLF_LSTM`, sedangkan baseline `ARIMA` diimplementasikan melalui `Arima/arima_ohlc_experiment.py` dan `Arima/arima_rolling_runner.py`.

Seluruh model diuji dengan pendekatan temporal agar urutan waktu tetap terjaga. Untuk pembahasan hasil utama pada bagian ini, skenario yang dipakai adalah **walk-forward fixed 72 bulan data latih dan 1 bulan data uji** pada lima fold terakhir, yaitu fold 17 sampai fold 21. Pemilihan skenario ini dimaksudkan agar perbandingan antar model dilakukan pada horizon historis dan horizon pengujian yang sama.

## 2. Implementasi Modul 1

Modul pertama yang diimplementasikan adalah baseline statistik **ARIMA** sebagai pembanding klasik terhadap model deep learning. Karena ARIMA bersifat univariat, implementasi dilakukan dengan membangun **empat model ARIMA terpisah**, masing-masing untuk komponen `Open`, `High`, `Low`, dan `Close`. Dengan desain ini, baseline tetap dapat digunakan untuk tugas prediksi candle OHLC walaupun mekanisme pemodelannya tidak simultan seperti pada model `FLF-LSTM` dan `FLF-BiLSTM`.

Pada baseline ini, pemilihan orde ARIMA tidak ditetapkan secara tetap, tetapi dioptimasi pada data latih tiap fold menggunakan pencarian grid terbatas berbasis **AIC**. Ruang kandidat yang digunakan adalah `p = {0,1,2}`, `d = {0,1}`, dan `q = {0,1,2}`. Dengan demikian, pemilihan orde dilakukan hanya pada segmen data latih, sehingga data uji 1 bulan tetap diperlakukan sebagai data yang belum terlihat oleh model pada saat seleksi struktur ARIMA.

Hasil implementasi menunjukkan bahwa orde terbaik yang terpilih cukup stabil di berbagai fold, yaitu `ARIMA(1,0,0)` untuk seri `Open` dan `Close`, serta `ARIMA(2,1,2)` untuk seri `High` dan `Low`. Konsistensi ini menunjukkan bahwa struktur dasar dinamika harga historis cukup seragam pada fold yang diuji. Pada lima fold terakhir skema `72 bulan/1 bulan`, baseline ARIMA menghasilkan rata-rata **MAE(pips) = 15.3941**, dengan performa terbaik pada fold 21 sebesar **12.4482 pips**. Nilai tersebut kemudian dijadikan acuan baseline statistik untuk menilai apakah model berbasis Forex Loss Function benar-benar memberikan peningkatan kinerja yang substantif.

## 3. Implementasi Modul 2

Modul kedua adalah implementasi model deep learning berbasis **Forex Loss Function (FLF)**, yaitu `FLF-LSTM` dan `FLF-BiLSTM`. Kedua model dibangun untuk memprediksi empat komponen harga OHLC secara langsung dari urutan candle historis. Berbeda dengan baseline ARIMA yang memodelkan masing-masing seri secara terpisah, kedua model ini mempelajari dependensi antarkomponen harga secara bersama-sama melalui representasi sekuensial.

Pada jalur `FLF-LSTM`, konfigurasi terbaik yang digunakan dalam evaluasi `walk-forward 72 bulan/1 bulan` adalah `window = 12`, `units = 256`, `activation = relu`, `learning rate = 0.0009`, `lambda = 0.8`, `sigma = 0.1`, `batch = 128`, dan `epochs = 60`. Jumlah parameter model ini adalah **268,292**. Pada jalur `FLF-BiLSTM`, konfigurasi akhir yang digunakan adalah `window = 12`, `units = 256`, `activation = relu`, `learning rate = 0.0007`, `lambda = 0.8`, `sigma = 0.15`, `batch = 128`, dan `epochs = 50`, dengan jumlah parameter **536,580**.

Dari sisi implementasi, `FLF-BiLSTM` sudah memiliki pipeline eksperimen yang paling matang karena mencakup persiapan data, pembentukan sekuens, pelatihan model, tuning hyperparameter bertahap, evaluasi `walk-forward`, serta generator laporan hasil. Jalur `FLF-LSTM` juga telah tersedia dan berhasil digunakan sebagai model pembanding yang setara pada skema `72 bulan/1 bulan`. Dengan demikian, pada tahap ini kedua model FLF sudah dapat diposisikan sebagai model utama penelitian, sedangkan ARIMA berfungsi sebagai baseline statistik pembanding.

### 3.1 Optimasi Hyperparameter Model FLF

Implementasi tuning hyperparameter dilakukan melalui runner pipeline terpisah untuk `FLF-BiLSTM` dan `FLF-LSTM`, yaitu `eurusd_pipeline_runner.py` dan `FLF_LSTM/eurusd_lstm_pipeline_runner.py`. Kedua runner tersebut mengeksekusi serangkaian stage tuning, menyimpan ringkasan `MAE(pips)` tiap kandidat pada berkas summary CSV, lalu membawa konfigurasi terbaik dari satu stage ke stage berikutnya melalui berkas `best_progression.csv`. Dengan mekanisme ini, proses seleksi dapat diaudit ulang secara langsung dari artefak eksperimen.

Pada `FLF-BiLSTM`, stage tuning meliputi `window sweep`, `units sweep`, `activation dan learning rate`, `lambda dan sigma`, `batch size`, serta `epochs`. Pada `FLF-LSTM`, struktur stage hampir sama, tetapi `activation` dikunci pada `relu` sehingga stage ketiga hanya memfokuskan pada `learning rate`. Perbedaan ini mengikuti kondisi implementasi aktif pada masing-masing runner dan menjaga tuning tetap dekat dengan baseline yang benar-benar dijalankan.

| Model | Konfigurasi akhir hasil tuning | `MAE_avg_pips` pada holdout tuning | Perbaikan dari stage awal |
|---|---|---:|---:|
| FLF-BiLSTM | `window=12, units=256, activation=relu, lr=0.0007, lambda=0.8, sigma=0.15, batch=128, epochs=50` | 9.0889 | 0.2823 pips (3.01%) |
| FLF-LSTM | `window=12, units=256, activation=relu, lr=0.0009, lambda=0.8, sigma=0.1, batch=128, epochs=60` | 8.9525 | 0.2392 pips (2.60%) |

Hasil tersebut menunjukkan bahwa kontribusi tuning tidak hanya berasal dari penambahan kapasitas jaringan, tetapi terutama dari penyesuaian dinamika optimisasi dan bobot `FLF`. Pada `FLF-BiLSTM`, perubahan paling menentukan terjadi pada tahap `activation` dan `learning rate` serta tahap `lambda` dan `sigma`. Pada `FLF-LSTM`, perbaikan terbesar muncul pada `learning rate` dan `epochs`. Konfigurasi hasil tuning inilah yang kemudian digunakan dalam evaluasi `walk-forward` lima fold terakhir pada bagian Pengujian dan Analisis.

## 4. Pengujian dan Analisis

Pengujian utama pada bagian ini difokuskan pada lima fold terakhir skema **walk-forward fixed 72 bulan data latih dan 1 bulan data uji**, yaitu fold 17 sampai fold 21. Pemilihan lima fold terakhir dilakukan untuk merepresentasikan kondisi pasar terbaru yang lebih relevan bagi penggunaan operasional pada analisis trading forex, sekaligus memastikan bahwa `ARIMA`, `FLF-LSTM`, dan `FLF-BiLSTM` dibandingkan pada horizon historis dan horizon pengujian yang sama. Dengan demikian, hasil pada bagian ini diinterpretasikan sebagai **kinerja komparatif model pada rezim pasar terbaru**, bukan sebagai satu-satunya estimasi performa model pada seluruh horizon historis. Metrik utama yang digunakan adalah **MAE(pips)**, karena metrik ini tetap proporsional terhadap MAE asli namun lebih mudah diinterpretasikan dalam konteks forex. Untuk perbandingan `FLF-LSTM` dan `FLF-BiLSTM`, analisis juga dilengkapi dengan `corr² avg HLC` dan `Directional Accuracy` body candle.

Tabel berikut merangkum hasil pengujian utama ketiga model.

| Model | Basis Implementasi | Mean MAE(pips) | Best Fold | Best-Fold MAE(pips) | Corr² Avg HLC | Directional Accuracy (%) | Jumlah Parameter |
|---|---|---:|---:|---:|---:|---:|---:|
| ARIMA | 4 model univariat + seleksi orde AIC | 15.3941 | 21 | 12.4482 | - | - | - |
| FLF-BiLSTM | Prediksi simultan OHLC berbasis FLF | 12.3993 | 21 | 10.6951 | 0.9164 | 53.3870 | 536,580 |
| FLF-LSTM | Prediksi simultan OHLC berbasis FLF | 11.3646 | 21 | 8.9525 | 0.9224 | 51.5688 | 268,292 |

Berdasarkan tabel tersebut, model `FLF-LSTM` memberikan performa agregat terbaik pada skema pengujian yang sama. Dibandingkan baseline `ARIMA`, `FLF-LSTM` menurunkan rata-rata error sebesar **4.0295 pips** atau sekitar **26.18%**. Sementara itu, `FLF-BiLSTM` juga masih lebih baik daripada `ARIMA`, dengan penurunan error sebesar **2.9948 pips** atau sekitar **19.45%**.

Jika `FLF-LSTM` dibandingkan langsung dengan `FLF-BiLSTM`, model `FLF-LSTM` masih unggul pada rata-rata error sebesar **1.0347 pips** atau sekitar **8.34%** relatif terhadap `FLF-BiLSTM`. Selain itu, `FLF-LSTM` mencapai `corr² avg HLC` yang sedikit lebih tinggi, yaitu **0.9224** dibanding **0.9164** pada `FLF-BiLSTM`. Di sisi lain, `FLF-BiLSTM` masih menunjukkan `Directional Accuracy` yang sedikit lebih baik, yaitu **53.3870%** dibanding **51.5688%** pada `FLF-LSTM`.

### 4.1 Analisis Konvergensi, Overfitting, dan Diagnostik Residual Model

Analisis tambahan dilakukan untuk memastikan bahwa perbedaan performa antar model tidak hanya dibaca dari `MAE(pips)`, tetapi juga dari perilaku pelatihan dan kualitas residual forecast. Pada `FLF-LSTM` dan `FLF-BiLSTM`, analisis dilakukan melalui kurva `loss` dan `val_loss`, sedangkan pada `ARIMA` analisis yang setara dilakukan melalui diagnostik residual, AIC/BIC, dan uji autokorelasi residual. Adapun grafik `gradient loss` pada artefak repo diperlakukan sebagai proxy perubahan loss per epoch (`dLoss`), sehingga fungsinya adalah membaca kecepatan konvergensi dan plateau, bukan gradient parameter model.

Pada `FLF-LSTM`, rata-rata epoch berjalan selama lima fold terakhir adalah **53.8**, dengan rata-rata `loss_end = 0.000485`, `val_end = 0.000547`, dan gap akhir `val_loss - loss = 0.000062`. Validation loss umumnya mencapai titik minimum pada bagian akhir training, yaitu sekitar **95.89%** dari total epoch, dan hanya menunjukkan pola overfitting ringan pada **3 dari 5 fold**. Temuan ini menunjukkan bahwa `FLF-LSTM` berhasil konvergen, tidak underfit, dan memiliki generalisasi yang relatif stabil pada skenario `walk-forward 72 bulan / 1 bulan`.

Pada `FLF-BiLSTM`, rata-rata epoch berjalan adalah **37.6**, dengan `loss_end = 0.000498`, `val_end = 0.000597`, dan gap akhir `0.000099`. Validation loss cenderung mencapai titik minimum lebih awal, yaitu sekitar **89.30%** dari total epoch, lalu pada beberapa fold meningkat kembali ketika training loss masih menurun. Pola tersebut muncul pada **3 dari 5 fold**, sehingga `FLF-BiLSTM` dapat dinyatakan tetap konvergen dan tidak underfit, tetapi lebih rentan terhadap overfitting dibanding `FLF-LSTM`.

Untuk baseline `ARIMA`, analisis tidak dilakukan melalui kurva loss berbasis epoch karena mekanisme modelnya berbeda dari neural network. Evaluasi yang lebih tepat adalah diagnostik residual forecast. Pada lima fold terakhir, `ARIMA` menghasilkan `combined close residual bias = 0.7157 pips`, `combined residual std = 22.3251 pips`, `lag-1 autocorrelation = -0.0018`, `Ljung-Box p-value lag 10 = 0.0566`, dan `lag 20 = 0.0500`. Hasil ini menunjukkan bahwa residual forecast `ARIMA` relatif stabil dan tidak memperlihatkan bukti kuat autokorelasi serial pada lag pendek, walaupun pada horizon gabungan yang lebih panjang masih terdapat indikasi lemah bahwa sebagian struktur temporal belum sepenuhnya hilang.

Secara komparatif, analisis ini mendukung hasil `MAE(pips)` sebelumnya. `FLF-LSTM` bukan hanya memberikan error agregat terendah, tetapi juga menunjukkan dinamika pelatihan yang paling sehat dan stabil. `FLF-BiLSTM` masih kompetitif dan tetap mengungguli `ARIMA`, namun generalisasinya lebih cepat jenuh. Sementara itu, `ARIMA` tetap valid sebagai baseline statistik yang stabil dan mudah diinterpretasikan, tetapi belum mampu menyamai kualitas generalisasi model FLF pada skenario evaluasi utama.

## 5. Pembahasan Hasil

Hasil implementasi menunjukkan bahwa penggunaan model berbasis **Forex Loss Function** memberikan keuntungan yang nyata dibandingkan baseline statistik klasik `ARIMA`. Hal ini terlihat dari penurunan `MAE(pips)` yang konsisten pada skema pengujian yang sama. Secara substantif, temuan ini menunjukkan bahwa pemodelan sekuensial berbasis jaringan saraf lebih mampu menangkap hubungan nonlinier antarkomponen harga dibanding pendekatan ARIMA yang memodelkan masing-masing seri secara terpisah.

Di antara dua model utama penelitian, `FLF-LSTM` menunjukkan kompromi terbaik antara akurasi dan efisiensi model. Walaupun `FLF-BiLSTM` memanfaatkan informasi dua arah pada urutan sekuens, pada horizon uji pendek 1 bulan model tersebut tidak menghasilkan rata-rata error yang lebih rendah daripada `FLF-LSTM`. Sebaliknya, `FLF-BiLSTM` membutuhkan parameter sekitar dua kali lebih besar, yaitu **536,580** dibanding **268,292** pada `FLF-LSTM`. Dengan demikian, untuk skenario evaluasi `72 bulan train / 1 bulan test`, model `FLF-LSTM` dapat dipandang sebagai model yang paling efisien sekaligus paling akurat.

Namun demikian, `FLF-BiLSTM` tetap memiliki nilai metodologis penting. Model ini masih mampu mengungguli `ARIMA` dan memperlihatkan `Directional Accuracy` yang sedikit lebih tinggi. Artinya, arsitektur dua arah tersebut masih relevan ketika fokus penelitian tidak hanya pada kecilnya error numerik, tetapi juga pada konsistensi arah pergerakan candle. Oleh karena itu, `FLF-BiLSTM` tetap layak dipertahankan sebagai model pembanding arsitektural terhadap `FLF-LSTM`.

Sementara itu, baseline `ARIMA` tetap penting sebagai pembanding karena mewakili pendekatan statistik yang lebih sederhana dan lebih mudah diinterpretasikan. Hasil implementasi menunjukkan bahwa ARIMA masih mampu menghasilkan prediksi yang stabil, terutama setelah orde model dipilih secara adaptif dengan AIC pada data latih. Akan tetapi, rata-rata error yang lebih tinggi menunjukkan bahwa baseline ini kurang mampu mengikuti kompleksitas dinamika OHLC EURUSD H4 dibanding model FLF. Dengan demikian, posisi ARIMA dalam penelitian ini lebih tepat sebagai **baseline referensi**, bukan sebagai kandidat model utama.

Secara keseluruhan, hasil implementasi dan pengujian pada bagian ini mendukung kesimpulan bahwa penelitian telah berhasil membangun tiga jalur eksperimen yang saling melengkapi, yaitu baseline statistik `ARIMA`, model utama `FLF-LSTM`, dan model pembanding `FLF-BiLSTM`. Pada tahap baseline OHLC-only, `FLF-LSTM` menjadi model dengan performa terbaik untuk skema pengujian utama, `FLF-BiLSTM` menjadi pembanding deep learning yang masih kompetitif, sedangkan `ARIMA` menjadi baseline klasik yang diperlukan untuk menunjukkan besarnya peningkatan yang diberikan oleh pendekatan berbasis Forex Loss Function.

## Catatan Pemakaian

Draft ini disusun berdasarkan artefak implementasi yang tersedia pada saat penulisan, khususnya:

1. hasil perbandingan tiga model pada `results/comparison/comparison_models_wf72_test1_last5.html`,
2. hasil tuning `FLF-LSTM` dan `FLF-BiLSTM` pada `bukuThesis/optimasi_hyperparameter_flf_bilstm_lstm_eurusd.md`,
3. baseline `ARIMA` pada `Arima/result/arima_wf72_test1_last5/`, dan
4. analisis konvergensi dan diagnostik model pada `bukuThesis/bahan/analisis_konvergensi_dan_diagnostik_model_wf72_test1_last5.md`.

Jika bagian ini akan dimasukkan ke dokumen tesis final, maka penomoran bab dan gaya heading dapat disesuaikan kembali dengan template utama Word yang digunakan.
