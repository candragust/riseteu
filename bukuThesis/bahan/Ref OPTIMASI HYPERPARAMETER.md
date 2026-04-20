# OPTIMASI HYPERPARAMETER

## Subbab: Optimasi Hyperparameter pada Model Prediksi Time Series Keuangan

Optimasi hyperparameter merupakan tahap yang sangat penting dalam pengembangan model prediksi time series keuangan karena performa model sangat sensitif terhadap konfigurasi arsitektur, skema pelatihan, struktur input, dan strategi evaluasi. Hal ini menjadi semakin penting pada data finansial yang bersifat nonstasioner, ber-noise tinggi, dan kerap mengalami perubahan rezim pasar. Dalam konteks ini, hyperparameter tidak dipelajari secara langsung selama training, melainkan ditentukan sebelum pelatihan dimulai. Hyperparameter tersebut dapat mencakup panjang jendela input (*lookback window*), jumlah layer dan unit, learning rate, batch size, epoch, dropout, regularisasi, horizon prediksi, dan strategi prapemrosesan seperti dekomposisi sinyal atau seleksi fitur.

Pada prediksi finansial, optimasi hyperparameter tidak semata-mata bertujuan menurunkan error seperti MAE atau RMSE, tetapi juga meningkatkan stabilitas generalisasi, menjaga robustitas model terhadap noise, dan mengurangi overfitting. Karena itu, metode optimasi hyperparameter harus dibahas secara eksplisit sebagai bagian dari metodologi penelitian, bukan dianggap sekadar detail implementasi teknis.

## 1. Pola Pendekatan Optimasi pada Literatur

Berdasarkan telaah literatur pada kumpulan paper yang tersedia dalam folder referensi, strategi optimasi hyperparameter dapat dipetakan ke dalam empat pola utama.

### (a) Metaheuristic atau optimization algorithm sebagai mesin pencari konfigurasi terbaik

Sejumlah studi menggunakan algoritma optimasi eksplisit untuk mencari konfigurasi terbaik pada ruang pencarian yang luas dan nonlinier. Abu-Doush (2023) menggunakan **archive-based Harris Hawks Optimizer** untuk meningkatkan kinerja MLP. Perla et al. (2023) juga menunjukkan pola serupa melalui pendekatan **hybrid neural network + optimization algorithm** untuk forecasting dan trend detection pada pasar forex. Pada kelompok ini, algoritma optimasi menjadi komponen utama yang mengarahkan pencarian konfigurasi.

### (b) Tuning hyperparameter pada model ML klasik setelah feature extraction

Beberapa studi menempatkan optimasi hyperparameter pada model prediktor downstream setelah tahap ekstraksi fitur oleh model deep learning. Simsek et al. (2024), misalnya, menggunakan LSTM sebagai ekstraktor fitur dan kemudian mengoptimasi model XGBoost melalui **random search**. Pola ini menunjukkan bahwa optimasi hyperparameter dapat muncul di lebih dari satu tingkat pipeline.

### (c) Seleksi variabel dan struktur input sebagai bentuk optimasi

Pada sebagian penelitian, yang dioptimasi bukan hanya arsitektur model, tetapi juga komposisi input. Xu et al. (2025) menekankan **multivariate selection + nonlinear combination**, sedangkan Abolmakarem et al. (2024) menunjukkan kerangka **multi-stage** yang memanfaatkan engineered and derivative indices. Dalam kelompok ini, optimasi diarahkan pada kualitas representasi input dan pengurangan noise.

### (d) Optimasi konfigurasi arsitektur dan pelatihan pada deep learning

Mayoritas studi deep learning pada domain finansial sangat bergantung pada keputusan hyperparameter arsitektur dan pelatihan. Harbaoui and Elhadjamor (2024) menekankan pentingnya pengaturan unit, layer, dan parameter pelatihan pada model LSTM-RNN fusion. Liang et al. (2022) menunjukkan bahwa model hybrid berbasis ICEEMDAN dan LSTM-CNN-CBAM menambah ruang hyperparameter melalui parameter dekomposisi, jaringan, dan attention. Koo and Kim (2024) juga memperlihatkan bahwa strategi dekomposisi pada LSTM memperkenalkan hyperparameter tambahan di luar konfigurasi LSTM standar. Dakalbab et al. (2025) memperluas isu ini pada model multimodal forex dengan attention, sehingga metodologi tuning perlu dijelaskan dengan jelas agar hasil replikatif.

## 2. Metode Optimasi yang Relevan untuk Penelitian Tesis

Literatur pada folder referensi juga menunjukkan bahwa metode optimasi hyperparameter dapat dikelompokkan menurut **mesin pencarinya**.

### (a) Simple search-based tuning

Kelompok ini mencakup **grid search** dan **random search**. Cohen and Aiche (2023) secara eksplisit menyebut penggunaan **cross-validation dan grid search** untuk memilih hyperparameter terbaik. Chen and Huang (2021) menjelaskan Bayesian optimization dengan cara membandingkannya terhadap random dan grid search, yang secara tidak langsung menegaskan bahwa grid search merupakan baseline tuning yang sah dan lazim digunakan.

### (b) Bayesian optimization

Chen and Huang (2021) serta Pagnottoni and Spelta (2024) menunjukkan pemakaian **Bayesian optimization** sebagai strategi yang lebih efisien dalam beberapa kasus, terutama ketika biaya evaluasi model mahal dan ruang hyperparameter cukup besar.

### (c) Metaheuristic optimization

Kelompok ini mencakup Harris Hawks Optimizer, Grey Wolf Optimization, NSGA-II, Genetic Algorithm, dan algoritma optimasi sejenis. Paper seperti Abu-Doush (2023), Xu et al. (2024), dan Sadeghi et al. (2021) menempatkan algoritma optimasi sebagai komponen sentral dalam pencarian konfigurasi terbaik.

### (d) Modular atau staged tuning

Dakalbab et al. (2025) menyatakan bahwa **setiap modality dituning secara independen**, sedangkan Abolmakarem et al. (2024) bekerja dengan kerangka **multi-stage approach**. Kedua paper ini penting karena memberikan justifikasi bahwa proses tuning dapat dilakukan secara **bertahap** atau **per-komponen**, bukan harus sekaligus pada semua parameter.

## 3. Posisi Metode yang Digunakan dalam Penelitian Ini

Berdasarkan klasifikasi di atas, metode yang digunakan pada penelitian tesis ini **tidak** termasuk:

- Bayesian optimization,
- metaheuristic optimization,
- atau feature selection sebagai mesin optimasi utama.

Metode yang digunakan paling tepat diposisikan sebagai:

**prosedur tuning hyperparameter berbasis grid search bertahap pada model deep learning dengan evaluasi temporal pada data time series**

Secara substantif, metode penelitian ini berada pada kategori:

- **optimasi hyperparameter arsitektur dan pelatihan deep learning**

Secara prosedural, metode penelitian ini berada pada kategori:

- **simple search-based tuning**, khususnya **grid search bertahap**

Penting ditekankan bahwa istilah **`stage-wise sequential grid search`** yang digunakan dalam dokumen penelitian **bukan nama algoritma baku yang dapat ditelusuri ke satu paper pencetus tunggal**, melainkan **label deskriptif** untuk prosedur yang dipakai.

Dengan demikian, formulasi yang paling aman dan tepat untuk penulisan tesis adalah:

penelitian ini menggunakan **prosedur optimasi hyperparameter berbasis grid search bertahap**, di mana satu kelompok hyperparameter divariasikan pada setiap tahap, sedangkan konfigurasi terbaik dari tahap sebelumnya digunakan sebagai baseline untuk tahap berikutnya, dengan evaluasi yang tetap mempertahankan urutan temporal data.

## 4. Mengapa Pendekatan Ini Layak Digunakan

Pendekatan ini layak dipakai dalam tesis karena beberapa alasan.

1. **Transparan secara metodologis**  
   Efek setiap kelompok hyperparameter terhadap performa model dapat diamati secara jelas.

2. **Mudah direplikasi**  
   Prosedur staged tuning lebih mudah diikuti, diuji ulang, dan diaudit dibanding strategi pencarian yang terlalu kompleks.

3. **Sesuai dengan karakter data time series finansial**  
   Evaluasi dilakukan secara kronologis sehingga tidak mencampurkan masa lalu dan masa depan.

4. **Efisien secara komputasi**  
   Pencarian penuh atas seluruh kombinasi hyperparameter akan sangat mahal dan tidak efisien untuk konteks penelitian terapan.

5. **Cukup kuat sebagai metodologi tesis**  
   Meskipun tidak menjamin optimum global, pendekatan ini sudah memadai untuk menghasilkan model yang kuat dan dapat dipertanggungjawabkan.

## 5. Keterbatasan yang Perlu Diakui

Sebagaimana metode tuning heuristik lainnya, pendekatan ini memiliki beberapa keterbatasan:

- tidak menjamin solusi global terbaik;
- sensitif terhadap baseline awal;
- sensitif terhadap urutan stage;
- tidak mengeksplorasi seluruh interaksi lintas-parameter secara penuh;
- sangat dipengaruhi oleh rancangan grid yang dipilih peneliti.

Oleh sebab itu, secara metodologis pendekatan ini lebih tepat disebut:

**structured heuristic tuning procedure**

daripada algoritma optimasi global.

## 6. Implikasi untuk Penulisan Tesis

Untuk naskah tesis, istilah yang direkomendasikan adalah:

**optimasi hyperparameter arsitektur dan pelatihan deep learning dengan pendekatan grid search bertahap pada data time series yang dipisah secara temporal**

Jika ingin memakai istilah Inggris, versi yang masih aman adalah:

**a stage-wise sequential grid-search procedure**

Namun istilah Inggris tersebut harus dijelaskan sebagai **deskripsi prosedur penelitian**, bukan sebagai nama algoritma baku yang sudah mapan di literatur.

## Kesimpulan

Literatur pada folder referensi ini memberikan dasar yang cukup kuat bahwa prosedur tuning yang dipakai dalam penelitian ini mempunyai legitimasi metodologis. Landasan itu tidak berasal dari satu paper yang menyebut frasa persis **“stage-wise sequential grid search”**, melainkan dari kombinasi beberapa garis bukti:

- penggunaan **grid search** sebagai baseline tuning yang sah;
- praktik **modular atau independent tuning** pada model atau modality tertentu;
- evaluasi hyperparameter yang dilakukan secara sistematis;
- dan penggunaan **time-series-aware validation** pada lingkungan data finansial.

Dengan demikian, metode penelitian ini dapat dipertanggungjawabkan secara akademik sebagai **prosedur optimasi hyperparameter berbasis grid search bertahap pada model deep learning untuk data time series keuangan**.
