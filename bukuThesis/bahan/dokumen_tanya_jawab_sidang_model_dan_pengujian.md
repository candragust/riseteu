# DOKUMEN TANYA-JAWAB SIDANG

## Prediksi EUR/USD dengan ARIMA, FLF-LSTM, dan FLF-BiLSTM

Dokumen ini disusun untuk membantu menjawab pertanyaan kritis dosen pembimbing dan dosen penguji terkait desain metodologi, optimasi hyperparameter, baseline ARIMA, feature engineering, dan skenario pengujian pada penelitian prediksi candlestick EUR/USD.

Struktur jawaban pada setiap butir terdiri atas:

- **Jawaban akademik**: versi formal yang aman untuk naskah ilmiah atau jawaban panjang saat sidang.
- **Jawaban singkat lisan**: versi ringkas yang dapat disampaikan secara langsung.
- **Status bukti**:
  - **Kuat**: sudah didukung implementasi dan hasil yang tersedia.
  - **Cukup dengan batas klaim**: dapat dipertahankan selama klaim dibatasi dengan jujur.
  - **Perlu penguatan lanjutan**: sebaiknya diikuti eksperimen tambahan atau dinyatakan sebagai keterbatasan.

## A. Klasifikasi Desain Penelitian

### 1. Apa perbedaan antara feature engineering, hyperparameter tuning, periode testing, dan multi-timeframe?

**Jawaban akademik**

Keempat komponen tersebut memiliki fungsi metodologis yang berbeda. **Feature engineering** berkaitan dengan desain fitur input yang diberikan ke model, misalnya baseline `OHLC-only` atau skenario lanjutan `OHLC + RSI + MACD + ATR`. **Hyperparameter tuning** berkaitan dengan penyesuaian parameter model dan proses training, seperti `window`, `units`, `learning rate`, `epochs`, `batch`, `lambda`, dan `sigma`. **Periode testing** adalah bagian dari protokol evaluasi temporal, yaitu berapa panjang horizon data uji pada skema walk-forward, yang pada penelitian ini ditetapkan 1 bulan. **Multi-timeframe** adalah faktor desain data, misalnya penggunaan data 4H dan 1D, yang memengaruhi karakteristik observasi tetapi bukan bagian dari hyperparameter model.

**Jawaban singkat lisan**

Feature engineering itu soal fitur input, hyperparameter tuning itu soal parameter model dan training, periode testing itu protokol evaluasi, sedangkan multi-timeframe itu desain data.

**Status bukti**: **Kuat**

### 2. Mengapa penelitian dimulai dari baseline OHLC-only?

**Jawaban akademik**

Baseline `OHLC-only` dipilih agar kemampuan dasar model dalam mempelajari struktur candlestick dapat diukur terlebih dahulu tanpa campur tangan informasi turunan dari indikator teknikal. Dengan baseline ini, kontribusi model dan fungsi loss dapat diamati secara lebih murni. Setelah baseline mapan, penambahan RSI, MACD, dan ATR dapat dievaluasi sebagai kontribusi tambahan dari feature engineering, bukan sebagai faktor yang sejak awal bercampur dengan kontribusi arsitektur model dan optimasi training.

**Jawaban singkat lisan**

Saya mulai dari `OHLC-only` supaya baseline-nya bersih. Dengan begitu, kalau nanti fitur tambahan dipasang, peningkatannya bisa diatribusikan dengan lebih jelas.

**Status bukti**: **Kuat**

### 3. Mengapa RSI, MACD, dan ATR tidak langsung dijadikan input utama sejak awal?

**Jawaban akademik**

Jika indikator teknikal langsung digunakan sejak tahap awal, maka akan sulit membedakan apakah peningkatan performa berasal dari arsitektur model, fungsi loss, atau dari informasi tambahan yang dibawa fitur turunan. Oleh karena itu, penelitian disusun bertahap: tahap pertama membangun baseline `OHLC-only`, kemudian tahap lanjutan menambahkan `RSI + MACD + ATR` sebagai feature engineering. Strategi ini lebih defensif secara metodologis karena menjaga pemisahan kontribusi antar komponen penelitian.

**Jawaban singkat lisan**

Kalau indikator langsung dimasukkan dari awal, sumber peningkatan performa jadi bercampur. Karena itu saya pisahkan baseline dulu, baru feature engineering.

**Status bukti**: **Kuat**

## B. Optimasi Hyperparameter Model FLF

### 4. Hyperparameter apa saja yang benar-benar dituning pada FLF-LSTM dan FLF-BiLSTM?

**Jawaban akademik**

Pada implementasi aktif, hyperparameter utama yang dituning meliputi `window`, `units`, `activation` atau `learning rate` sesuai jalur model, `lambda_coef`, `sigma_coef`, `batch`, dan `epochs`. Pada jalur `FLF-BiLSTM`, stage tuning mencakup `window sweep`, `units sweep`, `activation + learning rate`, `lambda / sigma`, `batch size`, dan `epochs`. Pada jalur `FLF-LSTM`, struktur stage hampir sama, tetapi `activation` dikunci pada `relu`, sehingga stage ketiga berfokus pada `learning rate`. Dengan demikian, tuning berfokus pada konteks temporal input, kapasitas representasi, dinamika optimisasi, dan pembobotan pada Forex Loss Function.

**Jawaban singkat lisan**

Yang saya tuning adalah `window`, `units`, `learning rate`, `activation` pada BiLSTM, `lambda`, `sigma`, `batch`, dan `epochs`.

**Status bukti**: **Kuat**

### 5. Mengapa tuning dilakukan dengan prosedur grid search bertahap, bukan full grid search atau random search?

**Jawaban akademik**

Penelitian ini menggunakan **stage-wise sequential grid search** agar proses seleksi tetap transparan, dapat diaudit, dan sesuai dengan keterbatasan komputasi. Pada setiap tahap, satu kelompok hyperparameter diuji sementara hyperparameter lain dipertahankan pada konfigurasi terbaik sementara dari tahap sebelumnya. Jika seluruh kombinasi diuji sekaligus melalui full grid search, ruang eksperimen menjadi sangat besar dan tidak efisien untuk penelitian ini. Sementara itu, random search memang efisien, tetapi kurang mudah dijelaskan dalam bentuk tahapan eksplisit yang dapat dipertanggungjawabkan secara pedagogis pada tesis. Karena itu, prosedur bertahap dipilih sebagai kompromi antara keterlacakan metodologis dan efisiensi komputasi.

**Jawaban singkat lisan**

Saya pilih grid search bertahap karena lebih transparan, lebih ringan secara komputasi, dan lebih mudah diaudit daripada full grid search.

**Status bukti**: **Cukup dengan batas klaim**

### 6. Apakah prosedur tuning ini membuktikan konfigurasi global terbaik?

**Jawaban akademik**

Tidak. Prosedur tuning ini tidak dimaksudkan untuk membuktikan bahwa konfigurasi akhir merupakan **optimum global** dari seluruh ruang kemungkinan hyperparameter. Klaim yang sah adalah bahwa penelitian berhasil memperoleh konfigurasi yang **lebih baik secara sistematis** dalam ruang kandidat yang didefinisikan. Batas klaim ini penting agar interpretasi hasil tetap jujur secara metodologis.

**Jawaban singkat lisan**

Tidak, saya tidak mengklaim optimum global. Klaim saya adalah konfigurasi terbaik dalam ruang kandidat yang saya definisikan.

**Status bukti**: **Kuat**

### 7. Apa dasar pemilihan ruang kandidat hyperparameter?

**Jawaban akademik**

Ruang kandidat dipilih secara terbatas di sekitar konfigurasi yang masih masuk akal untuk data EUR/USD H4 dan ukuran model yang dapat dijalankan secara stabil pada lingkungan komputasi penelitian. Sebagai contoh, `window` dipilih pada rentang pendek-menengah agar model tetap fokus pada konteks temporal terbaru, `units` dipilih pada kapasitas yang masih realistis untuk pelatihan, dan `lambda` serta `sigma` dipilih untuk mengeksplorasi keseimbangan antara error harga dan struktur candlestick pada Forex Loss Function. Dengan demikian, ruang pencarian tidak dimaksudkan sebagai enumerasi seluruh kemungkinan, tetapi sebagai ruang kandidat yang terarah dan dapat dipertanggungjawabkan.

**Jawaban singkat lisan**

Grid saya pilih secara terarah, bukan sembarang, yaitu pada rentang yang masih masuk akal untuk konteks EUR/USD H4 dan beban komputasi penelitian.

**Status bukti**: **Cukup dengan batas klaim**

### 8. Mengapa `recurrent_activation`, `dropout`, `recurrent_dropout`, dan `l2_reg` tidak ikut dituning?

**Jawaban akademik**

Parameter `recurrent_activation`, `dropout`, `recurrent_dropout`, dan `l2_reg` tersedia pada implementasi, tetapi tidak dimasukkan ke ruang pencarian utama. Pertimbangannya adalah agar tuning difokuskan terlebih dahulu pada komponen yang paling langsung memengaruhi kualitas prediksi pada eksperimen ini, yaitu panjang konteks input, kapasitas representasi, dinamika optimisasi, dan pembobotan FLF. Selain itu, penambahan parameter tersebut ke ruang pencarian akan memperbesar dimensi tuning secara signifikan. Oleh karena itu, parameter-parameter tersebut dipertahankan pada nilai default implementasi dan dapat diposisikan sebagai ruang eksplorasi lanjutan, bukan fokus utama tahap progress ini.

**Jawaban singkat lisan**

Parameter itu ada di kode, tetapi belum saya jadikan ruang pencarian utama supaya tuning tetap fokus dan tidak meledakkan kompleksitas eksperimen.

**Status bukti**: **Cukup dengan batas klaim**

### 9. Apa metrik seleksi utama pada tuning, dan mengapa dipilih MAE(pips)?

**Jawaban akademik**

Metrik seleksi utama antar kandidat pada tahap tuning adalah `MAE(pips)`. Metrik ini dipilih karena tetap proporsional terhadap MAE asli dalam satuan harga, namun lebih mudah diinterpretasikan pada konteks forex. Dengan menggunakan satuan pips, besarnya error dapat langsung dibaca terhadap ukuran gerak harga yang lazim dipakai dalam praktik trading. Penggunaan `MAE(pips)` juga konsisten dengan metrik komparatif utama yang dipakai pada evaluasi akhir.

**Jawaban singkat lisan**

Saya pakai `MAE(pips)` karena tetap setara secara proporsional dengan MAE asli, tetapi jauh lebih interpretatif untuk forex.

**Status bukti**: **Kuat**

### 10. Apa bukti empiris bahwa tuning benar-benar berkontribusi terhadap performa model?

**Jawaban akademik**

Pada `FLF-BiLSTM`, nilai `MAE_avg_pips` turun dari `9.3712` pada stage awal menjadi `9.0889` pada konfigurasi praktis terbaik, atau membaik sebesar `0.2823` pips (`3.01%`). Pada `FLF-LSTM`, `MAE_avg_pips` turun dari `9.1917` menjadi `8.9525`, atau membaik sebesar `0.2392` pips (`2.60%`). Walaupun persentasenya tidak sangat besar, penurunan ini konsisten dan relevan karena dicapai pada setting out-of-sample temporal holdout yang sama. Dengan demikian, tuning berperan sebagai komponen yang memberi kontribusi nyata, bukan sekadar penyesuaian teknis minor.

**Jawaban singkat lisan**

Secara empiris tuning menurunkan error sekitar `3.01%` pada BiLSTM dan `2.60%` pada LSTM, jadi kontribusinya nyata.

**Status bukti**: **Kuat**

### 11. Mengapa stage tuning LSTM dan BiLSTM tidak identik?

**Jawaban akademik**

Tahap tuning disusun mengikuti kondisi implementasi aktif masing-masing model. Pada `FLF-BiLSTM`, fungsi aktivasi masih dieksplorasi bersama learning rate, sedangkan pada `FLF-LSTM`, fungsi aktivasi sudah dikunci pada `relu` dan stage ketiga hanya memfokuskan pada learning rate. Perbedaan ini bukan inkonsistensi metodologis, melainkan konsekuensi dari baseline konfigurasi yang memang berbeda pada kedua pipeline. Yang tetap dijaga adalah logika seleksi bertahap, pembagian data temporal, dan metrik evaluasi yang konsisten.

**Jawaban singkat lisan**

Stage-nya tidak identik karena baseline implementasi kedua runner memang berbeda, tetapi logika tuning dan metrik seleksinya tetap sama.

**Status bukti**: **Kuat**

### 12. Mengapa `seed = 42` dipakai, dan apakah hasilnya stabil?

**Jawaban akademik**

Seed tetap digunakan agar eksperimen dapat direproduksi dengan kondisi inisialisasi yang sama. Dalam implementasi saat ini, `seed = 42` mengontrol komponen acak pada Python, NumPy, dan TensorFlow. Namun demikian, penggunaan satu seed belum cukup untuk mengklaim stabilitas penuh hasil terhadap variasi inisialisasi. Oleh karena itu, jawaban yang aman adalah bahwa seed tetap diperlukan untuk reproduktibilitas, sedangkan uji multi-seed merupakan penguatan lanjutan yang belum menjadi fokus tahap progress ini.

**Jawaban singkat lisan**

`Seed = 42` saya pakai untuk reproduktibilitas. Untuk stabilitas lintas seed, itu masih ruang penguatan lanjutan.

**Status bukti**: **Perlu penguatan lanjutan**

## C. Desain Evaluasi Walk-Forward

### 13. Mengapa skenario evaluasi utama memakai 72 bulan data latih dan 1 bulan data uji?

**Jawaban akademik**

Skema `72 bulan train / 1 bulan test` dipilih sebagai **parameter desain evaluasi**, bukan hyperparameter model. Pertimbangannya adalah menyediakan jumlah sampel latih yang cukup besar bagi model deep learning, terutama jika penelitian diperluas ke timeframe `1D`, sekaligus menjaga relevansi operasional horizon uji pada pasar forex. Horizon uji 1 bulan juga memudahkan pembandingan lintas model dan tetap cukup panjang untuk menghasilkan sampel evaluasi yang bermakna pada data 4H. Dengan demikian, pilihan `72/1` merupakan keputusan metodologis yang rasional, bukan klaim bahwa kombinasi tersebut sudah pasti paling optimal.

**Jawaban singkat lisan**

`72/1` saya pakai sebagai desain evaluasi yang rasional: train cukup panjang, test tetap relevan secara operasional, dan fair untuk semua model.

**Status bukti**: **Cukup dengan batas klaim**

### 14. Apakah 72 bulan dibuktikan lebih baik daripada 40 bulan?

**Jawaban akademik**

Tidak. Penelitian ini tidak mengklaim bahwa `72 bulan` telah dibuktikan lebih baik daripada `40 bulan` atau horizon train lainnya. Posisi metodologis yang tepat adalah bahwa `72 bulan` dipilih sebagai skenario evaluasi utama yang dinilai rasional berdasarkan kecukupan sampel dan konsistensi lintas eksperimen. Jika peneliti ingin mengklaim bahwa `72 bulan` adalah horizon terbaik, maka diperlukan eksperimen sensitivitas tambahan terhadap beberapa panjang jendela train.

**Jawaban singkat lisan**

Saya tidak mengklaim `72 bulan` paling optimal. Saya mengklaim itu sebagai desain evaluasi utama yang rasional.

**Status bukti**: **Kuat**

### 15. Mengapa periode testing dipilih 1 bulan, bukan 14 hari?

**Jawaban akademik**

Periode testing 1 bulan dipilih agar horizon evaluasi tetap konsisten dan menghasilkan jumlah sampel yang lebih memadai, terutama jika penelitian dibandingkan lintas timeframe seperti `4H` dan `1D`. Horizon uji yang terlalu pendek memang bisa terasa lebih dekat dengan kondisi pasar terbaru, tetapi juga cenderung menghasilkan estimasi error yang lebih volatil karena jumlah sampelnya sedikit. Oleh karena itu, pada penelitian ini 1 bulan dipilih sebagai kompromi yang lebih stabil dan lebih mudah dibela secara metodologis.

**Jawaban singkat lisan**

Saya pilih 1 bulan karena lebih stabil dan lebih fair untuk evaluasi, terutama jika nanti dibandingkan lintas timeframe.

**Status bukti**: **Kuat**

### 16. Mengapa hasil utama hanya memakai lima fold terakhir?

**Jawaban akademik**

Lima fold terakhir, yaitu fold `17–21`, dipilih untuk merepresentasikan **rezim pasar terbaru** yang lebih relevan bagi penggunaan operasional pada analisis trading forex. Pemilihan ini dapat dibenarkan selama ketiga model dibandingkan pada fold yang sama, horizon train-test yang sama, dan hasilnya diinterpretasikan sebagai performa pada rezim pasar terbaru, bukan sebagai satu-satunya estimasi performa historis penuh. Dengan kata lain, fokus pada lima fold terakhir sah secara metodologis selama batas interpretasinya dinyatakan secara eksplisit.

**Jawaban singkat lisan**

Lima fold terakhir saya pakai untuk merepresentasikan kondisi pasar terbaru, bukan untuk mengklaim seluruh histori sudah terwakili.

**Status bukti**: **Cukup dengan batas klaim**

### 17. Apakah fold terakhir benar-benar memuat 1 bulan penuh?

**Jawaban akademik**

Ya. Fold terakhir dibentuk sebagai horizon uji satu bulan kalender trading dengan batas awal inklusif dan batas akhir eksklusif. Pada implementasi aktif, rentang uji fold terakhir dimulai pada `2025-09-02` dan berjalan sampai batas eksklusif awal Oktober 2025, dengan observasi trading terakhir pada `2025-10-01 20:00`. Karena data forex 4H tidak memuat periode non-trading seperti akhir pekan, jumlah candle tidak sama dengan seluruh jam kalender, tetapi secara konstruksi fold tersebut tetap merepresentasikan 1 bulan penuh data trading.

**Jawaban singkat lisan**

Ya, secara konstruksi itu 1 bulan penuh data trading, walaupun tentu tidak memuat jam non-trading seperti weekend.

**Status bukti**: **Kuat**

### 18. Apakah ada potensi tumpang tindih antara validation dan test pada pipeline FLF?

**Jawaban akademik**

Pada implementasi progress saat ini, segmen holdout temporal pada fold yang sama digunakan sebagai `validation_data` untuk `EarlyStopping` sekaligus sebagai data evaluasi prediksi. Artinya, pemisahan antara validasi untuk model selection dan pengujian akhir belum sepenuhnya dipisahkan seperti pada skema `train-validation-test` yang lebih ketat. Oleh karena itu, jawaban yang aman adalah bahwa hasil saat ini dapat dipakai sebagai **evaluasi komparatif operasional dengan protokol yang sama antar model**, tetapi belum sekuat skema evaluasi final yang sepenuhnya bebas dari model selection overlap. Perbaikan lanjutan yang disarankan adalah memisahkan train, validation, dan test secara eksplisit atau menggunakan nested walk-forward.

**Jawaban singkat lisan**

Ada catatan metodologis di sini: validation untuk early stopping dan evaluasi akhir masih berada pada fold holdout yang sama. Jadi hasilnya fair antar model, tetapi untuk evaluasi final masih idealnya perlu pemisahan yang lebih ketat.

**Status bukti**: **Perlu penguatan lanjutan**

## D. Baseline ARIMA

### 19. Mengapa ARIMA dipakai sebagai baseline?

**Jawaban akademik**

ARIMA dipakai sebagai baseline karena mewakili pendekatan statistik time series klasik yang kuat, mapan, dan mudah diinterpretasikan. Dengan menghadirkan ARIMA, penelitian tidak hanya membandingkan dua arsitektur deep learning, tetapi juga menunjukkan apakah pendekatan berbasis Forex Loss Function benar-benar memberikan peningkatan di atas baseline statistik yang lebih sederhana. Kehadiran ARIMA membuat kontribusi penelitian lebih meyakinkan secara komparatif.

**Jawaban singkat lisan**

ARIMA saya pakai sebagai baseline statistik klasik agar peningkatan model deep learning bisa diukur terhadap pembanding yang mapan.

**Status bukti**: **Kuat**

### 20. Mengapa ARIMA dibuat menjadi empat model univariat, bukan satu model multivariat?

**Jawaban akademik**

ARIMA standar pada dasarnya adalah model **univariat**, sehingga satu model hanya memprediksi satu seri waktu. Karena target penelitian adalah `OHLC`, maka baseline ARIMA diimplementasikan sebagai empat model terpisah untuk `Open`, `High`, `Low`, dan `Close`. Pendekatan ini paling aman dan paling sesuai dengan definisi model ARIMA yang diminta sebagai baseline. Jika baseline statistik multivariat ingin digunakan, model yang lebih tepat justru adalah `VAR`, bukan ARIMA univariat.

**Jawaban singkat lisan**

Karena ARIMA standar itu univariat, maka untuk OHLC saya bangun empat model terpisah.

**Status bukti**: **Kuat**

### 21. Apa arti `p`, `d`, `q` pada ARIMA?

**Jawaban akademik**

Dalam notasi `ARIMA(p,d,q)`, parameter `p` menyatakan orde **autoregressive**, yaitu banyaknya lag nilai masa lalu yang digunakan untuk memprediksi nilai saat ini. Parameter `d` menyatakan orde **differencing**, yaitu jumlah pembedaan yang diterapkan untuk mengurangi non-stasioneritas seri. Parameter `q` menyatakan orde **moving average**, yaitu banyaknya lag residual masa lalu yang digunakan dalam model. Kombinasi ketiganya menentukan struktur dasar ARIMA dalam menangkap ketergantungan temporal dan pola error historis.

**Jawaban singkat lisan**

`p` adalah autoregressive, `d` differencing, dan `q` moving average.

**Status bukti**: **Kuat**

### 22. Apa fungsi AIC dan BIC pada baseline ARIMA?

**Jawaban akademik**

`AIC` dan `BIC` adalah kriteria informasi untuk memilih model yang menyeimbangkan kualitas kecocokan terhadap data dan kompleksitas model. Nilai yang lebih kecil menunjukkan model yang lebih baik setelah penalti kompleksitas diperhitungkan. Pada penelitian ini, `AIC` dipakai sebagai kriteria utama seleksi orde ARIMA pada data latih karena lebih sesuai untuk pencarian kandidat model, sedangkan `BIC` dicatat sebagai ukuran pendukung untuk menilai kecenderungan kompleksitas model yang terpilih.

**Jawaban singkat lisan**

`AIC` dan `BIC` dipakai untuk memilih orde ARIMA yang paling baik dengan tetap memberi penalti pada model yang terlalu kompleks.

**Status bukti**: **Kuat**

### 23. Bagaimana optimasi ARIMA dilakukan agar fair dengan model FLF?

**Jawaban akademik**

Pada setiap fold walk-forward, orde `ARIMA(p,d,q)` dipilih di **data latih saja** melalui grid terbatas `p={0,1,2}`, `d={0,1}`, dan `q={0,1,2}` menggunakan `AIC`. Setelah orde terbaik dipilih, model digunakan untuk menghasilkan prediksi pada horizon uji 1 bulan. Dengan demikian, baseline ARIMA mengikuti protokol train-test yang sama secara prinsip, yaitu pelatihan pada segmen historis dan evaluasi pada segmen holdout temporal berikutnya.

**Jawaban singkat lisan**

Setiap fold ARIMA di-fit pada 72 bulan train, orde dipilih dengan `AIC` di train set, lalu diuji pada 1 bulan test yang sama seperti model FLF.

**Status bukti**: **Kuat**

### 24. Apakah pembandingan ARIMA dengan FLF-LSTM dan FLF-BiLSTM fair?

**Jawaban akademik**

Pembandingan dapat dinyatakan fair dalam arti **protokol evaluasinya sama**, yaitu sama-sama menggunakan skema `72 bulan train / 1 bulan test`, fold yang sama, horizon prediksi satu langkah ke depan, dan metrik utama `MAE(pips)`. Namun demikian, fairness ini tidak berarti ketiga model identik dari sisi mekanisme pemodelan. ARIMA tetap merupakan baseline statistik univariat, sedangkan FLF-LSTM dan FLF-BiLSTM adalah model deep learning multivariat yang memprediksi OHLC secara simultan. Oleh karena itu, perbandingan ini lebih tepat dipahami sebagai perbandingan **baseline klasik vs model utama penelitian** dalam protokol evaluasi yang sama.

**Jawaban singkat lisan**

Fair dari sisi protokol evaluasi, bukan karena mekanismenya identik. ARIMA tetap baseline klasik, sedangkan FLF adalah model utama penelitian.

**Status bukti**: **Kuat**

## E. Multi-Timeframe dan Feature Engineering Lanjutan

### 25. Bagaimana posisi timeframe 4H dan 1D dalam penelitian ini?

**Jawaban akademik**

Timeframe `4H` dan `1D` diposisikan sebagai **faktor desain data**, bukan hyperparameter model. Pada tahap progress saat ini, eksperimen komparatif utama yang telah dijalankan secara penuh difokuskan pada `4H`. Timeframe `1D` tetap dipertahankan sebagai skenario perluasan penelitian agar konsistensi model pada granularitas waktu yang lebih panjang dapat diuji pada tahap lanjutan.

**Jawaban singkat lisan**

`4H` saat ini adalah fokus implementasi utama, sedangkan `1D` saya tempatkan sebagai perluasan eksperimen berikutnya.

**Status bukti**: **Kuat**

### 26. Jika nanti RSI, MACD, dan ATR ditambahkan sebagai fitur, apakah hyperparameter harus dituning ulang?

**Jawaban akademik**

Secara metodologis, ya. Ketika ruang input berubah dari `OHLC-only` menjadi `OHLC + RSI + MACD + ATR`, distribusi informasi yang masuk ke model juga berubah. Oleh karena itu, konfigurasi hyperparameter yang optimal pada baseline belum tentu tetap optimal pada skenario feature engineering. Untuk perbandingan yang fair, setidaknya perlu dilakukan retuning terbatas atau dinyatakan secara eksplisit bahwa eksperimen feature engineering menggunakan konfigurasi yang diwariskan dari baseline.

**Jawaban singkat lisan**

Idealnya harus dituning ulang, karena saat fitur input berubah, konfigurasi terbaik model juga bisa ikut berubah.

**Status bukti**: **Kuat**

## F. Klaim Ilmiah, Kontribusi, dan Keterbatasan

### 27. Apa kontribusi utama penelitian yang paling aman untuk diklaim?

**Jawaban akademik**

Kontribusi utama yang aman adalah bahwa penelitian ini membangun dan membandingkan tiga jalur eksperimen yang saling melengkapi, yaitu baseline statistik `ARIMA`, model utama `FLF-LSTM`, dan model pembanding `FLF-BiLSTM`, pada protokol evaluasi temporal yang konsisten. Selain itu, penelitian menunjukkan bahwa optimasi hyperparameter bertahap memberi penurunan error yang nyata pada baseline `OHLC-only`, serta bahwa model berbasis Forex Loss Function mampu mengungguli baseline ARIMA pada rezim pasar terbaru yang diuji. Klaim ini lebih defensif daripada menyatakan bahwa penelitian telah menemukan model terbaik mutlak untuk seluruh kondisi pasar.

**Jawaban singkat lisan**

Kontribusi utama saya adalah membangun baseline statistik dan dua model FLF dalam protokol evaluasi yang konsisten, lalu menunjukkan bahwa model FLF memberi peningkatan nyata terhadap ARIMA.

**Status bukti**: **Kuat**

### 28. Apa keterbatasan penelitian yang harus diakui dengan jujur?

**Jawaban akademik**

Keterbatasan yang perlu diakui secara jujur meliputi: pertama, evaluasi komparatif utama saat ini difokuskan pada lima fold terakhir sehingga hasilnya merepresentasikan rezim pasar terbaru, bukan keseluruhan histori. Kedua, timeframe `1D` belum menjadi jalur komparatif utama yang selesai dijalankan penuh. Ketiga, skenario feature engineering `RSI + MACD + ATR` masih merupakan tahap lanjutan, sehingga hasil utama saat ini masih bertumpu pada baseline `OHLC-only`. Keempat, pada implementasi progress, segmen holdout temporal yang sama masih digunakan untuk `EarlyStopping` dan evaluasi prediksi, sehingga skema validasi dan pengujian belum sepenuhnya dipisahkan. Pengakuan atas keterbatasan ini justru memperkuat kredibilitas penelitian.

**Jawaban singkat lisan**

Keterbatasan utama saya ada pada fokus lima fold terakhir, `1D` yang belum penuh, feature engineering yang masih tahap lanjutan, dan pemisahan validation-test yang masih perlu diperketat.

**Status bukti**: **Kuat**

### 29. Jika penguji menekan dengan pertanyaan, “Jadi apa yang benar-benar sudah terbukti pada penelitian ini?”, apa jawaban yang paling aman?

**Jawaban akademik**

Yang benar-benar sudah terbukti pada tahap ini adalah:  
- baseline `ARIMA` sudah diimplementasikan dan dievaluasi pada protokol `72 bulan train / 1 bulan test`;  
- `FLF-LSTM` dan `FLF-BiLSTM` sudah dituning secara bertahap pada baseline `OHLC-only`;  
- pada lima fold terakhir rezim pasar terbaru, kedua model FLF mengungguli baseline ARIMA dalam `MAE(pips)`;  
- di antara dua model utama, `FLF-LSTM` memberikan performa agregat terbaik pada skenario komparatif yang sama.  
Adapun hal-hal seperti sensitivitas terhadap panjang train window lain, efek penuh feature engineering, dan stabilitas lintas seed masih merupakan ruang pengembangan lanjutan.

**Jawaban singkat lisan**

Yang sudah terbukti adalah ARIMA, FLF-LSTM, dan FLF-BiLSTM sudah dibandingkan secara fair pada protokol yang sama, dan model FLF mengungguli ARIMA pada rezim pasar terbaru yang diuji.

**Status bukti**: **Kuat**

## G. Rumus Jawaban Aman Saat Sidang

### 30. Bagaimana pola menjawab pertanyaan kritis agar tetap kuat secara metodologis?

**Jawaban akademik**

Pola jawaban yang aman adalah:

- mulai dari **definisi**: jelaskan terlebih dahulu apa yang dimaksud dalam konteks penelitian;
- lanjutkan ke **keputusan metodologis**: jelaskan mengapa pilihan tersebut diambil;
- nyatakan **batas klaim** secara jujur: bedakan antara hal yang sudah dibuktikan dan yang belum;
- tutup dengan **arah penguatan lanjutan** jika memang masih ada keterbatasan.

Pola ini penting karena banyak pertanyaan penguji sebenarnya tidak menuntut jawaban absolut, melainkan menuntut kejelasan logika, konsistensi klaim, dan kejujuran ilmiah.

**Jawaban singkat lisan**

Definisikan dulu, jelaskan alasannya, batasi klaimnya, lalu akui kalau memang ada ruang penguatan lanjutan.

**Status bukti**: **Kuat**

## Ringkasan Cepat untuk Diingat

- **Baseline utama saat ini**: `OHLC-only`
- **Feature engineering lanjutan**: `RSI + MACD + ATR`
- **Hyperparameter FLF yang dituning**: `window`, `units`, `activation` atau `learning rate`, `lambda`, `sigma`, `batch`, `epochs`
- **Skenario evaluasi utama**: `walk-forward fixed 72 bulan train / 1 bulan test`
- **Fokus komparatif hasil utama**: `5 fold terakhir (17–21)`
- **Timeframe implementasi utama saat ini**: `4H`
- **Baseline statistik**: `ARIMA` dengan `4` model univariat
- **Konfigurasi akhir FLF-LSTM**: `window=12, units=256, activation=relu, lr=0.0009, lambda=0.8, sigma=0.1, batch=128, epochs=60`
- **Konfigurasi akhir FLF-BiLSTM**: `window=12, units=256, activation=relu, lr=0.0007, lambda=0.8, sigma=0.15, batch=128, epochs=50`
- **Hasil komparatif mean MAE(pips)**:
  - `ARIMA = 15.3941`
  - `FLF-BiLSTM = 12.3993`
  - `FLF-LSTM = 11.3646`

## Catatan Penutup

Dokumen ini sebaiknya dibaca bersama dengan:

- `progress_bab-metodologiPenelitian_updated.docx`
- `progress_bab-implementasi_dan_hasil_pembahasan_updated.docx`
- `progres_BAba Progress PEnelitian dan Skenario pengujian_updated.docx`

Tujuannya agar jawaban lisan saat sidang tetap konsisten dengan narasi tertulis pada dokumen proposal dan progress tesis.
