# Analisis Konvergensi, Overfitting, dan Diagnostik Residual Model

## Saran Judul

Judul yang paling saya sarankan untuk analisis ini adalah:

**Analisis Konvergensi, Overfitting, dan Diagnostik Residual Model**

Alternatif yang masih aman:
- Analisis Dinamika Pelatihan dan Diagnostik Error/Residual Model
- Analisis Generalisasi Model pada Skenario Walk-Forward 72 Bulan/1 Bulan

## Ruang Lingkup Analisis

Analisis ini mencakup tiga hal yang berbeda secara metodologis:

1. **FLF-LSTM dan FLF-BiLSTM** dianalisis melalui kurva `loss` dan `val_loss` untuk melihat konvergensi, indikasi overfitting, dan stabilitas generalisasi.
2. **Gradient loss** pada report repo ini diperlakukan sebagai **proxy perubahan loss per epoch (`dLoss`)**, sehingga fungsinya adalah membaca kecepatan konvergensi dan plateau, bukan gradient parameter model.
3. **ARIMA** tidak memiliki kurva loss epoch seperti neural network, sehingga analisis yang setara dilakukan melalui **diagnostik residual forecast**, AIC/BIC, dan uji autokorelasi residual.

Skenario evaluasi yang dipakai sama untuk semua model, yaitu **walk-forward fixed 72 bulan training, 1 bulan testing, fold 17-21**.

## Hasil FLF-LSTM

- Rata-rata epoch berjalan: **53.8**
- Rata-rata `loss_end`: **0.000485**
- Rata-rata `val_end`: **0.000547**
- Rata-rata gap akhir `val_loss - loss`: **0.000062**
- Fold yang menunjukkan pola overfitting setelah epoch terbaik: **3 dari 5 fold**
- Rata-rata posisi epoch terbaik validation terhadap total epoch: **95.89%**

Interpretasi:

**FLF-LSTM konvergen dengan overfitting ringan.**

Secara umum training loss dan validation loss sama-sama turun hingga sangat kecil. Tidak ada indikasi underfitting dan tidak ada tanda divergensi training. Overfitting yang muncul bersifat ringan, karena validation loss pada beberapa fold hanya memburuk tipis setelah mencapai titik minimum. Plateau di akhir training juga jelas, ditunjukkan oleh rata-rata perubahan loss 5 epoch terakhir yang sangat kecil.

## Hasil FLF-BiLSTM

- Rata-rata epoch berjalan: **37.6**
- Rata-rata `loss_end`: **0.000498**
- Rata-rata `val_end`: **0.000597**
- Rata-rata gap akhir `val_loss - loss`: **0.000099**
- Fold yang menunjukkan pola overfitting setelah epoch terbaik: **3 dari 5 fold**
- Rata-rata posisi epoch terbaik validation terhadap total epoch: **89.30%**

Interpretasi:

**FLF-BiLSTM konvergen dengan overfitting ringan hingga moderat.**

Model ini juga tidak underfit dan berhasil konvergen. Namun, dibanding FLF-LSTM, titik minimum validation loss lebih sering tercapai lebih awal, lalu validation loss bergerak naik ketika training loss masih turun. Itu menunjukkan kecenderungan overfitting yang lebih nyata daripada FLF-LSTM. Dengan kata lain, BiLSTM masih belajar, tetapi generalisasi mulai jenuh lebih cepat.

## Perbandingan LSTM vs BiLSTM

- Kedua model **tidak underfit**
- Kedua model **tidak menunjukkan training yang gagal atau divergen**
- Keduanya berada pada rezim **konvergen**
- Perbedaannya ada pada **stabilitas generalisasi**

Kesimpulan komparatif:

**FLF-LSTM memperlihatkan dinamika pelatihan yang lebih sehat dan generalisasi yang lebih stabil daripada FLF-BiLSTM pada skenario WF72m/1m fold 17-21.**

## Hasil ARIMA

- Mean MAE avg: **15.3941 pips**
- Mean close bias per fold: **0.7396 pips**
- Combined close bias: **0.7157 pips**
- Combined close residual std: **22.3251 pips**
- Combined lag-1 autocorrelation residual close: **-0.0018**
- Combined Ljung-Box p-value lag 10: **0.0566**
- Combined Ljung-Box p-value lag 20: **0.0500**

Interpretasi:

**ARIMA forecast residual relatif stabil tanpa bukti kuat autokorelasi serial pada lag 10.**

Karena ARIMA bukan model berbasis epoch, maka ia tidak dianalisis dengan kurva `loss`/`val_loss`. Diagnostik yang relevan adalah apakah residual forecast masih menyimpan pola serial. Pada hasil ini, p-value Ljung-Box per fold masih berada di atas 0.05, sehingga tidak ada bukti kuat autokorelasi residual close pada masing-masing fold. Namun ketika residual digabung lintas fold, nilai lag 20 berada sangat dekat dengan batas 0.05, sehingga masih ada indikasi lemah bahwa struktur temporal belum sepenuhnya hilang.

## Makna Metodologis

1. **Analisis LSTM/BiLSTM** sebaiknya disebut:
   - analisis konvergensi dan overfitting
   - atau analisis dinamika pelatihan dan generalisasi
2. **Analisis ARIMA** sebaiknya disebut:
   - analisis diagnostik residual
   - atau analisis error/residual forecast
3. Jika ingin semua digabung dalam satu subbagian, istilah paling aman adalah:
   - **analisis konvergensi, overfitting, dan diagnostik residual model**

## Kesimpulan Akhir

- **FLF-LSTM**: konvergen, tidak underfit, dengan overfitting ringan
- **FLF-BiLSTM**: konvergen, tidak underfit, tetapi lebih rentan overfitting
- **ARIMA**: tidak memakai analisis loss curve; evaluasi yang tepat adalah diagnostik residual, AIC/BIC, dan autokorelasi residual

Dokumen ini paling tepat ditempatkan sebagai bahan untuk:
- pembahasan hasil eksperimen
- jawaban sidang/penguji
- atau lampiran analisis teknis model
