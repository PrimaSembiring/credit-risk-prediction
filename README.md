# Credit Card Default Prediction

Project ini berfokus pada analisis dan prediksi risiko *default* pada nasabah kartu kredit menggunakan pendekatan data science. Dataset mencakup profil nasabah, limit kredit, serta riwayat pembayaran.

Melalui exploratory data analysis (EDA), dilakukan identifikasi pola dan variabel yang berpengaruh terhadap default. Selanjutnya, model deep learning berbasis **PyTorch** digunakan untuk melakukan prediksi.

---

## Deployment dengan Streamlit

Model telah di-deploy menggunakan Streamlit untuk antarmuka web yang mudah digunakan.

### Persyaratan Sistem
- Python 3.8+
- PyTorch
- Streamlit
- scikit-learn
- pandas
- numpy

### Instalasi Dependencies
```bash
pip install streamlit torch scikit-learn pandas numpy imbalanced-learn joblib
```

### Menjalankan Aplikasi
1. Pastikan file `best_model.pth` dan `default of credit card clients.csv` ada di direktori yang sama.
2. Jalankan perintah berikut:
```bash
streamlit run app.py
```
3. Buka browser dan akses `http://localhost:8501`

### Fitur Aplikasi
- **Input Form**: Masukkan data nasabah termasuk limit kredit, demografi, riwayat pembayaran, dan jumlah tagihan/pembayaran.
- **Prediksi Real-time**: Model akan memprediksi probabilitas default secara instan.
- **Risk Assessment**: Menampilkan level risiko (Very Low, Low, Medium, High, Very High) berdasarkan probabilitas.
- **Input Summary**: Ringkasan data yang dimasukkan untuk verifikasi.

---

## Sumber Dataset

Dataset yang digunakan dalam project ini adalah **Default of Credit Card Clients** yang diperoleh dari UCI Machine Learning Repository.

Referensi:
Yeh, I. (2009). *Default of Credit Card Clients*.  
UCI Machine Learning Repository.  
https://doi.org/10.24432/C55S3H  

---

## Penjelasan Dataset

Dataset ini menggunakan variabel target berupa *default payment*, yaitu:
- `1` = nasabah mengalami gagal bayar (default)  
- `0` = nasabah tidak mengalami gagal bayar  

Untuk melakukan prediksi, dataset ini terdiri dari 23 variabel yang merepresentasikan karakteristik nasabah dan perilaku keuangannya.


### Informasi Umum Nasabah
- `LIMIT_BAL` → jumlah batas kredit yang diberikan (termasuk kredit individu dan tambahan keluarga)  
- `SEX` → jenis kelamin (1 = laki-laki, 2 = perempuan)  
- `EDUCATION` → tingkat pendidikan  
  - 1 = graduate school  
  - 2 = university  
  - 3 = high school  
  - 4 = lainnya  
- `MARRIAGE` → status pernikahan  
  - 1 = menikah  
  - 2 = single  
  - 3 = lainnya  
- `AGE` → usia nasabah  


### Riwayat Pembayaran (PAY_X)
- `PAY_0` sampai `PAY_6` → status pembayaran dalam 6 bulan terakhir  

Skala keterlambatan:

- `-2`, `-1`, `0` = pembayaran tepat waktu  
- `1` = telat 1 bulan  
- `2` = telat 2 bulan  
- ...  
- `9` = telat ≥ 9 bulan  

Variabel ini sangat penting karena menggambarkan perilaku pembayaran nasabah.


### Tagihan Bulanan (BILL_AMT_X)
- `BILL_AMT1` – `BILL_AMT6` → jumlah tagihan kartu kredit setiap bulan  

Menunjukkan total penggunaan kredit oleh nasabah dalam periode tertentu.


### Pembayaran (PAY_AMT_X)
- `PAY_AMT1` – `PAY_AMT6` → jumlah pembayaran yang dilakukan setiap bulan  
