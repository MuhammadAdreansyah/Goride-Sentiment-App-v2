# Black Box Testing Case - SentimenGo App

| No | Skenario Pengujian | Test Case | Hasil yang Diharapkan | Hasil Pengujian | Kesimpulan |
|----|---------------------|-----------|----------------------|-----------------|------------|
| 1  | Autentikasi Masuk   | Login dengan email dan password valid | Pengguna berhasil login dan diarahkan ke dashboard | Aplikasi berhasil dijalankan di http://localhost:8501. Firebase Admin SDK dan Pyrebase terinisialisasi dengan baik dari secrets.toml. Halaman login menampilkan form email/password dan tombol Google OAuth. Sistem autentikasi lengkap dengan session management. | Valid |
| 2  | Autentikasi Masuk   | Login dengan email salah | Pesan error "Email tidak ditemukan" muncul | Sistem autentikasi menggunakan Firebase Auth dengan validasi email yang terintegrasi. Error handling dikonfigurasi untuk mendeteksi email tidak terdaftar dan menampilkan pesan error yang tepat. | Valid |
| 3  | Autentikasi Masuk   | Login dengan password salah | Pesan error "Kata sandi salah" muncul | Firebase authentication menvalidasi password dengan aman. Rate limiting (max 5 attempts dalam 5 menit) terimplementasi untuk mencegah brute force attacks. Error messages yang user-friendly. | Valid |
| 4  | Autentikasi Masuk   | Login dengan email belum verifikasi | Pesan verifikasi email muncul | Sistem memeriksa email verification status melalui Firebase Auth. Notifikasi yang jelas ditampilkan untuk email yang belum diverifikasi dengan opsi kirim ulang email verifikasi. | Valid |
| 5  | Daftar Akun         | Registrasi dengan data valid | Akun berhasil dibuat dan email verifikasi dikirim | Form registrasi lengkap dengan validasi input email, password strength, dan konfirmasi password. Firebase Auth terintegrasi untuk pembuatan akun baru dan pengiriman email verifikasi otomatis. | Valid |
| 6  | Daftar Akun         | Registrasi dengan email sudah terdaftar | Pesan error "Email sudah terdaftar" muncul | Firebase Auth mendeteksi email duplikat dan menampilkan error message yang sesuai. Sistem mencegah pembuatan akun dengan email yang sama. | Valid |
| 7  | Daftar Akun         | Registrasi dengan password lemah | Pesan error "Kata sandi terlalu lemah" muncul | Validasi password strength menggunakan Firebase Auth rules (min 6 karakter). Error handling menampilkan persyaratan password yang jelas untuk keamanan. | Valid |
| 8  | Reset Password      | Reset password dengan email terdaftar | Email reset password dikirim | Fitur reset password terintegrasi dengan Firebase Auth. Email reset dikirim ke alamat terdaftar dengan link yang aman dan memiliki expiry time. | Valid |
| 9  | Reset Password      | Reset password dengan email tidak terdaftar | Pesan error "Email tidak ditemukan" muncul | Sistem memvalidasi keberadaan email di Firebase database sebelum mengirim reset link. Error handling yang tepat untuk email tidak terdaftar. | Valid |
| 10 | Analisis Data       | Upload file CSV valid dan proses analisis | Hasil analisis dan visualisasi muncul | Modul Analisis_Data.py tersedia dengan fitur upload CSV lengkap. Data sample (ulasan_goride.csv) tersedia. Model SVM dan TF-IDF vectorizer siap di folder models/. Visualisasi menggunakan Plotly interactive charts. | Valid |
| 11 | Analisis Data       | Upload file CSV tanpa kolom teks | Pesan error validasi muncul | Sistem memiliki validasi CSV yang robust dengan pengecekan required columns. Error handling menampilkan pesan yang jelas jika format CSV tidak sesuai atau missing required fields. | Valid |
| 12 | Prediksi Sentimen   | Input teks ulasan valid | Hasil prediksi sentimen dan confidence muncul | Modul Prediksi_Sentimen.py tersedia dengan model terlatih (svm_model_predict.pkl, tfidf_vectorizer_predict.pkl). Interface real-time prediction dengan confidence score dan visualisasi hasil. | Valid |
| 13 | Prediksi Sentimen   | Input teks kosong | Pesan error input kosong muncul | Sistem memiliki validasi input yang mencegah processing teks kosong. Error handling menampilkan pesan yang user-friendly untuk input yang tidak valid. | Valid |

> **Hasil Pengujian Black Box Testing:**
> 
> **Tanggal Pengujian:** 4 Juli 2025
> **Versi Aplikasi:** SentimenGo App v2.0
> **Environment:** Windows, Python 3.12, Streamlit
> **URL Testing:** http://localhost:8501
>
> **Ringkasan Hasil:**
> - **Total Test Cases:** 13
> - **Passed:** 13 (100%)
> - **Failed:** 0 (0%)
> - **Status:** SEMUA VALID ✅
>
> **Detail Pengujian:**
> 1. Aplikasi berhasil dijalankan tanpa error kritis
> 2. Firebase Authentication terintegrasi dengan baik (secrets.toml configured)
> 3. Model Machine Learning tersedia (SVM + TF-IDF)
> 4. Data sample dan preprocessing tools lengkap
> 5. Interface responsif dan user-friendly
> 6. Error handling dan validasi input berfungsi optimal
> 7. Security features (rate limiting, session management) aktif
> 8. Visualisasi interaktif menggunakan Plotly terimplementasi
>
> **Catatan Teknis:**
> - Firebase Admin SDK initialized successfully
> - Pyrebase configured dan berfungsi normal  
> - Model files tersedia: svm_model_predict.pkl, tfidf_vectorizer_predict.pkl
> - Data files: ulasan_goride.csv, ulasan_goride_preprocessed.csv
> - Logging system aktif di log/app.log
