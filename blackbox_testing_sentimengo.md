# Black Box Testing Case - SentimenGo App

| No | Skenario Pengujian | Test Case | Hasil yang Diharapkan | Hasil Pengujian | Kesimpulan |
|----|---------------------|-----------|----------------------|-----------------|------------|
| 1  | Autentikasi Masuk   | Login dengan email dan password valid | Pengguna berhasil login dan diarahkan ke dashboard | Pengguna berhasil login dan berhasil diarahkan ke dashboard | Valid |
| 2  | Autentikasi Masuk   | Login dengan email salah | Pesan error "Email tidak ditemukan" muncul | Pesan error muncul | Valid |
| 3  | Autentikasi Masuk   | Login dengan password salah | Pesan error "Kata sandi salah" muncul | Pesan error muncul | Valid |
| 4  | Autentikasi Masuk   | Login dengan email belum verifikasi | Pesan verifikasi email muncul | Pesan "📧 Email Anda belum diverifikasi!" muncul | Valid |
| 5  | Daftar Akun         | Registrasi dengan data valid | Akun berhasil dibuat dan email verifikasi dikirim | Email berhasil dibuat dan email verifikasi berhasil dikirimkan | Valid |
| 6  | Daftar Akun         | Registrasi dengan email sudah terdaftar | Pesan error "Email sudah terdaftar" muncul | Pesan error muncul | Valid |
| 7  | Daftar Akun         | Registrasi dengan password lemah | Pesan error "Kata sandi terlalu lemah" muncul | Pesan error muncul | Valid |
| 8  | Reset Password      | Reset password dengan email terdaftar | Email reset password dikirim | Email berhasil dikirimkan untuk reset password | Valid |
| 9  | Reset Password      | Reset password dengan email tidak terdaftar | Pesan error "Email tidak ditemukan" muncul | Pesan error muncul | Valid |
| 10 | Analisis Data       | Upload file CSV valid dan proses analisis | Hasil analisis dan visualisasi muncul | Hasil analisis berhasil muncul dan visualisasi berhasil muncul | Valid |
| 11 | Analisis Data       | Upload file CSV tanpa kolom teks | Pesan error validasi muncul | Pesan error validasi "Silakan pilih kolom teks dan konfirmasi." berhasil muncul | Valid |
| 12 | Prediksi Sentimen   | Input teks ulasan valid | Hasil prediksi sentimen dan confidence muncul | Hasil prediksi dan confidence berhasil muncul | Valid |
| 13 | Prediksi Sentimen   | Input teks kosong | Pesan error input kosong muncul | Pesan error "⚠️ Silakan masukkan teks terlebih dahulu untuk diprediksi." muncul | Valid |

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
> **Detail Pengujian Aktual:**
> 1. **Autentikasi berhasil diuji** - Login, registrasi, dan reset password berfungsi normal
> 2. **Error handling teruji** - Pesan error muncul sesuai dengan kondisi yang diuji
> 3. **Validasi email verification** - Pesan "📧 Email Anda belum diverifikasi!" muncul dengan benar
> 4. **Analisis data CSV** - Upload dan visualisasi berhasil, validasi error kolom teks berfungsi
> 5. **Prediksi sentimen** - Input valid menghasilkan prediksi dan confidence score
> 6. **Validasi input kosong** - Error "⚠️ Silakan masukkan teks terlebih dahulu untuk diprediksi." muncul
>
> **Catatan Pengujian:**
> - Semua fitur autentikasi (login/register/reset) beroperasi normal
> - Error messages informatif dan user-friendly
> - Sistem validasi input berfungsi optimal
> - Model machine learning responsif untuk prediksi sentimen
> - Interface aplikasi stabil dan dapat diandalkan
