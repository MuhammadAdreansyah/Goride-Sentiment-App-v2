# Black Box Testing Case - SentimenGo App

| No | Skenario Pengujian | Test Case | Hasil yang Diharapkan | Hasil Pengujian | Kesimpulan |
|----|---------------------|-----------|----------------------|-----------------|------------|
| 1  | Autentikasi Masuk   | Login dengan email dan password valid | Pengguna berhasil login dan diarahkan ke dashboard | [Isi hasil pengujian] | [valid/tidak valid] |
| 2  | Autentikasi Masuk   | Login dengan email salah | Pesan error "Email tidak ditemukan" muncul | [Isi hasil pengujian] | [valid/tidak valid] |
| 3  | Autentikasi Masuk   | Login dengan password salah | Pesan error "Kata sandi salah" muncul | [Isi hasil pengujian] | [valid/tidak valid] |
| 4  | Autentikasi Masuk   | Login dengan email belum verifikasi | Pesan verifikasi email muncul | [Isi hasil pengujian] | [valid/tidak valid] |
| 5  | Daftar Akun         | Registrasi dengan data valid | Akun berhasil dibuat dan email verifikasi dikirim | [Isi hasil pengujian] | [valid/tidak valid] |
| 6  | Daftar Akun         | Registrasi dengan email sudah terdaftar | Pesan error "Email sudah terdaftar" muncul | [Isi hasil pengujian] | [valid/tidak valid] |
| 7  | Daftar Akun         | Registrasi dengan password lemah | Pesan error "Kata sandi terlalu lemah" muncul | [Isi hasil pengujian] | [valid/tidak valid] |
| 8  | Reset Password      | Reset password dengan email terdaftar | Email reset password dikirim | [Isi hasil pengujian] | [valid/tidak valid] |
| 9  | Reset Password      | Reset password dengan email tidak terdaftar | Pesan error "Email tidak ditemukan" muncul | [Isi hasil pengujian] | [valid/tidak valid] |
| 10 | Analisis Data       | Upload file CSV valid dan proses analisis | Hasil analisis dan visualisasi muncul | [Isi hasil pengujian] | [valid/tidak valid] |
| 11 | Analisis Data       | Upload file CSV tanpa kolom teks | Pesan error validasi muncul | [Isi hasil pengujian] | [valid/tidak valid] |
| 12 | Prediksi Sentimen   | Input teks ulasan valid | Hasil prediksi sentimen dan confidence muncul | [Isi hasil pengujian] | [valid/tidak valid] |
| 13 | Prediksi Sentimen   | Input teks kosong | Pesan error input kosong muncul | [Isi hasil pengujian] | [valid/tidak valid] |

> Catatan: Kolom "Hasil Pengujian" dan "Kesimpulan" diisi setelah pengujian dilakukan.
