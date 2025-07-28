# 📁 Log Directory

Folder ini berisi file log dari aplikasi SentimenGo.

## 📄 File Log yang Dihasilkan

### 1. app.log
Log utama aplikasi yang mencatat:
- Aktivitas pengguna
- Error dan exception
- Proses autentikasi
- Operasi CRUD

### 2. analisis_data.log  
Log khusus untuk proses analisis data yang mencatat:
- Proses preprocessing teks
- Hasil prediksi sentimen
- Performa model
- Statistik analisis

## ⚡ Auto-Generated

File log ini dibuat secara otomatis oleh aplikasi dan tidak perlu di-commit ke repository karena bersifat sementara dan dapat berisi informasi sensitif pengguna.

## 🔍 Format Log

Log menggunakan format standar Python logging dengan level:
- DEBUG: Informasi detail untuk debugging
- INFO: Informasi umum operasi aplikasi  
- WARNING: Peringatan yang tidak menghentikan proses
- ERROR: Error yang perlu perhatian
- CRITICAL: Error kritis yang menghentikan aplikasi
