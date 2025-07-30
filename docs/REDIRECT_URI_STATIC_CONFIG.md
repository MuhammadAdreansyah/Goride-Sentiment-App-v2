# Static Redirect URI Configuration Guide

## Overview
Aplikasi SentimenGo sekarang menggunakan konfigurasi redirect URI statis untuk Google OAuth, menggantikan sistem deteksi otomatis yang sebelumnya digunakan.

## Konfigurasi Redirect URI

### 1. Lokasi Konfigurasi
File: `.streamlit/secrets.toml`

### 2. Parameter yang Digunakan
```toml
# Static redirect URI - manually change based on deployment environment
REDIRECT_URI = "http://localhost:8501/oauth2callback"  # Change to production URL when deploying
```

### 3. Environment-Specific Configuration

#### Development (Local)
```toml
REDIRECT_URI = "http://localhost:8501/oauth2callback"
```

#### Production (Streamlit Cloud)
```toml
REDIRECT_URI = "https://sentimentgo.streamlit.app/oauth2callback"
```

## Cara Mengubah Environment

### Saat Development Local:
1. Buka file `.streamlit/secrets.toml`
2. Pastikan `REDIRECT_URI` berisi:
   ```toml
   REDIRECT_URI = "http://localhost:8501/oauth2callback"
   ```

### Saat Deploy ke Production:
1. Di Streamlit Cloud Dashboard, masuk ke App Settings > Secrets
2. Update nilai `REDIRECT_URI` menjadi:
   ```toml
   REDIRECT_URI = "https://sentimentgo.streamlit.app/oauth2callback"
   ```

## Keuntungan Sistem Statis

### ✅ Advantages:
- **Sederhana dan Jelas**: Tidak ada logika kompleks untuk deteksi environment
- **Mudah Debug**: Developer tahu persis URI mana yang digunakan
- **Kontrol Penuh**: Developer mengontrol kapan dan bagaimana mengubah URI
- **Konsisten**: Selalu menggunakan URI yang sama untuk environment yang sama
- **Troubleshooting Mudah**: Tidak ada ambiguitas dalam pemilihan URI

### ❌ Removed Complexity:
- Tidak ada lagi deteksi hostname otomatis
- Tidak ada lagi pengecekkan environment variables
- Tidak ada lagi fallback mechanisms yang kompleks
- Tidak ada lagi manual override switches

## Migration dari Sistem Lama

### Perubahan yang Dilakukan:
1. **secrets.toml**: 
   - Dihapus: `REDIRECT_URI_PRODUCTION`, `REDIRECT_URI_DEVELOPMENT`
   - Ditambah: `REDIRECT_URI` (single static URI)
   
2. **auth.py**:
   - Fungsi `get_redirect_uri()` disederhanakan drastis
   - Dihapus: `debug_environment_variables()` function
   - Diperbarui: `is_config_valid()` function

## Best Practices

### 1. Documentation
- Selalu dokumentasikan URI mana yang digunakan untuk environment apa
- Buat checklist deployment yang mencakup update redirect URI

### 2. Testing
- Test OAuth flow setelah mengubah redirect URI
- Verifikasi URL callback di Google Cloud Console

### 3. Security
- Pastikan redirect URI di Google Cloud Console sesuai dengan yang dikonfigurasi
- Hanya gunakan HTTPS untuk production environment

## Troubleshooting

### Problem: OAuth redirect mismatch error
**Solution**: 
1. Periksa nilai `REDIRECT_URI` di secrets.toml
2. Pastikan URI tersebut terdaftar di Google Cloud Console
3. Pastikan tidak ada typo atau trailing slash

### Problem: OAuth tidak bekerja setelah deployment
**Solution**:
1. Verify bahwa `REDIRECT_URI` sudah diubah ke production URL
2. Pastikan Streamlit Cloud secrets sudah diupdate
3. Restart aplikasi di Streamlit Cloud

## Contact
Untuk pertanyaan lebih lanjut mengenai konfigurasi ini, hubungi team developer SentimenGo.
