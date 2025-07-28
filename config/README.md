# 📁 Config Directory

Folder ini berisi file konfigurasi untuk aplikasi SentimenGo.

## 🔒 File yang Diperlukan (Tidak disertakan dalam repository)

### 1. Google OAuth Client Secret
```
client_secret_[CLIENT_ID].apps.googleusercontent.com.json
```
File ini berisi konfigurasi OAuth untuk autentikasi Google. Dapatkan dari [Google Cloud Console](https://console.cloud.google.com/).

### 2. Firebase Admin SDK
```
[PROJECT_NAME]-firebase-adminsdk-[KEY_ID].json
```
File ini berisi service account key untuk Firebase Admin SDK. Dapatkan dari [Firebase Console](https://console.firebase.google.com/).

## 📋 Template

Gunakan file `secrets.toml.example` di root directory sebagai template untuk konfigurasi aplikasi.

## ⚠️ Keamanan

**JANGAN PERNAH** commit file konfigurasi yang berisi:
- API Keys
- Client Secrets
- Private Keys
- Database Credentials

File-file tersebut sudah di-ignore dalam `.gitignore` untuk menjaga keamanan.
