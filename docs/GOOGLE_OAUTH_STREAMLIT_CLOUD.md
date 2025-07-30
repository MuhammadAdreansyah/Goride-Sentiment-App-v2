# Konfigurasi Google OAuth untuk Streamlit Cloud

## Masalah yang Diperbaiki

Sebelumnya, fitur "Lanjutkan dengan Google" tidak bekerja di Streamlit Cloud karena menggunakan `meta refresh` HTML yang tidak kompatibel. Sekarang telah diperbaiki dengan menggunakan `st.link_button` yang native dan kompatibel dengan Streamlit Cloud.

## Perubahan yang Dilakukan

### 1. Mengganti Meta Refresh dengan st.link_button

**Sebelum (Tidak Kompatibel):**
```python
# Meta refresh - TIDAK BEKERJA di Streamlit Cloud
st.markdown(f"""
    <meta http-equiv="refresh" content="0;url={google_url}">
""", unsafe_allow_html=True)
```

**Sesudah (Kompatibel):**
```python
# Menggunakan st.link_button native Streamlit
try:
    st.link_button(
        "🚀 Login dengan Google",
        url=google_url,
        use_container_width=True,
        type="primary"
    )
except AttributeError:
    # Fallback untuk versi Streamlit lama
    st.markdown(f"""
        <div style="text-align: center;">
            <a href="{google_url}" target="_blank">
                🚀 Login dengan Google
            </a>
        </div>
    """, unsafe_allow_html=True)
```

### 2. Perbaikan Deteksi Environment

Fungsi `get_redirect_uri()` telah diperbaiki untuk lebih baik mendeteksi environment Streamlit Cloud vs Local:

```python
def get_redirect_uri() -> str:
    """Deteksi environment dan return redirect URI yang tepat"""
    try:
        # Method 1: Check localhost (LOCAL DEVELOPMENT)
        current_hostname = socket.gethostname()
        local_ip = socket.gethostbyname(current_hostname)
        
        if (current_hostname.lower() in ['localhost', '127.0.0.1'] or 
            local_ip.startswith('127.') or
            local_ip.startswith('192.168.')):
            return st.secrets.get("REDIRECT_URI_DEVELOPMENT", "http://localhost:8501/oauth2callback")
        
        # Method 2: Check Streamlit Cloud environment variables
        if os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true' or os.getenv('STREAMLIT_CLOUD'):
            return st.secrets.get("REDIRECT_URI_PRODUCTION", "https://sentimentgo.streamlit.app/oauth2callback")
            
        # Method 3: Check Python execution path (Cloud backup)
        python_path = sys.executable.lower()
        if "/mount/src" in python_path or "streamlit" in python_path:
            return st.secrets.get("REDIRECT_URI_PRODUCTION", "https://sentimentgo.streamlit.app/oauth2callback")
        
        # Default: LOCAL DEVELOPMENT
        return st.secrets.get("REDIRECT_URI_DEVELOPMENT", "http://localhost:8501/oauth2callback")
        
    except Exception as e:
        logger.error(f"Environment detection failed: {e}")
        return st.secrets.get("REDIRECT_URI_DEVELOPMENT", "http://localhost:8501/oauth2callback")
```

## Konfigurasi yang Diperlukan

### 1. Google Cloud Console

1. **Buka Google Cloud Console** → APIs & Services → Credentials
2. **Buat OAuth 2.0 Client ID** atau edit yang existing
3. **Tambahkan Authorized redirect URIs:**
   - Local: `http://localhost:8501/oauth2callback`
   - Production: `https://your-app-name.streamlit.app/oauth2callback`

### 2. Streamlit Secrets (`.streamlit/secrets.toml`)

**Local Development:**
```toml
# Google OAuth Configuration
GOOGLE_CLIENT_ID = "your-client-id.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "your-client-secret"

# Redirect URIs
REDIRECT_URI_DEVELOPMENT = "http://localhost:8501/oauth2callback"
REDIRECT_URI_PRODUCTION = "https://your-app-name.streamlit.app/oauth2callback"

# Firebase Configuration
[firebase]
type = "service_account"
project_id = "your-project-id"
# ... other firebase configs
```

**Streamlit Cloud:**
1. **Pergi ke Streamlit Cloud Dashboard**
2. **Edit App → Advanced settings → Secrets**
3. **Tambahkan konfigurasi yang sama seperti di atas**

### 3. URL Callback Handling

Pastikan aplikasi Anda dapat handle callback dari Google:

```python
# Di main app atau auth handler
def handle_google_callback():
    """Handle Google OAuth callback"""
    if 'code' in st.query_params:
        code = st.query_params.get('code')
        # Process Google OAuth code here
        ...
```

## Fitur Baru

### 1. Multiple Fallback Methods

- **Primary:** `st.link_button` (native Streamlit)
- **Fallback:** HTML anchor dengan styling
- **Manual:** Copy-paste URL untuk troubleshooting

### 2. Better User Experience

- Progress indicator saat redirect
- Clear instructions untuk user
- Troubleshooting expander jika ada masalah

### 3. Robust Error Handling

- Graceful fallback jika `st.link_button` tidak tersedia
- Logging untuk debugging
- Manual URL option sebagai backup terakhir

## Testing

### Local Testing
1. Run aplikasi di localhost:8501
2. Test login Google → harus redirect ke Google
3. Setelah authorize → harus kembali ke localhost:8501/oauth2callback

### Streamlit Cloud Testing
1. Deploy ke Streamlit Cloud
2. Test login Google → harus redirect ke Google
3. Setelah authorize → harus kembali ke https://your-app.streamlit.app/oauth2callback

## Troubleshooting

### Problem: Tombol Google tidak muncul
**Solution:** Check version Streamlit (butuh 1.32+)

### Problem: Redirect URI mismatch
**Solution:** 
1. Check Google Cloud Console authorized URIs
2. Verify secrets.toml REDIRECT_URI settings
3. Check environment detection logs

### Problem: Infinite redirect loop
**Solution:**
1. Clear browser cache dan cookies
2. Check callback handler implementation
3. Verify Google OAuth configuration

## Verifikasi Deployment

1. **Local:** ✅ Meta refresh replacement tidak perlu (sudah bekerja)
2. **Streamlit Cloud:** ✅ st.link_button kompatibel
3. **Fallback:** ✅ HTML anchor untuk compatibility
4. **UX:** ✅ Progress indicator dan instructions
5. **Error Handling:** ✅ Graceful degradation

Dengan perubahan ini, fitur "Lanjutkan dengan Google" sekarang akan bekerja dengan baik di kedua environment: local development dan Streamlit Cloud.
