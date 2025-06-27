"""
Sistem Autentikasi Streamlit dengan Integrasi Firebase

Aplikasi ini menyediakan sistem autentikasi lengkap dengan fitur:
- Autentikasi email/kata sandi
- Login OAuth Google  
- Registrasi pengguna baru
- Reset kata sandi
- Manajemen sesi yang aman
- Verifikasi email
- Rate limiting dan keamanan

Author: SentimenGo App Team
Created: 2024
Last Modified: 2025-06-24
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import streamlit as st
import re
import uuid
import asyncio
import httpx
import time
import base64
import logging
import firebase_admin
import pyrebase
from firebase_admin import credentials, auth, firestore
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Union, Any
from urllib.parse import urlencode

from streamlit_cookies_controller import CookieController

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

class AuthConfig:
    """Konfigurasi autentikasi dengan error handling yang lebih baik"""
    
    # Default values
    DEFAULT_SESSION_TIMEOUT = 3600  # 1 jam dalam detik
    DEFAULT_MAX_LOGIN_ATTEMPTS = 5
    DEFAULT_RATE_LIMIT_WINDOW = 300  # 5 menit dalam detik
    DEFAULT_EMAIL_VERIFICATION_LIMIT = 50  # per jam
    DEFAULT_REMEMBER_ME_DURATION = 30 * 24 * 60 * 60  # 30 hari
    DEFAULT_LAST_EMAIL_DURATION = 90 * 24 * 60 * 60  # 90 hari
    
    def __init__(self):
        self._load_config()
    
    def _load_config(self) -> None:
        """Load konfigurasi dengan fallback ke default values"""
        try:
            # Firebase dan Google OAuth konfigurasi
            self.GOOGLE_CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID", "")
            self.GOOGLE_CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET", "")
            self.REDIRECT_URI = st.secrets.get("REDIRECT_URI", "")
            self.FIREBASE_API_KEY = st.secrets.get("FIREBASE_API_KEY", "")
            
            # Timeout dan limit konfigurasi dengan fallback
            self.SESSION_TIMEOUT = st.secrets.get("SESSION_TIMEOUT", self.DEFAULT_SESSION_TIMEOUT)
            self.MAX_LOGIN_ATTEMPTS = st.secrets.get("MAX_LOGIN_ATTEMPTS", self.DEFAULT_MAX_LOGIN_ATTEMPTS)
            self.RATE_LIMIT_WINDOW = st.secrets.get("RATE_LIMIT_WINDOW", self.DEFAULT_RATE_LIMIT_WINDOW)
            self.EMAIL_VERIFICATION_LIMIT = st.secrets.get("EMAIL_VERIFICATION_LIMIT", self.DEFAULT_EMAIL_VERIFICATION_LIMIT)
            self.REMEMBER_ME_DURATION = st.secrets.get("REMEMBER_ME_DURATION", self.DEFAULT_REMEMBER_ME_DURATION)
            self.LAST_EMAIL_DURATION = st.secrets.get("LAST_EMAIL_DURATION", self.DEFAULT_LAST_EMAIL_DURATION)
            
            # Validasi konfigurasi kritis
            self._validate_config()
            
        except Exception as e:
            logger.critical(f"Configuration loading failed: {e}")
            self._use_fallback_config()
    
    def _validate_config(self) -> None:
        """Validasi konfigurasi kritis"""
        required_fields = {
            "GOOGLE_CLIENT_ID": self.GOOGLE_CLIENT_ID,
            "GOOGLE_CLIENT_SECRET": self.GOOGLE_CLIENT_SECRET,
            "REDIRECT_URI": self.REDIRECT_URI,
            "FIREBASE_API_KEY": self.FIREBASE_API_KEY
        }
        
        missing_fields = [field for field, value in required_fields.items() if not value]
        
        if missing_fields:
            logger.error(f"Missing critical configuration: {', '.join(missing_fields)}")
            if not st.session_state.get('config_error_shown', False):
                st.error(f"❌ **Konfigurasi tidak lengkap:** {', '.join(missing_fields)}")
                st.session_state['config_error_shown'] = True
    
    def _use_fallback_config(self) -> None:
        """Gunakan konfigurasi fallback jika terjadi error"""
        logger.warning("Using fallback configuration")
        self.GOOGLE_CLIENT_ID = ""
        self.GOOGLE_CLIENT_SECRET = ""
        self.REDIRECT_URI = ""
        self.FIREBASE_API_KEY = ""
        self.SESSION_TIMEOUT = self.DEFAULT_SESSION_TIMEOUT
        self.MAX_LOGIN_ATTEMPTS = self.DEFAULT_MAX_LOGIN_ATTEMPTS
        self.RATE_LIMIT_WINDOW = self.DEFAULT_RATE_LIMIT_WINDOW
        self.EMAIL_VERIFICATION_LIMIT = self.DEFAULT_EMAIL_VERIFICATION_LIMIT
        self.REMEMBER_ME_DURATION = self.DEFAULT_REMEMBER_ME_DURATION
        self.LAST_EMAIL_DURATION = self.DEFAULT_LAST_EMAIL_DURATION
    
    def is_valid(self) -> bool:
        """Check apakah konfigurasi valid untuk operasi"""
        return bool(
            self.GOOGLE_CLIENT_ID and 
            self.GOOGLE_CLIENT_SECRET and 
            self.REDIRECT_URI and 
            self.FIREBASE_API_KEY
        )
    
    def get_firebase_config(self) -> Dict[str, Any]:
        """Dapatkan konfigurasi Firebase yang terstruktur"""
        try:
            if "firebase" not in st.secrets:
                return {}
            
            service_account = dict(st.secrets["firebase"])
            return {
                "apiKey": self.FIREBASE_API_KEY,
                "authDomain": f"{service_account.get('project_id', '')}.firebaseapp.com",
                "projectId": service_account.get('project_id', ''),
                "databaseURL": f"https://{service_account.get('project_id', '')}-default-rtdb.firebaseio.com",
                "storageBucket": f"{service_account.get('project_id', '')}.appspot.com"
            }
        except Exception as e:
            logger.error(f"Failed to get Firebase config: {e}")
            return {}

# Initialize global config
Config = AuthConfig()

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

cookie_controller = CookieController()

# =============================================================================
# SESSION MANAGEMENT FUNCTIONS
# =============================================================================

def initialize_session_state() -> None:
    """Inisialisasi state sesi dengan nilai default"""
    default_values = {
        'logged_in': False,
        'login_attempts': 0,
        'firebase_initialized': False,
        'auth_type': '🔒 Masuk',
        'auth_type_changed': False,
        'user_email': None,
        'remember_me': False,
        'models_prepared': False,
        'model_preparation_started': False,
        'model_preparation_completed': False,
        'ready_for_tools': False,
        'login_time': None
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def verify_environment() -> bool:
    """Verifikasi bahwa semua variabel lingkungan yang diperlukan telah diset"""
    if not Config.is_valid():
        logger.error("Configuration validation failed")
        return False
    
    # Periksa apakah Firebase config ada
    if "firebase" not in st.secrets:
        logger.error("Firebase configuration missing")
        return False
    
    return True

def sync_login_state() -> None:
    """Sinkronisasi status login dari cookie ke session_state dengan error handling yang lebih baik"""
    try:
        # Tunggu sejenak untuk memastikan cookie controller siap
        if not hasattr(cookie_controller, 'get'):
            logger.warning("Cookie controller not ready, skipping sync")
            return
        
        is_logged_in_cookie = cookie_controller.get('is_logged_in')
        user_email_cookie = cookie_controller.get('user_email')
        remember_me_cookie = cookie_controller.get('remember_me')
        
        # Validasi data cookie sebelum sync
        if is_logged_in_cookie == 'True' and user_email_cookie:
            # Validasi format email dari cookie
            is_valid_email, _ = validate_email_format(user_email_cookie)
            if is_valid_email:
                st.session_state['logged_in'] = True
                st.session_state['user_email'] = user_email_cookie
                if remember_me_cookie == 'True':
                    st.session_state['remember_me'] = True
                logger.info(f"Login state synced from cookies for user: {user_email_cookie}")
            else:
                logger.warning(f"Invalid email format in cookie: {user_email_cookie}")
                # Clear invalid cookies
                clear_remember_me_cookies()
                st.session_state['logged_in'] = False
        else:
            st.session_state['logged_in'] = False
            
    except Exception as e:
        logger.error(f"Error syncing login state: {e}")
        st.session_state['logged_in'] = False
        # Clear potentially corrupted cookies
        try:
            clear_remember_me_cookies()
        except:
            pass

def set_remember_me_cookies(email: str, remember: bool = False) -> None:
    """Set cookies untuk fungsionalitas 'ingat saya'"""
    try:
        if remember:
            # Set cookies dengan masa berlaku yang dikonfigurasi
            cookie_controller.set('is_logged_in', 'True', max_age=Config.REMEMBER_ME_DURATION)
            cookie_controller.set('user_email', email, max_age=Config.REMEMBER_ME_DURATION)
            cookie_controller.set('remember_me', 'True', max_age=Config.REMEMBER_ME_DURATION)
            cookie_controller.set('last_email', email, max_age=Config.LAST_EMAIL_DURATION)
        else:
            # Set session cookies (berakhir saat browser ditutup)
            cookie_controller.set('is_logged_in', 'True')
            cookie_controller.set('user_email', email)
            cookie_controller.set('remember_me', 'False')
            
    except Exception as e:
        logger.error(f"Error setting cookies: {e}")

def get_remembered_email() -> str:
    """Dapatkan email terakhir yang diingat untuk kemudahan pengguna"""
    try:
        remembered_email = cookie_controller.get('last_email') or ""
        # Validasi email yang diingat
        if remembered_email:
            is_valid, _ = validate_email_format(remembered_email)
            if is_valid:
                return remembered_email
            else:
                logger.warning(f"Invalid remembered email format: {remembered_email}")
                # Clear invalid email
                try:
                    cookie_controller.remove('last_email')
                except:
                    pass
                return ""
        return ""
    except Exception as e:
        logger.error(f"Error getting remembered email: {e}")
        return ""

def is_app_ready() -> bool:
    """Check apakah aplikasi sudah siap untuk digunakan"""
    return (
        st.session_state.get('firebase_initialized', False) and
        st.session_state.get('firebase_auth') is not None and
        st.session_state.get('firestore') is not None
    )

def clear_remember_me_cookies() -> None:
    """Bersihkan semua cookies terkait autentikasi"""
    try:
        cookie_controller.remove('is_logged_in')
        cookie_controller.remove('user_email')
        cookie_controller.remove('remember_me')
    except Exception as e:
        logger.error(f"Error clearing cookies: {e}")

# =============================================================================
# =============================================================================
# CENTRALIZED ERROR HANDLING
# =============================================================================

def handle_firebase_error(error: Exception, context: str = "") -> Tuple[str, str]:
    """
    Centralized Firebase error handling dengan pesan yang konsisten
    
    Args:
        error: Exception yang terjadi
        context: Konteks dimana error terjadi (login, register, reset, dll)
    
    Returns:
        Tuple[toast_message, detailed_message]
    """
    error_str = str(error).upper()
    
    # Firebase Authentication Errors
    error_mappings = {
        "INVALID_EMAIL": ("Format email tidak valid", "Format email tidak valid. Periksa kembali alamat email Anda."),
        "USER_NOT_FOUND": _get_user_not_found_message(context),
        "INVALID_LOGIN_CREDENTIALS": _get_invalid_credentials_message(context),
        "WRONG_PASSWORD": ("Kata sandi salah", "Kata sandi salah. Silakan coba lagi atau reset kata sandi."),
        "INVALID_PASSWORD": ("Kata sandi salah", "Kata sandi salah. Silakan coba lagi atau reset kata sandi."),
        "USER_DISABLED": ("Akun dinonaktifkan", "Akun Anda telah dinonaktifkan. Hubungi administrator."),
        "EMAIL_EXISTS": ("Email sudah terdaftar", "Email ini sudah terdaftar. Silakan gunakan email lain atau login dengan akun yang ada."),
        "EMAIL_ALREADY_IN_USE": ("Email sudah terdaftar", "Email ini sudah terdaftar. Silakan gunakan email lain atau login dengan akun yang ada."),
        "WEAK_PASSWORD": ("Kata sandi terlalu lemah", "Kata sandi terlalu lemah. Gunakan minimal 8 karakter dengan kombinasi huruf besar, kecil, angka dan simbol."),
        "TOO_MANY_REQUESTS": _get_too_many_requests_message(context),
        "NETWORK_REQUEST_FAILED": ("Koneksi bermasalah", "Koneksi internet bermasalah. Periksa koneksi Anda dan coba lagi."),
        "QUOTA_EXCEEDED": ("Batas tercapai", "Batas pengiriman email Firebase tercapai. Coba lagi nanti."),
        "INVALID_ID_TOKEN": ("Token tidak valid", "Token tidak valid. Silakan registrasi ulang."),
        "OPERATION_NOT_ALLOWED": ("Operasi tidak diizinkan", "Operasi tidak diizinkan. Hubungi administrator.")
    }
    
    # Cari error yang cocok
    for error_key, messages in error_mappings.items():
        if error_key in error_str:
            if isinstance(messages, tuple):
                return messages
            else:
                return messages()  # Untuk function yang mengembalikan tuple
    
    # Generic error jika tidak ada yang cocok
    return f"{context.title()} gagal", f"{context.title()} gagal: {str(error)}"

def _get_user_not_found_message(context: str) -> Tuple[str, str]:
    """Helper function untuk pesan user not found berdasarkan context"""
    if context == "login":
        return "Email tidak terdaftar", "Email tidak terdaftar. Silakan daftar terlebih dahulu atau periksa ejaan."
    elif context == "reset":
        return "Akun tidak ditemukan", "Tidak ada akun yang ditemukan dengan alamat email ini."
    else:
        return "User tidak ditemukan", "User tidak ditemukan dalam sistem."

def _get_invalid_credentials_message(context: str) -> Tuple[str, str]:
    """Helper function untuk pesan invalid credentials berdasarkan context"""
    if context == "login":
        return "Email tidak terdaftar", "Email tidak terdaftar dalam sistem kami. Silakan daftar terlebih dahulu atau periksa ejaan email Anda."
    else:
        return "Kredensial tidak valid", "Kredensial login tidak valid. Periksa email dan kata sandi Anda."

def _get_too_many_requests_message(context: str) -> Tuple[str, str]:
    """Helper function untuk pesan too many requests berdasarkan context"""
    if context == "login":
        return "Terlalu banyak percobaan", "Terlalu banyak percobaan login. Tunggu beberapa menit sebelum mencoba lagi."
    elif context == "reset":
        return "Terlalu banyak permintaan", "Terlalu banyak permintaan reset password. Tunggu beberapa menit sebelum mencoba lagi."
    else:
        return "Terlalu banyak permintaan", "Terlalu banyak permintaan. Tunggu beberapa saat sebelum mencoba lagi."

def show_error_with_context(error: Exception, context: str, progress_container: Any = None, message_container: Any = None) -> None:
    """
    Tampilkan error dengan konteks dan UI feedback yang konsisten
    
    Args:
        error: Exception yang terjadi
        context: Konteks error (login, register, reset)
        progress_container: Container untuk progress (opsional)
        message_container: Container untuk pesan (opsional)
    """
    toast_msg, detailed_msg = handle_firebase_error(error, context)
    
    # Clear progress jika ada
    if progress_container:
        try:
            progress_container.empty()
        except Exception as e:
            logger.warning(f"Failed to clear progress container: {e}")
    
    # Tampilkan pesan error dengan fallback
    try:
        if message_container:
            message_container.error(f"❌ {detailed_msg}")
        else:
            st.error(f"❌ {detailed_msg}")
    except Exception as e:
        logger.error(f"Failed to display error message: {e}")
        # Ultimate fallback
        st.write(f"❌ {detailed_msg}")
    
    # Tampilkan toast
    show_error_toast(toast_msg)
    
    # Log error dengan context
    logger.error(f"{context.title()} failed: {str(error)}")

def handle_validation_errors(errors: list, progress_container: Any = None, message_container: Any = None) -> None:
    """
    Handle validation errors dengan display yang konsisten
    
    Args:
        errors: List of validation error messages
        progress_container: Container untuk progress (opsional)
        message_container: Container untuk pesan (opsional)
    """
    if not errors:
        return
    
    if progress_container:
        progress_container.empty()
    
    combined_errors = "\n".join(errors)
    
    if message_container:
        message_container.error("❌ Validasi data gagal:")
        for error in errors:
            message_container.error(error)
    else:
        st.error("❌ Validasi data gagal:")
        for error in errors:
            st.error(error)
    
    show_error_toast("Data tidak valid")
    logger.warning(f"Validation failed: {combined_errors}")

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_email_format(email: str) -> Tuple[bool, str]:
    """Validasi format email dengan aturan yang komprehensif"""
    if not email:
        return False, "Email tidak boleh kosong"
    
    # Pola email dasar
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        return False, "Format email tidak valid. Contoh: nama@domain.com"
    
    # Pemeriksaan tambahan
    if len(email) > 254:  # Batas RFC 5321
        return False, "Email terlalu panjang (maksimal 254 karakter)"
    
    local_part, domain = email.rsplit('@', 1)
    if len(local_part) > 64:  # Batas RFC 5321
        return False, "Bagian lokal email terlalu panjang (maksimal 64 karakter)"
    
    # Periksa titik berturut-turut
    if '..' in email:
        return False, "Email tidak boleh mengandung titik berturut-turut"
    
    # Periksa jika diawali atau diakhiri dengan titik
    if local_part.startswith('.') or local_part.endswith('.'):
        return False, "Email tidak boleh diawali atau diakhiri dengan titik"
    
    return True, ""

def validate_name_format(name: str, field_name: str) -> Tuple[bool, str]:
    """Validasi format nama"""
    if not name:
        return False, f"{field_name} tidak boleh kosong"
    
    if len(name) < 2:
        return False, f"{field_name} minimal 2 karakter"
    
    if len(name) > 50:
        return False, f"{field_name} maksimal 50 karakter"
    
    # Izinkan huruf, spasi, dan karakter nama umum
    name_pattern = r'^[a-zA-Z\s\'-]+$'
    if not re.match(name_pattern, name):
        return False, f"{field_name} hanya boleh mengandung huruf, spasi, apostrof, dan tanda hubung"
    
    return True, ""

def validate_password(password: str) -> Tuple[bool, str]:
    """Validasi persyaratan kekuatan kata sandi"""
    if len(password) < 8:
        return False, "Kata sandi minimal 8 karakter"
    if not any(c.isupper() for c in password):
        return False, "Kata sandi harus mengandung huruf besar"
    if not any(c.islower() for c in password):
        return False, "Kata sandi harus mengandung huruf kecil"  
    if not any(c.isdigit() for c in password):
        return False, "Kata sandi harus mengandung angka"
    return True, ""

# =============================================================================
# SECURITY FUNCTIONS
# =============================================================================

def check_rate_limit(user_email: str) -> bool:
    """Periksa apakah pengguna telah melebihi batas laju untuk percobaan login"""
    now = datetime.now()
    rate_limit_key = f'ratelimit_{user_email}'
    attempts = st.session_state.get(rate_limit_key, [])

    # Hapus percobaan di luar jendela
    valid_attempts = [
        attempt for attempt in attempts 
        if (now - attempt) < timedelta(seconds=Config.RATE_LIMIT_WINDOW)
    ]

    if len(valid_attempts) >= Config.MAX_LOGIN_ATTEMPTS:
        return False

    valid_attempts.append(now)
    st.session_state[rate_limit_key] = valid_attempts
    return True

def check_session_timeout() -> bool:
    """Periksa apakah sesi pengguna telah kedaluwarsa"""
    if 'login_time' in st.session_state and st.session_state['login_time']:
        elapsed = (datetime.now() - st.session_state['login_time']).total_seconds()
        if elapsed > Config.SESSION_TIMEOUT:
            logout()
            return False
    return True

def check_email_verification_quota() -> Tuple[bool, str]:
    """Periksa kuota verifikasi email untuk mencegah spam"""
    try:
        now = datetime.now()
        quota_key = 'email_verification_attempts'
        attempts = st.session_state.get(quota_key, [])
        
        # Hapus upaya yang lebih dari 1 jam
        valid_attempts = [
            attempt for attempt in attempts 
            if (now - attempt) < timedelta(hours=1)
        ]
        
        if len(valid_attempts) >= Config.EMAIL_VERIFICATION_LIMIT:
            return False, "Batas pengiriman email tercapai. Silakan coba lagi dalam 1 jam."
        
        valid_attempts.append(now)
        st.session_state[quota_key] = valid_attempts
        return True, ""
        
    except Exception as e:
        logger.error(f"Error checking email quota: {e}")
        return False, "Error checking email quota"

# =============================================================================
# FIREBASE FUNCTIONS
# =============================================================================

def initialize_firebase() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Inisialisasi Firebase Admin SDK dan Pyrebase dengan penanganan error yang lebih baik
    dan logika percobaan ulang. Ambil semua konfigurasi dari st.secrets.
    """
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Cek apakah Firebase sudah diinisialisasi sebelumnya
            if st.session_state.get('firebase_initialized', False):
                firebase_auth = st.session_state.get('firebase_auth')
                firestore_client = st.session_state.get('firestore')
                
                # Validasi bahwa objects masih valid
                if firebase_auth and firestore_client:
                    logger.info("Using existing Firebase initialization")
                    return firebase_auth, firestore_client
                else:
                    logger.warning("Firebase objects invalid, reinitializing...")
                    st.session_state['firebase_initialized'] = False

            # Verifikasi environment dan konfigurasi
            if not verify_environment():
                logger.error("Environment verification failed")
                return None, None
                
            # Periksa konfigurasi Firebase terlebih dahulu
            if "firebase" not in st.secrets:
                logger.critical("Firebase configuration not found in secrets")
                st.error("""
                *🔥 Konfigurasi Firebase tidak ditemukan!*
                
                Aplikasi memerlukan konfigurasi Firebase service account untuk beroperasi.
                Hubungi administrator sistem untuk konfigurasi yang diperlukan.
                """)
                return None, None
            
            # Ambil konfigurasi service account dari st.secrets
            service_account = dict(st.secrets["firebase"])
            
            # Periksa field yang diperlukan untuk service account
            required_fields = ["project_id", "client_email", "private_key"]
            missing_fields = [field for field in required_fields if field not in service_account]
            
            if missing_fields:
                logger.critical(f"Missing Firebase config fields: {missing_fields}")
                st.error(f"""
                *🔥 Konfigurasi Firebase tidak lengkap!*
                
                *Field yang diperlukan:* {', '.join(missing_fields)}
                
                Hubungi administrator sistem untuk melengkapi konfigurasi Firebase.
                """)
                return None, None
            
            # Inisialisasi Firebase Admin SDK menggunakan service account dari secrets
            if not firebase_admin._apps:
                cred = credentials.Certificate(service_account)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized successfully from secrets")
            
            # Konfigurasi Pyrebase menggunakan konfigurasi yang terstruktur
            config = Config.get_firebase_config()
            if not config:
                logger.error("Failed to get Firebase configuration")
                return None, None
            
            # Inisialisasi Pyrebase
            pb = pyrebase.initialize_app(config)
            firebase_auth = pb.auth()
            logger.info("Pyrebase initialized successfully")
            
            # Inisialisasi Firestore client
            firestore_client = firestore.client()
            logger.info("Firestore client initialized successfully")
            
            # Simpan ke session state untuk penggunaan selanjutnya
            st.session_state['firebase_auth'] = firebase_auth
            st.session_state['firestore'] = firestore_client
            st.session_state['firebase_initialized'] = True
            
            logger.info("Firebase initialized successfully (from secrets)")
            return firebase_auth, firestore_client
            
        except Exception as e:
            logger.error(f"Firebase initialization attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying Firebase initialization in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                # Gagal setelah semua percobaan
                logger.critical("Failed to initialize Firebase after all retries")
                st.error(f"Gagal menginisialisasi Firebase setelah {max_retries} percobaan. Silakan periksa konfigurasi Anda.")
                
                # Tampilkan detail error untuk debugging
                if "FIREBASE_API_KEY" in str(e):
                    st.error("❌ *FIREBASE_API_KEY* tidak ditemukan atau tidak valid")
                elif "project_id" in str(e):
                    st.error("❌ *project_id* tidak ditemukan dalam konfigurasi Firebase")
                elif "private_key" in str(e):
                    st.error("❌ *private_key* tidak valid dalam konfigurasi Firebase")
                
                return None, None
    
    return None, None

def send_email_verification_safe(firebase_auth: Any, id_token: str, email: str) -> Tuple[bool, str]:
    """Kirim verifikasi email dengan penanganan error yang komprehensif"""
    try:
        # Periksa kuota terlebih dahulu
        can_send, quota_message = check_email_verification_quota()
        if not can_send:
            return False, quota_message
        
        # Kirim verifikasi email
        firebase_auth.send_email_verification(id_token)
        logger.info(f"Email verification sent to: {email}")
        return True, "Email verifikasi berhasil dikirim"
        
    except Exception as e:
        logger.error(f"Failed to send email verification to {email}: {e}")
        toast_msg, detailed_msg = handle_firebase_error(e, "email_verification")
        return False, detailed_msg

def verify_user_exists(user_email: str, firestore_client: Any) -> bool:
    """Verifikasi bahwa pengguna ada dan memiliki data yang valid di Firestore"""
    try:
        firebase_user = auth.get_user_by_email(user_email)
        user_doc = firestore_client.collection('users').document(firebase_user.uid).get()
        
        if user_doc.exists:
            logger.info(f"User {user_email} verified successfully")
            return True
        
        logger.warning(f"User {user_email} has no Firestore data")
        return False

    except auth.UserNotFoundError:
        logger.warning(f"User {user_email} not found in Firebase Auth")
        return False
    except Exception as e:
        logger.error(f"Error verifying user {user_email}: {str(e)}")
        return False

# =============================================================================
# GOOGLE OAUTH FUNCTIONS
# =============================================================================

def get_google_authorization_url() -> str:
    """Hasilkan URL otorisasi Google OAuth dengan cakupan yang diperlukan"""
    base_url = 'https://accounts.google.com/o/oauth2/v2/auth'
    params = {
        'client_id': Config.GOOGLE_CLIENT_ID,
        'redirect_uri': Config.REDIRECT_URI,
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'offline',
        'prompt': 'consent'
    }
    return f"{base_url}?{urlencode(params)}"

async def exchange_google_token(code: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Tukar kode otorisasi Google untuk informasi pengguna"""
    async with httpx.AsyncClient() as client:
        token_url = 'https://oauth2.googleapis.com/token'
        payload = {
            'client_id': Config.GOOGLE_CLIENT_ID,
            'client_secret': Config.GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': Config.REDIRECT_URI
        }

        try:
            # Tukar kode untuk token
            token_response = await client.post(token_url, data=payload)
            token_data = token_response.json()
            
            if 'access_token' not in token_data:
                logger.error(f"Token exchange failed: {token_data}")
                return None, None
            
            # Gunakan token untuk mendapatkan info pengguna
            user_info_url = f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={token_data['access_token']}"
            user_response = await client.get(user_info_url)
            user_info = user_response.json()
            
            if 'email' not in user_info:
                logger.error(f"User info incomplete: {user_info}")
                return None, None
                
            return user_info['email'], user_info

        except Exception as e:
            logger.error(f"Google token exchange error: {e}")
            return None, None

def handle_google_login_callback() -> bool:
    """Tangani callback Google OAuth setelah autentikasi pengguna"""
    try:
        if 'code' not in st.query_params:
            return True  # Tidak ada callback Google, lanjutkan normal
            
        code = st.query_params.get('code')
        if not code or not isinstance(code, str):
            st.error("Kode otorisasi Google tidak valid")
            return False

        async def async_token_exchange():
            return await exchange_google_token(code)

        user_email, user_info = asyncio.run(async_token_exchange())
        if not user_email or not user_info:
            st.error("Gagal mendapatkan informasi pengguna dari Google")
            return False

        # Verifikasi pengguna ada di sistem
        firebase_auth, firestore_client = initialize_firebase()
        if not firebase_auth or not firestore_client:
            st.error("Gagal menginisialisasi Firebase")
            return False
            
        try:
            # Cek apakah user sudah terdaftar
            firebase_user = auth.get_user_by_email(user_email)
            user_doc = firestore_client.collection('users').document(firebase_user.uid).get()
            
            if user_doc.exists:
                # User ada, cek verifikasi email untuk keamanan ekstra
                user_data = user_doc.to_dict()
                is_google_user = user_data.get('auth_provider') == 'google'
                is_email_verified = user_data.get('email_verified', False)
                
                # User Google atau email sudah verified, login berhasil
                if is_google_user or is_email_verified:
                    # Update status verifikasi untuk user Google jika belum ter-set
                    if is_google_user and not is_email_verified:
                        try:
                            firestore_client.collection('users').document(firebase_user.uid).update({
                                'email_verified': True,
                                'last_login': datetime.now().isoformat()
                            })
                            logger.info(f"Updated email verification status for Google user: {user_email}")
                        except Exception as update_error:
                            logger.warning(f"Failed to update email verification for Google user {user_email}: {update_error}")
                    
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = user_email
                    st.session_state['login_time'] = datetime.now()
                    set_remember_me_cookies(user_email, True)
                    
                    logger.info(f"Google login successful for: {user_email}")
                    st.success("Login Google berhasil!")
                    st.rerun()
                    return True
                else:
                    # Email belum diverifikasi untuk user non-Google
                    show_warning_toast("Akses Ditolak: Email Belum Diverifikasi")
                    st.warning("📧 **Email Anda belum diverifikasi!**\n\n"
                              "Silakan periksa kotak masuk email Anda dan klik link verifikasi.")
                    return False
            else:
                # User tidak ada di Firestore, arahkan ke registrasi
                st.session_state['google_auth_error'] = True
                st.session_state['google_auth_email'] = user_email
                st.query_params.clear()
                return False

        except auth.UserNotFoundError:
            # User tidak ada di Firebase Auth, arahkan ke registrasi
            st.session_state['google_auth_error'] = True
            st.session_state['google_auth_email'] = user_email
            st.query_params.clear()
            return False

    except Exception as e:
        logger.error(f"Google login callback error: {str(e)}")
        st.error("Terjadi kesalahan saat memproses login Google")
        if 'logged_in' in st.session_state:
            st.session_state['logged_in'] = False
        return False

# =============================================================================
# AUTHENTICATION FUNCTIONS
# =============================================================================

def sync_email_verified_to_firestore(firebase_auth, firestore_client, user):
    """Ambil status email_verified dari Firebase Auth dan update ke Firestore."""
    try:
        # Refresh token untuk mendapatkan data terbaru
        firebase_auth.refresh(user['refreshToken'])
        user_info = firebase_auth.get_account_info(user['idToken'])
        email_verified = user_info['users'][0].get('emailVerified', False)
        
        # Update status ke Firestore
        firestore_client.collection('users').document(user['localId']).update({'email_verified': email_verified})
        
        logger.info(f"Email verification status synced for user {user.get('email', 'unknown')}: {email_verified}")
        return email_verified
    except Exception as e:
        logger.error(f"Gagal sync email_verified: {e}")
        return False

def login_user(email: str, password: str, firebase_auth: Any, firestore_client: Any, 
               remember: bool, progress_container: Any, message_container: Any) -> bool:
    """Proses login pengguna dengan feedback yang ditampilkan di lokasi yang konsisten"""
    
    # Validasi format email
    is_valid_email, email_message = validate_email_format(email)
    if not is_valid_email:
        progress_container.empty()
        show_error_toast("Format email tidak valid")
        message_container.error(email_message)
        return False
    
    # Cek rate limiting
    if not check_rate_limit(email):
        progress_container.empty()
        show_error_toast("Terlalu banyak percobaan")
        message_container.error("Terlalu banyak percobaan login. Silakan coba lagi nanti.")
        return False
    
    try:
        # Step 1: Validating credentials
        progress_container.progress(0.2)
        message_container.caption("🔐 Memvalidasi kredensial...")
        
        # Coba login dengan Firebase
        user = firebase_auth.sign_in_with_email_and_password(email, password)
        
        # Step 2: Checking email verification status
        progress_container.progress(0.5)
        message_container.caption("� Memeriksa status verifikasi email...")
        
        # Sync dan cek status verifikasi email dari Firebase Auth
        email_verified = sync_email_verified_to_firestore(firebase_auth, firestore_client, user)
        
        if not email_verified:
            progress_container.empty()
            show_warning_toast("Email belum diverifikasi")
            message_container.warning(
                "📧 **Email Anda belum diverifikasi!**\n\n"
                "Silakan periksa kotak masuk email Anda dan klik link verifikasi yang telah dikirim. "
                "Setelah verifikasi, silakan coba login kembali.\n\n"
                "💡 *Tip: Periksa juga folder spam/junk email*"
            )
            return False
        
        # Step 3: Verifying user data
        progress_container.progress(0.7)
        message_container.caption("👤 Memverifikasi data pengguna...")
        
        # Verifikasi pengguna ada di Firestore
        if not verify_user_exists(email, firestore_client):
            progress_container.empty()
            show_error_toast("Data pengguna tidak ditemukan")
            message_container.error("Data pengguna tidak ditemukan di sistem. Silakan hubungi administrator.")
            return False
        # Step 3: Setting up session
        progress_container.progress(0.9)
        message_container.caption("⚙️ Menyiapkan sesi pengguna...")
        
        # Set status login
        st.session_state['logged_in'] = True
        st.session_state['user_email'] = email
        st.session_state['login_time'] = datetime.now()
        st.session_state['login_attempts'] = 0
        
        # Set cookies
        set_remember_me_cookies(email, remember)
        
        # Step 4: Complete
        progress_container.progress(1.0)
        message_container.caption("✅ Login berhasil!")
        message_container.success("🎉 Login berhasil! Selamat datang kembali!")
        
        logger.info(f"Login successful for: {email}")
        show_success_toast("Login berhasil! Selamat datang kembali!")
        time.sleep(1.2)  # Beri waktu untuk menampilkan progress completion
        return True
        
    except Exception as e:
        st.session_state['login_attempts'] = st.session_state.get('login_attempts', 0) + 1
        
        # Enhanced error handling untuk memberikan pesan yang lebih spesifik
        error_str = str(e).upper()
        
        # Khusus untuk INVALID_LOGIN_CREDENTIALS, berikan pesan seperti Google OAuth
        if "INVALID_LOGIN_CREDENTIALS" in error_str:
            progress_container.empty()
            
            # Tampilkan pesan error langsung di message_container
            show_error_toast(f"Email {email} tidak terdaftar dalam sistem kami")
            message_container.error(
                f"**Akun Email Tidak Terdaftar**\n\n"
                f"Email {email} belum terdaftar dalam sistem kami."
            )
            message_container.info(
                f"💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' "
                f"atau periksa ejaan email Anda."
            )
            
            # Log error
            logger.warning(f"Login failed - email not registered: {email}")
            return False
        else:
            # Gunakan centralized error handling untuk error lainnya
            show_error_with_context(e, "login", progress_container, message_container)
            return False

def register_user(first_name: str, last_name: str, email: str, password: str, 
                 firebase_auth: Any, firestore_client: Any, is_google: bool, 
                 progress_container: Any, message_container: Any) -> Tuple[bool, str]:
    """Proses registrasi pengguna dengan feedback yang ditampilkan di lokasi yang konsisten"""
    
    # Step 1: Input validation
    progress_container.progress(0.1)
    message_container.caption("📝 Memvalidasi data input...")
    
    # Validasi input
    validation_errors = []
    
    # Validasi nama
    is_valid_fname, fname_message = validate_name_format(first_name.strip(), "Nama Depan")
    if not is_valid_fname:
        validation_errors.append(f"❌ {fname_message}")
        
    is_valid_lname, lname_message = validate_name_format(last_name.strip(), "Nama Belakang")
    if not is_valid_lname:
        validation_errors.append(f"❌ {lname_message}")
    
    # Validasi email
    is_valid_email, email_message = validate_email_format(email.strip())
    if not is_valid_email:
        validation_errors.append(f"❌ {email_message}")
    
    # Validasi password (hanya untuk registrasi non-Google)
    if not is_google:
        is_valid_password, password_message = validate_password(password)
        if not is_valid_password:
            validation_errors.append(f"❌ {password_message}")
    
    if validation_errors:
        handle_validation_errors(validation_errors, progress_container, message_container)
        return False, "\n".join(validation_errors)
    
    try:
        # Step 2: Checking email availability
        progress_container.progress(0.3)
        message_container.caption("📧 Memeriksa ketersediaan email...")
        
        # Cek apakah email sudah terdaftar
        try:
            existing_user = auth.get_user_by_email(email)
            progress_container.empty()
            message_container.error("❌ Email ini sudah terdaftar. Silakan gunakan email lain atau login dengan akun yang ada.")
            show_error_toast("Email sudah terdaftar")
            return False, "❌ Email ini sudah terdaftar. Silakan gunakan email lain atau login dengan akun yang ada."
        except auth.UserNotFoundError:
            pass  # Email belum terdaftar, lanjutkan
        
        # Step 3: Creating Firebase account
        progress_container.progress(0.5)
        message_container.caption("🔐 Membuat akun Firebase...")
        
        # Buat user di Firebase Auth
        if is_google:
            auto_password = f"Google-{uuid.uuid4().hex[:16]}"
            user = firebase_auth.create_user_with_email_and_password(email, auto_password)
        else:
            user = firebase_auth.create_user_with_email_and_password(email, password)
        
        # Step 4: Email verification
        progress_container.progress(0.7)
        if not is_google:
            message_container.caption("📬 Mengirim email verifikasi...")
        else:
            message_container.caption("✅ Memproses akun Google...")
        
        # Kirim email verifikasi untuk registrasi non-Google
        email_verification_sent = False
        if not is_google:
            verification_success, verification_message = send_email_verification_safe(
                firebase_auth, user['idToken'], email
            )
            email_verification_sent = verification_success
        
        # Step 5: Saving user data
        progress_container.progress(0.9)
        message_container.caption("💾 Menyimpan data pengguna...")
        
        # Simpan data user ke Firestore
        user_data = {
            "first_name": first_name.strip(),
            "last_name": last_name.strip(),
            "email": email.strip(),
            "auth_provider": "google" if is_google else "email",
            "created_at": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat(),
            "email_verified": is_google
        }
        
        firestore_client.collection('users').document(user['localId']).set(user_data)
        
        # Step 6: Complete
        progress_container.progress(1.0)
        message_container.caption("✅ Registrasi berhasil!")
        
        if is_google:
            success_message = "🎉 Akun Google berhasil didaftarkan! Anda sekarang dapat login dan menggunakan semua fitur aplikasi."
        else:
            if email_verification_sent:
                success_message = f"✅ Akun berhasil dibuat untuk {email}!\n\n📧 Email verifikasi telah dikirim. Silakan periksa kotak masuk (dan folder spam) untuk mengaktifkan akun Anda."
            else:
                success_message = f"✅ Akun berhasil dibuat untuk {email}! Anda dapat login sekarang dan mulai menggunakan aplikasi."
        
        message_container.success(success_message)
        logger.info(f"Successfully created account for: {email}")
        show_success_toast("Registrasi berhasil")
        time.sleep(1.2)
        
        return True, success_message
                
    except Exception as e:
        show_error_with_context(e, "register", progress_container, message_container)
        return False, f"❌ Pendaftaran gagal: {str(e)}"

def reset_password(email: str, firebase_auth: Any, progress_container: Any, message_container: Any) -> bool:
    """Proses reset password dengan feedback yang ditampilkan di lokasi yang konsisten"""
    
    # Step 1: Input validation
    progress_container.progress(0.2)
    message_container.caption("📝 Memvalidasi alamat email...")
    
    # Validasi format email
    is_valid_email, email_message = validate_email_format(email.strip())
    if not is_valid_email:
        progress_container.empty()
        message_container.error(f"❌ {email_message}")
        show_error_toast("Format email tidak valid")
        return False
    
    # Step 2: Rate limiting check
    progress_container.progress(0.4)
    message_container.caption("🔒 Memeriksa batas permintaan...")
    
    # Cek rate limiting
    if not check_rate_limit(f"reset_{email}"):
        progress_container.empty()
        message_container.error("⚠️ Terlalu banyak percobaan reset password. Silakan tunggu 5 menit sebelum mencoba lagi.")
        show_warning_toast("Terlalu banyak percobaan reset")
        return False
    
    try:
        # Step 3: Checking user existence
        progress_container.progress(0.6)
        message_container.caption("👤 Memeriksa keberadaan akun...")
        
        # Cek apakah user ada
        try:
            auth.get_user_by_email(email)
        except auth.UserNotFoundError:
            progress_container.empty()
            message_container.error("❌ Tidak ada akun yang ditemukan dengan alamat email ini.")
            show_error_toast("❌ Akun tidak ditemukan!")
            return False
        
        # Step 4: Sending reset email
        progress_container.progress(0.8)
        message_container.caption("📧 Mengirim email reset password...")
        
        # Kirim email reset password
        firebase_auth.send_password_reset_email(email)
        logger.info(f"Password reset email sent to: {email}")
        
        # Step 5: Complete
        progress_container.progress(1.0)
        message_container.caption("✅ Email reset berhasil dikirim!")
        
        success_message = f"📧 **Petunjuk reset password telah dikirim ke {email}**\n\nSilakan periksa kotak masuk email Anda (dan folder spam) untuk link reset password.\n\nLink akan aktif selama 1 jam."
        message_container.success(success_message)
        
        show_success_toast("Link reset password berhasil dikirim")
        time.sleep(1.2)
        return True
        
    except Exception as e:
        show_error_with_context(e, "reset", progress_container, message_container)
        return False

def logout() -> None:
    """Tangani logout pengguna dengan pembersihan sesi"""
    try:
        user_email = st.session_state.get('user_email')
        logger.info(f"Logging out user: {user_email}")
        
        # Simpan objek firebase yang diperlukan
        fb_auth = st.session_state.get('firebase_auth', None)
        fs_client = st.session_state.get('firestore', None)
        fb_initialized = st.session_state.get('firebase_initialized', False)
        
        # Bersihkan session state
        st.session_state.clear()
        
        # Kembalikan objek firebase jika ada
        if fb_auth:
            st.session_state['firebase_auth'] = fb_auth
        if fs_client:
            st.session_state['firestore'] = fs_client
        if fb_initialized:
            st.session_state['firebase_initialized'] = fb_initialized
        
        # Reset status login
        st.session_state['logged_in'] = False
        st.session_state['user_email'] = None
        st.session_state["logout_success"] = True
        
        # Reset model preparation status
        st.session_state['models_prepared'] = False
        st.session_state['model_preparation_started'] = False
        st.session_state['model_preparation_completed'] = False
        st.session_state['ready_for_tools'] = False
        
        # Clear URL params
        st.query_params.clear()
        
        # Clear cookies but keep last_email for convenience
        clear_remember_me_cookies()
        
        logger.info("Logout completed successfully")
        
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        show_error_toast(f"Logout failed: {str(e)}")

# =============================================================================
# UI FEEDBACK FUNCTIONS
# =============================================================================

def show_toast_notification(message: str, icon: str = "ℹ") -> None:
    """Tampilkan notifikasi toast dengan gaya yang konsisten"""
    try:
        st.toast(message, icon=icon)
        logger.debug(f"Toast displayed: {icon} {message}")
    except Exception as e:
        logger.error(f"Failed to show toast: {e}")
        # Fallback: tampilkan sebagai st.info jika toast gagal
        st.info(f"{icon} {message}")

def show_success_toast(message: str) -> None:
    """Tampilkan notifikasi toast sukses"""
    show_toast_notification(message, "✅")

def show_error_toast(message: str) -> None:
    """Tampilkan notifikasi toast error"""
    show_toast_notification(message, "❌")

def show_warning_toast(message: str) -> None:
    """Tampilkan notifikasi toast peringatan"""
    show_toast_notification(message, "⚠️")

def show_info_toast(message: str) -> None:
    """Tampilkan notifikasi toast info"""
    show_toast_notification(message, "ℹ")

def show_loading_toast(message: str) -> None:
    """Tampilkan notifikasi toast loading"""
    show_toast_notification(message, "⏳")

# =============================================================================
# ENHANCED LOADING AND FEEDBACK FUNCTIONS
# =============================================================================

def create_auth_spinner_context(message: str, success_message: str = ""):
    """Context manager untuk spinner autentikasi dengan feedback yang konsisten"""
    class AuthSpinnerContext:
        def __init__(self, msg: str, success_msg: str):
            self.message = msg
            self.success_message = success_msg
            self.spinner = None
            
        def __enter__(self):
            self.spinner = st.spinner(self.message)
            self.spinner.__enter__()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.spinner:
                self.spinner.__exit__(exc_type, exc_val, exc_tb)
            if exc_type is None and self.success_message:
                show_loading_toast(self.success_message)
                time.sleep(0.3)  # Brief pause untuk user feedback
    
    return AuthSpinnerContext(message, success_message)

def display_auth_tips(auth_type: str) -> None:
    """Tampilkan tips berguna berdasarkan jenis autentikasi"""
    tips = {
        "login": [
            "💡 Gunakan fitur 'Ingat Saya' untuk login otomatis",
            "🔒 Pastikan kata sandi Anda aman dan unik",
            "📱 Gunakan login Google untuk kemudahan akses"
        ],
        "register": [
            "📧 Periksa email spam jika verifikasi tidak diterima",
            "🔐 Gunakan kata sandi yang kuat: 8+ karakter, angka, simbol",
            "✅ Login Google lebih cepat dan aman"
        ],
        "reset": [
            "📧 Link reset berlaku selama 1 jam",
            "🗂️ Periksa folder spam/junk email",
            "⏰ Tunggu 5 menit sebelum meminta link baru"
        ]
    }
    
    if auth_type in tips:
        with st.expander("💡 Tips Berguna", expanded=False):
            for tip in tips[auth_type]:
                st.markdown(f"• {tip}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def display_config_debug_info() -> None:
    """Tampilkan informasi debug konfigurasi (hanya untuk development)"""
    # Aktifkan debug mode jika query parameter ?debug=1 ada
    if st.query_params.get('debug') == '1':
        st.session_state['debug_mode'] = True
    
    if st.session_state.get('debug_mode', False):
        with st.expander("🔧 Configuration Debug", expanded=False):
            debug_info = {
                "config_valid": Config.is_valid(),
                "config_type": type(Config).__name__,
                "session_timeout": Config.SESSION_TIMEOUT,
                "max_login_attempts": Config.MAX_LOGIN_ATTEMPTS,
                "rate_limit_window": Config.RATE_LIMIT_WINDOW,
                "email_verification_limit": Config.EMAIL_VERIFICATION_LIMIT,
                "remember_me_duration_days": Config.REMEMBER_ME_DURATION // (24*60*60),
                "last_email_duration_days": Config.LAST_EMAIL_DURATION // (24*60*60),
                "firebase_initialized": st.session_state.get('firebase_initialized', False),
                "logged_in": st.session_state.get('logged_in', False),
                "user_email": st.session_state.get('user_email', 'None'),
                "login_attempts": st.session_state.get('login_attempts', 0),
                "has_google_client_id": bool(Config.GOOGLE_CLIENT_ID),
                "has_firebase_api_key": bool(Config.FIREBASE_API_KEY),
                "has_redirect_uri": bool(Config.REDIRECT_URI)
            }
            st.json(debug_info)
            
            # Tombol untuk clear debug mode
            if st.button("Clear Debug Mode"):
                st.session_state['debug_mode'] = False
                st.rerun()

def validate_app_configuration() -> Tuple[bool, list]:
    """
    Validasi konfigurasi aplikasi secara menyeluruh
    
    Returns:
        Tuple[is_valid, list_of_issues]
    """
    issues = []
    
    # Cek konfigurasi dasar
    if not Config.is_valid():
        issues.append("Konfigurasi Firebase/Google OAuth tidak lengkap")
    
    # Cek Firebase configuration
    firebase_config = Config.get_firebase_config()
    if not firebase_config:
        issues.append("Konfigurasi Firebase tidak dapat dimuat")
    
    # Cek secrets yang diperlukan
    required_secrets = ["firebase", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "REDIRECT_URI", "FIREBASE_API_KEY"]
    for secret in required_secrets:
        if secret not in st.secrets:
            issues.append(f"Secret '{secret}' tidak ditemukan")
    
    # Cek konfigurasi numerik
    numeric_configs = [
        ("SESSION_TIMEOUT", Config.SESSION_TIMEOUT),
        ("MAX_LOGIN_ATTEMPTS", Config.MAX_LOGIN_ATTEMPTS),
        ("RATE_LIMIT_WINDOW", Config.RATE_LIMIT_WINDOW),
        ("EMAIL_VERIFICATION_LIMIT", Config.EMAIL_VERIFICATION_LIMIT)
    ]
    
    for name, value in numeric_configs:
        if not isinstance(value, (int, float)) or value <= 0:
            issues.append(f"Konfigurasi {name} tidak valid: {value}")
    
    return len(issues) == 0, issues

def get_config_summary() -> Dict[str, Any]:
    """
    Dapatkan ringkasan konfigurasi untuk logging dan debugging
    
    Returns:
        Dictionary dengan informasi konfigurasi
    """
    return {
        "config_class": type(Config).__name__,
        "is_valid": Config.is_valid(),
        "timeout_settings": {
            "session_timeout_hours": Config.SESSION_TIMEOUT / 3600,
            "rate_limit_window_minutes": Config.RATE_LIMIT_WINDOW / 60,
            "remember_me_duration_days": Config.REMEMBER_ME_DURATION / (24*60*60)
        },
        "limits": {
            "max_login_attempts": Config.MAX_LOGIN_ATTEMPTS,
            "email_verification_limit": Config.EMAIL_VERIFICATION_LIMIT
        },
        "oauth_configured": bool(Config.GOOGLE_CLIENT_ID and Config.GOOGLE_CLIENT_SECRET),
        "firebase_configured": bool(Config.FIREBASE_API_KEY),
        "redirect_configured": bool(Config.REDIRECT_URI)
    }

# =============================================================================
# UI COMPONENTS
# =============================================================================

def display_login_form(firebase_auth: Any, firestore_client: Any) -> None:
    """Tampilkan dan tangani formulir login"""
    
    # Check app readiness
    app_ready = is_app_ready()
    
    with st.form("login_form", clear_on_submit=False):
        st.markdown("### Masuk")

        # Input email dengan nilai yang diingat
        remembered_email = get_remembered_email()
        email = st.text_input(
            "Email",
            value=remembered_email,
            placeholder="email.anda@contoh.com",
            help="Masukkan alamat email terdaftar Anda",
            disabled=not app_ready
        )

        # Validasi email secara real-time (dengan debouncing dan app readiness check)
        if app_ready and email and email.strip() and email != remembered_email and len(email.strip()) > 5:
            is_valid_email, email_message = validate_email_format(email.strip())
            if not is_valid_email:
                st.error(f"❌ {email_message}")

        # Input password
        password = st.text_input(
            "Kata Sandi",
            type="password",
            placeholder="Masukkan kata sandi Anda",
            help="Masukkan kata sandi yang aman",
            disabled=not app_ready
        )

        # Checkbox remember me
        col1, col2 = st.columns([1, 2])
        with col1:
            remember = st.checkbox(
                "Ingat saya", 
                value=True, 
                help=f"Simpan login selama {Config.REMEMBER_ME_DURATION // (24*60*60)} hari",
                disabled=not app_ready
            )

        # Status app readiness
        if not app_ready:
            st.info("🔄 Sistem sedang mempersiapkan diri... Mohon tunggu sebentar.")

        # Tombol login email
        email_login_clicked = st.form_submit_button(
            "Lanjutkan dengan Email", 
            use_container_width=True, 
            type="primary",
            disabled=not app_ready
        )

        # Divider
        st.markdown("""
            <div class='auth-divider-custom'>
                <div class='divider-line-custom'></div>
                <span class='divider-text-custom'>ATAU</span>
                <div class='divider-line-custom'></div>
            </div>
        """, unsafe_allow_html=True)

        # Tombol login Google
        google_login_clicked = st.form_submit_button(
            "Lanjutkan dengan Google", 
            use_container_width=True, 
            type="primary",
            disabled=not app_ready
        )
        
        # Placeholder untuk pesan feedback dan progress di bawah tombol Google
        feedback_placeholder = st.empty()

    # Tampilkan pesan error Google OAuth jika ada - di placeholder yang konsisten
    if st.session_state.get('google_auth_error', False):
        email_error = st.session_state.get('google_auth_email', '')
        with feedback_placeholder.container():
            st.error(f"**Akun Google Tidak Terdaftar**\n\n"
                    f"Akun Google {email_error} belum terdaftar dalam sistem kami.")
            st.info(f"💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' atau gunakan akun email yang sudah terdaftar.")
        show_error_toast(f"Akun Google {email_error} tidak terdaftar dalam sistem kami.")
        del st.session_state['google_auth_error']
        if 'google_auth_email' in st.session_state:
            del st.session_state['google_auth_email']

    # Handle tombol login email di luar form
    if email_login_clicked:
        if email and password:
            # Validasi ulang email sebelum proses login
            email_clean = email.strip()
            is_valid_email, email_message = validate_email_format(email_clean)
            
            if not is_valid_email:
                with feedback_placeholder.container():
                    st.error(f"❌ {email_message}")
                    show_error_toast("Format email tidak valid")
                return
            
            # Pastikan Firebase sudah siap
            if not firebase_auth or not firestore_client:
                with feedback_placeholder.container():
                    st.error("❌ Sistem belum siap. Silakan tunggu beberapa detik dan coba lagi.")
                    show_error_toast("Sistem belum siap")
                return
            
            # Simpan email terakhir untuk kemudahan
            try:
                cookie_controller.set('last_email', email_clean, max_age=Config.LAST_EMAIL_DURATION)
            except Exception as e:
                logger.warning(f"Failed to save last email: {e}")
            
            # Tampilkan progress di placeholder yang konsisten
            with feedback_placeholder.container():
                progress_container = st.empty()
                message_container = st.empty()
                
                # Progress indicator
                progress_container.progress(0.1)
                message_container.caption("🔐 Memulai proses login...")
                
                # Proses login dengan email yang sudah divalidasi
                try:
                    result = login_user(email_clean, password, firebase_auth, firestore_client, remember, progress_container, message_container)
                    if result:
                        st.rerun()
                except Exception as login_error:
                    # Fallback error handling jika login_user tidak menampilkan error
                    progress_container.empty()
                    logger.error(f"Login process failed: {login_error}")
                    
                    error_str = str(login_error).upper()
                    if "INVALID_LOGIN_CREDENTIALS" in error_str:
                        show_error_toast(f"Email {email_clean} tidak terdaftar dalam sistem kami")
                        message_container.error(
                            f"**Akun Email Tidak Terdaftar**\n\n"
                            f"Email {email_clean} belum terdaftar dalam sistem kami."
                        )
                        message_container.info(
                            f"💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' "
                            f"atau periksa ejaan email Anda."
                        )
                    else:
                        show_error_toast("Login gagal")
                        message_container.error(f"❌ Login gagal: {str(login_error)}")
        else:
            with feedback_placeholder.container():
                st.warning("⚠️ Silakan isi kolom email dan kata sandi.")
                show_warning_toast("Silakan isi kolom email dan kata sandi.")

    # Handle tombol login Google di luar form  
    if google_login_clicked:
        with feedback_placeholder.container():
            progress_container = st.empty()
            message_container = st.empty()
            
            progress_container.progress(0.1)
            message_container.caption("🔗 Mengalihkan ke Google OAuth...")
            
            try:
                google_url = get_google_authorization_url()
                progress_container.progress(0.8)
                message_container.caption("✅ Berhasil mengalihkan ke Google...")
                st.markdown(f'<meta http-equiv="refresh" content="0; url={google_url}">', unsafe_allow_html=True)
                time.sleep(1)  # Beri waktu untuk redirect
            except Exception as e:
                logger.error(f"Google OAuth redirect failed: {e}")
                progress_container.empty()
                message_container.error("❌ Gagal mengalihkan ke Google. Silakan coba lagi.")
                show_error_toast("❌ Gagal mengalihkan ke Google. Silakan coba lagi.")
    
    # Tampilkan tips untuk login
    display_auth_tips("login")

def display_register_form(firebase_auth: Any, firestore_client: Any) -> None:
    """Tampilkan dan tangani formulir registrasi pengguna"""
    
    google_email = st.session_state.get('google_auth_email', '')
    
    # Inisialisasi data formulir di state sesi
    if 'register_form_data' not in st.session_state:
        st.session_state['register_form_data'] = {
            'first_name': '',
            'last_name': '',
            'email': google_email,
            'terms_accepted': False
        }
    
    # Perbarui email jika google_email diset
    if google_email and st.session_state['register_form_data']['email'] != google_email:
        st.session_state['register_form_data']['email'] = google_email

    with st.form("register_form", clear_on_submit=False):
        st.markdown("### Daftar")

        # Input nama
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input(
                "Nama Depan", 
                value=st.session_state['register_form_data']['first_name'],
                placeholder="John"
            )
                    
        with col2:
            last_name = st.text_input(
                "Nama Belakang", 
                value=st.session_state['register_form_data']['last_name'],
                placeholder="Doe"
            )

        # Input email
        email = st.text_input(
            "Email",
            value=st.session_state['register_form_data']['email'],
            placeholder="email.anda@contoh.com",
            help="Kami akan mengirimkan link verifikasi ke email ini"
        )

        # Input password (hanya untuk non-Google)
        if not google_email:
            col3, col4 = st.columns(2)
            with col3:
                password = st.text_input(
                    "Kata Sandi",
                    type="password",
                    placeholder="Buat kata sandi yang kuat",
                    help="Gunakan 8+ karakter dengan campuran huruf, angka & simbol"
                )
                        
            with col4:
                confirm_password = st.text_input(
                    "Konfirmasi Kata Sandi",
                    type="password",
                    placeholder="Masukkan ulang kata sandi"
                )
        else:
            password = st.text_input(
                "Kata Sandi (Dibuat otomatis untuk akun Google)",
                type="password",
                value=f"Google-{uuid.uuid4().hex[:8]}",
                disabled=True
            )
            confirm_password = password
            st.info("Karena Anda mendaftar dengan akun Google, kami akan mengelola kata sandi dengan aman.")

        # Checkbox syarat layanan
        terms = st.checkbox(
            "Saya setuju dengan Syarat Layanan dan Kebijakan Privasi",
            value=st.session_state['register_form_data']['terms_accepted']
        )
        
        button_text = "Daftar dengan Google" if google_email else "Buat Akun"

        register_clicked = st.form_submit_button(button_text, use_container_width=True, type="primary")
        
        # Placeholder untuk pesan feedback dan progress di bawah tombol registrasi
        feedback_placeholder = st.empty()

    # Handle tombol registrasi di luar form
    if register_clicked:
        # Perbarui state sesi dengan nilai formulir saat ini
        st.session_state['register_form_data'].update({
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'terms_accepted': terms
        })
        
        # Validasi dasar
        if not terms:
            with feedback_placeholder.container():
                st.warning("⚠️ Silakan terima Syarat Layanan untuk melanjutkan.")
            show_warning_toast("Silakan terima Syarat Layanan untuk melanjutkan.")
            return

        if not all([first_name, last_name, email, password]):
            with feedback_placeholder.container():
                st.warning("⚠️ Silakan isi semua kolom yang diperlukan.")
            show_error_toast("⚠️ Silakan isi semua kolom yang diperlukan.")
            return

        if not google_email and password != confirm_password:
            with feedback_placeholder.container():
                st.error("❌ Kata sandi tidak cocok! Silakan periksa kembali.")
            show_error_toast("Kata sandi tidak cocok! Silakan periksa kembali.")
            return
            
        # Proses registrasi dengan progress steps yang konsisten
        with feedback_placeholder.container():
            progress_container = st.empty()
            message_container = st.empty()
            
            # Progress indicator
            progress_container.progress(0.05)
            message_container.caption("📝 Memulai proses registrasi...")
            
            # Proses registrasi tanpa spinner bawaan
            success, message = register_user(
                first_name or "", last_name or "", email or "", password or "", 
                firebase_auth, firestore_client, bool(google_email),
                progress_container, message_container
            )
            
            if success:
                # Hapus data formulir setelah registrasi berhasil
                if 'register_form_data' in st.session_state:
                    del st.session_state['register_form_data']
                
                # Hapus google auth email jika ada
                if 'google_auth_email' in st.session_state:
                    del st.session_state['google_auth_email']
                
                # Simpan status untuk fitur pengiriman ulang
                st.session_state['last_registration_email'] = email
                time.sleep(0.5)  # Beri waktu untuk membaca pesan
    
    # Tampilkan tips untuk registrasi
    display_auth_tips("register")

def display_reset_password_form(firebase_auth: Any) -> None:
    """Tampilkan dan tangani formulir reset kata sandi"""
    
    with st.form("reset_form", clear_on_submit=True):
        st.markdown("### Reset Kata Sandi")
        st.info("Masukkan alamat email Anda di bawah ini dan kami akan mengirimkan petunjuk untuk mereset kata sandi Anda.")

        email = st.text_input(
            "Alamat Email",
            placeholder="email.anda@contoh.com",
            help="Masukkan alamat email yang terkait dengan akun Anda"
        )

        # Validasi email real-time
        if email and email.strip():
            is_valid_email, email_message = validate_email_format(email.strip())
            if not is_valid_email:
                st.error(f"❌ {email_message}")

        reset_clicked = st.form_submit_button("Kirim Link Reset", use_container_width=True, type="primary")
        
        # Placeholder untuk pesan feedback dan progress di bawah tombol reset
        feedback_placeholder = st.empty()

    # Handle tombol reset di luar form
    if reset_clicked:
        if not email or not email.strip():
            with feedback_placeholder.container():
                st.warning("⚠️ Silakan masukkan alamat email Anda.")
            show_warning_toast("Silakan masukkan alamat email Anda.")
            return
            
        # Proses reset password dengan progress steps yang konsisten
        with feedback_placeholder.container():
            progress_container = st.empty()
            message_container = st.empty()
            
            # Progress indicator
            progress_container.progress(0.1)
            message_container.caption("📧 Memulai proses reset password...")
            
            # Proses reset password tanpa spinner bawaan
            result = reset_password(email.strip(), firebase_auth, progress_container, message_container)
    
    # Tampilkan tips untuk reset password
    display_auth_tips("reset")

def tampilkan_header_sambutan():
    """Menampilkan header sambutan dan logo aplikasi"""
    try:
        logo_path = "ui/icon/logo_app.png"
        with open(logo_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
            <div class="welcome-header">
                <img src="data:image/png;base64,{img_base64}" alt="SentimenGo Logo" style="width:170px; display:block; margin:0 auto 1rem auto;">
                <div style="text-align:center; font-size:1.8rem; font-weight:bold; margin-bottom:1rem; color:#2E8B57;">Selamat Datang di SentimenGo!</div>
                <div style="text-align:center; font-size:1rem; color:#666; margin-bottom:2rem;">Sistem Analisis Sentimen GoRide</div>
            </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
            <div class="welcome-header">
                <div style='text-align:center; font-size:2.5rem; margin-bottom:1rem; color:#2E8B57;'>🛵 SentimenGo</div>
                <div style='text-align:center; font-size:1.8rem; font-weight:bold; margin-bottom:1rem; color:#333;'>Selamat Datang!</div>
                <div style="text-align:center; font-size:1rem; color:#666; margin-bottom:2rem;">Sistem Analisis Sentimen GoRide</div>
            </div>
        """, unsafe_allow_html=True)

def tampilkan_pilihan_autentikasi(firebase_auth, firestore_client):
    """Menampilkan selectbox pilihan metode autentikasi"""
    previous_auth_type = st.session_state.get('auth_type', '')
    
    # Tampilkan selectbox untuk memilih metode autentikasi
    auth_type = st.selectbox(
        "Pilih metode autentikasi",
        ["🔐 Masuk", "📝 Daftar", "🔑 Reset Kata Sandi"],
        index=0,
        help="Pilih metode autentikasi Anda"
    )
    
    # Reset form data jika tipe auth berubah
    if previous_auth_type != auth_type:
        st.session_state['auth_type'] = auth_type
        st.session_state['auth_type_changed'] = True
        if 'register_form_data' in st.session_state:
            del st.session_state['register_form_data']
    
    # Tampilkan form sesuai pilihan
    if auth_type == "🔐 Masuk":
        display_login_form(firebase_auth, firestore_client)
    elif auth_type == "📝 Daftar":
        display_register_form(firebase_auth, firestore_client)
    elif auth_type == "🔑 Reset Kata Sandi":
        display_reset_password_form(firebase_auth)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    """Titik masuk utama aplikasi"""
    try:
        # Inisialisasi
        sync_login_state()
        initialize_session_state()
        
        # Validasi konfigurasi aplikasi
        config_valid, config_issues = validate_app_configuration()
        if not config_valid:
            logger.warning(f"Configuration issues detected: {config_issues}")
        
        # Log configuration summary
        config_summary = get_config_summary()
        logger.info(f"Application started with config: {config_summary}")
        
        # Tampilkan debug info jika diminta
        if st.query_params.get('debug') == '1':
            display_config_debug_info()
        
        # CSS Styles
        st.markdown("""
            <style>
            /* Reset dan viewport configuration */

            html, body {
                height: 100vh !important;
                max-height: 100vh !important;
                overflow: hidden !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            
            /* Streamlit container fixes */
            .main .block-container {
                padding-top: 1rem !important;
                padding-bottom: 1rem !important;
                max-height: 100vh !important;
                overflow: hidden !important;
            }
            
            /* Main content area */
            section.main {
                height: 100vh !important;
                max-height: 100vh !important;
                overflow: hidden !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: center !important;
                align-items: center !important;
                padding: 0 !important;
            }
            
            /* Content wrapper untuk memastikan semua konten terlihat */
            .auth-content-wrapper {
                width: 100%;
                max-width: 500px;
                height: auto;
                max-height: 95vh;
                overflow-y: auto;
                overflow-x: hidden;
                padding: 1rem;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            
            /* Welcome header kompak */
            .welcome-header {
                text-align: center;
                margin-bottom: 1rem;
            }
            
            /* Selectbox styling */
            .stSelectbox {
                margin-bottom: 1rem !important;
                width: 100%;
            }
            
            /* Form styling yang lebih kompak */
            div[data-testid="stForm"] {
                border: 1px solid #f0f2f6;
                padding: 1.2rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 0.5rem;
                width: 100%;
                box-sizing: border-box;
            }
            
            /* Button styling */
            .stButton button {
                width: 100%;
                border-radius: 20px;
                height: 2.8rem;
                font-weight: bold;
                margin: 0.3rem 0;
            }
            
            /* Input field spacing */
            .stTextInput {
                margin-bottom: 0.8rem;
            }
            
            /* Column spacing yang lebih rapat */
            .stColumns {
                gap: 0.5rem;
            }
            
            /* Divider custom untuk "ATAU" */
            .auth-divider-custom {
                display: flex;
                align-items: center;
                margin: 1rem 0;
            }
            .divider-line-custom {
                flex: 1;
                height: 1px;
                background: #e0e0e0;
            }
            .divider-text-custom {
                margin: 0 1rem;
                color: #888;
                font-weight: 600;
                letter-spacing: 1px;
                font-size: 0.9rem;
            }
            
            /* Hide scrollbar for webkit browsers */
            .auth-content-wrapper::-webkit-scrollbar {
                width: 4px;
            }
            .auth-content-wrapper::-webkit-scrollbar-track {
                background: transparent;
            }
            .auth-content-wrapper::-webkit-scrollbar-thumb {
                background: #ccc;
                border-radius: 2px;
            }
            
            /* Responsive adjustments */
            @media (max-height: 700px) {
                .welcome-header {
                    margin-bottom: 0.5rem;
                }
                div[data-testid="stForm"] {
                    padding: 1rem;
                }
                .stButton button {
                    height: 2.5rem;
                }
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Inisialisasi Firebase dengan retry dan status feedback
        firebase_auth, firestore_client = None, None
        initialization_container = st.empty()
        
        with initialization_container.container():
            with st.spinner("🔥 Menginisialisasi Firebase..."):
                firebase_auth, firestore_client = initialize_firebase()
        
        # Clear initialization message setelah selesai
        initialization_container.empty()
        
        # Verifikasi Firebase berhasil diinisialisasi
        if firebase_auth and firestore_client:
            logger.info("Firebase successfully initialized and ready")
        else:
            logger.error("Firebase initialization failed")
        
        # Cek status login (hanya jika Firebase tersedia dan user sudah login)
        if firebase_auth and firestore_client and st.session_state.get('logged_in', False) and not st.session_state.get('force_auth_page', False):
            if check_session_timeout():
                user_email = st.session_state.get('user_email')
                if user_email and verify_user_exists(user_email, firestore_client):
                    # User terautentikasi, keluar dari fungsi auth
                    return
                else:
                    logger.warning(f"Pengguna {user_email} gagal verifikasi, melakukan logout paksa")
                    st.error("Masalah autentikasi terdeteksi. Silakan login kembali.")
                    logout()
                    st.session_state['force_auth_page'] = True
                    st.rerun()
          # Tampilkan UI autentikasi dalam container yang tepat
        with st.container():
            st.markdown('<div class="auth-content-wrapper">', unsafe_allow_html=True)
            tampilkan_header_sambutan()
            
            # Handle logout message
            if st.query_params.get("logout") == "1":
                st.toast("Anda telah berhasil logout.", icon="✅")
                st.query_params.clear()
              # Handle Google OAuth callback atau tampilkan form autentikasi
            if firebase_auth and firestore_client:
                # Handle Google OAuth callback jika ada
                handle_google_login_callback()
                
                # Selalu tampilkan pilihan autentikasi jika user belum login
                if not st.session_state.get('logged_in', False):
                    tampilkan_pilihan_autentikasi(firebase_auth, firestore_client)
            else:
                # Firebase tidak tersedia - tampilkan error konfigurasi untuk produksi
                st.error("🔥 *Kesalahan Konfigurasi Firebase*")
                st.error("""
                *Aplikasi tidak dapat berjalan tanpa konfigurasi Firebase yang valid.*
                
                Silakan pastikan:
                • File .streamlit/secrets.toml tersedia dan lengkap
                • Konfigurasi Firebase service account benar
                • Semua kredensial telah dikonfigurasi dengan benar
                
                Hubungi administrator sistem untuk bantuan konfigurasi.
                """)
            
            # Close the content wrapper
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        logger.critical(f"Aplikasi crash: {str(e)}", exc_info=True)
        st.error("Terjadi kesalahan yang tidak terduga. Silakan coba lagi nanti.")
        st.session_state.clear()
        initialize_session_state()
        st.rerun()

if __name__ == "_main_":
    main()