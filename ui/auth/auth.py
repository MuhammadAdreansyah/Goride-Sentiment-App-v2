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
Last Modified: 2025-07-06
"""

import streamlit as st
import streamlit.components.v1 as components
import re
import asyncio
import httpx
import time
import base64
import logging
import secrets
import firebase_admin
import pyrebase
from firebase_admin import credentials, auth, firestore
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
from urllib.parse import urlencode
from streamlit_cookies_controller import CookieController

# Simplified logging configuration
import sys
from pathlib import Path

class StreamlitAuthFormatter(logging.Formatter):
    """Lightweight formatter optimized for auth module - unified format"""
    
    def format(self, record):
        # Simple timestamp formatting only - let log_event handle icons and prefixes
        timestamp = self.formatTime(record, '%H:%M:%S')
        
        # For log_event calls that already have icons, insert timestamp after icon
        message = record.getMessage()
        if message.startswith(('ℹ️', '✅', '⚠️', '❌', '🚨', ' ')):
            # Find the first space after the icon to split properly
            space_index = message.find(' ')
            if space_index > 0:
                icon = message[:space_index]  # Get icon part
                rest = message[space_index + 1:]  # Get everything after first space
                return f"{icon} [{timestamp}] {rest}"
            else:
                # Fallback if no space found
                return f"{message[:2]} [{timestamp}] {message[2:].strip()}"
        else:
            # For direct logger calls (fallback), use minimal format with icon
            level_icons = {'INFO': 'ℹ️', 'WARNING': '⚠️', 'ERROR': '❌', 'CRITICAL': '🚨', 'DEBUG': '🔍'}
            icon = level_icons.get(record.levelname, 'ℹ️')
            return f"{icon} [{timestamp}] {message}"

def setup_auth_logger():
    """Setup optimized logger for auth module"""
    logger = logging.getLogger('auth_module')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Ensure log directory exists
    Path('log').mkdir(exist_ok=True)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(StreamlitAuthFormatter())
    
    # File handler
    file_handler = logging.FileHandler('log/auth.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(funcName)s | %(message)s'
    ))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger

# Initialize logger
logger = setup_auth_logger()

# Configuration constants
SESSION_TIMEOUT = 3600  # 1 hour
MAX_LOGIN_ATTEMPTS = 5
RATE_LIMIT_WINDOW = 300  # 5 minutes
EMAIL_VERIFICATION_LIMIT = 50  # per hour
REMEMBER_ME_DURATION = 30 * 24 * 60 * 60  # 30 days
LAST_EMAIL_DURATION = 90 * 24 * 60 * 60  # 90 days

# UI/UX timing constants
SUCCESS_DISPLAY_DURATION = 1.2  # seconds to show success messages
REDIRECT_PAUSE_DURATION = 0.5   # seconds before redirect
PROGRESS_ANIMATION_DELAY = 0.02  # seconds between progress steps

# Initialize cookie controller
cookie_controller = CookieController()

# =============================================================================
# UNIFIED LOGGING SYSTEM
# =============================================================================

def log_event(category: str, operation: str, level: str = "info", email: str = "", details: str = "", error: Optional[Exception] = None) -> None:
    """Unified logging system for all auth operations
    
    Args:
        category: Type of operation (auth, security, system, firebase, user, session, config)
        operation: Description of the operation
        level: Log level (info, success, warning, error, critical)
        email: User email (optional)
        details: Additional details (optional)
        error: Exception object for error logging (optional)
    """
    # Single primary icon based on level (most important)
    primary_icons = {
        "info": "ℹ️", "success": "✅", "warning": "⚠️", 
        "error": "❌", "critical": " ", "start": " "
    }
    
    # Category prefixes (text-based, no icons)
    category_prefixes = {
        "auth": "AUTHENTICATION", "security": "SECURITY", "system": "SYSTEM", 
        "firebase": "FIREBASE", "user": "USER", "session": "SESSION", "config": "CONFIG"
    }
    
    # Build message components
    icon = primary_icons.get(level, "ℹ️")
    prefix = category_prefixes.get(category, "LOG")
    email_info = f" [{email}]" if email else ""
    details_info = f" | {details}" if details else ""
    error_info = f" | {str(error)}" if error else ""
    
    # Format message: ICON [TIME] PREFIX: operation [email] | details
    message = f"{icon} {prefix}: {operation}{email_info}{details_info}{error_info}"
    
    # Log with appropriate level - message already contains icon
    if level in ["error", "critical"] or error:
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)

# Convenience functions for common logging patterns
def log_auth_event(operation: str, level: str = "info", email: str = "", details: str = "", error: Optional[Exception] = None) -> None:
    """Log authentication events"""
    log_event("auth", operation, level, email, details, error)

def log_security_event(event: str, email: str = "", details: str = "") -> None:
    """Log security events"""
    log_event("security", event, "warning", email, details)

def log_system_event(event: str, details: str = "") -> None:
    """Log system events"""
    log_event("system", event, "info", "", details)

def log_firebase_operation(operation: str, status: str, details: str = "") -> None:
    """Log Firebase operations"""
    level = "success" if status.lower() == "success" else "error"
    log_event("firebase", f"{operation}: {status.upper()}", level, "", details)

def log_user_action(action: str, email: str = "", result: str = "success") -> None:
    """Log user actions"""
    level = "success" if result == "success" else "warning"
    details = result if result != "success" else ""
    log_event("user", f"Action: {action}", level, email, details)

# Backward compatibility wrappers for existing code
def log_auth_start(operation: str, email: str = "") -> None:
    """Log auth start events"""
    log_auth_event(operation, "start", email)

def log_auth_success(operation: str, email: str = "", details: str = "") -> None:
    """Log auth success events"""
    log_auth_event(operation, "success", email, details)

def log_auth_failure(operation: str, email: str = "", reason: str = "") -> None:
    """Log auth failure events"""
    log_auth_event(operation, "warning", email, reason)

def log_auth_error(operation: str, error: Exception, email: str = "") -> None:
    """Log auth error events"""
    log_auth_event(operation, "error", email, error=error)

def log_config_event(event: str, status: str = "INFO") -> None:
    """Log configuration events"""
    level = "error" if status == "ERROR" else "warning" if status == "WARNING" else "info"
    log_event("config", event, level)

def log_session_event(event: str, email: str = "", details: str = "") -> None:
    """Log session events"""
    log_event("session", event, "info", email, details)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_redirect_uri() -> str:
    """Static redirect URI configuration
    
    Returns the configured redirect URI from secrets.
    Change REDIRECT_URI in secrets.toml based on deployment environment.
    """
    try:
        redirect_uri = st.secrets.get("REDIRECT_URI", "http://localhost:8501/oauth2callback")
        log_config_event(f"Redirect URI loaded: {redirect_uri}")
        return redirect_uri
        
    except Exception as e:
        log_config_event(f"Failed to load redirect URI: {str(e)}", "ERROR")
        # Emergency fallback
        fallback_uri = "http://localhost:8501/oauth2callback"
        log_config_event(f"Using fallback redirect URI: {fallback_uri}", "WARNING")
        return fallback_uri

def get_firebase_config() -> Dict[str, Any]:
    """Dapatkan konfigurasi Firebase yang terstruktur"""
    try:
        if "firebase" not in st.secrets:
            log_config_event("Firebase config section missing from secrets", "ERROR")
            return {}
        
        service_account = dict(st.secrets["firebase"])
        firebase_api_key = st.secrets.get("FIREBASE_API_KEY", "")
        
        config = {
            "apiKey": firebase_api_key,
            "authDomain": f"{service_account.get('project_id', '')}.firebaseapp.com",
            "projectId": service_account.get('project_id', ''),
            "databaseURL": f"https://{service_account.get('project_id', '')}-default-rtdb.firebaseio.com",
            "storageBucket": f"{service_account.get('project_id', '')}.appspot.com"
        }
        
        log_config_event(f"Firebase config loaded for project: {service_account.get('project_id', 'unknown')}")
        return config
        
    except Exception as e:
        log_config_event(f"Failed to load Firebase config: {str(e)}", "ERROR")
        return {}

def is_config_valid() -> bool:
    """Check apakah konfigurasi valid untuk operasi"""
    return bool(
        st.secrets.get("GOOGLE_CLIENT_ID") and 
        st.secrets.get("GOOGLE_CLIENT_SECRET") and 
        st.secrets.get("REDIRECT_URI") and 
        st.secrets.get("FIREBASE_API_KEY")
    )

# UNIFIED SESSION MANAGEMENT SYSTEM
# =============================================================================

class SessionManager:
    """Centralized session and cookie management"""
    
    @staticmethod
    def initialize() -> None:
        """Initialize session state with default values"""
        defaults = {
            'logged_in': False, 'login_attempts': 0, 'firebase_initialized': False,
            'auth_type': '🔒 Masuk', 'user_email': None, 'remember_me': False, 'login_time': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def sync_from_cookies() -> None:
        """Sync login state from cookies to session_state"""
        try:
            if not hasattr(cookie_controller, 'get'):
                log_event("SYSTEM", "cookie_sync", "info", details="Controller not ready")
                return
            
            is_logged_in = cookie_controller.get('is_logged_in')
            user_email = cookie_controller.get('user_email')
            remember_me = cookie_controller.get('remember_me')
            
            if is_logged_in == 'True' and user_email and SessionManager._is_valid_email(user_email):
                st.session_state.update({
                    'logged_in': True, 'user_email': user_email,
                    'remember_me': remember_me == 'True'
                })
                log_event("SESSION", "cookie_sync", "info", email=user_email, details="State restored from cookies")
            else:
                st.session_state['logged_in'] = False
                if user_email and not SessionManager._is_valid_email(user_email):
                    SessionManager._clear_cookies()
                    log_event("SECURITY", "invalid_email_cookie", "warning", email=user_email, details="Cookies cleared")
                    
        except Exception as e:
            st.session_state['logged_in'] = False
            SessionManager._clear_cookies()
            log_event("ERROR", "cookie_sync", "error", error=e)
    
    @staticmethod
    def set_login_cookies(email: str, remember: bool = False) -> None:
        """Set authentication cookies"""
        try:
            max_age = REMEMBER_ME_DURATION if remember else None
            cookie_controller.set('is_logged_in', 'True', max_age=max_age)
            cookie_controller.set('user_email', email, max_age=max_age)
            cookie_controller.set('remember_me', str(remember), max_age=max_age)
            cookie_controller.set('last_email', email, max_age=LAST_EMAIL_DURATION)
            
            duration_desc = f"{REMEMBER_ME_DURATION//86400} days" if remember else "session only"
            log_event("SESSION", "login_cookies_set", "info", email=email, details=duration_desc)
        except Exception as e:
            log_event("ERROR", "cookie_setting", "error", email=email, error=e)
    
    @staticmethod
    def get_remembered_email() -> str:
        """Get last remembered email for auto-fill"""
        try:
            email = cookie_controller.get('last_email') or ""
            if email and SessionManager._is_valid_email(email):
                log_event("SESSION", "email_remembered", "info", email=email)
                return email
            elif email:
                cookie_controller.remove('last_email')
                log_event("SECURITY", "invalid_remembered_email", "warning", email=email)
        except Exception as e:
            log_event("ERROR", "remember_email_retrieval", "error", error=e)
        return ""
    
    @staticmethod
    def clear_auth_data() -> None:
        """Clear all authentication data"""
        SessionManager._clear_cookies()
        st.session_state.update({
            'logged_in': False, 'user_email': None, 'remember_me': False,
            'login_attempts': 0, 'login_time': None
        })
        log_event("SESSION", "auth_cleared", "info", details="Complete logout")
    
    @staticmethod
    def is_app_ready() -> bool:
        """Check if app is ready for use"""
        return all([
            st.session_state.get('firebase_initialized', False),
            st.session_state.get('firebase_auth') is not None,
            st.session_state.get('firestore') is not None
        ])
    
    @staticmethod
    def _clear_cookies() -> None:
        """Internal method to clear cookies"""
        try:
            for cookie in ['is_logged_in', 'user_email', 'remember_me']:
                cookie_controller.remove(cookie)
        except Exception as e:
            log_event("ERROR", "cookie_clearing", "error", error=e)
    
    @staticmethod
    def _is_valid_email(email: str) -> bool:
        """Internal email validation"""
        is_valid, _ = validate_email_format(email)
        return is_valid

# Backward compatibility functions
def initialize_session_state() -> None:
    """Legacy wrapper for SessionManager.initialize()"""
    SessionManager.initialize()

def sync_login_state() -> None:
    """Legacy wrapper for SessionManager.sync_from_cookies()"""
    SessionManager.sync_from_cookies()

def set_remember_me_cookies(email: str, remember: bool = False) -> None:
    """Legacy wrapper for SessionManager.set_login_cookies()"""  
    SessionManager.set_login_cookies(email, remember)

def get_remembered_email() -> str:
    """Legacy wrapper for SessionManager.get_remembered_email()"""
    return SessionManager.get_remembered_email()

def clear_remember_me_cookies() -> None:
    """Legacy wrapper for SessionManager.clear_auth_data()"""
    SessionManager.clear_auth_data()

def is_app_ready() -> bool:
    """Legacy wrapper for SessionManager.is_app_ready()"""
    return SessionManager.is_app_ready()

# =============================================================================
# CENTRALIZED ERROR HANDLING
# =============================================================================

# UNIFIED ERROR HANDLING SYSTEM
# =============================================================================

def handle_auth_error(error: Optional[Exception] = None, context: str = "", 
                     validation_errors: Optional[List[str]] = None,
                     progress_container: Any = None, message_container: Any = None) -> Tuple[str, str]:
    """
    Unified error handling for all authentication operations
    
    Args:
        error: Exception object for Firebase/system errors
        context: Operation context (login, register, etc.)
        validation_errors: List of validation error messages
        progress_container: Streamlit container for progress display
        message_container: Streamlit container for messages
    
    Returns:
        Tuple of (toast_message, detailed_message)
    """
    # Clear progress if provided
    if progress_container:
        try:
            progress_container.empty()
        except:
            pass
    
    # Handle validation errors
    if validation_errors:
        toast_msg = "Data tidak valid"
        detailed_msg = "Validasi data gagal"
        
        if message_container:
            message_container.error(f"❌ {detailed_msg}:")
            for err in validation_errors:
                message_container.error(f"• {err}")
        else:
            st.error(f"❌ {detailed_msg}:")
            for err in validation_errors:
                st.error(f"• {err}")
        
        show_toast("error", toast_msg)
        log_event("ERROR", "validation", "warning", details=f"Validation failed: {'; '.join(validation_errors)}")
        return toast_msg, detailed_msg
    
    # Handle Firebase/system errors
    if error:
        error_str = str(error).upper()
        
        # Firebase error mapping
        error_map = {
            "INVALID_EMAIL": ("Format email tidak valid", "Format email tidak valid. Periksa kembali alamat email Anda."),
            "USER_NOT_FOUND": ("Email tidak terdaftar", "Email tidak terdaftar dalam sistem kami. Silakan daftar terlebih dahulu."),
            "INVALID_LOGIN_CREDENTIALS": ("Login gagal", "Email atau password salah. Periksa kembali data Anda."),
            "WRONG_PASSWORD": ("Kata sandi salah", "Kata sandi salah. Silakan coba lagi atau reset kata sandi."),
            "INVALID_PASSWORD": ("Kata sandi salah", "Kata sandi salah. Silakan coba lagi atau reset kata sandi."),
            "USER_DISABLED": ("Akun dinonaktifkan", "Akun Anda telah dinonaktifkan. Hubungi administrator."),
            "EMAIL_EXISTS": ("Email sudah terdaftar", "Email ini sudah terdaftar. Silakan gunakan email lain atau login."),
            "EMAIL_ALREADY_IN_USE": ("Email sudah terdaftar", "Email ini sudah terdaftar. Silakan gunakan email lain atau login."),
            "WEAK_PASSWORD": ("Kata sandi terlalu lemah", "Kata sandi terlalu lemah. Gunakan minimal 8 karakter dengan kombinasi huruf dan angka."),
            "TOO_MANY_REQUESTS": ("Terlalu banyak percobaan", "Terlalu banyak percobaan. Tunggu beberapa menit sebelum mencoba lagi."),
            "NETWORK_REQUEST_FAILED": ("Koneksi bermasalah", "Koneksi internet bermasalah. Periksa koneksi Anda dan coba lagi."),
            "QUOTA_EXCEEDED": ("Batas tercapai", "Batas pengiriman email Firebase tercapai. Coba lagi nanti.")
        }
        
        # Find matching error
        for key, (toast_msg, detailed_msg) in error_map.items():
            if key in error_str:
                break
        else:
            toast_msg = f"{context.title()} gagal" if context else "Operasi gagal"
            detailed_msg = f"{context.title()} gagal: {str(error)}" if context else f"Operasi gagal: {str(error)}"
        
        # Display error
        try:
            if message_container:
                message_container.error(f"❌ {detailed_msg}")
            else:
                st.error(f"❌ {detailed_msg}")
        except:
            st.write(f"❌ {detailed_msg}")
        
        # Show toast and log
        show_toast("error", toast_msg)
        log_event("ERROR", context or "operation", "error", details=str(error), error=error)
        
        return toast_msg, detailed_msg
    
    # Default case
    return "Operasi gagal", "Terjadi kesalahan yang tidak diketahui"

# Backward compatibility wrappers
def handle_firebase_error(error: Exception, context: str = "") -> Tuple[str, str]:
    """Legacy wrapper for handle_auth_error"""
    return handle_auth_error(error=error, context=context)

def show_error_with_context(error: Exception, context: str, progress_container: Any = None, message_container: Any = None) -> None:
    """Legacy wrapper for handle_auth_error"""
    handle_auth_error(error=error, context=context, progress_container=progress_container, message_container=message_container)

def handle_validation_errors(errors: list, progress_container: Any = None, message_container: Any = None) -> None:
    """Legacy wrapper for handle_auth_error"""
    handle_auth_error(validation_errors=errors, progress_container=progress_container, message_container=message_container)

# UNIFIED VALIDATION SYSTEM
# =============================================================================

class ValidationManager:
    """Centralized validation for all input data"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Comprehensive email validation"""
        if not email:
            return False, "Email tidak boleh kosong"
        
        # Check length limits
        if len(email) > 254:
            return False, "Email terlalu panjang (maksimal 254 karakter)"
        
        # Basic email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return False, "Format email tidak valid. Contoh: nama@domain.com"
        
        # Additional checks
        local_part, domain = email.rsplit('@', 1)
        if len(local_part) > 64:
            return False, "Bagian lokal email terlalu panjang (maksimal 64 karakter)"
        
        if '..' in email or local_part.startswith('.') or local_part.endswith('.'):
            return False, "Format email tidak valid"
        
        return True, ""
    
    @staticmethod
    def validate_name(name: str, field_name: str = "Nama") -> Tuple[bool, str]:
        """Name validation with flexible rules"""
        if not name:
            return False, f"{field_name} tidak boleh kosong"
        
        name = name.strip()
        if len(name) < 2:
            return False, f"{field_name} minimal 2 karakter"
        if len(name) > 50:
            return False, f"{field_name} maksimal 50 karakter"
        
        # Allow letters, spaces, apostrophes, and hyphens
        if not re.match(r'^[a-zA-Z\s\'-]+$', name):
            return False, f"{field_name} hanya boleh mengandung huruf, spasi, apostrof, dan tanda hubung"
        
        return True, ""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """Password strength validation"""
        if len(password) < 8:
            return False, "Kata sandi minimal 8 karakter"
        
        checks = [
            (any(c.isupper() for c in password), "harus mengandung huruf besar"),
            (any(c.islower() for c in password), "harus mengandung huruf kecil"),
            (any(c.isdigit() for c in password), "harus mengandung angka")
        ]
        
        for check, msg in checks:
            if not check:
                return False, f"Kata sandi {msg}"
        
        return True, ""
    
    @staticmethod
    def validate_registration_data(email: str, password: str, first_name: str, last_name: str) -> List[str]:
        """Complete registration data validation"""
        errors = []
        
        validators = [
            (ValidationManager.validate_email(email), "Email"),
            (ValidationManager.validate_password(password), "Password"),
            (ValidationManager.validate_name(first_name, "Nama depan"), "Nama depan"),
            (ValidationManager.validate_name(last_name, "Nama belakang"), "Nama belakang")
        ]
        
        for (is_valid, error_msg), field in validators:
            if not is_valid:
                errors.append(f"{field}: {error_msg}")
        
        return errors

class SecurityManager:
    """Rate limiting and session security"""
    
    @staticmethod
    def check_rate_limit(user_email: str) -> bool:
        """Check if user exceeded login rate limit"""
        now = datetime.now()
        rate_limit_key = f'ratelimit_{user_email}'
        attempts = st.session_state.get(rate_limit_key, [])
        
        # Filter valid attempts within window
        valid_attempts = [
            attempt for attempt in attempts 
            if (now - attempt) < timedelta(seconds=RATE_LIMIT_WINDOW)
        ]
        
        if len(valid_attempts) >= MAX_LOGIN_ATTEMPTS:
            return False
        
        valid_attempts.append(now)
        st.session_state[rate_limit_key] = valid_attempts
        return True
    
    @staticmethod
    def check_session_timeout() -> bool:
        """Check if user session has expired"""
        login_time = st.session_state.get('login_time')
        if not login_time:
            return True
        
        elapsed = datetime.now() - login_time
        return elapsed < timedelta(seconds=SESSION_TIMEOUT)

# Backward compatibility wrappers
def validate_email_format(email: str) -> Tuple[bool, str]:
    """Legacy wrapper for ValidationManager.validate_email()"""
    return ValidationManager.validate_email(email)

def validate_name_format(name: str, field_name: str) -> Tuple[bool, str]:
    """Legacy wrapper for ValidationManager.validate_name()"""
    return ValidationManager.validate_name(name, field_name)

def validate_password(password: str) -> Tuple[bool, str]:
    """Legacy wrapper for ValidationManager.validate_password()"""
    return ValidationManager.validate_password(password)

def check_rate_limit(user_email: str) -> bool:
    """Legacy wrapper for SecurityManager.check_rate_limit()"""
    return SecurityManager.check_rate_limit(user_email)

def check_session_timeout() -> bool:
    """Legacy wrapper for SecurityManager.check_session_timeout()"""
    return SecurityManager.check_session_timeout()

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
        
        if len(valid_attempts) >= EMAIL_VERIFICATION_LIMIT:
            return False, "Batas pengiriman email tercapai. Silakan coba lagi dalam 1 jam."
        
        valid_attempts.append(now)
        st.session_state[quota_key] = valid_attempts
        return True, ""
        
    except Exception as e:
        log_event("system", "Email quota check", "error", error=e)
        return False, "Error checking email quota"

# =============================================================================
# FIREBASE FUNCTIONS
# =============================================================================

def initialize_firebase() -> Tuple[Optional[Any], Optional[Any]]:
    """Inisialisasi Firebase Admin SDK dan Pyrebase"""
    try:
        # Cek apakah Firebase sudah diinisialisasi sebelumnya
        if st.session_state.get('firebase_initialized', False):
            firebase_auth = st.session_state.get('firebase_auth')
            firestore_client = st.session_state.get('firestore')
            
            if firebase_auth and firestore_client:
                log_system_event("Using existing Firebase initialization")
                return firebase_auth, firestore_client
            else:
                log_event("firebase", "Firebase validation", "warning", 
                         details="Firebase objects invalid, reinitializing...")
                st.session_state['firebase_initialized'] = False

        # Verifikasi environment dan konfigurasi
        if not is_config_valid():
            log_event("config", "Configuration validation", "error")
            return None, None
            
        # Periksa konfigurasi Firebase
        if "firebase" not in st.secrets:
            log_event("config", "Firebase configuration", "critical", 
                     details="Firebase configuration not found in secrets")
            st.error("🔥 Konfigurasi Firebase tidak ditemukan!")
            return None, None
        
        # Ambil konfigurasi service account
        service_account = dict(st.secrets["firebase"])
        
        # Periksa field yang diperlukan
        required_fields = ["project_id", "client_email", "private_key"]
        missing_fields = [field for field in required_fields if field not in service_account]
        
        if missing_fields:
            log_event("config", "Firebase configuration", "critical", 
                     details=f"Missing Firebase config fields: {missing_fields}")
            st.error(f"🔥 Konfigurasi Firebase tidak lengkap! Field yang diperlukan: {', '.join(missing_fields)}")
            return None, None
        
        # Inisialisasi Firebase Admin SDK
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account)
            firebase_admin.initialize_app(cred)
            log_firebase_operation("Admin SDK", "success", "Initialized successfully")
        
        # Konfigurasi Pyrebase
        config = get_firebase_config()
        if not config:
            log_event("config", "Firebase configuration", "error")
            return None, None
        
        # Inisialisasi Pyrebase
        pb = pyrebase.initialize_app(config)
        firebase_auth = pb.auth()
        log_firebase_operation("Pyrebase", "success", "Initialized successfully")
        
        # Inisialisasi Firestore client
        firestore_client = firestore.client()
        log_firebase_operation("Firestore client", "success", "Initialized successfully")
        
        # Simpan ke session state
        st.session_state['firebase_auth'] = firebase_auth
        st.session_state['firestore'] = firestore_client
        st.session_state['firebase_initialized'] = True
        
        log_system_event("Firebase initialization complete", "All services ready")
        return firebase_auth, firestore_client
        
    except Exception as e:
        log_event("firebase", "Firebase initialization", "critical", error=e)
        st.error(f"Gagal menginisialisasi Firebase: {str(e)}")
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
        log_firebase_operation("Email verification", "success", f"Sent to {email}")
        return True, "Email verifikasi berhasil dikirim"
        
    except Exception as e:
        log_event("firebase", "Email verification", "error", 
                 details=f"Failed to send to {email}", error=e)
        toast_msg, detailed_msg = handle_firebase_error(e, "email_verification")
        return False, detailed_msg

def verify_user_exists(user_email: str, firestore_client: Any) -> bool:
    """Verifikasi bahwa pengguna ada dan memiliki data yang valid di Firestore"""
    try:
        firebase_user = auth.get_user_by_email(user_email)
        user_doc = firestore_client.collection('users').document(firebase_user.uid).get()
        
        if user_doc.exists:
            log_user_action("User verification", user_email, "success")
            return True
        
        log_event("firebase", "User data", "warning", 
                 details=f"User {user_email} has no Firestore data")
        return False

    except auth.UserNotFoundError:
        log_event("firebase", "User data", "warning", 
                 details=f"User {user_email} not found in Firebase Auth")
        return False
    except Exception as e:
        log_event("firebase", "User verification", "error", 
                 details=f"Error verifying user {user_email}", error=e)
        return False

# =============================================================================
# GOOGLE OAUTH FUNCTIONS
# =============================================================================

def get_google_authorization_url(popup: bool = False) -> str:
    """Bangun URL otorisasi Google OAuth dengan dukungan state untuk mode popup/normal."""
    base_url = 'https://accounts.google.com/o/oauth2/v2/auth'
    # Generate dan simpan state (anti-CSRF)
    state_token = secrets.token_urlsafe(16)
    st.session_state['oauth_state'] = state_token
    state_value = ("popup|" if popup else "normal|") + state_token

    params = {
        'client_id': st.secrets.get("GOOGLE_CLIENT_ID", ""),
        'redirect_uri': get_redirect_uri(),
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'offline',
        'prompt': 'consent',
        'state': state_value,
    }
    return f"{base_url}?{urlencode(params)}"

def _parse_oauth_state(state: str) -> Tuple[str, str]:
    """Parse state menjadi (mode, token). Kembalikan ('normal','') bila gagal."""
    try:
        if '|' in state:
            mode, token = state.split('|', 1)
            return mode, token
        return 'normal', state
    except Exception:
        return 'normal', ''

def render_oauth_popup_listener():
    """Pasang JS listener untuk menerima code dari popup lalu navigasi ulang halaman utama."""
    components.html(
        """
        <script>
        (function(){
            if (window.__oauthListenerInstalled) return;
            window.__oauthListenerInstalled = true;
            window.addEventListener('message', function(event){
                try {
                    if (event.origin !== window.location.origin) return;
                    var data = event.data || {};
                    if (data.type !== 'oauth-code' || !data.code) return;
                    var state = data.state || '';
                    if (typeof state === 'string' && state.indexOf('popup|') === 0) {
                        state = 'normal|' + state.split('|')[1];
                    }
                    var url = new URL(window.location.href);
                    url.searchParams.set('code', data.code);
                    if (state) url.searchParams.set('state', state);
                    window.location.href = url.toString();
                } catch(e) { console.error('OAuth listener error:', e); }
            }, false);
        })();
        </script>
        """,
        height=0,
    )

async def exchange_google_token(code: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Tukar kode otorisasi Google untuk informasi pengguna"""
    async with httpx.AsyncClient() as client:
        token_url = 'https://oauth2.googleapis.com/token'
        payload = {
            'client_id': st.secrets.get("GOOGLE_CLIENT_ID", ""),
            'client_secret': st.secrets.get("GOOGLE_CLIENT_SECRET", ""),
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': get_redirect_uri()
        }

        try:
            # Tukar kode untuk token
            token_response = await client.post(token_url, data=payload)
            token_data = token_response.json()
            
            if 'access_token' not in token_data:
                log_event("auth", "Google token exchange", "error", 
                         details=f"Token exchange failed: {token_data}")
                return None, None
            
            # Gunakan token untuk mendapatkan info pengguna
            user_info_url = f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={token_data['access_token']}"
            user_response = await client.get(user_info_url)
            user_info = user_response.json()
            
            if 'email' not in user_info:
                log_event("auth", "User info validation", "error", 
                         details=f"User info incomplete: {user_info}")
                return None, None
                
            return user_info['email'], user_info

        except Exception as e:
            log_event("auth", "Google token exchange", "error", error=e)
            return None, None

def handle_google_login_callback() -> bool:
    """Tangani callback Google OAuth setelah autentikasi pengguna dengan progress feedback"""
    try:
        if 'code' not in st.query_params:
            return True  # Tidak ada callback Google, lanjutkan normal
            
        code = st.query_params.get('code')
        if not code or not isinstance(code, str):
            st.error("Kode otorisasi Google tidak valid")
            return False

        # Deteksi dan validasi state; jika mode popup, kirim code ke opener dan tutup popup
        state = st.query_params.get('state', '')
        mode, token = _parse_oauth_state(state) if state else ('normal', '')
        expected = st.session_state.get('oauth_state')
        if token and expected and token != expected:
            st.error("State OAuth tidak valid. Silakan coba lagi.")
            return False

        if mode == 'popup':
            log_event("auth", "Google login callback (popup)", "info", details="Posting code to opener and closing popup")
            components.html(
                """
                <script>
                    try {
                        if (window.opener) {
                            window.opener.postMessage({
                                type: 'oauth-code',
                                code: new URLSearchParams(window.location.search).get('code'),
                                state: new URLSearchParams(window.location.search).get('state')
                            }, window.location.origin);
                        }
                    } catch(e) { console.error('postMessage failed', e); }
                    setTimeout(function(){ window.close(); }, 100);
                </script>
                """,
                height=0,
            )
            st.stop()

        # Tampilkan progress untuk Google callback processing
        callback_progress = st.empty()
        callback_message = st.empty()
        
        with callback_progress.container():
            progress_container = st.empty()
            message_container = st.empty()
            
            # Step 1: Token exchange
            progress_container.progress(0.2)
            message_container.caption("🔄 Memproses token Google...")
            
            async def async_token_exchange():
                return await exchange_google_token(code)

            user_email, user_info = asyncio.run(async_token_exchange())
            if not user_email or not user_info:
                progress_container.empty()
                message_container.error("❌ Gagal mendapatkan informasi pengguna dari Google")
                show_error_toast("Gagal memproses login Google")
                return False

            # Step 2: Firebase initialization
            progress_container.progress(0.4)
            message_container.caption("🔥 Menginisialisasi Firebase...")
            
            # Verifikasi pengguna ada di sistem
            firebase_auth, firestore_client = initialize_firebase()
            if not firebase_auth or not firestore_client:
                progress_container.empty()
                message_container.error("❌ Gagal menginisialisasi Firebase")
                show_error_toast("Gagal menginisialisasi sistem")
                return False
            
            # Step 3: User verification
            progress_container.progress(0.6)
            message_container.caption("👤 Memverifikasi pengguna...")
            
            try:
                # Cek apakah user sudah terdaftar
                firebase_user = auth.get_user_by_email(user_email)
                user_doc = firestore_client.collection('users').document(firebase_user.uid).get()
                
                if user_doc.exists:
                    # Step 4: Processing login
                    progress_container.progress(0.8)
                    message_container.caption("🔐 Memproses login...")
                    
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
                                log_event("firebase", "Email verification update", "info", 
                                         details=f"Updated email verification status for Google user: {user_email}")
                            except Exception as update_error:
                                log_event("firebase", "Email verification update", "warning", 
                                         details=f"Failed to update email verification for Google user {user_email}", 
                                         error=update_error)
                        
                        # Step 5: Complete
                        progress_container.progress(1.0)
                        message_container.caption("✅ Login Google berhasil!")
                        
                        st.session_state['logged_in'] = True
                        st.session_state['user_email'] = user_email
                        st.session_state['login_time'] = datetime.now()
                        set_remember_me_cookies(user_email, True)
                        
                        log_auth_success("Google login", user_email, "Authentication successful")
                        
                        # Clear progress dan tampilkan pesan sukses
                        time.sleep(1.0)
                        progress_container.empty()
                        message_container.success("🎉 Login Google berhasil! Mengarahkan ke dashboard...")
                        show_success_toast("Login Google berhasil! Mengarahkan ke dashboard...")
                        
                        # Auto-redirect ke halaman tools setelah login berhasil
                        time.sleep(1.0)  # Beri waktu untuk membaca pesan
                        callback_progress.empty()
                        st.session_state['should_redirect'] = True
                        st.session_state['login_success'] = True  # Flag untuk menampilkan toast di main app
                        st.query_params.clear()  # Clear OAuth params
                        st.rerun()  # Rerun untuk trigger redirect logic
                        return True
                    else:
                        # Email belum diverifikasi untuk user non-Google
                        progress_container.empty()
                        message_container.warning(
                            f"📧 **Email Anda belum diverifikasi!**\n\n"
                            f"Email {user_email} belum diverifikasi. "
                            f"Silakan periksa kotak masuk email Anda dan klik link verifikasi yang telah dikirim."
                        )
                        show_warning_toast("Email belum diverifikasi")
                        
                        # Simpan error di session state untuk ditampilkan di feedback_placeholder
                        st.session_state['google_auth_verification_error'] = True
                        st.session_state['google_auth_email'] = user_email
                        st.query_params.clear()
                        time.sleep(2.0)
                        callback_progress.empty()
                        return False
                else:
                    # User tidak ada di Firestore, arahkan ke registrasi
                    progress_container.empty()
                    message_container.error(
                        f"**Akun Google Tidak Terdaftar**\n\n"
                        f"Akun Google {user_email} belum terdaftar dalam sistem kami."
                    )
                    message_container.info(
                        "💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' "
                        "atau gunakan akun email yang sudah terdaftar."
                    )
                    show_error_toast(f"Akun Google {user_email} tidak terdaftar")
                    
                    st.session_state['google_auth_error'] = True
                    st.session_state['google_auth_email'] = user_email
                    st.query_params.clear()
                    time.sleep(2.0)
                    callback_progress.empty()
                    return False

            except auth.UserNotFoundError:
                # User tidak ada di Firebase Auth, arahkan ke registrasi
                progress_container.empty()
                message_container.error(
                    f"**Akun Google Tidak Terdaftar**\n\n"
                    f"Akun Google {user_email} belum terdaftar dalam sistem kami."
                )
                message_container.info(
                    "💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' "
                    "atau gunakan akun email yang sudah terdaftar."
                )
                show_error_toast(f"Akun Google {user_email} tidak terdaftar")
                
                st.session_state['google_auth_error'] = True
                st.session_state['google_auth_email'] = user_email
                st.query_params.clear()
                time.sleep(2.0)
                callback_progress.empty()
                return False

    except Exception as e:
        log_event("auth", "Google login callback", "error", error=e)
        if 'callback_progress' in locals():
            callback_progress.empty()
        st.error("❌ Terjadi kesalahan saat memproses login Google")
        show_error_toast("Terjadi kesalahan saat memproses login Google")
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
        
        log_firebase_operation("Email verification sync", "success", f"User {user.get('email', 'unknown')}: {email_verified}")
        return email_verified
    except Exception as e:
        log_event("firebase", "Email sync", "error", 
                 details="Gagal sync email_verified", error=e)
        return False

def login_user(email: str, password: str, firebase_auth: Any, firestore_client: Any, 
               remember: bool, progress_container: Any, message_container: Any) -> bool:
    """Proses login pengguna dengan feedback yang ditampilkan di lokasi yang konsisten"""
    
    log_auth_start("Login process", email)
    
    # Validasi format email
    is_valid_email, email_message = validate_email_format(email)
    if not is_valid_email:
        progress_container.empty()
        show_error_toast("Format email tidak valid")
        message_container.error(email_message)
        log_auth_failure("Login validation", email, "Invalid email format")
        return False
    
    # Cek rate limiting
    if not check_rate_limit(email):
        progress_container.empty()
        show_error_toast("Terlalu banyak percobaan")
        message_container.error("Terlalu banyak percobaan login. Silakan coba lagi nanti.")
        log_security_event("Rate limit exceeded", email, "Login blocked")
        return False
    
    try:
        # Step 1: Validating credentials
        progress_container.progress(0.2)
        message_container.caption("🔐 Memvalidasi kredensial...")
        
        # Coba login dengan Firebase
        user = firebase_auth.sign_in_with_email_and_password(email, password)
        log_firebase_operation("Authentication", "success", f"User {email} authenticated")
        
        # Step 2: Checking email verification status
        progress_container.progress(0.5)
        message_container.caption("⚠️ Memeriksa status verifikasi email...")
        
        # Sync dan cek status verifikasi email dari Firebase Auth
        email_verified = sync_email_verified_to_firestore(firebase_auth, firestore_client, user)
        
        if not email_verified:
            progress_container.empty()
            show_warning_toast("Email belum diverifikasi")
            message_container.warning(
                f"📧 **Email Anda belum diverifikasi!**\n\n"
                f"Email {email} belum diverifikasi. "
                f"Silakan periksa kotak masuk email Anda dan klik link verifikasi yang telah dikirim. "
                f"Setelah verifikasi, silakan coba login kembali.\n\n"
                f"💡 *Tip: Periksa juga folder spam/junk email*"
            )
            log_auth_failure("Login verification", email, "Email not verified")
            return False
        
        # Step 3: Verifying user data
        progress_container.progress(0.7)
        message_container.caption("👤 Memverifikasi data pengguna...")
        
        # Verifikasi pengguna ada di Firestore
        if not verify_user_exists(email, firestore_client):
            progress_container.empty()
            show_error_toast("Data pengguna tidak ditemukan")
            message_container.error("Data pengguna tidak ditemukan di sistem. Silakan hubungi administrator.")
            log_auth_failure("User verification", email, "User data not found in Firestore")
            return False
            
        # Step 4: Setting up session
        progress_container.progress(0.9)
        message_container.caption("⚙️ Menyiapkan sesi pengguna...")
        
        # Set status login
        st.session_state['logged_in'] = True
        st.session_state['user_email'] = email
        st.session_state['login_time'] = datetime.now()
        st.session_state['login_attempts'] = 0
        
        # Set cookies
        set_remember_me_cookies(email, remember)
        
        # Step 5: Complete
        progress_container.progress(1.0)
        message_container.success("🎉 Login berhasil! Mengarahkan ke dashboard...")
        
        log_auth_success("Login process", email, f"Remember me: {remember}")
        log_user_action("User logged in", email)
        show_success_toast("Login berhasil! Mengarahkan ke dashboard...")
        
        # Clear progress setelah menampilkan pesan sukses, tapi biarkan message tetap
        time.sleep(SUCCESS_DISPLAY_DURATION)  # Beri waktu untuk menampilkan progress completion
        progress_container.empty()
        
        # Auto-redirect ke halaman tools setelah login berhasil
        st.session_state['should_redirect'] = True
        st.session_state['login_success'] = True  # Flag untuk menampilkan toast di main app
        time.sleep(REDIRECT_PAUSE_DURATION)  # Brief pause untuk user experience
        st.rerun()  # Rerun untuk trigger redirect logic
        
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
            
            log_auth_failure("Login authentication", email, "Email not registered")
            return False
        else:
            # Gunakan centralized error handling untuk error lainnya
            log_auth_error("Login process", e, email)
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
            auto_password = f"Google-{secrets.token_hex(8)}"
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
        log_event("auth", "Account creation", "info", 
                 details=f"Successfully created account for: {email}")
        show_success_toast("Registrasi berhasil")
        
        # Clear progress setelah menampilkan pesan sukses, tapi biarkan message tetap
        time.sleep(SUCCESS_DISPLAY_DURATION)
        progress_container.empty()
        
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
        log_event("auth", "Password reset", "info", 
                 details=f"Password reset email sent to: {email}")
        
        # Step 5: Complete
        progress_container.progress(1.0)
        message_container.caption("✅ Email reset berhasil dikirim!")
        
        success_message = f"📧 **Petunjuk reset password telah dikirim ke {email}**\n\nSilakan periksa kotak masuk email Anda (dan folder spam) untuk link reset password.\n\nLink akan aktif selama 1 jam."
        message_container.success(success_message)
        
        show_success_toast("Link reset password berhasil dikirim")
        
        # Clear progress setelah menampilkan pesan sukses, tapi biarkan message tetap
        time.sleep(SUCCESS_DISPLAY_DURATION)
        progress_container.empty()
        return True
        
    except Exception as e:
        show_error_with_context(e, "reset", progress_container, message_container)
        return False

def logout() -> None:
    """Tangani logout pengguna dengan pembersihan sesi"""
    try:
        user_email = st.session_state.get('user_email')
        log_event("auth", "Logout", "info", 
                 details=f"Logging out user: {user_email}")
        
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
        
        # Clear URL params
        st.query_params.clear()
        
        # Clear cookies but keep last_email for convenience
        clear_remember_me_cookies()
        
        log_event("auth", "Logout", "info", 
                 details="Logout completed successfully")
        
    except Exception as e:
        log_event("auth", "Logout", "error", error=e)
        show_error_toast(f"Logout failed: {str(e)}")

def show_toast(message: str, toast_type: str = "info") -> None:
    """Unified toast notification system
    
    Args:
        message: Message to display
        toast_type: Type of toast (success, error, warning, info)
    """
    icons = {
        "success": "✅", "error": "❌", "warning": "⚠️", 
        "info": "ℹ️", "critical": "🚨"
    }
    
    icon = icons.get(toast_type, "ℹ️")
    
    try:
        st.toast(message, icon=icon)
    except Exception as e:
        log_event("system", "Toast notification", "error", error=e)
        st.info(f"{icon} {message}")

# Backward compatibility wrappers
def show_toast_notification(message: str, icon: str = "ℹ") -> None:
    """Legacy toast function for compatibility"""
    show_toast(message, "info")

def show_success_toast(message: str) -> None:
    """Show success toast"""
    show_toast(message, "success")

def show_error_toast(message: str) -> None:
    """Show error toast"""
    show_toast(message, "error")

def show_warning_toast(message: str) -> None:
    """Show warning toast"""
    show_toast(message, "warning")

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
# UI COMPONENTS
# =============================================================================

def display_login_form(firebase_auth: Any, firestore_client: Any) -> None:
    """Tampilkan dan tangani formulir login"""
    
    # Check app readiness
    app_ready = is_app_ready()
    
    # Initialize feedback containers untuk layout stability
    feedback_placeholder = st.empty()
    progress_container = None
    message_container = None


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
                st.error(f"❌ {email_message}")  # Tetap gunakan st.error untuk real-time validation

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
                help=f"Simpan login selama {REMEMBER_ME_DURATION // (24*60*60)} hari",
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

        # Tombol Login Google via Popup (di dalam form agar UI rapi dan tidak submit form)
        google_popup_url = get_google_authorization_url(popup=True)
        _popup_html_in_form = """
            <div style='width:100%'>
                <button id=\"google-login-popup\" type=\"button\" style=\"width:100%;border:none;border-radius:20px;height:44px;font-weight:700;background:#0d6efd;color:white;cursor:pointer\">Lanjutkan dengan Google</button>
            </div>
            <script>
                (function(){
                    var btn = document.getElementById('google-login-popup');
                    if (btn && !btn.__bound) {
                        btn.__bound = true;
                        btn.addEventListener('click', function(){
                            var w = 520, h = 600;
                            var y = window.top.outerHeight / 2 + window.top.screenY - ( h / 2);
                            var x = window.top.outerWidth / 2 + window.top.screenX - ( w / 2);
                            window.open('__OAUTH_URL__', 'oauth_popup', 'popup=yes,toolbar=no,location=no,status=no,menubar=no,scrollbars=yes,resizable=yes,width='+w+',height='+h+',top='+y+',left='+x);
                        });
                    }
                })();
            </script>
        """.replace("__OAUTH_URL__", google_popup_url)
        components.html(_popup_html_in_form, height=60)

        # Placeholder untuk pesan feedback dan progress di bawah tombol Google
        # Gunakan single placeholder dengan containers untuk konsistensi layout
        feedback_placeholder = st.empty()
        
        # Pre-allocate containers untuk mencegah layout shift
        with feedback_placeholder.container():
            progress_container = st.empty()
            message_container = st.empty()

    # Pasang listener postMessage di halaman utama (sekali)
    render_oauth_popup_listener()


    # Tampilkan pesan error Google OAuth jika ada - menggunakan feedback placeholder
    if st.session_state.get('google_auth_error', False):
        email_error = st.session_state.get('google_auth_email', '')
        with feedback_placeholder.container():
            progress_container.empty()  # Clear any existing progress
            message_container.error(f"**Akun Google Tidak Terdaftar**\n\n"
                    f"Akun Google {email_error} belum terdaftar dalam sistem kami.")
            st.info(f"💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' atau gunakan akun email yang sudah terdaftar.")
        show_error_toast(f"Akun Google {email_error} tidak terdaftar dalam sistem kami.")
        del st.session_state['google_auth_error']
        if 'google_auth_email' in st.session_state:
            del st.session_state['google_auth_email']

    # Tampilkan pesan error verifikasi Google OAuth jika ada - menggunakan feedback placeholder
    if st.session_state.get('google_auth_verification_error', False):
        email_error = st.session_state.get('google_auth_email', '')
        with feedback_placeholder.container():
            progress_container.empty()  # Clear any existing progress
            message_container.warning(
                f"📧 **Email Anda belum diverifikasi!**\n\n"
                f"Email {email_error} belum diverifikasi. "
                f"Silakan periksa kotak masuk email Anda dan klik link verifikasi yang telah dikirim. "
                f"Setelah verifikasi, silakan coba login kembali.\n\n"
                f"💡 *Tip: Periksa juga folder spam/junk email*"
            )
        show_warning_toast("Email belum diverifikasi")
        del st.session_state['google_auth_verification_error']
        if 'google_auth_email' in st.session_state:
            del st.session_state['google_auth_email']


    # Handle tombol login email di luar form
    if email_login_clicked:
        if email and password:
            # Validasi ulang email sebelum proses login
            email_clean = email.strip()
            is_valid_email, email_message = validate_email_format(email_clean)
            if not is_valid_email:
                # Gunakan container yang sudah ada untuk error display
                progress_container.empty()
                message_container.error(f"❌ {email_message}")
                show_error_toast("Format email tidak valid")
                return
            # Pastikan Firebase sudah siap
            if not firebase_auth or not firestore_client:
                progress_container.empty()
                message_container.error("❌ Sistem belum siap. Silakan tunggu beberapa detik dan coba lagi.")
                show_error_toast("Sistem belum siap")
                return
            # Simpan email terakhir untuk kemudahan
            try:
                cookie_controller.set('last_email', email_clean, max_age=LAST_EMAIL_DURATION)
            except Exception as e:
                log_event("system", "Email storage", "warning", 
                         details="Failed to save last email", error=e)
            
            # Gunakan containers yang sudah di-allocate untuk progress
            progress_container.progress(0.1)
            message_container.caption("🔐 Memulai proses login...")
            # Proses login dengan email yang sudah divalidasi
            try:
                result = login_user(email_clean, password, firebase_auth, firestore_client, remember, progress_container, message_container)
                if result:
                    progress_container.empty()
                    st.rerun()
            except Exception as login_error:
                progress_container.empty()
                log_event("auth", "Login process", "error", 
                         details=f"Login process failed: {login_error}")
                error_str = str(login_error).upper()
                if "INVALID_LOGIN_CREDENTIALS" in error_str:
                    show_error_toast(f"Email {email_clean} tidak terdaftar dalam sistem kami")
                    message_container.error(
                        f"**Akun Email Tidak Terdaftar**\n\n"
                        f"Email {email_clean} belum terdaftar dalam sistem kami."
                    )
                    st.info(
                        f"💡 **Saran:** Silakan daftar terlebih dahulu menggunakan tab 'Daftar' "
                        f"atau periksa ejaan email Anda."
                    )
                else:
                    show_error_toast("Login gagal")
                    message_container.error(f"❌ Login gagal: {str(login_error)}")
        else:
            # Clear existing content dan tampilkan warning
            progress_container.empty()
            message_container.warning("⚠️ Silakan isi kolom email dan kata sandi.")
            show_warning_toast("Silakan isi kolom email dan kata sandi.")


    # Tidak ada handler klik untuk Google; tombol popup membuka jendela dan listener menangani balasan
    
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
                value=f"Google-{secrets.token_hex(4)}",
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
            show_error_toast("Silakan isi semua kolom yang diperlukan.")
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
                
                # Clear progress setelah registrasi berhasil, tapi biarkan message tetap
                time.sleep(SUCCESS_DISPLAY_DURATION)  # Beri waktu untuk membaca pesan
                progress_container.empty()
    
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
                st.error(f"❌ {email_message}")  # Tetap gunakan st.error untuk real-time validation

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
            
            # Clear progress setelah reset password selesai, tapi biarkan message tetap
            if result:
                time.sleep(SUCCESS_DISPLAY_DURATION)  # Beri waktu untuk membaca pesan sukses
                progress_container.empty()
    
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
    # Tampilkan selectbox untuk memilih metode autentikasi
    auth_type = st.selectbox(
        "Pilih metode autentikasi",
        ["🔐 Masuk", "📝 Daftar", "🔑 Reset Kata Sandi"],
        index=0,
        help="Pilih metode autentikasi Anda"
    )
    
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
        log_system_event("Auth module startup initiated")
        sync_login_state()
        initialize_session_state()
        
        log_system_event("Session and login state initialized")
        
        # CSS Styles - Optimized with Layout Stability
        st.markdown("""
            <style>
            /* Main layout optimizations */
            html, body { height: 100vh !important; overflow: hidden !important; margin: 0 !important; }
            .main .block-container { padding-top: 1rem !important; max-height: 100vh !important; }
            section.main { height: 100vh !important; display: flex !important; flex-direction: column !important; 
                          justify-content: center !important; align-items: center !important; }
            
            /* Content wrapper with consistent spacing */
            .auth-content-wrapper { width: 100%; max-width: 500px; max-height: 95vh; overflow-y: auto; 
                                   padding: 1rem; display: flex; flex-direction: column; align-items: center; }
            
            /* Form and UI styling with layout stability */
            .welcome-header { text-align: center; margin-bottom: 1rem; }
            .stSelectbox { margin-bottom: 1rem !important; width: 100%; }
            div[data-testid="stForm"] { border: 1px solid #f0f2f6; padding: 1.2rem; border-radius: 10px; 
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0.5rem; width: 100%; }
            .stButton button { width: 100%; border-radius: 20px; height: 2.8rem; font-weight: bold; margin: 0.3rem 0; }
            .stTextInput { margin-bottom: 0.8rem; }
            
            /* Feedback container untuk mencegah layout shift */
            .element-container { min-height: 2rem; }
            .stEmpty > div { min-height: 1px; }
            
            /* Divider styling */
            .auth-divider-custom { display: flex; align-items: center; margin: 1rem 0; }
            .divider-line-custom { flex: 1; height: 1px; background: #e0e0e0; }
            .divider-text-custom { margin: 0 1rem; color: #888; font-weight: 600; letter-spacing: 1px; font-size: 0.9rem; }
            
            /* Scrollbar */
            .auth-content-wrapper::-webkit-scrollbar { width: 4px; }
            .auth-content-wrapper::-webkit-scrollbar-thumb { background: #ccc; border-radius: 2px; }
            
            /* Responsive */
            @media (max-height: 700px) {
                .welcome-header { margin-bottom: 0.5rem; }
                div[data-testid="stForm"] { padding: 1rem; }
                .stButton button { height: 2.5rem; }
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Inisialisasi Firebase dengan retry dan status feedback
        log_system_event("Firebase initialization starting")
        firebase_auth, firestore_client = None, None
        initialization_container = st.empty()
        
        with initialization_container.container():
            with st.spinner("🔥 Menginisialisasi Firebase..."):
                firebase_auth, firestore_client = initialize_firebase()
        
        # Clear initialization message setelah selesai
        initialization_container.empty()
        
        # Verifikasi Firebase berhasil diinisialisasi
        if not (firebase_auth and firestore_client):
            log_firebase_operation("Initialization", "failed", "Missing auth or firestore client")
            st.error("🔥 *Kesalahan Konfigurasi Firebase*")
            st.error("""
            *Aplikasi tidak dapat berjalan tanpa konfigurasi Firebase yang valid.*
            
            Silakan pastikan:
            • File .streamlit/secrets.toml tersedia dan lengkap
            • Konfigurasi Firebase service account benar
            • Semua kredensial telah dikonfigurasi dengan benar
            
            Hubungi administrator sistem untuk bantuan konfigurasi.
            """)
            return
        
        log_firebase_operation("Initialization", "success", "Auth and Firestore clients ready")

        # Cek status login
        if firebase_auth and firestore_client and st.session_state.get('logged_in', False):
            if check_session_timeout():
                user_email = st.session_state.get('user_email')
                if user_email and verify_user_exists(user_email, firestore_client):
                    log_session_event("User session validated", user_email, "Active session found")
                    
                    # Auto-redirect ke halaman tools setelah login berhasil
                    if st.session_state.get('should_redirect', False):
                        # Clear redirect flag
                        st.session_state['should_redirect'] = False
                        log_system_event("Redirecting to dashboard", f"User: {user_email}")
                        
                        # Show brief redirect message
                        with st.container():
                            st.success("🎉 Login berhasil! Mengarahkan ke dashboard...")
                            
                            # Simple progress animation
                            redirect_progress = st.progress(0)
                            for i in range(0, 101, 5):  # Faster progress
                                redirect_progress.progress(i / 100)
                                time.sleep(PROGRESS_ANIMATION_DELAY)  # Very fast animation
                            
                            redirect_progress.empty()
                        
                        # Clear any query params
                        st.query_params.clear()
                        
                        # Set ready flag untuk main app
                        st.session_state['ready_for_redirect'] = True
                        
                        # Force immediate redirect dengan JavaScript
                        st.markdown("""
                            <script>
                                // Force immediate reload untuk trigger main app workflow
                                setTimeout(function() {
                                    window.location.reload();
                                }, 100);
                            </script>
                        """, unsafe_allow_html=True)
                        
                        time.sleep(REDIRECT_PAUSE_DURATION)  # Brief pause
                        st.stop()  # Stop execution untuk mencegah loading form auth
                    
                    # User sudah login dan verified, keluar dari auth.py
                    # Ini mengindikasikan bahwa auth sudah selesai, main app harus handle routing
                    log_system_event("User authenticated, exiting auth module", user_email)
                    return
                else:
                    log_security_event("User verification failed", user_email or "unknown", "Session terminated")
                    st.error("Masalah autentikasi terdeteksi. Silakan login kembali.")
                    logout()
                    st.rerun()
        # Tampilkan UI autentikasi dalam container yang tepat
        with st.container():
            st.markdown('<div class="auth-content-wrapper">', unsafe_allow_html=True)
            tampilkan_header_sambutan()

            # Handle logout message
            if st.query_params.get("logout") == "1":
                log_user_action("User logout completed", result="Logout page accessed")
                st.toast("Anda telah berhasil logout.", icon="✅")
                st.query_params.clear()
                
            # Handle Google OAuth callback atau tampilkan form autentikasi
            if firebase_auth and firestore_client:
                # Handle Google OAuth callback jika ada
                handle_google_login_callback()

                # Selalu tampilkan pilihan autentikasi jika user belum login
                if not st.session_state.get('logged_in', False):
                    log_system_event("Displaying authentication forms")
                    tampilkan_pilihan_autentikasi(firebase_auth, firestore_client)
            else:
                # Firebase tidak tersedia - tampilkan error konfigurasi
                log_firebase_operation("Runtime check", "failed", "Firebase services unavailable")
                st.error("🔥 *Kesalahan Konfigurasi Firebase*")
                st.error("*Aplikasi tidak dapat berjalan tanpa konfigurasi Firebase yang valid.*")

            # Close the content wrapper
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        log_auth_error("Application runtime", e)
        st.error("Terjadi kesalahan yang tidak terduga. Silakan coba lagi nanti.")
        st.session_state.clear()
        initialize_session_state()
        st.rerun()

if __name__ == "__main__":
    main()
