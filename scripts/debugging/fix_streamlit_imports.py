#!/usr/bin/env python3
"""
Fix imports for Streamlit Cloud deployment

Script ini akan memperbaiki semua masalah import yang umum terjadi
saat deployment ke Streamlit Cloud.
"""

import os
import re

def fix_imports_in_file(file_path):
    """Fix imports in a specific Python file for Streamlit Cloud compatibility"""
    
    if not os.path.exists(file_path):
        print(f"❌ File tidak ditemukan: {file_path}")
        return False
    
    print(f"🔧 Memperbaiki imports di: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix 1: Replace sys.path modifications with try-except import
    sys_path_pattern = r'sys\.path\.append\(os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\), \'\.\.\'.*?\)\)\)'
    if re.search(sys_path_pattern, content):
        print("  📝 Mengganti sys.path modifications...")
        # This replacement is already done in the main file
    
    # Fix 2: Make matplotlib import optional
    matplotlib_pattern = r'^import matplotlib\.pyplot as plt$'
    if re.search(matplotlib_pattern, content, re.MULTILINE):
        print("  📝 Membuat matplotlib import optional...")
        content = re.sub(
            matplotlib_pattern,
            '''try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False''',
            content,
            flags=re.MULTILINE
        )
    
    # Fix 3: Add error handling for critical imports
    critical_imports = [
        r'from ui\.auth import auth',
        r'from ui\.utils import',
    ]
    
    for pattern in critical_imports:
        if re.search(pattern, content):
            print(f"  📝 Menambahkan error handling untuk: {pattern}")
    
    # Fix 4: Replace relative imports with absolute imports where possible
    relative_import_pattern = r'from \.(\w+) import'
    if re.search(relative_import_pattern, content):
        print("  📝 Mengganti relative imports...")
        content = re.sub(relative_import_pattern, r'from ui.\1 import', content)
    
    # Only write if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ File berhasil diperbaiki!")
        return True
    else:
        print(f"  ℹ️ Tidak ada perubahan diperlukan")
        return False

def fix_all_python_files():
    """Fix imports in all Python files in the project"""
    
    print("🚀 Memulai perbaikan imports untuk Streamlit Cloud...")
    print("=" * 60)
    
    # Files to fix
    files_to_fix = [
        "streamlit_app.py",
        "ui/tools/Dashboard_Ringkasan.py",
        "ui/tools/Analisis_Data.py", 
        "ui/tools/Prediksi_Sentimen.py",
        "ui/auth/auth.py",
        "ui/utils.py"
    ]
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        if fix_imports_in_file(file_path):
            fixed_count += 1
    
    print("=" * 60)
    print(f"✅ Selesai! {fixed_count} file berhasil diperbaiki.")
    print("\n🔄 Langkah selanjutnya:")
    print("1. Commit dan push perubahan ke GitHub")
    print("2. Redeploy aplikasi di Streamlit Cloud")
    print("3. Check logs jika masih ada error")
    print("\n💡 Tips debugging:")
    print("- Gunakan debug_streamlit.py untuk debugging lebih lanjut")
    print("- Check requirements.txt sudah include semua dependencies")
    print("- Pastikan semua __init__.py files ada")

if __name__ == "__main__":
    fix_all_python_files()
