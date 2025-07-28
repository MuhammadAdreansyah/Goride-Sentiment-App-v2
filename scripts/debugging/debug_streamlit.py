#!/usr/bin/env python3
"""
Debug script untuk mengidentifikasi masalah import di Streamlit Cloud

Jalankan ini jika masih ada masalah deployment
"""

import sys
import os

def debug_streamlit_cloud():
    """Debug function untuk Streamlit Cloud deployment"""
    
    print("=== STREAMLIT CLOUD DEBUG INFO ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    
    print("\n=== DIRECTORY STRUCTURE ===")
    def show_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        try:
            items = sorted(os.listdir(path))
            for i, item in enumerate(items):
                if item.startswith('.'):
                    continue
                item_path = os.path.join(path, item)
                is_last = i == len(items) - 1
                print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
                if os.path.isdir(item_path) and current_depth < max_depth - 1:
                    extension = "    " if is_last else "│   "
                    show_tree(item_path, prefix + extension, max_depth, current_depth + 1)
        except PermissionError:
            pass
    
    show_tree(".", max_depth=3)
    
    print("\n=== PYTHON PATH ===")
    for i, path in enumerate(sys.path):
        print(f"{i}: {path}")
    
    print("\n=== IMPORT TESTS ===")
    
    # Test basic imports
    test_imports = [
        "streamlit",
        "pandas", 
        "numpy",
        "plotly.express",
        "ui",
        "ui.auth",
        "ui.utils",
        "ui.tools"
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Check requirements.txt includes all dependencies")
    print("2. Ensure all __init__.py files exist")
    print("3. Use absolute imports instead of relative imports")
    print("4. Check Streamlit Cloud logs for detailed error messages")

if __name__ == "__main__":
    debug_streamlit_cloud()
