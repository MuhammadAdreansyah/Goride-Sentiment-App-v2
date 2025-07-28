#!/usr/bin/env python3
"""
Test Script for SentimenGo Cloud Deployment Fixes
================================================

This script tests all the fixes we've implemented for Streamlit Cloud deployment:
1. Scikit-learn version compatibility
2. NLTK data downloading
3. Model loading with fallback strategies
4. Import system reliability

Run this before deploying to cloud to catch any remaining issues.
"""

import os
import sys
import warnings
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ sklearn version {sklearn.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ sklearn import failed: {e}")
        return False
    
    try:
        import nltk
        print(f"✅ NLTK version {nltk.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False
        
    try:
        import pandas as pd
        import numpy as np
        import joblib
        print("✅ Core data science libraries imported successfully")
    except ImportError as e:
        print(f"❌ Core libraries import failed: {e}")
        return False
    
    return True

def test_sklearn_compatibility():
    """Test sklearn version compatibility"""
    print("\n🧪 Testing sklearn compatibility...")
    
    try:
        import sklearn
        from sklearn.svm import SVC
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        version = sklearn.__version__
        major, minor, patch = map(int, version.split('.'))
        
        print(f"📊 sklearn version: {version}")
        
        if major == 1 and 3 <= minor <= 5:
            print("✅ sklearn version is in compatible range (1.3.x - 1.5.x)")
        else:
            print(f"⚠️ sklearn version {version} may have compatibility issues")
        
        # Test basic functionality
        vectorizer = TfidfVectorizer(max_features=100)
        svm = SVC(kernel='linear', probability=True)
        
        # Test with dummy data
        texts = ["positive sentiment", "negative sentiment", "neutral text"]
        labels = [1, 0, 1]
        
        X = vectorizer.fit_transform(texts)
        svm.fit(X, labels)
        
        # Test prediction
        test_text = ["test prediction"]
        test_X = vectorizer.transform(test_text)
        prediction = svm.predict(test_X)
        
        print("✅ sklearn basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ sklearn compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_nltk_setup():
    """Test NLTK setup and data downloading"""
    print("\n🧪 Testing NLTK setup...")
    
    try:
        import nltk
        
        # Test data directory
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir, exist_ok=True)
            print(f"📁 Created NLTK data directory: {nltk_data_dir}")
        
        # Test downloads
        required_data = ['punkt', 'punkt_tab', 'stopwords']
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}' if 'punkt' in data_name else f'corpora/{data_name}')
                print(f"✅ {data_name} data already available")
            except LookupError:
                print(f"📥 Downloading {data_name}...")
                try:
                    nltk.download(data_name, quiet=True)
                    print(f"✅ {data_name} downloaded successfully")
                except Exception as download_error:
                    print(f"❌ Failed to download {data_name}: {download_error}")
                    return False
        
        # Test tokenization
        from nltk.tokenize import word_tokenize
        test_text = "This is a test sentence for tokenization."
        tokens = word_tokenize(test_text)
        print(f"✅ Tokenization test passed: {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ NLTK setup test failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading functions"""
    print("\n🧪 Testing model loading...")
    
    try:
        # Test if model files exist
        model_files = [
            "models/svm_model_predict.pkl",
            "models/tfidf_vectorizer_predict.pkl"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"✅ Found model file: {model_file}")
            else:
                print(f"⚠️ Model file not found: {model_file}")
        
        # Test if we can import our utilities
        try:
            from ui.utils import load_prediction_model, check_sklearn_version_compatibility
            print("✅ Utility functions imported successfully")
            
            # Test compatibility check
            is_compatible = check_sklearn_version_compatibility()
            print(f"📊 sklearn compatibility check: {'✅ Compatible' if is_compatible else '⚠️ May have issues'}")
            
            # Test model loading (if files exist)
            if all(os.path.exists(f) for f in model_files):
                svm_model, tfidf_vectorizer = load_prediction_model()
                if svm_model is not None and tfidf_vectorizer is not None:
                    print("✅ Models loaded successfully")
                else:
                    print("⚠️ Models exist but failed to load")
            else:
                print("ℹ️ Model files not found - this is expected for fresh deployment")
                
        except ImportError as import_error:
            print(f"❌ Failed to import utility functions: {import_error}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        traceback.print_exc()
        return False

def test_ui_imports():
    """Test UI module imports"""
    print("\n🧪 Testing UI module imports...")
    
    try:
        # Test auth module
        from ui.auth import auth
        print("✅ Auth module imported successfully")
        
        # Test tools modules 
        ui_modules = [
            'ui.tools.Dashboard_Ringkasan',
            'ui.tools.Prediksi_Sentimen', 
            'ui.tools.Analisis_Data'
        ]
        
        for module_name in ui_modules:
            try:
                __import__(module_name)
                print(f"✅ {module_name} imported successfully")
            except ImportError as e:
                print(f"❌ Failed to import {module_name}: {e}")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ UI imports test failed: {e}")
        traceback.print_exc()  
        return False

def main():
    """Run all tests"""
    print("🚀 SentimenGo Cloud Deployment Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("sklearn Compatibility", test_sklearn_compatibility), 
        ("NLTK Setup", test_nltk_setup),
        ("Model Loading", test_model_loading),
        ("UI Imports", test_ui_imports)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for cloud deployment.")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
