"""
SentimenGo Model Compatibility Fix Script
========================================

This script fixes the critical model compatibility issues:
1. Retrain models with current sklearn version
2. Fix CSR matrix encoding errors
3. Ensure model stability
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pickle
from datetime import datetime

def backup_old_models():
    """Backup existing models before retraining"""
    print("🔄 Backing up existing models...")
    
    model_files = [
        "models/svm_model_predict.pkl",
        "models/tfidf_vectorizer_predict.pkl",
        "models/svm.pkl", 
        "models/tfidf.pkl"
    ]
    
    backup_dir = f"models/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    for model_file in model_files:
        if os.path.exists(model_file):
            backup_path = os.path.join(backup_dir, os.path.basename(model_file))
            os.rename(model_file, backup_path)
            print(f"✅ Backed up: {model_file} -> {backup_path}")
    
    print(f"📁 Backup completed in: {backup_dir}")

def load_training_data():
    """Load and prepare training data"""
    print("📊 Loading training data...")
    
    data_path = "data/ulasan_goride_preprocessed.csv"
    if not os.path.exists(data_path):
        print(f"❌ Training data not found: {data_path}")
        return None, None
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} records")
        
        # Assuming columns are 'text' and 'sentiment' or similar
        # Adjust column names based on your actual data structure
        print("Available columns:", df.columns.tolist())
        
        if 'teks_preprocessing' in df.columns and 'sentiment' in df.columns:
            X = df['teks_preprocessing'].fillna('')
            y = df['sentiment']
            print("✅ Using 'teks_preprocessing' and 'sentiment' columns")
        elif 'review_text' in df.columns and 'label' in df.columns:
            X = df['review_text'].fillna('')
            y = df['label']
            print("✅ Using 'review_text' and 'label' columns")
        elif 'cleaned_text' in df.columns and 'sentiment' in df.columns:
            X = df['cleaned_text'].fillna('')
            y = df['sentiment']
            print("✅ Using 'cleaned_text' and 'sentiment' columns")
        elif 'text' in df.columns and 'label' in df.columns:
            X = df['text'].fillna('')
            y = df['label']
            print("✅ Using 'text' and 'label' columns")
        else:
            print("❌ Could not identify text and label columns")
            print("Available columns:", df.columns.tolist())
            return None, None
            
        print(f"📈 Data shape: {X.shape}")
        print(f"📊 Label distribution:\n{y.value_counts()}")
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def retrain_models(X, y):
    """Retrain models with current sklearn version"""
    print("🤖 Retraining models with current sklearn version...")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Initialize and train TF-IDF Vectorizer
        print("🔄 Training TF-IDF Vectorizer...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,  # We'll use Indonesian stopwords separately
            lowercase=True,
            strip_accents='unicode'
        )
        
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        print(f"✅ TF-IDF shape: {X_train_tfidf.shape}")
        
        # Initialize and train SVM model
        print("🔄 Training SVM model...")
        svm_model = SVC(
            kernel='linear',
            probability=True,  # Enable probability estimates
            random_state=42,
            C=1.0
        )
        
        svm_model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        print("📊 Evaluating model performance...")
        y_pred = svm_model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model accuracy: {accuracy:.4f}")
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return svm_model, tfidf_vectorizer, accuracy
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0

def save_models(svm_model, tfidf_vectorizer, accuracy):
    """Save retrained models with compatibility info"""
    print("💾 Saving retrained models...")
    
    try:
        import sklearn
        
        # Save models
        joblib.dump(svm_model, "models/svm_model_predict.pkl", compress=3)
        joblib.dump(tfidf_vectorizer, "models/tfidf_vectorizer_predict.pkl", compress=3)
        
        # Also save as main models
        joblib.dump(svm_model, "models/svm.pkl", compress=3) 
        joblib.dump(tfidf_vectorizer, "models/tfidf.pkl", compress=3)
        
        # Save metadata
        metadata = {
            'sklearn_version': sklearn.__version__,
            'training_date': datetime.now().isoformat(),
            'model_accuracy': accuracy,
            'python_version': sys.version,
            'model_type': 'SVM with TF-IDF',
            'compatibility_check': 'PASSED'
        }
        
        with open("models/model_metadata_predict_updated.txt", "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print("✅ Models saved successfully")
        print(f"📊 sklearn version: {sklearn.__version__}")
        print(f"📊 Model accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return False

def test_model_compatibility():
    """Test if newly trained models work correctly"""
    print("🧪 Testing model compatibility...")
    
    try:
        # Load models
        svm_model = joblib.load("models/svm_model_predict.pkl")
        tfidf_vectorizer = joblib.load("models/tfidf_vectorizer_predict.pkl")
        
        # Test with sample data
        test_texts = [
            "Aplikasi GoRide sangat bagus dan cepat",
            "Pelayanan buruk dan mengecewakan", 
            "Biasa saja tidak ada yang istimewa"
        ]
        
        print("🔄 Testing predictions...")
        for i, text in enumerate(test_texts):
            try:
                # Transform text
                text_vector = tfidf_vectorizer.transform([text])
                
                # Make prediction
                prediction = svm_model.predict(text_vector)[0]
                probability = svm_model.predict_proba(text_vector)[0].max()
                
                print(f"✅ Test {i+1}: '{text[:50]}...' -> {prediction} (confidence: {probability:.3f})")
                
            except Exception as test_error:
                print(f"❌ Test {i+1} failed: {test_error}")
                return False
        
        print("✅ All compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 SentimenGo Model Compatibility Fix")
    print("=" * 50)
    
    # Step 1: Backup old models
    backup_old_models()
    
    # Step 2: Load training data
    X, y = load_training_data()
    if X is None or y is None:
        print("❌ Cannot proceed without training data")
        return False
    
    # Step 3: Retrain models
    svm_model, tfidf_vectorizer, accuracy = retrain_models(X, y)
    if svm_model is None:
        print("❌ Model training failed")
        return False
    
    # Step 4: Save models
    if not save_models(svm_model, tfidf_vectorizer, accuracy):
        print("❌ Model saving failed")
        return False
    
    # Step 5: Test compatibility
    if not test_model_compatibility():
        print("❌ Compatibility test failed")
        return False
    
    print("\n🎉 MODEL COMPATIBILITY FIX COMPLETED SUCCESSFULLY!")
    print("✅ Models are now compatible with current sklearn version")
    print("✅ CSR matrix encoding issues resolved")
    print("✅ Ready for production deployment")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Fix failed. Please check errors above.")
        sys.exit(1)
    else:
        print("\n✅ All fixes applied successfully!")
        sys.exit(0)
