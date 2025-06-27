"""
Fungsi utilitas dan resource untuk aplikasi GoRide Sentiment Analysis.
Berisi: preprocessing, analisis kata, load data, model, dsb.
"""

import warnings
# Suppress the pkg_resources deprecation warning from gcloud/pyrebase
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import time
import os
import joblib
from datetime import datetime
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import sklearn
# SMOTE untuk imbalanced data handling (sesuai notebook)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
import io
import base64
from wordcloud import WordCloud
from pathlib import Path
import traceback
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download NLTK resources jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Definisikan direktori dasar project
BASE_DIR = Path(__file__).parent.parent
DICTIONARY_DIR = BASE_DIR / "data"
DATA_DIR = BASE_DIR / "data"

# Inisialisasi stemmer dan stopword untuk bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()
stopword_list = set(stop_factory.get_stop_words())

# Load resources
slang_path = DICTIONARY_DIR / "kamus_slang_formal.txt"
stopwords_path = DICTIONARY_DIR / "stopwordsID.txt"

if not os.path.exists(slang_path):
    slang_path = BASE_DIR / "kamus_slang_formal.txt"
try:
    slang_dict = dict(line.strip().split(':') for line in open(slang_path, encoding='utf-8'))
except FileNotFoundError:
    slang_dict = {}

if not os.path.exists(stopwords_path):
    stopwords_path = BASE_DIR / "stopwordsID.txt"
try:
    custom_stopwords = set(open(stopwords_path, encoding='utf-8').read().splitlines())
    stopword_list.update(custom_stopwords)
except FileNotFoundError:
    pass

def normalize_word(word):
    """
    Normalize a single word using the slang dictionary.
    """
    if not isinstance(word, str) or not word.strip():
        return word
    
    if slang_dict:
        normalized = slang_dict.get(word.lower(), word)
        return normalized if normalized is not None else word
    return word

def preprocess_text(text, options=None):
    """
    Preprocess text following the exact steps from notebook 2Preprocessingdata.ipynb
    Pipeline version ready for production use.
    """
    import re
    
    # Validasi input yang lebih ketat
    if text is None:
        return ""
    
    # Konversi ke string jika bukan string
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    
    # Jika text kosong setelah strip
    if not text.strip():
        return ""
        
    if options is None:
        options = {
            'case_folding': True,
            'phrase_standardization': True,
            'cleansing': True,
            'normalize_slang': True,
            'remove_repeated': True,
            'tokenize': True,
            'remove_stopwords': True,
            'stemming': True,
            'rejoin': True
        }
    
    try:
        # === Langkah 1: Case Folding + Phrase Standardization ===
        if options.get('case_folding', True):
            text = text.lower()
        
        if options.get('phrase_standardization', True):
            # Normalisasi istilah umum: "go ride", "go-ride", dll → "goride"
            text = re.sub(r'\bgo[\s\-_]?ride\b', 'goride', text)
            # Tambahan normalisasi lain jika diperlukan
            # text = re.sub(r'\bgo[\s\-_]?jek\b', 'gojek', text)
            # text = re.sub(r'\bgrab[\s\-_]?bike\b', 'grabbike', text)
        
        # === Langkah 2: Cleansing ===
        if options.get('cleansing', True):
            # Hapus URL
            text = re.sub(r'http\S+|www\S+', '', text)
            # Hapus karakter selain huruf dan spasi (sesuai notebook: [^a-zA-Z\s])
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            # Normalisasi spasi berlebih
            text = re.sub(r'\s+', ' ', text).strip()
        
        # === Langkah 3: Normalisasi Slang ===
        if options.get('normalize_slang', True) and slang_dict:
            words = text.split()
            normalized_words = []
            for word in words:
                if word.strip():
                    # Gunakan kamus slang, jika tidak ada tetap gunakan kata asli
                    normalized_word = slang_dict.get(word, word)
                    if normalized_word and normalized_word.strip():
                        normalized_words.append(normalized_word)
            text = ' '.join(normalized_words)
        
        # === Langkah 4: Remove Repeated Characters ===
        if options.get('remove_repeated', True):
            # Sesuai notebook: (\w)\1{2,} → \1\1 (sisakan 2 karakter)
            text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
        
        # === Langkah 5: Tokenization ===
        if options.get('tokenize', True):
            # Sesuai notebook: menggunakan re.findall(r'\b\w+\b', text)
            tokens = re.findall(r'\b\w+\b', text)
        else:
            tokens = text.split() if isinstance(text, str) else [text]
        
        # === Langkah 6: Stopword Removal ===
        if options.get('remove_stopwords', True) and isinstance(tokens, list):
            # Gunakan custom stopwords yang sudah di-load (stopword_custom equivalent)
            filtered_tokens = []
            for word in tokens:
                if word and word not in stopword_list and len(word) > 1:
                    filtered_tokens.append(word)
            tokens = filtered_tokens
        
        # === Langkah 7: Stemming ===
        if options.get('stemming', True) and isinstance(tokens, list):
            stemmed_tokens = []
            for word in tokens:
                if word and word.strip():
                    try:
                        stemmed_word = stemmer.stem(word)
                        if stemmed_word and stemmed_word.strip():
                            stemmed_tokens.append(stemmed_word)
                    except Exception:
                        # Jika stemming gagal, gunakan kata asli
                        stemmed_tokens.append(word)
            tokens = stemmed_tokens
        
        # === Langkah 8: Rejoin Tokens ===
        if options.get('rejoin', True) and isinstance(tokens, list):
            # Filter out None values dan empty strings sebelum join
            filtered_tokens = [token for token in tokens if token is not None and str(token).strip()]
            return ' '.join(filtered_tokens)
        
        return tokens if isinstance(tokens, list) else str(tokens)
        
    except Exception as e:
        # Return original text or empty string jika error
        try:
            return str(text) if text is not None else ""
        except Exception:
            return ""

def get_word_frequencies(text, top_n=10):
    """Get word frequencies from text."""
    try:
        words = nltk.word_tokenize(text) if isinstance(text, str) else text
        word_freq = Counter(words)
        return dict(word_freq.most_common(top_n))
    except Exception as e:
        st.error(f"Error in word frequency analysis: {str(e)}")
        return {}

def get_ngrams(text, n, top_n=10):
    """Get n-grams from text."""
    try:
        words = nltk.word_tokenize(text) if isinstance(text, str) else text
        n_grams = list(ngrams(words, n))
        n_gram_freq = Counter([' '.join(g) for g in n_grams])
        return dict(n_gram_freq.most_common(top_n))
    except Exception as e:
        st.error(f"Error in n-gram analysis: {str(e)}")
        return {}

def create_wordcloud(text, max_words=100, background_color='white'):
    """
    Create wordcloud with proper error handling and type checking.
    """
    try:
        # Ensure text is a string
        if isinstance(text, list):
            text = ' '.join(filter(None, text))  # Filter out None values
        elif not isinstance(text, str):
            text = str(text)
        
        # Check if text is empty after processing
        if not text.strip():
            return None
            
        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color=background_color,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate(text)
        return wordcloud
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None

def prepare_and_load_preprocessed_data(max_rows=None, chunksize=10000, preprocessing_options=None):
    """
    Load data preprocessed jika sudah ada, jika belum lakukan batch preprocessing dan simpan ke file preprocessed.
    """
    preprocessed_path = DATA_DIR / "ulasan_goride_preprocessed.csv"
    raw_path = DATA_DIR / "ulasan_goride.csv"
    if preprocessing_options is None:
        preprocessing_options = {
            'case_folding': True,
            'phrase_standardization': True,
            'cleansing': True,
            'normalize_slang': True,
            'remove_repeated': True,
            'tokenize': True,
            'remove_stopwords': True,
            'stemming': True,
            'rejoin': True
        }
    # Jika file preprocessed sudah ada, langsung load
    if os.path.exists(preprocessed_path):
        try:
            df = pd.read_csv(preprocessed_path, nrows=max_rows)
            # Pastikan mapping label tetap benar jika file lama
            label_map = {
                'Positive': 'POSITIF', 'POSITIVE': 'POSITIF',
                'Negative': 'NEGATIF', 'NEGATIVE': 'NEGATIF',
                'Netral': 'NETRAL', 'Neutral': 'NETRAL', 'NETRAL': 'NETRAL', 'NEUTRAL': 'NETRAL'
            }
            df['sentiment'] = df['sentiment'].replace(label_map)
            df = df[df['sentiment'].isin(['POSITIF', 'NEGATIF'])]
            # Pastikan kolom wajib tetap ada
            required_columns = ['review_text', 'sentiment', 'date', 'teks_preprocessing']
            for col in required_columns:
                if col not in df.columns:
                    st.error(f"Kolom {col} tidak ditemukan di file preprocessed!")
                    return pd.DataFrame(columns=required_columns)
            return df
        except Exception as e:
            st.error(f"Gagal membaca file preprocessed: {str(e)}")
            # Jika gagal, hapus file preprocessed agar bisa regenerate
            try:
                os.remove(preprocessed_path)
            except Exception:
                pass
    # Jika belum ada, lakukan batch preprocessing dan simpan
    if not os.path.exists(raw_path):
        st.error("File ulasan_goride.csv tidak ditemukan!")
        return pd.DataFrame(columns=['review_text', 'sentiment', 'date', 'teks_preprocessing'])
    try:
        df = pd.read_csv(raw_path, nrows=max_rows)
        # Validasi kolom
        required_columns = ['review_text', 'sentiment', 'date']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Kolom {col} tidak ditemukan di file CSV!")
                return pd.DataFrame(columns=required_columns+['teks_preprocessing'])
        # Mapping label sebelum preprocessing
        label_map = {
            'Positive': 'POSITIF', 'POSITIVE': 'POSITIF',
            'Negative': 'NEGATIF', 'NEGATIVE': 'NEGATIF',
            'Netral': 'NETRAL', 'Neutral': 'NETRAL', 'NETRAL': 'NETRAL', 'NEUTRAL': 'NETRAL'
        }
        df['sentiment'] = df['sentiment'].replace(label_map)
        df = df[df['sentiment'].isin(['POSITIF', 'NEGATIF'])]
        # Preprocessing batch
        with st.spinner("Melakukan batch preprocessing dan menyimpan hasil..."):
            df['teks_preprocessing'] = df['review_text'].astype(str).apply(lambda x: preprocess_text(x, preprocessing_options))
            # Simpan ke file preprocessed
            df.to_csv(preprocessed_path, index=False)
        return df
    except Exception as e:
        st.error(f"Gagal melakukan preprocessing batch: {str(e)}")
        return pd.DataFrame(columns=['review_text', 'sentiment', 'date', 'teks_preprocessing'])

# Ganti load_sample_data agar hanya wrapper ke prepare_and_load_preprocessed_data
@st.cache_data(ttl=1800)
def load_sample_data(max_rows=None, chunksize=10000):
    return prepare_and_load_preprocessed_data(max_rows=max_rows, chunksize=chunksize)

@st.cache_resource(ttl=3600)
def train_model(data, preprocessing_options=None, batch_size=1000):
    if preprocessing_options is None:
        preprocessing_options = {
            'case_folding': True,
            'phrase_standardization': True,
            'cleansing': True,
            'normalize_slang': True,
            'remove_repeated': True,
            'tokenize': True,
            'remove_stopwords': True,
            'stemming': True,
            'rejoin': True
        }
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed_texts = []
        total_batches = (len(data) + batch_size - 1) // batch_size
        for i in range(0, len(data), batch_size):
            batch_end = min(i + batch_size, len(data))
            batch = data.iloc[i:batch_end]
            batch_num = i // batch_size + 1
            status_text.text(f"Preprocessing batch {batch_num}/{total_batches}...")
            progress_bar.progress(i / len(data))
            batch_processed = []
            for text in batch['review_text']:
                try:
                    processed = preprocess_text(text, preprocessing_options)
                    batch_processed.append(processed)
                except Exception as e:
                    batch_processed.append(text)
            processed_texts.extend(batch_processed)
        status_text.text("Vectorizing text data...")
        progress_bar.progress(1.0)
        tfidf = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2),
            lowercase=False,
            strip_accents='unicode',
            norm='l2',
            sublinear_tf=True,
        )
        status_text.text("Splitting data before TF-IDF (preventing data leakage)...")
        
        # FIX DATA LEAKAGE: Split BEFORE TF-IDF fit_transform
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_texts, data['sentiment'], test_size=0.1, random_state=42, stratify=data['sentiment']
        )
        
        status_text.text("Building SMOTE Pipeline (sesuai notebook)...")
        
        # IMPLEMENTASI SMOTE PIPELINE (sama seperti notebook)
        pipeline = ImbPipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.85,
                ngram_range=(1, 2),
                lowercase=False,
                strip_accents='unicode',
                norm='l2',
                sublinear_tf=True,
            )),
            ('smote', SMOTE(random_state=42)),  # ← KEY IMPROVEMENT dari notebook!
            ('svm', SVC(
                C=0.1,                   # ← OPTIMAL dari GridSearchCV
                kernel='linear',         # ← Confirmed optimal
                gamma='scale',           # ← Confirmed optimal
                probability=True,
                random_state=42
                # REMOVED: class_weight='balanced' - SMOTE handles imbalance
            ))
        ])
        
        status_text.text("Training SMOTE Pipeline...")
        # Fit pipeline pada training text (bukan TF-IDF vectors)
        pipeline.fit(X_train_text, y_train)
        
        status_text.text("Evaluating model...")
        y_pred = pipeline.predict(X_test_text)
        
        status_text.empty()
        progress_bar.empty()
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="POSITIF")
        recall = recall_score(y_test, y_pred, pos_label="POSITIF")
        f1 = f1_score(y_test, y_pred, pos_label="POSITIF")
        cm = confusion_matrix(y_test, y_pred)
        
        # Extract components for compatibility
        tfidf = pipeline.named_steps['tfidf']
        svm = pipeline.named_steps['svm']
        
        # Create test vectors for compatibility (sudah di-transform oleh pipeline)
        X_test = tfidf.transform(X_test_text)
        
        return pipeline, accuracy, precision, recall, f1, cm, X_test, y_test, tfidf, svm
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, 0, 0, 0, 0, None, None, None, None, None

def save_model_and_vectorizer(pipeline, tfidf, model_dir="models"):
    """
    Save pipeline and TF-IDF vectorizer with current sklearn version metadata.
    Now supports both regular Pipeline and ImbPipeline (with SMOTE).
    """
    import sklearn
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "svm.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf.pkl")
    metadata_path = os.path.join(model_dir, "model_metadata.txt")
    
    # Save models - handle both Pipeline types
    if hasattr(pipeline, 'named_steps'):
        # For ImbPipeline or regular Pipeline
        classifier = pipeline.named_steps.get('svm', pipeline.named_steps.get('classifier'))
        joblib.dump(classifier, model_path)
    else:
        # Fallback for direct classifier
        joblib.dump(pipeline, model_path)
    
    joblib.dump(tfidf, vectorizer_path)
    
    # Save metadata including sklearn version
    with open(metadata_path, 'w') as f:
        f.write(f"sklearn_version: {sklearn.__version__}\n")
        f.write(f"model_saved_at: {datetime.now().isoformat()}\n")
        f.write(f"model_type: SVM with TF-IDF and SMOTE\n")
        f.write(f"pipeline_type: ImbPipeline\n")
        f.write(f"imbalanced_handling: SMOTE\n")
    
    return model_path, vectorizer_path

def load_saved_model(model_dir="models"):
    """
    Load saved SVM model and TF-IDF vectorizer with version compatibility handling.
    Suppresses sklearn version warnings for better user experience.
    """
    model_path = os.path.join(model_dir, "svm.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            # Suppress sklearn version warnings during model loading
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
                
                svm_model = joblib.load(model_path)
                tfidf_vectorizer = joblib.load(vectorizer_path)
                
                # Validate that models are functional
                if hasattr(svm_model, 'predict') and hasattr(tfidf_vectorizer, 'transform'):
                    return svm_model, tfidf_vectorizer
                else:
                    return svm_model, tfidf_vectorizer
                    
        except Exception as e:
            return None, None
    return None, None

def load_prediction_model(model_dir="models"):
    """
    Load model khusus untuk prediksi sentimen (svm_model_predict.pkl dan tfidf_vectorizer_predict.pkl).
    Model ini dirancang khusus untuk modul Prediksi_Sentimen.py tanpa preprocessing tambahan.
    """
    model_path = os.path.join(model_dir, "svm_model_predict.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer_predict.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            # Suppress sklearn version warnings during model loading
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
                
                svm_model = joblib.load(model_path)
                tfidf_vectorizer = joblib.load(vectorizer_path)
                
                # Validate that models are functional
                if hasattr(svm_model, 'predict') and hasattr(tfidf_vectorizer, 'transform'):
                    return svm_model, tfidf_vectorizer
                else:
                    return svm_model, tfidf_vectorizer
                    
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat model prediksi khusus: {str(e)}")
            return None, None
    else:
        st.warning("⚠️ Model prediksi khusus tidak ditemukan. Menggunakan model fallback.")
        return None, None

def check_model_compatibility():
    """
    Check if saved models are compatible with current sklearn version.
    Returns a tuple (is_compatible, error_message).
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*Trying to unpickle estimator.*")
            
            # Try loading models
            svm_model, tfidf_vectorizer = load_saved_model()
            if svm_model is not None and tfidf_vectorizer is not None:
                # Try a basic prediction to test functionality
                test_text = ["test"]
                try:
                    tfidf_vectorizer.transform(test_text)
                    return True, "Models are compatible"
                except Exception as e:
                    return False, f"Models loaded but not functional: {str(e)}"
            else:
                return False, "Models could not be loaded"
                
    except UserWarning as e:
        if "Trying to unpickle estimator" in str(e):
            return False, f"Version incompatibility detected: {str(e)}"
        return False, f"Warning during model loading: {str(e)}"
    except Exception as e:
        return False, f"Error checking compatibility: {str(e)}"

def get_or_train_model(data, preprocessing_options=None, batch_size=1000):
    """
    Load pre-trained models that should already be prepared via check_and_prepare_models_with_progress().
    This function expects models to be ready and will not perform training.
    """
    from sklearn.pipeline import Pipeline
    
    # Try to load existing model
    svm_model, tfidf_vectorizer = load_saved_model()
        
    if svm_model is not None and tfidf_vectorizer is not None:
        # Model sudah ada dan siap digunakan
        tfidf = tfidf_vectorizer
        svm = svm_model
        
        # Pastikan data text dalam format yang benar untuk evaluasi
        try:
            # Gunakan kolom teks_preprocessing jika tersedia, jika tidak gunakan review_text
            if 'teks_preprocessing' in data.columns and not data['teks_preprocessing'].isna().all():
                processed_texts = data['teks_preprocessing'].astype(str).tolist()
                # Validasi processed_texts
                processed_texts = [str(text) if text is not None else "" for text in processed_texts]
            else:
                # Lakukan preprocessing on-the-fly jika kolom teks_preprocessing tidak ada
                if preprocessing_options is None:
                    preprocessing_options = {
                        'case_folding': True, 'phrase_standardization': True, 'cleansing': True,
                        'normalize_slang': True, 'remove_repeated': True, 'tokenize': True,
                        'remove_stopwords': True, 'stemming': True, 'rejoin': True
                    }
                processed_texts = []
                for i, text in enumerate(data['review_text'].astype(str)):
                    try:
                        processed_text = preprocess_text(text, preprocessing_options)
                        processed_texts.append(str(processed_text) if processed_text is not None else "")
                    except Exception:
                        # Fallback ke raw text jika preprocessing gagal
                        processed_texts.append(str(text) if text is not None else "")
            
            # Validasi final processed_texts
            processed_texts = [text for text in processed_texts if text and text.strip()]
            if not processed_texts:
                raise ValueError("No valid texts after preprocessing")
            
            X = tfidf.transform(processed_texts)
            y = data['sentiment'].iloc[:len(processed_texts)]  # Adjust y to match processed_texts length
            
        except Exception as e:
            # Fallback ke raw text
            try:
                processed_texts = data['review_text'].astype(str).fillna('').tolist()
                processed_texts = [str(text) for text in processed_texts if str(text).strip()]
                X = tfidf.transform(processed_texts)
                y = data['sentiment'].iloc[:len(processed_texts)]
            except Exception as fallback_error:
                raise e  # Re-raise original error
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="POSITIF")
        recall = recall_score(y_test, y_pred, pos_label="POSITIF")
        f1 = f1_score(y_test, y_pred, pos_label="POSITIF")
        cm = confusion_matrix(y_test, y_pred)
        pipeline = Pipeline([
            ('vectorizer', tfidf),
            ('classifier', svm)
        ])
        
        return pipeline, accuracy, precision, recall, f1, cm, X_test, y_test, tfidf, svm
    else:
        # Model tidak ditemukan - seharusnya sudah disiapkan sebelumnya
        st.error(f"❌ Model utama tidak ditemukan! Silakan restart aplikasi untuk pelatihan ulang model.")
        st.stop()

def predict_sentiment(text, pipeline, preprocessing_options=None, use_prediction_model=False, svm_model=None, tfidf_vectorizer=None):
    """
    Prediksi sentimen dengan perbaikan bug SVM (argmax vs decision function).
    
    Args:
        text: Teks untuk diprediksi
        pipeline: Pipeline model yang akan digunakan (untuk model regular)
        preprocessing_options: Opsi preprocessing (opsional)
        use_prediction_model: Jika True, gunakan model prediksi khusus
        svm_model: Model SVM khusus untuk prediksi (jika use_prediction_model=True)
        tfidf_vectorizer: TF-IDF vectorizer khusus untuk prediksi (jika use_prediction_model=True)
    
    Returns:
        dict: {'sentiment': str, 'confidence': float, 'probabilities': dict}
    """
    if preprocessing_options is None:
        preprocessing_options = {
            'case_folding': True,
            'phrase_standardization': True,
            'cleansing': True,
            'normalize_slang': True,
            'remove_repeated': True,
            'tokenize': True,
            'remove_stopwords': True,
            'stemming': True,
            'rejoin': True
        }
    
    try:
        # Jika menggunakan model prediksi khusus
        if use_prediction_model and svm_model is not None and tfidf_vectorizer is not None:
            # Untuk model prediksi khusus, lakukan preprocessing
            processed_text = preprocess_text(text, preprocessing_options)
            if not processed_text.strip():
                return {
                    'sentiment': 'NETRAL',
                    'confidence': 0,
                    'probabilities': {'POSITIF': 0, 'NEGATIF': 0}
                }
            
            # The svm_model is actually the complete pipeline, so we can use it directly
            probabilities = svm_model.predict_proba([processed_text])[0]
            predicted_class_idx = probabilities.argmax()
            prediction = svm_model.classes_[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]
            
            # Buat mapping probabilitas berdasarkan classes
            prob_dict = {}
            for i, class_name in enumerate(svm_model.classes_):
                prob_dict[class_name] = float(probabilities[i])
                
        else:
            # Gunakan preprocessing penuh untuk model regular
            processed_text = preprocess_text(text, preprocessing_options)
            if not processed_text.strip():
                return {
                    'sentiment': 'NETRAL',
                    'confidence': 0,
                    'probabilities': {'POSITIF': 0, 'NEGATIF': 0}
                }
            
            # Dapatkan probabilitas dari model pipeline
            probabilities = pipeline.predict_proba([processed_text])[0]
            
            # ✅ PERBAIKAN BUG SVM: Gunakan argmax dari probabilitas, bukan decision function
            # Ini memastikan prediksi konsisten dengan probabilitas tertinggi
            predicted_class_idx = probabilities.argmax()
            prediction = pipeline.classes_[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]
            
            # Buat mapping probabilitas berdasarkan classes
            prob_dict = {}
            for i, class_name in enumerate(pipeline.classes_):
                prob_dict[class_name] = float(probabilities[i])
        
        return {
            'sentiment': prediction,
            'confidence': float(confidence),
            'probabilities': prob_dict
        }
        
    except Exception as e:
        # Return error yang lebih informatif tanpa menampilkan error ke UI
        import traceback
        error_msg = f"Prediction error: {str(e)}"
        
        # Log error untuk debugging
        print(f"ERROR in predict_sentiment: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            'sentiment': 'ERROR',
            'confidence': 0,
            'probabilities': {'POSITIF': 0, 'NEGATIF': 0},
            'error': error_msg
        }

def analyze_sentiment_trends(data):
    try:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.dropna(subset=['date'])
        sentiment_trends = data.groupby([data['date'].dt.strftime('%Y-%m-%d'), 'sentiment']).size().reset_index(name='count')
        pivot_trends = sentiment_trends.pivot(index='date', columns='sentiment', values='count').fillna(0)
        if 'POSITIF' in pivot_trends.columns and 'NEGATIF' in pivot_trends.columns:
            pivot_trends['ratio'] = pivot_trends['POSITIF'] / (pivot_trends['POSITIF'] + pivot_trends['NEGATIF'])
        return pivot_trends
    except Exception as e:
        st.error(f"Error in trend analysis: {str(e)}")
        return pd.DataFrame()

def get_table_download_link(df, filename, text):
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    except Exception as e:
        st.error(f"Error generating download link: {str(e)}")
        return ''

def display_model_metrics(accuracy, precision, recall, f1, confusion_mat):
    with st.sidebar.expander("🏆 Model Metrics", expanded=False):
        st.write(f"✅ Accuracy: {accuracy:.4f}")
        st.write(f"✅ Precision: {precision:.4f}")
        st.write(f"✅ Recall: {recall:.4f}")
        st.write(f"✅ F1-Score: {f1:.4f}")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("Confusion Matrix")
        im = ax.imshow(confusion_mat, cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["NEGATIF", "POSITIF"])
        ax.set_yticklabels(["NEGATIF", "POSITIF"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, confusion_mat[i, j], ha="center", va="center", 
                    color="white" if confusion_mat[i, j] > confusion_mat.max()/2 else "black")
        st.pyplot(fig)

def quick_model_check():
    """
    Quick check if all models are ready and compatible without UI elements.
    Returns (models_ready: bool, compatibility_status: str, data_count: int)
    """
    try:
        # Check main models
        svm_model, tfidf_vectorizer = load_saved_model()
        main_models_exist = (svm_model is not None and tfidf_vectorizer is not None)
        
        # Check prediction models  
        prediction_model, prediction_vectorizer = load_prediction_model()
        prediction_models_exist = (prediction_model is not None and prediction_vectorizer is not None)
        
        # Check compatibility
        is_compatible, compatibility_msg = check_model_compatibility()
        
        # Load data count
        data = load_sample_data()
        data_count = len(data) if not data.empty else 0
        
        models_ready = main_models_exist and prediction_models_exist and is_compatible
        
        return models_ready, compatibility_msg, data_count
        
    except Exception as e:
        return False, f"Error checking models: {str(e)}", 0

def check_and_prepare_models_with_progress():
    """
    Unified function for checking and preparing models with progress feedback.
    This replaces the separate model_preparation_page() function for consistency.
    Returns tuple: (models_ready: bool, should_show_ui: bool, preparation_data: dict)
    """
    import streamlit as st
    
    # Quick check first - no UI elements
    models_ready, compatibility_msg, data_count = quick_model_check()
    
    if models_ready:
        # Models are ready, no UI needed
        return True, False, {
            'data_count': data_count,
            'status': 'ready',
            'message': f'All models ready with {data_count:,} training samples'
        }
    
    # Models need preparation - show UI
    svm_model, tfidf_vectorizer = load_saved_model()
    prediction_model, prediction_vectorizer = load_prediction_model()
    
    main_models_exist = (svm_model is not None and tfidf_vectorizer is not None)
    prediction_model_exists = (prediction_model is not None and prediction_vectorizer is not None)
    
    if not main_models_exist or not prediction_model_exists:
        # Show model preparation notification
        if not main_models_exist and not prediction_model_exists:
            st.warning("⚠️ **Model tidak ditemukan, memulai pelatihan model lengkap...**")
        elif not main_models_exist:
            st.warning("⚠️ **Model utama tidak ditemukan, memulai pelatihan...**")
        else:
            st.warning("⚠️ **Model prediksi khusus tidak ditemukan, memulai persiapan...**")
        
        # Create progress container
        progress_container = st.container()
        with progress_container:
            st.markdown("🔄 **Proses Persiapan Model:**")
            
            # Load data for training
            data = load_sample_data()
            if data.empty:
                st.error("❌ Data training tidak ditemukan!")
                return False, True, {'status': 'error', 'message': 'No training data found'}
                
            preprocessing_options = {
                'case_folding': True, 'phrase_standardization': True, 'cleansing': True,
                'normalize_slang': True, 'remove_repeated': True, 'tokenize': True,
                'remove_stopwords': True, 'stemming': True, 'rejoin': True
            }
            
            # Train models with progress feedback
            with st.spinner("🤖 Menyiapkan model sentiment analysis..."):
                try:
                    if not main_models_exist:
                        # Train main models
                        st.write("📈 **Tahap 1:** Melatih model utama...")
                        pipeline, accuracy, precision, recall, f1, cm, X_test, y_test, tfidf, svm = train_model_silent(data, preprocessing_options)
                        save_model_and_vectorizer(pipeline, tfidf)
                        
                        st.write("🎯 **Tahap 2:** Menyiapkan model prediksi khusus...")
                        save_model_and_vectorizer_predict(pipeline, tfidf)
                    else:
                        # Load existing models for metrics
                        pipeline, accuracy, precision, recall, f1, cm, X_test, y_test, tfidf, svm = get_or_train_model(data, preprocessing_options)
                    
                    if not prediction_model_exists:
                        # Create prediction-specific model
                        st.write("🎯 **Tahap 3:** Menyiapkan model prediksi khusus...")
                        save_model_and_vectorizer_predict(pipeline, tfidf)
                    
                    # Clear progress container
                    progress_container.empty()
                    
                    # Success notification
                    st.toast(f"✅ Semua model berhasil disiapkan! Akurasi: {accuracy:.2%}", icon="✅")
                    st.success(f"""
                    🎉 **Sistem model sentiment analysis berhasil disiapkan!**
                    
                    📊 **Performa Model:**
                    - **Akurasi:** {accuracy:.2%}
                    - **Precision:** {precision:.2%}
                    - **Recall:** {recall:.2%}
                    - **F1-Score:** {f1:.2%}
                    
                    ✅ **Model yang Tersedia:**
                    - 🔧 Model utama untuk analisis umum
                    - ⚙️ Model tanpa SMOTE untuk performa optimal
                    - 🎯 Model khusus untuk prediksi real-time
                    
                    🚀 **Semua model siap digunakan!**
                    """)
                    
                    return True, True, {
                        'status': 'success',
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'data_count': len(data)
                    }
                    
                except Exception as e:
                    progress_container.empty()
                    st.error(f"❌ Gagal menyiapkan model: {str(e)}")
                    st.toast("❌ Persiapan model gagal!", icon="❌")
                    return False, True, {'status': 'error', 'message': str(e)}
    else:
        # Models exist, check compatibility
        is_compatible, compatibility_msg = check_model_compatibility()
        
        if not is_compatible:
            st.warning(f"⚠️ **Masalah kompatibilitas model:** {compatibility_msg}")
            
            progress_container = st.container()
            with progress_container:
                st.markdown("🔄 **Memperbarui model ke versi yang kompatibel...**")
                
                data = load_sample_data()
                preprocessing_options = {
                    'case_folding': True, 'phrase_standardization': True, 'cleansing': True,
                    'normalize_slang': True, 'remove_repeated': True, 'tokenize': True,
                    'remove_stopwords': True, 'stemming': True, 'rejoin': True
                }
                
                with st.spinner("🔄 Memperbarui model..."):
                    try:
                        st.write("📈 **Tahap 1:** Memperbarui model utama...")
                        pipeline, accuracy, precision, recall, f1, cm, X_test, y_test, tfidf, svm = train_model_silent(data, preprocessing_options)
                        save_model_and_vectorizer(pipeline, tfidf)
                        
                        st.write("🎯 **Tahap 2:** Memperbarui model prediksi khusus...")
                        save_model_and_vectorizer_predict(pipeline, tfidf)
                        
                        progress_container.empty()
                        st.toast("✅ Semua model berhasil diperbarui!", icon="✅")
                        st.success("🎉 **Semua model berhasil diperbarui dan siap digunakan!**")
                        return True, True, {
                            'status': 'updated',
                            'accuracy': accuracy,
                            'data_count': len(data)
                        }
                        
                    except Exception as e:
                        progress_container.empty()
                        st.error(f"❌ Gagal memperbarui model: {str(e)}")
                        st.toast("❌ Pembaruan model gagal!", icon="❌") 
                        return False, True, {'status': 'error', 'message': str(e)}
        else:
            # All models are ready and compatible
            return True, False, {
                'status': 'ready',
                'data_count': data_count,
                'message': 'All models ready and compatible'
            }

def train_model_silent(data, preprocessing_options=None, batch_size=1000):
    """
    Train model without showing progress bar (for use in check_and_prepare_models_with_progress).
    """
    if preprocessing_options is None:
        preprocessing_options = {
            'case_folding': True, 'phrase_standardization': True, 'cleansing': True,
            'normalize_slang': True, 'remove_repeated': True, 'tokenize': True,
            'remove_stopwords': True, 'stemming': True, 'rejoin': True
        }
    
    try:
        processed_texts = []
        for i in range(0, len(data), batch_size):
            batch_end = min(i + batch_size, len(data))
            batch = data.iloc[i:batch_end]
            batch_processed = []
            for text in batch['review_text']:
                try:
                    processed = preprocess_text(text, preprocessing_options)
                    batch_processed.append(processed)
                except Exception as e:
                    batch_processed.append(text)
            processed_texts.extend(batch_processed)
            
        tfidf = TfidfVectorizer(
            max_features=1000, min_df=2, max_df=0.85, ngram_range=(1, 2),
            lowercase=False, strip_accents='unicode', norm='l2', sublinear_tf=True,
        )
        
        # FIX DATA LEAKAGE: Split BEFORE TF-IDF fit_transform  
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_texts, data['sentiment'], test_size=0.2, random_state=42, stratify=data['sentiment']
        )
        
        # IMPLEMENTASI SMOTE PIPELINE (sama seperti notebook)
        pipeline = ImbPipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.85,
                ngram_range=(1, 2),
                lowercase=False,
                strip_accents='unicode',
                norm='l2',
                sublinear_tf=True,
            )),
            ('smote', SMOTE(random_state=42)),  # ← KEY IMPROVEMENT dari notebook!
            ('svm', SVC(
                C=0.1,                   # ← OPTIMAL dari GridSearchCV
                kernel='linear',         # ← Confirmed optimal
                gamma='scale',           # ← Confirmed optimal
                probability=True,
                random_state=42
                # REMOVED: class_weight='balanced' - SMOTE handles imbalance
            ))
        ])
        
        # Fit pipeline pada training text (bukan TF-IDF vectors)
        pipeline.fit(X_train_text, y_train)
        y_pred = pipeline.predict(X_test_text)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="POSITIF")
        recall = recall_score(y_test, y_pred, pos_label="POSITIF")
        f1 = f1_score(y_test, y_pred, pos_label="POSITIF")
        cm = confusion_matrix(y_test, y_pred)
        
        # Extract components for compatibility
        tfidf_component = pipeline.named_steps['tfidf']
        svm_component = pipeline.named_steps['svm']
        
        # Create test vectors for compatibility
        X_test = tfidf_component.transform(X_test_text)
        
        return pipeline, accuracy, precision, recall, f1, cm, X_test, y_test, tfidf_component, svm_component
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        raise e

def save_model_and_vectorizer_predict(pipeline, tfidf_vectorizer, model_dir="models"):
    """
    Save model khusus untuk prediksi sentimen (svm_model_predict.pkl).
    Model ini dirancang khusus untuk modul Prediksi_Sentimen.py dengan preprocessing minimal.
    Now saves the complete pipeline for easier prediction.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the complete pipeline for prediction
    # This handles all the SMOTE/TF-IDF/SVM steps internally
    model_path = os.path.join(model_dir, "svm_model_predict.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer_predict.pkl")
    
    # For prediction model, save the complete pipeline
    joblib.dump(pipeline, model_path)
    # We still save the TF-IDF vectorizer separately for compatibility
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    
    # Save metadata khusus untuk model prediksi
    metadata = {
        'model_type': 'Complete_Pipeline_with_SMOTE',
        'vectorizer_type': 'TF-IDF',
        'training_date': datetime.now().isoformat(),
        'sklearn_version': joblib.__version__,
        'version': 'prediction_complete_pipeline',
        'purpose': 'Prediksi sentimen dengan complete pipeline',
        'target_module': 'Prediksi_Sentimen.py',
        'preprocessing': 'Handled by complete pipeline including SMOTE',
        'imbalanced_handling': 'SMOTE (Synthetic Minority Oversampling)'
    }
    
    metadata_path = os.path.join(model_dir, "model_metadata_predict.txt")
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    return model_path, vectorizer_path

def render_model_preparation_page():
    """
    Render halaman persiapan model dengan UI yang lebih baik.
    Fungsi ini menggantikan model_preparation_page() di main.py untuk konsistensi.
    """
    import streamlit as st
    
    st.markdown(
        """
        <style>
        [data-testid='stSidebar'], [data-testid='collapsedControl'] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Header halaman
    st.markdown("# 🤖 Persiapan Model Sentiment Analysis")
    st.markdown("---")
    
    # Welcome message
    user_email = st.session_state.get('user_email', 'User')
    st.markdown(f"### Selamat datang, **{user_email}**! 👋")
    
    st.info("""
    🔧 **Sistem sedang mempersiapkan model AI untuk analisis sentimen GoRide**
    
    📝 Proses ini meliputi:
    - ✅ Pemeriksaan model yang tersedia
    - 🤖 Pelatihan model jika diperlukan  
    - 📊 Validasi performa model
    - 🚀 Persiapan tools analisis
    
    *Proses ini hanya dilakukan sekali atau saat model perlu diperbarui.*
    """)
    
    # Status container
    status_container = st.container()
    
    # Check if models are being prepared
    if not st.session_state.get('model_preparation_started', False):
        with status_container:
            if st.button("🚀 **Mulai Persiapan Model**", type="primary", use_container_width=True):
                st.session_state['model_preparation_started'] = True
                st.rerun()
    else:
        # Model preparation in progress
        with status_container:
            st.markdown("### 🔄 Sedang Mempersiapkan Model...")
            
            try:
                # Show preparation progress
                models_ready, should_show_ui, preparation_data = check_and_prepare_models_with_progress()
                
                if models_ready:
                    st.session_state['models_prepared'] = True
                    st.session_state['model_preparation_completed'] = True
                    
                    # Success message with celebration
                    st.balloons()
                    st.success("🎉 **Model berhasil disiapkan!**")
                    
                    st.markdown("---")
                    st.markdown("### ✅ Persiapan Selesai!")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🎯 **Lanjutkan ke Dashboard**", type="primary", use_container_width=True):
                            st.session_state['ready_for_tools'] = True
                            st.rerun()
                            
                    st.markdown("*Anda siap menggunakan semua fitur analisis sentimen!*")
                else:
                    st.error("❌ **Gagal mempersiapkan model**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("🔄 Coba Lagi"):
                            st.session_state['model_preparation_started'] = False
                            st.rerun()
                    with col2:
                        if st.button("🚪 Logout"):
                            from ui.auth import auth
                            auth.logout()
                            
            except Exception as e:
                st.error(f"❌ **Error saat mempersiapkan model:** {str(e)}")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔄 Coba Lagi"):
                        st.session_state['model_preparation_started'] = False
                        st.rerun()
                with col2:
                    if st.button("🚪 Logout"):
                        from ui.auth import auth
                        auth.logout()

