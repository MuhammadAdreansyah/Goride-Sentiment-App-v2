"""
GoRide Sentiment Analysis - Data Analysis Module
===============================================

This module handles CSV file upload, text preprocessing, sentiment prediction,
and comprehensive analysis visualization for the GoRide sentiment analysis application.

Features:
- CSV file upload and validation
- Configurable text preprocessing
- Sentiment prediction with confidence scores
- Interactive visualizations (pie chart, gauge, bar charts)
- Word frequency and N-gram analysis
- Word cloud generation
- Text summarization
- Export functionality

Author: Mhd Adreansyah
Version: 2.0.0 (Rebuilt)
License: Copyright Protected (Tugas Akhir/Skripsi)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
import os
import traceback
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
# (NLTK dihapus – tidak lagi diperlukan setelah refactor)

# Authentication and utilities
from ui.auth import auth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    load_sample_data, get_or_train_model, predict_sentiment,
    preprocess_text, get_word_frequencies, get_ngrams, create_wordcloud
)

# Caching ringan untuk menghindari model retrain & data reload tiap interaksi UI
@st.cache_resource(show_spinner=False)
def _cached_model_and_metrics():
    """Load model and metrics with PyCharm compatibility fixes."""
    data = load_sample_data()
    preprocessing_options = DEFAULT_PREPROCESSING_OPTIONS.copy()
    pipeline, accuracy, precision, recall, f1, *_ = get_or_train_model(data, preprocessing_options)
    return data, preprocessing_options, pipeline, (accuracy, precision, recall, f1)

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# Default preprocessing options (compatible with utils.py)
DEFAULT_PREPROCESSING_OPTIONS = {
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

# UI Configuration
SENTIMENT_COLORS = {
    'POSITIF': 'green',
    'NEGATIF': 'red'
}

WORDCLOUD_COLOR_SCHEMES = [
    "viridis", "plasma", "inferno", "magma", 
    "cividis", "YlGnBu", "YlOrRd"
]

# Required columns for analysis
DISPLAY_COLUMNS = [
    'review_text', 'teks_preprocessing', 
    'predicted_sentiment', 'confidence'
]

# ==============================================================================
# HELPER FUNCTIONS FOR CALCULATIONS
# ==============================================================================

def calculate_sentiment_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Hitung statistik inti sentimen (jumlah, persentase, rata-rata confidence)."""
    total_count = len(df)
    pos_count = len(df[df['predicted_sentiment'] == 'POSITIF'])
    neg_count = len(df[df['predicted_sentiment'] == 'NEGATIF'])
    
    pos_percentage = (pos_count / total_count * 100) if total_count > 0 else 0
    neg_percentage = (neg_count / total_count * 100) if total_count > 0 else 0
    avg_confidence = df['confidence'].mean() * 100 if not df['confidence'].empty else 0
    
    dominant_sentiment = "Positif" if pos_count >= neg_count else "Negatif"
    dominant_percentage = max(pos_percentage, neg_percentage)
    
    return {
        'total_count': total_count,
        'pos_count': pos_count,
        'neg_count': neg_count,
        'pos_percentage': pos_percentage,
        'neg_percentage': neg_percentage,
        'avg_confidence': avg_confidence,
        'dominant_sentiment': dominant_sentiment,
        'dominant_percentage': dominant_percentage
    }


def safe_progress_cleanup(progress_bar) -> None:
    """Bersihkan progress bar secara aman (abaikan error jika sudah dihapus)."""
    if progress_bar is not None:
        try:
            progress_bar.empty()
        except Exception:
            pass

def reset_analysis_state() -> None:
    """Reset state analisis (membersihkan hasil & pilihan sebelumnya)."""
    st.session_state.analysis_complete = False
    for key in ['csv_results', 'csv_preprocessed', 'preprocess_options', 'selected_text_column']:
        if key in st.session_state:
            del st.session_state[key]


def initialize_session_state() -> None:
    """Inisialisasi variabel state utama bila belum ada (refactor)."""
    defaults = {
        'analysis_complete': False,
        'csv_results': None,
        'preprocess_options': {},
        'selected_text_column': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str, Optional[str]]:
    """
    Validate uploaded DataFrame and identify text column.
    
    Args:
        df: Uploaded DataFrame
        
    Returns:
        Tuple of (is_valid, message, text_column_name)
    """
    if df.empty:
        return False, "❌ File CSV kosong!", None
    
    # Check if 'review_text' column exists
    if 'review_text' in df.columns:
        return True, "✅ File CSV valid dengan kolom 'review_text'", 'review_text'
    
    # Look for potential text columns
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Text columns are usually object type
            # Check if column contains meaningful text (not just numbers/short strings)
            sample_text = str(df[col].iloc[0]) if len(df) > 0 else ""
            if len(sample_text.split()) > 2:  # Has more than 2 words
                text_columns.append(col)
    
    if text_columns:
        return True, f"✅ File CSV valid. Kolom teks ditemukan: {', '.join(text_columns)}", None
    else:
        return False, "❌ Tidak ditemukan kolom teks yang sesuai dalam file CSV!", None


def create_preprocessing_options_ui() -> Dict[str, bool]:
    """
    Create preprocessing options UI and return user selections.
    
    Returns:
        Dictionary of preprocessing options
    """
    st.write("### 🛠️ Opsi Preprocessing Teks")
    
    with st.expander("Pengaturan Preprocessing", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            case_folding = st.checkbox(
                "Konversi ke huruf kecil", 
                value=True, 
                key="analisis_data_case_folding",
                help="Mengubah semua huruf menjadi huruf kecil"
            )
            cleansing = st.checkbox(
                "Cleansing teks (URL, karakter khusus)", 
                value=True, 
                key="analisis_data_cleansing",
                help="Menghapus URL, emoji, dan karakter non-alfabetik"
            )
            normalize_slang = st.checkbox(
                "Normalisasi kata gaul/slang", 
                value=True, 
                key="analisis_data_normalize_slang",
                help="Mengubah kata gaul menjadi kata formal"
            )
            remove_repeated = st.checkbox(
                "Hapus karakter berulang", 
                value=True, 
                key="analisis_data_remove_repeated",
                help="Mengurangi karakter berulang (misal: 'bagusssss' → 'baguss')"
            )
            tokenize = st.checkbox(
                "Tokenisasi teks", 
                value=True, 
                key="analisis_data_tokenize",
                help="Memecah teks menjadi token/kata individual"
            )
            
        with col2:
            remove_stopwords = st.checkbox(
                "Hapus stopwords", 
                value=True, 
                key="analisis_data_remove_stopwords",
                help="Menghapus kata-kata umum yang kurang bermakna"
            )
            stemming = st.checkbox(
                "Stemming (Sastrawi)", 
                value=True, 
                key="analisis_data_stemming",
                help="Mengubah kata ke bentuk dasarnya"
            )
            phrase_standardization = st.checkbox(
                "Standardisasi frasa", 
                value=True, 
                key="analisis_data_phrase_standardization",
                help="Menormalisasi frasa umum (misal: 'go-ride' → 'goride')"
            )
            rejoin = st.checkbox(
                "Gabungkan kembali token", 
                value=True, 
                key="analisis_data_rejoin",
                help="Menggabungkan token kembali menjadi teks"
            )
    
    return {
        'case_folding': case_folding,
        'phrase_standardization': phrase_standardization,
        'cleansing': cleansing,
        'normalize_slang': normalize_slang,
        'remove_repeated': remove_repeated,
        'tokenize': tokenize,
        'remove_stopwords': remove_stopwords,
        'stemming': stemming,
        'rejoin': rejoin
    }


def process_uploaded_file(uploaded_file, preprocess_options: Dict[str, bool], 
                         pipeline) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """
    Process uploaded CSV file and perform sentiment analysis.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        preprocess_options: Preprocessing configuration
        pipeline: Trained sentiment analysis pipeline
        
    Returns:
        Tuple of (success, dataframe, message)
    """
    progress_bar = None
    
    try:
        # Progress bar setup
        progress_bar = st.progress(0, text="Memproses file CSV...")
        
        # Read CSV file with error handling
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
                st.warning("⚠️ File dibaca dengan encoding latin-1. Pastikan karakter khusus tertampil dengan benar.")
            except Exception as e:
                safe_progress_cleanup(progress_bar)
                return False, None, f"❌ Gagal membaca file CSV: {str(e)}"
        
        progress_bar.progress(25, text="File berhasil diunggah...")
        
        # Validate file
        is_valid, message, text_col = validate_dataframe(df)
        if not is_valid:
            safe_progress_cleanup(progress_bar)
            return False, None, message
        
        # Handle column selection if needed
        if text_col is None:
            # Let user select text column
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if not text_columns:
                safe_progress_cleanup(progress_bar)
                return False, None, "❌ Tidak ditemukan kolom teks dalam file CSV!"
            
            # Create column selector with better state management
            if 'selected_text_column' not in st.session_state:
                safe_progress_cleanup(progress_bar)
                
                st.write("**🎯 Pilih kolom yang berisi teks ulasan:**")
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        selected_col = st.selectbox(
                            "Kolom teks:", 
                            text_columns, 
                            key="text_column_selector",
                            help="Pilih kolom yang berisi teks ulasan untuk dianalisis"
                        )
                    with col2:
                        if st.button("✅ Konfirmasi", key="confirm_column", use_container_width=True):
                            st.session_state.selected_text_column = selected_col
                            st.rerun()
                
                # Show preview of selected column
                if selected_col and len(df) > 0:
                    st.write("**👀 Preview data dari kolom yang dipilih:**")
                    preview_data = df[selected_col].head(3).tolist()
                    for i, text in enumerate(preview_data):
                        st.text(f"{i+1}. {str(text)[:100]}...")
                
                return False, None, "Silakan pilih kolom teks dan konfirmasi untuk melanjutkan."
            else:
                df['review_text'] = df[st.session_state.selected_text_column]
        
        # Ensure review_text column exists
        if 'review_text' not in df.columns:
            safe_progress_cleanup(progress_bar)
            return False, None, "❌ Kolom review_text tidak ditemukan!"
        
        # Validate data quality
        df = df.dropna(subset=['review_text'])  # Remove empty reviews
        df = df[df['review_text'].astype(str).str.strip() != '']  # Remove empty strings
        
        if len(df) == 0:
            safe_progress_cleanup(progress_bar)
            return False, None, "❌ Tidak ada data teks yang valid untuk dianalisis!"
        
        progress_bar.progress(50, text="Melakukan preprocessing teks...")
        
        # Batch preprocess text for better performance
        try:
            df['teks_preprocessing'] = df['review_text'].astype(str).apply(
                lambda x: preprocess_text(x, preprocess_options)
            )
        except Exception as e:
            safe_progress_cleanup(progress_bar)
            return False, None, f"❌ Gagal melakukan preprocessing: {str(e)}"
        
        progress_bar.progress(75, text="Memprediksi sentimen...")
        
        # Predict sentiment with better error handling
        predicted_results = []
        successful_predictions = 0
        
        for i, text in enumerate(df['teks_preprocessing']):
            try:
                result = predict_sentiment(text, pipeline, preprocess_options)
                predicted_results.append(result)
                successful_predictions += 1
            except Exception as e:
                # Handle prediction errors gracefully
                predicted_results.append({
                    'sentiment': 'ERROR',
                    'confidence': 0.0,
                    'probabilities': {'POSITIF': 0.0, 'NEGATIF': 0.0}
                })
            
            # Update progress for large datasets
            if i % 100 == 0 and i > 0:
                current_progress = 75 + (i / len(df)) * 20
                progress_bar.progress(int(current_progress), text=f"Memprediksi sentimen... ({i}/{len(df)})")
        
        # Extract results
        df['predicted_sentiment'] = [result['sentiment'] for result in predicted_results]
        df['confidence'] = [result['confidence'] for result in predicted_results]
        
        # Filter out error predictions
        error_count = len(df[df['predicted_sentiment'] == 'ERROR'])
        if error_count > 0:
            st.warning(f"⚠️ {error_count} teks gagal diprediksi dan akan diabaikan.")
            df = df[df['predicted_sentiment'] != 'ERROR']
        
        if len(df) == 0:
            safe_progress_cleanup(progress_bar)
            return False, None, "❌ Semua prediksi gagal. Periksa kualitas data input."
        
        progress_bar.progress(100, text="Analisis selesai!")
        time.sleep(0.5)
        safe_progress_cleanup(progress_bar)
        
        success_rate = (successful_predictions / len(predicted_results)) * 100
        return True, df, f"✅ Berhasil menganalisis {len(df)} ulasan! (Tingkat keberhasilan: {success_rate:.1f}%)"
        
    except Exception as e:
        safe_progress_cleanup(progress_bar)
        error_msg = f"❌ Terjadi kesalahan: {str(e)}"
        st.error(f"Debug info: {traceback.format_exc()}")
        return False, None, error_msg

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def create_sentiment_metrics(df: pd.DataFrame, stats: Optional[Dict[str, Any]] = None) -> None:
    """Create clean and structured sentiment metrics display."""
    st.write("### 📊 Hasil Analisis Sentimen")


def create_visualization_charts(df: pd.DataFrame, stats: Optional[Dict[str, Any]] = None) -> None:
    """Create clean and focused visualization charts for sentiment analysis."""
    
    # Use provided stats or calculate if not provided
    if stats is None:
        stats = calculate_sentiment_statistics(df)
    
    # Create two columns for clean layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for sentiment distribution
        sentiment_counts = df['predicted_sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        fig_pie = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map=SENTIMENT_COLORS,
            title="📊 Distribusi Sentimen",
            hover_data=['Count']
        )
        fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=14
        )
        fig_pie.update_layout(
            showlegend=True, 
            height=400,
            title_font_size=16,
            title_x=0.5
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Gauge chart for positive sentiment percentage
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stats['pos_percentage'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "📈 Persentase Sentimen Positif", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if stats['pos_percentage'] >= 50 else "red"},
                'steps': [
                    {'range': [0, 33], 'color': 'lightgray'},
                    {'range': [33, 66], 'color': 'gray'},
                    {'range': [66, 100], 'color': 'darkgray'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': stats['pos_percentage']
                }
            },
            number={'suffix': "%", 'valueformat': ".1f", 'font': {'size': 20}}
        ))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==============================================================================
# TAB CONTENT FUNCTIONS
# ==============================================================================

def render_results_table_tab(df: pd.DataFrame) -> None:
    """Render the results table tab with improved filtering and validation."""
    st.subheader("📋 Tabel Hasil Prediksi Sentimen")
    
    if df.empty:
        st.warning("⚠️ Tidak ada data untuk ditampilkan.")
        return
    
    # Filter options with better layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        filter_sentiment = st.selectbox(
            "Filter berdasarkan sentimen:",
            ["Semua", "POSITIF", "NEGATIF"],
            key="analisis_data_filter_sentiment",
            help="Filter hasil berdasarkan jenis sentimen"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Minimum confidence score:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="analisis_data_confidence_threshold",
            help="Tampilkan hanya prediksi dengan confidence di atas threshold"
        )
    
    # Apply filters with validation
    filtered_df = df.copy()
    
    if filter_sentiment != "Semua":
        filtered_df = filtered_df[filtered_df['predicted_sentiment'] == filter_sentiment]
    
    filtered_df = filtered_df[filtered_df['confidence'] >= confidence_threshold]
    
    # Display filtered data with status
    if len(filtered_df) == 0:
        st.warning("⚠️ Tidak ada data yang memenuhi kriteria filter. Coba kurangi threshold confidence atau ubah filter sentimen.")
        
        # Show current filter summary
        st.info(f"""
        **Filter Saat Ini:**
        - Sentimen: {filter_sentiment}
        - Minimum Confidence: {confidence_threshold:.2f}
        
        **Saran:** Kurangi nilai minimum confidence atau pilih "Semua" untuk sentimen.
        """)
        return
    
    st.success(f"**Menampilkan {len(filtered_df):,} dari {len(df):,} ulasan** ({len(filtered_df)/len(df)*100:.1f}%)")
    
    # Select columns to display with error handling
    available_cols = [col for col in DISPLAY_COLUMNS if col in filtered_df.columns]
    if not available_cols:
        st.error("❌ Kolom yang diperlukan tidak tersedia dalam data.")
        return
    
    # Enhanced dataframe display
    st.dataframe(
        filtered_df[available_cols], 
        use_container_width=True,
        hide_index=True,
        column_config={
            "review_text": st.column_config.TextColumn(
                "Teks Asli",
                help="Teks ulasan asli sebelum preprocessing",
                max_chars=100
            ),
            "teks_preprocessing": st.column_config.TextColumn(
                "Teks Preprocessing",
                help="Teks setelah preprocessing",
                max_chars=100
            ),
            "predicted_sentiment": st.column_config.TextColumn(
                "Prediksi Sentimen",
                help="Hasil prediksi sentimen"
            ),
            "confidence": st.column_config.NumberColumn(
                "Confidence Score",
                help="Tingkat kepercayaan prediksi (0-1)",
                format="%.3f"
            )
        }
    )
    
    # Enhanced download functionality
    if not filtered_df.empty:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Add timestamp to filename
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hasil_prediksi_goride_{timestamp}.csv"
            
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=csv,
                file_name=filename,
                mime="text/csv",
                use_container_width=True,
                help=f"Download {len(filtered_df)} data yang telah difilter"
            )


## Fungsi render_word_frequency_tab & render_ngram_analysis_tab dihapus (digantikan oleh render_word_analysis_tab)


def render_wordcloud_tab(df: pd.DataFrame) -> None:
    """Render dual word clouds (positive vs negative) with clean configuration and improved error handling."""
    st.subheader("☁️ Word Cloud Per Sentimen")
    st.info("Visualisasi kata dominan yang membedakan ulasan POSITIF dan NEGATIF untuk insights yang lebih mendalam.")

    if df is None or df.empty:
        st.warning("⚠️ Data tidak tersedia untuk membuat word cloud.")
        return

    if 'predicted_sentiment' not in df.columns:
        st.error("❌ Kolom 'predicted_sentiment' tidak ditemukan.")
        return

    # Pilih kolom teks
    text_col = 'teks_preprocessing' if 'teks_preprocessing' in df.columns else ('review_text' if 'review_text' in df.columns else None)
    if text_col is None:
        st.error("❌ Tidak ada kolom teks yang dapat digunakan.")
        return

    # Konfigurasi wordcloud dalam expander (collapsed by default)
    with st.expander("🔧 Pengaturan Word Cloud", expanded=False):
        st.markdown("**Sesuaikan parameter word cloud untuk hasil visualisasi optimal:**")
        
        col1, col2 = st.columns(2)
        with col1:
            max_words = st.slider(
                "Jumlah Kata Maksimal", 50, 500, 150, step=25, 
                key="wordcloud_max_words",
                help="Menentukan berapa banyak kata yang akan ditampilkan (150 kata optimal untuk keseimbangan detail dan readability)"
            )
            background_color = st.selectbox(
                "Warna Latar Belakang", ["white", "black", "lightgray"], index=0, 
                key="wordcloud_bg",
                help="Pilih warna latar belakang yang kontras dengan skema warna"
            )
        
        with col2:
            cmap_pos = st.selectbox(
                "Skema Warna Positif", WORDCLOUD_COLOR_SCHEMES, index=0, 
                key="wordcloud_cmap_pos",
                help="Skema warna untuk word cloud ulasan positif"
            )
            cmap_neg = st.selectbox(
                "Skema Warna Negatif", WORDCLOUD_COLOR_SCHEMES, index=3, 
                key="wordcloud_cmap_neg",
                help="Skema warna untuk word cloud ulasan negatif"
            )

    # Siapkan data dengan error handling
    try:
        pos_series = df[df['predicted_sentiment'] == 'POSITIF'][text_col].dropna().astype(str)
        neg_series = df[df['predicted_sentiment'] == 'NEGATIF'][text_col].dropna().astype(str)
        
        pos_text = " ".join(pos_series.tolist()) if len(pos_series) > 0 else ""
        neg_text = " ".join(neg_series.tolist()) if len(neg_series) > 0 else ""
    except Exception as e:
        st.error(f"❌ Gagal memproses teks untuk wordcloud: {str(e)}")
        return

    def sufficient(text: str, min_unique: int = 5) -> bool:
        """Check if text has enough unique words for wordcloud"""
        if not text or not text.strip():
            return False
        unique_words = {t.strip() for t in text.split() if t.strip() and len(t.strip()) > 1}
        return len(unique_words) >= min_unique

    # Layout wordcloud yang lebih bersih
    st.markdown("---")
    st.markdown("### 📊 Visualisasi Word Cloud")
    
    col_pos, col_neg = st.columns(2)
    any_generated = False
    
    # Wordcloud Positif
    with col_pos:
        st.markdown("#### 😊 Sentimen Positif")
        if sufficient(pos_text):
            try:
                with st.spinner("🎨 Membuat word cloud positif..."):
                    wc_pos = create_wordcloud(pos_text, max_words=max_words, background_color=background_color, colormap=cmap_pos)
                    if wc_pos is not None:
                        st.image(wc_pos.to_array(), use_container_width=True, caption="Word Cloud kata-kata dominan dalam ulasan positif")
                        any_generated = True
                        
                        # Tampilkan kata tersering dengan styling yang lebih baik
                        if hasattr(wc_pos, 'words_') and wc_pos.words_:
                            top_pos = max(wc_pos.words_.items(), key=lambda x: x[1])[0]
                            st.success(f"🏆 **Kata tersering:** {top_pos}")
                    else:
                        st.info("⚠️ Tidak dapat membentuk word cloud positif dengan parameter saat ini")
            except Exception as e:
                st.error(f"❌ Error membuat wordcloud positif: {str(e)}")
                st.info("💡 **Saran:** Coba ubah pengaturan atau periksa kualitas data preprocessing")
        else:
            st.warning(f"📝 **Data positif belum mencukupi** untuk word cloud\n\n" +
                      f"• Kata unik tersedia: < 5\n" +
                      f"• Ulasan positif: {len(pos_series):,}\n" +
                      f"• **Solusi:** Perbesar dataset atau kurangi preprocessing")
    
    # Wordcloud Negatif  
    with col_neg:
        st.markdown("#### 😞 Sentimen Negatif")
        if sufficient(neg_text):
            try:
                with st.spinner("🎨 Membuat word cloud negatif..."):
                    wc_neg = create_wordcloud(neg_text, max_words=max_words, background_color=background_color, colormap=cmap_neg)
                    if wc_neg is not None:
                        st.image(wc_neg.to_array(), use_container_width=True, caption="Word Cloud kata-kata dominan dalam ulasan negatif")
                        any_generated = True
                        
                        # Tampilkan kata tersering dengan styling yang lebih baik
                        if hasattr(wc_neg, 'words_') and wc_neg.words_:
                            top_neg = max(wc_neg.words_.items(), key=lambda x: x[1])[0]
                            st.error(f"⚠️ **Kata tersering:** {top_neg}")
                    else:
                        st.info("⚠️ Tidak dapat membentuk word cloud negatif dengan parameter saat ini")
            except Exception as e:
                st.error(f"❌ Error membuat wordcloud negatif: {str(e)}")
                st.info("💡 **Saran:** Coba ubah pengaturan atau periksa kualitas data preprocessing")
        else:
            st.warning(f"📝 **Data negatif belum mencukupi** untuk word cloud\n\n" +
                      f"• Kata unik tersedia: < 5\n" +
                      f"• Ulasan negatif: {len(neg_series):,}\n" +
                      f"• **Solusi:** Perbesar dataset atau kurangi preprocessing")

    # Status dan tips (hanya jika tidak ada word cloud yang berhasil)
    if not any_generated:
        st.markdown("---")
        st.warning("⚠️ **Tidak ada word cloud yang dapat ditampilkan**")
        with st.expander("💡 Tips untuk Mengoptimalkan Word Cloud", expanded=True):
            st.markdown("""
            **Kemungkinan penyebab dan solusi:**
            
            1. **Data terlalu sedikit:** 
               - Perluas rentang tanggal filter
               - Kurangi intensitas preprocessing
            
            2. **Preprocessing terlalu agresif:**
               - Nonaktifkan beberapa opsi preprocessing
               - Cek kualitas teks hasil preprocessing
            
            3. **Pengaturan word cloud:**
               - Kurangi jumlah kata minimal (saat ini: {max_words})
               - Coba skema warna yang berbeda
               
            4. **Kualitas data:**
               - Pastikan kolom teks berisi data bermakna
               - Periksa hasil preprocessing di tab lain
            """)
        
    # Footer informasi teknis yang sederhana
    st.divider()
    st.caption("""
    💡 **Catatan:** Word cloud menampilkan {max_words} kata teratas dengan minimal 5 kata unik per kategori. 
    Top 10 kata tersering diurutkan berdasarkan frekuensi kemunculan dalam word cloud.
    """.format(max_words=max_words))


## Fungsi ringkasan teks manual dihapus – digantikan oleh tab Insights & analisis aspek.

# ==============================================================================
# MAIN ANALYSIS TABS
# ==============================================================================

def render_analysis_tabs(df: pd.DataFrame, preprocessed_text: str) -> None:
    """Render semua tab analisis menggunakan st.tabs (persist otomatis oleh Streamlit)."""
    tabs = st.tabs([
        "📋 Tabel Hasil",
        "📝 Analisis Kata",
        "☁️ Word Cloud",
        "💡 Insights & Rekomendasi"
    ])

    with tabs[0]:
        render_results_table_tab(df)
    with tabs[1]:
        render_word_analysis_tab(df, preprocessed_text)
    with tabs[2]:
        render_wordcloud_tab(df)
    with tabs[3]:
        render_insights_recommendations_tab(df)


def render_insights_recommendations_tab(df: pd.DataFrame) -> None:
    """Render tab Insights & Rekomendasi (adaptasi dari dashboard ringkasan).

    Mengganti Ringkasan Teks dengan analisis aspek, status keseluruhan,
    temuan utama, rekomendasi dinamis, dan narasi ringkas.
    Kolom sentimen menggunakan 'predicted_sentiment' (bukan 'sentiment').
    """
    st.subheader("💡 Insights & Rekomendasi")

    if df is None or df.empty:
        st.warning("⚠️ Data tidak tersedia untuk menghasilkan insights.")
        return

    if 'predicted_sentiment' not in df.columns:
        st.error("❌ Kolom 'predicted_sentiment' tidak ada. Jalankan analisis terlebih dahulu.")
        return

    # =====================
    # 1. METRIK DASAR
    # =====================
    total_reviews = len(df)
    pos_count = int((df['predicted_sentiment'] == 'POSITIF').sum())
    neg_count = int((df['predicted_sentiment'] == 'NEGATIF').sum())
    pos_pct = (pos_count / total_reviews * 100) if total_reviews else 0
    neg_pct = (neg_count / total_reviews * 100) if total_reviews else 0

    # =====================
    # 2. TREN 7 HARI (jika ada kolom date)
    # =====================
    trend_change: float = 0.0
    if 'date' in df.columns:
        try:
            df_ts = df.copy()
            df_ts['date'] = pd.to_datetime(df_ts['date'], errors='coerce')
            recent_cut = pd.Timestamp.now() - pd.Timedelta(days=7)
            prev_cut = recent_cut - pd.Timedelta(days=7)
            recent = df_ts[df_ts['date'] >= recent_cut]
            prev = df_ts[(df_ts['date'] >= prev_cut) & (df_ts['date'] < recent_cut)]
            if len(prev) > 0 and len(recent) > 0:
                recent_pos = (recent['predicted_sentiment'] == 'POSITIF').mean() * 100
                prev_pos = (prev['predicted_sentiment'] == 'POSITIF').mean() * 100
                trend_change = recent_pos - prev_pos
        except Exception:
            trend_change = 0.0

    # =====================
    # 3. STATUS OVERALL
    # =====================
    if pos_pct >= 75 and trend_change >= 3:
        status_label = "Sangat Baik"
        status_desc = "Kepuasan tinggi dan meningkat"
        status_color = "#16a34a"
    elif pos_pct < 55 or trend_change <= -3:
        status_label = "Perlu Perhatian"
        status_desc = "Perlu tindakan perbaikan segera"
        status_color = "#dc2626"
    else:
        status_label = "Stabil"
        status_desc = "Sentimen relatif stabil"
        status_color = "#2563eb"

    # =====================
    # 4. ANALISIS ASPEK
    # =====================
    @st.cache_data(show_spinner=False)
    def build_aspect_lexicon():
        return {
            'driver': {'driver','pengemudi','ojek','kurir'},
            'aplikasi': {'aplikasi','app','apps','fitur','versi','update'},
            'harga': {'harga','tarif','biaya','ongkos','mahal','murah'},
            'waktu': {'waktu','lama','cepat','delay','menunggu','nunggu','tunggu'},
            'pembayaran': {'bayar','pembayaran','gopay','cash','tunai','saldo'},
            'promosi': {'promo','promosi','diskon','voucher','potongan'},
            'keamanan': {'aman','keamanan','bahaya','kecelakaan','nabrak'},
            'kenyamanan': {'nyaman','kenyamanan','bersih','panas','bau','helm'},
            'layanan': {'layanan','service','respon','customer','cs'},
            'performa': {'error','bug','lambat','lemot','lag','crash','force','close'}
        }

    STOP_TOKENS = {"nya","yang","itu","di","ke","ku","lah","pun","deh","dong","nih","ya","sih"}

    def normalize_token(tok: str) -> str:
        tok = tok.lower()
        tok = re.sub(r'(nya|lah|kah|pun)$','', tok)
        return tok

    def extract_aspect_stats(df_local: pd.DataFrame) -> dict:
        if 'teks_preprocessing' not in df_local.columns:
            return {}
        lex = build_aspect_lexicon()
        inverse = {w: a for a, ws in lex.items() for w in ws}
        def _new_stat():
            return {"pos": 0, "neg": 0, "total": 0, "pos_ids": set(), "neg_ids": set()}
        stats = defaultdict(_new_stat)
        for idx, row in df_local.iterrows():
            sent = row.get('predicted_sentiment','')
            tokens = str(row.get('teks_preprocessing','')).split()
            aspects_found = set()
            for t in tokens:
                t_norm = normalize_token(t)
                if t_norm in STOP_TOKENS or len(t_norm) < 2:
                    continue
                if t_norm in inverse:
                    aspects_found.add(inverse[t_norm])
            for a in aspects_found:
                stats[a]['total'] += 1
                if sent == 'POSITIF':
                    stats[a]['pos'] += 1
                    stats[a]['pos_ids'].add(idx)
                elif sent == 'NEGATIF':
                    stats[a]['neg'] += 1
                    stats[a]['neg_ids'].add(idx)
        return stats

    def score_aspects(stats: dict, total_neg: int):
        scored = []
        for a, s in stats.items():
            total_a = s['total']
            if total_a < max(5, int(0.01 * total_reviews)):
                continue
            pos_a, neg_a = s['pos'], s['neg']
            sentiment_score = (pos_a - neg_a) / max(1, total_a)
            impact = neg_a / max(1, total_neg)
            opportunity = impact * (1 - (sentiment_score + 1)/2)
            scored.append({
                'aspect': a,
                'total': total_a,
                'pos': pos_a,
                'neg': neg_a,
                'sentiment_score': sentiment_score,
                'impact': impact,
                'opportunity': opportunity
            })
        return scored

    def generate_dynamic_recommendations(improvement_areas: list) -> list:
        templates = {
            'driver': 'Perkuat pelatihan & quality control driver untuk konsistensi layanan.',
            'aplikasi': 'Optimalkan stabilitas & UX aplikasi; prioritaskan perbaikan bug paling sering.',
            'harga': 'Evaluasi struktur tarif & transparansi biaya perjalanan.',
            'waktu': 'Kurangi waktu tunggu dengan optimasi algoritma penugasan & estimasi.',
            'pembayaran': 'Perbaiki reliabilitas metode pembayaran & jelaskan kegagalan transaksi.',
            'promosi': 'Sesuaikan komunikasi promo agar relevan & mudah ditebus pengguna.',
            'keamanan': 'Tingkatkan edukasi & standar keselamatan perjalanan.',
            'kenyamanan': 'Pastikan standar kebersihan & kelayakan perlengkapan keselamatan.',
            'layanan': 'Tingkatkan kecepatan/respons tim dukungan & proaktif tangani keluhan.',
            'performa': 'Optimalkan performa aplikasi (load time, crash rate).'
        }
        recs = []
        for item in improvement_areas:
            a = item['aspect']
            base = templates.get(a, f'Perbaiki kualitas aspek {a}.')
            recs.append(f"{a.capitalize()}: {base}")
        if not recs:
            recs.append('Pertahankan kualitas layanan & eksplor program loyalitas untuk meningkatkan retensi.')
        return recs

    stats_aspect = extract_aspect_stats(df)
    total_neg = int((df['predicted_sentiment'] == 'NEGATIF').sum())
    scored = score_aspects(stats_aspect, total_neg)
    positive_highlights = sorted(
        [x for x in scored if x['sentiment_score'] > 0.25],
        key=lambda d: (d['sentiment_score'], d['total']), reverse=True)[:5]
    improvement_areas = sorted(
        [x for x in scored if x['sentiment_score'] < -0.15],
        key=lambda d: (d['opportunity'], d['neg']), reverse=True)[:5]

    # =====================
    # 5. KARTU STATUS
    # =====================
    st.markdown("#### 📊 Analisis Sentimen Saat Ini")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style='border-left:4px solid {status_color};padding:0.75rem 0.9rem;background:#11182712;border-radius:6px;'>
            <h5 style='margin:0;color:{status_color};'>Status: {status_label}</h5>
            <div style='font-size:0.85rem;color:#555;'>{status_desc}</div>
            <div style='margin-top:0.5rem;font-weight:600;'>{pos_pct:.1f}% Positif</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='border-left:4px solid #0d9488;padding:0.75rem 0.9rem;background:#11182712;border-radius:6px;'>
            <h5 style='margin:0;color:#0d9488;'>Volume Ulasan</h5>
            <div style='font-size:0.85rem;color:#555;'>Total ulasan dianalisis</div>
            <div style='margin-top:0.5rem;font-weight:600;'>{total_reviews:,} Ulasan</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        trend_color = '#16a34a' if trend_change > 0 else ('#dc2626' if trend_change < 0 else '#2563eb')
        st.markdown(f"""
        <div style='border-left:4px solid {trend_color};padding:0.75rem 0.9rem;background:#11182712;border-radius:6px;'>
            <h5 style='margin:0;color:{trend_color};'>Tren 7 Hari</h5>
            <div style='font-size:0.85rem;color:#555;'>Perubahan persentase positif</div>
            <div style='margin-top:0.5rem;font-weight:600;'>{trend_change:+.1f} p.p</div>
        </div>
        """, unsafe_allow_html=True)

    # =====================
    # 6. TEMUAN UTAMA
    # =====================
    st.markdown("---")
    st.markdown("#### 🔍 Temuan Utama Berbasis Aspek")
    if not scored:
        st.info("Belum ada aspek terdeteksi atau preprocessing belum menghasilkan teks.")
        return
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**✅ Aspek Positif yang Menonjol**")
        if positive_highlights:
            for item in positive_highlights:
                st.markdown(f"• **{item['aspect'].capitalize()}** – {item['total']} ulasan (skor {item['sentiment_score']:.2f})")
        else:
            st.markdown("_Tidak ada aspek dengan keunggulan signifikan saat ini_")
    with col_neg:
        st.markdown("**⚠️ Aspek yang Perlu Diperbaiki**")
        if improvement_areas:
            for item in improvement_areas:
                st.markdown(f"• **{item['aspect'].capitalize()}** – {item['neg']} keluhan (kontribusi {item['impact']*100:.1f}% negatif)")
        else:
            st.markdown("_Tidak ada aspek negatif dominan – fokus pada penguatan_")

    # =====================
    # 7. REKOMENDASI DINAMIS
    # =====================
    st.markdown("---")
    st.markdown("#### 🎯 Rekomendasi Tindakan Prioritas")
    recs = generate_dynamic_recommendations(improvement_areas)
    for r in recs:
        st.markdown(f"- {r}")

    # =====================
    # 8. NARASI RINGKAS
    # =====================
    st.markdown("---")
    narrative = []
    narrative.append(f"Dari {total_reviews} ulasan, {pos_pct:.1f}% bernada positif dan {neg_pct:.1f}% negatif.")
    if trend_change != 0:
        narrative.append(f"Sentimen {'meningkat' if trend_change>0 else 'menurun'} {abs(trend_change):.1f} p.p dalam 7 hari terakhir.")
    if improvement_areas:
        top_imp = improvement_areas[0]
        narrative.append(f"Aspek prioritas: {top_imp['aspect']} (kontribusi {top_imp['impact']*100:.1f}% keluhan).")
    st.markdown("**Ringkasan:** " + " ".join(narrative))
    st.caption("Aspek dihitung sekali per ulasan; token tidak bermakna seperti 'nya' diabaikan.")


def render_word_analysis_tab(df: pd.DataFrame, preprocessed_text: str) -> None:
    """Unified word analysis (frequency + N-gram) dengan tampilan visual yang bersih dan terstruktur."""

    st.subheader("📝 Analisis Kata & Frasa")
    st.info("Temukan kata paling sering muncul dan kombinasi kata (N-Gram) untuk memahami pola umum dalam ulasan.")

    if not preprocessed_text or not preprocessed_text.strip():
        st.warning("⚠️ Tidak ada teks yang tersedia untuk dianalisis.")
        return

    # =============================
    # 1. KONTROL UTAMA
    # =============================
    st.markdown("#### ⚙️ Pengaturan Analisis")
    
    col1, col2 = st.columns(2)
    with col1:
        top_n_words = st.slider(
            "Jumlah Kata Teratas", 5, 30, 15, step=5,
            key="word_analysis_top_n",
            help="Tentukan berapa banyak kata teratas yang akan ditampilkan"
        )
        
    with col2:
        segment_mode = st.selectbox(
            "Mode Analisis", ["Gabungan", "Per Sentimen"], index=0,
            key="word_analysis_segment_mode",
            help="Pilih 'Per Sentimen' untuk perbandingan kata positif vs negatif"
        )

    # Parameter internal yang disederhanakan
    min_token_len = 2  # Fixed value untuk konsistensi

    # Helper to build frequency from tokenized texts
    def build_freq(series: pd.Series) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for text in series.dropna().astype(str):
            for tok in text.split():
                if not tok or len(tok) < min_token_len:
                    continue
                freq[tok] = freq.get(tok, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

    # Ensure we have per-row preprocessed text; fallback ke preprocessed_text global
    if 'teks_preprocessing' in df.columns:
        overall_freq = build_freq(df['teks_preprocessing'])
        pos_freq = build_freq(df[df['predicted_sentiment'] == 'POSITIF']['teks_preprocessing']) if 'predicted_sentiment' in df.columns else {}
        neg_freq = build_freq(df[df['predicted_sentiment'] == 'NEGATIF']['teks_preprocessing']) if 'predicted_sentiment' in df.columns else {}
    else:
        # Fallback: gunakan gabungan teks global
        overall_freq = get_word_frequencies(preprocessed_text, top_n=top_n_words)
        pos_freq, neg_freq = {}, {}

    # =============================
    # 2. VISUALISASI FREKUENSI KATA
    # =============================
    st.markdown("---")
    st.markdown("#### 📊 Frekuensi Kata Dominan")

    if segment_mode == "Gabungan":
        if not overall_freq:
            st.info("📝 Tidak ada kata yang cukup untuk dianalisis.")
        else:
            # Visualisasi gabungan dengan horizontal bar yang lebih clean
            top_items = list(overall_freq.items())[:top_n_words]
            freq_df = pd.DataFrame(top_items, columns=["Kata", "Frekuensi"])
            freq_df_plot = freq_df.sort_values('Frekuensi', ascending=True)
            
            fig = px.bar(
                freq_df_plot,
                x='Frekuensi', y='Kata', orientation='h',
                title=f"📈 {len(freq_df_plot)} Kata Paling Sering Muncul",
                color='Frekuensi', 
                color_continuous_scale='Viridis',
                text='Frekuensi'
            )
            fig.update_layout(
                height=500, 
                showlegend=False,
                xaxis_title="Jumlah Kemunculan",
                yaxis_title="Kata",
                title_x=0.5,
                font=dict(size=12)
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics sederhana tanpa tabel
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Kata Ditampilkan", len(freq_df))
            with col2:
                st.metric("🔤 Total Kata Unik", f"{len(overall_freq):,}")
            with col3:
                coverage = (freq_df['Frekuensi'].sum() / max(sum(overall_freq.values()), 1)) * 100
                st.metric("📊 Cakupan", f"{coverage:.1f}%")
                
    else:
        # Mode perbandingan sentimen dengan layout yang lebih baik
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.markdown("##### 😊 Kata Dominan Positif")
            if not pos_freq:
                st.info("Tidak ada data positif tersedia")
            else:
                pos_items = list(pos_freq.items())[:top_n_words]
                pos_df = pd.DataFrame(pos_items, columns=["Kata", "Frekuensi"])
                pos_df_plot = pos_df.sort_values('Frekuensi', ascending=True)
                
                fig_pos = px.bar(
                    pos_df_plot,
                    x='Frekuensi', y='Kata', orientation='h',
                    color='Frekuensi', 
                    color_continuous_scale='Greens',
                    text='Frekuensi'
                )
                fig_pos.update_layout(
                    height=450, 
                    showlegend=False,
                    xaxis_title="Frekuensi",
                    yaxis_title="",
                    font=dict(size=10),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                fig_pos.update_traces(texttemplate='%{text}', textposition='outside')
                st.plotly_chart(fig_pos, use_container_width=True)
                
                st.metric("📊 Kata Positif", len(pos_df))
        
        with col_neg:
            st.markdown("##### 😞 Kata Dominan Negatif")
            if not neg_freq:
                st.info("Tidak ada data negatif tersedia")
            else:
                neg_items = list(neg_freq.items())[:top_n_words]
                neg_df = pd.DataFrame(neg_items, columns=["Kata", "Frekuensi"])
                neg_df_plot = neg_df.sort_values('Frekuensi', ascending=True)
                
                fig_neg = px.bar(
                    neg_df_plot,
                    x='Frekuensi', y='Kata', orientation='h',
                    color='Frekuensi', 
                    color_continuous_scale='Reds',
                    text='Frekuensi'
                )
                fig_neg.update_layout(
                    height=450, 
                    showlegend=False,
                    xaxis_title="Frekuensi",
                    yaxis_title="",
                    font=dict(size=10),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                fig_neg.update_traces(texttemplate='%{text}', textposition='outside')
                st.plotly_chart(fig_neg, use_container_width=True)
                
                st.metric("📊 Kata Negatif", len(neg_df))

    # =============================
    # 3. ANALISIS N-GRAM
    # =============================
    st.markdown("---")
    st.markdown("#### 🔄 Analisis Frasa (N-Gram)")
    
    # Pengaturan N-Gram yang disederhanakan
    col1, col2 = st.columns(2)
    with col1:
        ngram_type = st.radio(
            "Jenis Frasa", ["Bigram (2 Kata)", "Trigram (3 Kata)"], 
            horizontal=True, key="word_analysis_ngram_type",
            help="Bigram: kombinasi 2 kata, Trigram: kombinasi 3 kata"
        )
    with col2:
        ngram_scope = st.selectbox(
            "Data yang Dianalisis", ["Gabungan", "Hanya Positif", "Hanya Negatif"], 
            index=0, key="word_analysis_ngram_scope",
            help="Pilih subset data untuk analisis N-Gram"
        )

    # Tentukan korpus untuk N-gram
    if 'teks_preprocessing' in df.columns:
        if ngram_scope == "Hanya Positif" and 'predicted_sentiment' in df.columns:
            corpus_text = " ".join(df[df['predicted_sentiment'] == 'POSITIF']['teks_preprocessing'].astype(str))
        elif ngram_scope == "Hanya Negatif" and 'predicted_sentiment' in df.columns:
            corpus_text = " ".join(df[df['predicted_sentiment'] == 'NEGATIF']['teks_preprocessing'].astype(str))
        else:
            corpus_text = " ".join(df['teks_preprocessing'].astype(str))
    else:
        corpus_text = preprocessed_text

    # Generate N-gram dengan jumlah tetap untuk konsistensi visual
    n_val = 2 if "Bigram" in ngram_type else 3
    ngram_data = get_ngrams(corpus_text, n_val, top_n=12)

    if ngram_data:
        ngram_df = pd.DataFrame(list(ngram_data.items()), columns=['Frasa', 'Frekuensi'])
        ngram_df_plot = ngram_df.sort_values('Frekuensi', ascending=True)
        
        # Pilih color scheme berdasarkan scope
        if ngram_scope == "Hanya Positif":
            color_scheme = 'Greens'
            title_icon = "😊"
        elif ngram_scope == "Hanya Negatif":
            color_scheme = 'Reds'
            title_icon = "😞"
        else:
            color_scheme = 'Plasma'
            title_icon = "🔄"
        
        fig_ng = px.bar(
            ngram_df_plot,
            x='Frekuensi', y='Frasa', orientation='h',
            title=f"{title_icon} {len(ngram_df_plot)} Frasa {ngram_type.split()[0]} Teratas",
            color='Frekuensi', 
            color_continuous_scale=color_scheme,
            text='Frekuensi'
        )
        fig_ng.update_layout(
            height=550, 
            showlegend=False,
            xaxis_title="Jumlah Kemunculan",
            yaxis_title="Frasa",
            title_x=0.5,
            font=dict(size=11)
        )
        fig_ng.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig_ng, use_container_width=True)
        
        # Simple metric untuk N-gram
        st.metric("🎯 Frasa Unik Ditemukan", f"{len(ngram_data):,}")
        
    else:
        st.info(f"📝 Tidak cukup data untuk membentuk {ngram_type.split()[0]} pada subset '{ngram_scope}'")

    # Footer info yang lebih clean
    st.divider()
    st.caption("💡 Analisis kata & frasa berdasarkan teks hasil preprocessing. Token dengan panjang < 2 karakter diabaikan untuk akurasi hasil.")

# ==============================================================================
# FOOTER FUNCTIONS  
# ==============================================================================

def render_footer():
    """Render the application footer."""
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: rgba(0,0,0,0.05); border-radius: 0.5rem;">
        <p style="margin: 0; font-size: 0.9rem; color: #666;">
            © 2025 GoRide Sentiment Analysis Dashboard • Developed by Mhd Adreansyah
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #888;">
            🎓 Aplikasi ini merupakan bagian dari Tugas Akhir/Skripsi di bawah perlindungan Hak Cipta
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# MAIN RENDER FUNCTION
# ==============================================================================

def render_data_analysis() -> None:
    """
    Main function to render the data analysis page.
    
    This function handles the complete workflow:
    1. Authentication and session state initialization
    2. Model loading and validation
    3. File upload and preprocessing configuration
    4. Sentiment analysis processing
    5. Results visualization and analysis tabs
    """
    # Authentication check
    auth.sync_login_state()
    
    # Initialize session state
    initialize_session_state()
    
    # Page header (gunakan key agar tidak reset saat tab interaksi)
    st.title("📑 Analisis Data Teks GoRide")
    st.markdown("---")

    # Ambil model & metrik dari cache
    try:
        data, default_preproc_cached, pipeline, metrics = _cached_model_and_metrics()
        accuracy, precision, recall, f1 = metrics
        if data.empty:
            st.error("❌ Data training tidak tersedia untuk analisis!")
            st.stop()
        with st.sidebar:
            st.info(
                f"""🤖 **Model Siap Digunakan**\n\n📊 **Performa Model:**\n- Akurasi: {accuracy:.2%}\n- Precision: {precision:.2%}\n- Recall: {recall:.2%}\n- F1-Score: {f1:.2%}\n\n📈 **Data Training:** {len(data):,} ulasan"""
            )
    except Exception as e:
        st.error(f"❌ Gagal memuat model (cache): {str(e)}")
        st.stop()
    
    # File upload section
    st.write("### 📁 Upload File CSV")
    st.info("""
    📋 **Format File yang Didukung:**
    - File CSV dengan kolom teks ulasan
    - Encoding UTF-8 direkomendasikan
    - Kolom 'review_text' akan diprioritaskan, atau Anda dapat memilih kolom lain
    """)
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV untuk dianalisis:",
        type=["csv"],
        key="csv_uploader",
        help="Upload file CSV yang berisi data teks untuk dianalisis sentimen"
    )
    
    # Preprocessing options
    preprocess_options = create_preprocessing_options_ui()
    
    # Analysis button
    analyze_button = st.button(
        "🔍 Mulai Analisis Sentimen",
        type="primary",
        disabled=uploaded_file is None
    )
    
    # Handle file processing with better state management
    if uploaded_file is not None and analyze_button:
        # Validate that we have all required components
        if not uploaded_file:
            st.error("⚠️ File tidak tersedia untuk diproses.")
            return
            
        st.session_state.analysis_complete = True
        st.session_state.preprocess_options = preprocess_options
        
        # Clear any selected column from previous runs to avoid conflicts
        if 'selected_text_column' in st.session_state:
            del st.session_state.selected_text_column
        
        # Clear previous results to avoid displaying stale data
        if 'csv_results' in st.session_state:
            del st.session_state.csv_results
        
        # Process file with comprehensive error handling
        with st.spinner("🔄 Memproses file... Mohon tunggu..."):
            success, processed_df, message = process_uploaded_file(
                uploaded_file, preprocess_options, pipeline
            )
        
        if success and processed_df is not None and not processed_df.empty:
            st.session_state.csv_results = processed_df
            st.success(message)
            
            # Show immediate preview of results
            stats = calculate_sentiment_statistics(processed_df)
            st.info(f"""
            🎉 **Analisis Berhasil Diselesaikan!**
            
            📊 **Ringkasan Hasil:**
            - Total ulasan dianalisis: {stats['total_count']:,}
            - Sentimen Positif: {stats['pos_count']:,} ({stats['pos_percentage']:.1f}%)
            - Sentimen Negatif: {stats['neg_count']:,} ({stats['neg_percentage']:.1f}%)
            - Rata-rata Confidence: {stats['avg_confidence']:.1f}%
            """)
        else:
            st.error(message if message else "❌ Gagal memproses file.")
            st.session_state.analysis_complete = False
            
            # Clear any partial results
            if 'csv_results' in st.session_state:
                del st.session_state.csv_results
    
    elif analyze_button and uploaded_file is None:
        st.error("⚠️ Silakan upload file CSV terlebih dahulu!")
    
    # Display results if analysis is complete with comprehensive validation
    if (st.session_state.get('analysis_complete', False) and 
        st.session_state.get('csv_results') is not None):
        
        df = st.session_state.csv_results
        
        # Validate dataframe
        if df is None or df.empty:
            st.warning("⚠️ Tidak ada data yang berhasil diproses.")
            
            # Provide action options
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔄 Coba Lagi", use_container_width=True):
                    reset_analysis_state()
                    st.rerun()
            return
        
        # Validate required columns
        required_cols = ['predicted_sentiment', 'confidence', 'review_text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Kolom yang diperlukan tidak tersedia: {', '.join(missing_cols)}")
            
            # Show available columns for debugging
            with st.expander("🔍 Debug Info - Kolom yang Tersedia"):
                st.write("Kolom dalam dataframe:", list(df.columns))
            
            if st.button("🔄 Reset dan Coba Lagi"):
                reset_analysis_state()
                st.rerun()
            return
        
        # Continue with results display
        st.divider()
        
        # Calculate statistics once for all visualizations
        stats = calculate_sentiment_statistics(df)
        
        # Create sentiment metrics with error handling
        try:
            create_sentiment_metrics(df, stats)
        except Exception as e:
            st.error(f"❌ Gagal menampilkan metrics: {str(e)}")
        
        # Create visualizations with error handling
        try:
            create_visualization_charts(df, stats)
        except Exception as e:
            st.error(f"❌ Gagal membuat visualisasi: {str(e)}")
            
            # Fallback: show basic statistics (stats already calculated)
            st.write("**📊 Statistik Dasar (Mode Fallback):**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Ulasan", stats['total_count'])
                st.metric("Sentimen Positif", f"{stats['pos_count']} ({stats['pos_percentage']:.1f}%)")
            with col2:
                st.metric("Sentimen Negatif", f"{stats['neg_count']} ({stats['neg_percentage']:.1f}%)")
                st.metric("Rata-rata Confidence", f"{stats['avg_confidence']:.1f}%")
        
        # Gunakan kolom 'teks_preprocessing' yang sudah ada untuk agregasi cepat (hindari preprocessing ulang)
        try:
            if 'teks_preprocessing' in df.columns:
                preprocessed_all_text = " ".join(df['teks_preprocessing'].dropna().astype(str))
            else:
                # fallback: minimal join teks asli (tidak ideal tapi tetap jalan)
                preprocessed_all_text = " ".join(df['review_text'].dropna().astype(str)) if 'review_text' in df.columns else ""
        except Exception as e:
            st.warning(f"⚠️ Gagal menggabungkan teks untuk analisis kata: {e}")
            preprocessed_all_text = ""
        
        # Render analysis tabs with validation
        st.divider()
        st.write("### 🔍 Analisis Mendalam")
        
        if preprocessed_all_text.strip():
            try:
                render_analysis_tabs(df, preprocessed_all_text)
            except Exception as e:
                st.error(f"❌ Gagal menampilkan analisis mendalam: {str(e)}")
                
                # Fallback: show only results table
                st.write("**📋 Tabel Hasil (Mode Fallback):**")
                display_cols = [col for col in DISPLAY_COLUMNS if col in df.columns]
                if display_cols:
                    st.dataframe(df[display_cols], use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada teks yang tersedia untuk analisis mendalam.")
        
        # Reset button with improved styling
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Analisis File Baru", use_container_width=True, type="secondary"):
                reset_analysis_state()
                st.rerun()
    
    # Render footer
    render_footer()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    render_data_analysis()
