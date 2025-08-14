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
from typing import Dict, List, Optional, Tuple, Any

# NLTK for text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize  
from nltk import FreqDist

# Authentication and utilities
from ui.auth import auth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    load_sample_data, get_or_train_model, predict_sentiment,
    preprocess_text, get_word_frequencies, get_ngrams, create_wordcloud
)

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
    """
    Calculate comprehensive sentiment statistics from dataframe.
    
    Args:
        df: DataFrame with predicted_sentiment and confidence columns
        
    Returns:
        Dictionary containing all calculated statistics
    """
    total_count = len(df)
    pos_count = len(df[df['predicted_sentiment'] == 'POSITIF'])
    neg_count = len(df[df['predicted_sentiment'] == 'NEGATIF'])
    
    pos_percentage = (pos_count / total_count * 100) if total_count > 0 else 0
    neg_percentage = (neg_count / total_count * 100) if total_count > 0 else 0
    avg_confidence = df['confidence'].mean() * 100 if not df['confidence'].empty else 0
    
    dominant_sentiment = "Positif" if pos_count > neg_count else "Negatif"
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
    """Safely cleanup progress bar."""
    try:
        if progress_bar is not None:
            progress_bar.empty()
    except:
        pass


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def initialize_session_state() -> None:
    """Initialize all session state variables for the analysis module."""
    session_vars = {
        'analysis_complete': False,
        'csv_results': None,
        'preprocess_options': {},
        'selected_text_column': None
    }
    
    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def reset_analysis_state() -> None:
    """Reset all analysis-related session state variables."""
    st.session_state.analysis_complete = False
    
    # Clean up session state keys
    keys_to_remove = ['csv_results', 'csv_preprocessed', 'preprocess_options', 'selected_text_column']
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]


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
                key="csv_case_folding",
                help="Mengubah semua huruf menjadi huruf kecil"
            )
            cleansing = st.checkbox(
                "Cleansing teks (URL, karakter khusus)", 
                value=True, 
                key="csv_cleansing",
                help="Menghapus URL, emoji, dan karakter non-alfabetik"
            )
            normalize_slang = st.checkbox(
                "Normalisasi kata gaul/slang", 
                value=True, 
                key="csv_normalize_slang",
                help="Mengubah kata gaul menjadi kata formal"
            )
            remove_repeated = st.checkbox(
                "Hapus karakter berulang", 
                value=True, 
                key="csv_remove_repeated",
                help="Mengurangi karakter berulang (misal: 'bagusssss' → 'baguss')"
            )
            tokenize = st.checkbox(
                "Tokenisasi teks", 
                value=True, 
                key="csv_tokenize",
                help="Memecah teks menjadi token/kata individual"
            )
            
        with col2:
            remove_stopwords = st.checkbox(
                "Hapus stopwords", 
                value=True, 
                key="csv_remove_stopwords",
                help="Menghapus kata-kata umum yang kurang bermakna"
            )
            stemming = st.checkbox(
                "Stemming (Sastrawi)", 
                value=True, 
                key="csv_stemming",
                help="Mengubah kata ke bentuk dasarnya"
            )
            phrase_standardization = st.checkbox(
                "Standardisasi frasa", 
                value=True, 
                key="csv_phrase_standardization",
                help="Menormalisasi frasa umum (misal: 'go-ride' → 'goride')"
            )
            rejoin = st.checkbox(
                "Gabungkan kembali token", 
                value=True, 
                key="csv_rejoin",
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
            key="filter_sentiment",
            help="Filter hasil berdasarkan jenis sentimen"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Minimum confidence score:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="confidence_threshold",
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


def render_word_frequency_tab(preprocessed_text: str) -> None:
    """Render the word frequency analysis tab with improved validation."""
    st.subheader("📊 Analisis Frekuensi Kata")
    
    # Validate input text
    if not preprocessed_text or not preprocessed_text.strip():
        st.warning("⚠️ Tidak ada teks yang tersedia untuk analisis frekuensi kata.")
        return
    
    # Configuration with better defaults
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider(
            "Jumlah kata teratas:",
            min_value=5,
            max_value=50,
            value=15,
            key="word_freq_top_n",
            help="Pilih berapa banyak kata teratas yang ingin ditampilkan"
        )
    
    with col2:
        chart_type = st.radio(
            "Tipe visualisasi:",
            ["Horizontal Bar", "Vertical Bar"],
            key="word_freq_chart_type",
            help="Pilih orientasi chart yang diinginkan"
        )
    
    # Get word frequencies with error handling
    try:
        word_freq = get_word_frequencies(preprocessed_text, top_n=top_n)
    except Exception as e:
        st.error(f"❌ Gagal menganalisis frekuensi kata: {str(e)}")
        return
    
    if not word_freq:
        st.info("📝 Tidak cukup kata unik untuk analisis frekuensi setelah preprocessing.")
        
        # Provide suggestions
        with st.expander("💡 Saran untuk meningkatkan hasil analisis"):
            st.write("""
            - Pastikan data teks mengandung cukup kata yang bermakna
            - Periksa pengaturan preprocessing (mungkin terlalu ketat)
            - Coba dengan dataset yang lebih besar
            """)
        return
    
    # Create DataFrame with validation
    word_freq_df = pd.DataFrame(
        list(word_freq.items()), 
        columns=['Kata', 'Frekuensi']
    )
    
    if len(word_freq_df) == 0:
        st.warning("⚠️ Tidak ada data frekuensi kata untuk ditampilkan.")
        return
    
    # Create visualization with improved styling
    try:
        if chart_type == "Horizontal Bar":
            word_freq_df = word_freq_df.sort_values('Frekuensi', ascending=True)
            fig = px.bar(
                word_freq_df.tail(top_n),
                x='Frekuensi',
                y='Kata',
                orientation='h',
                title=f"Top {min(top_n, len(word_freq_df))} Kata Paling Sering Muncul",
                color='Frekuensi',
                color_continuous_scale='Viridis',
                labels={'Frekuensi': 'Jumlah Kemunculan', 'Kata': 'Kata'}
            )
        else:
            word_freq_df = word_freq_df.sort_values('Frekuensi', ascending=False)
            fig = px.bar(
                word_freq_df.head(top_n),
                x='Kata',
                y='Frekuensi',
                title=f"Top {min(top_n, len(word_freq_df))} Kata Paling Sering Muncul",
                color='Frekuensi',
                color_continuous_scale='Viridis',
                labels={'Frekuensi': 'Jumlah Kemunculan', 'Kata': 'Kata'}
            )
            fig.update_xaxes(tickangle=45)
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table with summary statistics
        st.write("**📋 Tabel Frekuensi Kata:**")
        
        # Add summary info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Kata Unik", len(word_freq_df))
        with col2:
            st.metric("Kata Tersering", word_freq_df.iloc[0]['Kata'] if len(word_freq_df) > 0 else "N/A")
        with col3:
            st.metric("Frekuensi Tertinggi", word_freq_df.iloc[0]['Frekuensi'] if len(word_freq_df) > 0 else 0)
        
        word_freq_df_sorted = word_freq_df.sort_values('Frekuensi', ascending=False)
        st.dataframe(
            word_freq_df_sorted, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Kata": st.column_config.TextColumn("Kata", help="Kata hasil preprocessing"),
                "Frekuensi": st.column_config.NumberColumn("Frekuensi", help="Jumlah kemunculan kata")
            }
        )
        
    except Exception as e:
        st.error(f"❌ Gagal membuat visualisasi: {str(e)}")
        
        # Fallback: show data in table only
        st.write("**📋 Data Frekuensi Kata (Mode Fallback):**")
        word_freq_df_sorted = word_freq_df.sort_values('Frekuensi', ascending=False)
        st.dataframe(word_freq_df_sorted, use_container_width=True)


def render_ngram_analysis_tab(preprocessed_text: str) -> None:
    """Render the N-gram analysis tab."""
    st.subheader("🔄 Analisis N-Gram")
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        n_gram_type = st.radio(
            "Pilih tipe N-gram:",
            ["Bigram (2 kata)", "Trigram (3 kata)"],
            key="ngram_type"
        )
    
    with col2:
        top_n_ngrams = st.slider(
            "Jumlah N-gram teratas:",
            min_value=5,
            max_value=25,
            value=10,
            key="ngram_top_n"
        )
    
    # Get N-grams
    n = 2 if n_gram_type == "Bigram (2 kata)" else 3
    n_gram_data = get_ngrams(preprocessed_text, n, top_n=top_n_ngrams)
    
    if n_gram_data:
        # Create DataFrame
        n_gram_df = pd.DataFrame(
            list(n_gram_data.items()), 
            columns=['N-gram', 'Frekuensi']
        )
        n_gram_df = n_gram_df.sort_values('Frekuensi', ascending=True)
        
        # Create visualization
        fig = px.bar(
            n_gram_df.tail(top_n_ngrams),
            x='Frekuensi',
            y='N-gram',
            orientation='h',
            title=f"Top {top_n_ngrams} {n_gram_type}",
            color='Frekuensi',
            color_continuous_scale='Plasma'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.write(f"**📋 Tabel {n_gram_type}:**")
        n_gram_df_sorted = n_gram_df.sort_values('Frekuensi', ascending=False)
        st.dataframe(n_gram_df_sorted, use_container_width=True)
        
    else:
        st.info(f"📝 Tidak cukup {n_gram_type.lower()} untuk dianalisis.")


def render_wordcloud_tab(df: pd.DataFrame) -> None:
    """Render dual word clouds (positive vs negative)."""
    st.subheader("☁️ Word Cloud Per Sentimen")
    st.caption("Perbandingan kata dominan antara ulasan POSITIF dan NEGATIF.")

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

    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    with col_cfg1:
        max_words = st.slider("Max Kata", 50, 300, 120, step=10, key="wc_max_words_dual")
    with col_cfg2:
        background_color = st.selectbox("Latar Belakang", ["white", "black", "lightgray"], index=0, key="wc_bg_dual")
    with col_cfg3:
        cmap_pos = st.selectbox("Skema Positif", WORDCLOUD_COLOR_SCHEMES, index=0, key="wc_cmap_pos")
    with col_cfg4:
        cmap_neg = st.selectbox("Skema Negatif", WORDCLOUD_COLOR_SCHEMES, index=3, key="wc_cmap_neg")

    pos_series = df[df['predicted_sentiment'] == 'POSITIF'][text_col].dropna().astype(str)
    neg_series = df[df['predicted_sentiment'] == 'NEGATIF'][text_col].dropna().astype(str)

    pos_text = " ".join(pos_series.tolist())
    neg_text = " ".join(neg_series.tolist())

    def sufficient(text: str, min_unique: int = 5) -> bool:
        return len({t for t in text.split() if t}) >= min_unique

    col_pos, col_neg = st.columns(2)
    any_generated = False
    with col_pos:
        st.markdown("### 😊 Positif")
        if sufficient(pos_text):
            wc_pos = create_wordcloud(pos_text, max_words=max_words, background_color=background_color, colormap=cmap_pos)
            if wc_pos is not None:
                st.image(wc_pos.to_array(), use_column_width=True)
                any_generated = True
                if wc_pos.words_:
                    top_pos = max(wc_pos.words_.items(), key=lambda x: x[1])[0]
                    st.caption(f"Kata tersering: {top_pos}")
            else:
                st.info("Tidak dapat membentuk word cloud positif.")
        else:
            st.info("Data positif belum cukup (kata unik < 5).")
    with col_neg:
        st.markdown("### 😞 Negatif")
        if sufficient(neg_text):
            wc_neg = create_wordcloud(neg_text, max_words=max_words, background_color=background_color, colormap=cmap_neg)
            if wc_neg is not None:
                st.image(wc_neg.to_array(), use_column_width=True)
                any_generated = True
                if wc_neg.words_:
                    top_neg = max(wc_neg.words_.items(), key=lambda x: x[1])[0]
                    st.caption(f"Kata tersering: {top_neg}")
            else:
                st.info("Tidak dapat membentuk word cloud negatif.")
        else:
            st.info("Data negatif belum cukup (kata unik < 5).")

    st.markdown("---")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Ulasan Positif", f"{len(pos_series):,}")
    with col_m2:
        st.metric("Ulasan Negatif", f"{len(neg_series):,}")
    with col_m3:
        vocab_union = len({*pos_text.split(), *neg_text.split()})
        st.metric("Total Vokab Unik Gabungan", f"{vocab_union:,}")

    if not any_generated:
        st.warning("⚠️ Word cloud belum dapat ditampilkan. Tambah data atau ubah preprocessing.")
    st.caption("Word cloud ditampilkan jika tiap kategori memiliki ≥5 kata unik.")


def render_text_summary_tab(preprocessed_text: str) -> None:
    """Render the text summary tab with comprehensive error handling."""
    st.subheader("📝 Ringkasan dan Statistik Teks")
    
    # Validate input
    if not preprocessed_text or not preprocessed_text.strip():
        st.warning("⚠️ Tidak ada teks yang tersedia untuk analisis.")
        return
    
    # Basic text statistics with error handling
    try:
        # Ensure NLTK data is available
        try:
            sentences = sent_tokenize(preprocessed_text)
            words = word_tokenize(preprocessed_text)
        except LookupError:
            st.error("❌ Data NLTK tidak tersedia. Menggunakan metode alternatif...")
            # Fallback to simple split
            sentences = preprocessed_text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            words = preprocessed_text.split()
        
        unique_words = set(words)
        
        word_count = len(words)
        char_count = len(preprocessed_text)
        sent_count = len(sentences)
        unique_word_count = len(unique_words)
        
        # Avoid division by zero
        avg_word_len = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        avg_sent_len = word_count / sent_count if sent_count > 0 else 0
        lexical_diversity = unique_word_count / word_count if word_count > 0 else 0
        
        # Display statistics with better formatting
        st.write("#### 📊 Statistik Dasar Teks")
        
        if word_count == 0:
            st.warning("⚠️ Tidak ada kata yang tersedia untuk analisis statistik.")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Kata", f"{word_count:,}")
            st.metric("Rata-rata Panjang Kata", f"{avg_word_len:.1f} karakter")
        
        with col2:
            st.metric("Total Karakter", f"{char_count:,}")
            st.metric("Rata-rata Panjang Kalimat", f"{avg_sent_len:.1f} kata")
        
        with col3:
            st.metric("Total Kalimat", f"{sent_count:,}")
            st.metric(
                "Keragaman Leksikal", 
                f"{lexical_diversity:.3f}",
                help="Rasio kata unik terhadap total kata (0-1). Nilai lebih tinggi = keragaman lebih besar."
            )
        
        # Additional statistics
        st.write("#### 📈 Statistik Lanjutan")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Kata Unik", f"{unique_word_count:,}")
            # Calculate readability estimate
            if sent_count > 0 and word_count > 0:
                flesch_approx = 206.835 - (1.015 * avg_sent_len) - (84.6 * (sum(1 for word in words if len(word) > 6) / word_count))
                st.metric("Skor Keterbacaan (Approx)", f"{max(0, min(100, flesch_approx)):.1f}")
        
        with col2:
            st.metric("Rasio Pengulangan", f"{(1-lexical_diversity):.3f}")
            if word_count > 0:
                long_words_ratio = sum(1 for word in words if len(word) > 6) / word_count
                st.metric("Rasio Kata Panjang (>6 huruf)", f"{long_words_ratio:.3f}")
        
        # Text summarization with improved algorithm
        if sent_count > 3:
            st.write("#### 📄 Ringkasan Ekstraktif Otomatis")
            
            summary_length = st.slider(
                "Persentase teks untuk ringkasan:",
                min_value=10,
                max_value=80,
                value=30,
                key="summary_length",
                help="Persentase kalimat yang akan dimasukkan dalam ringkasan"
            )
            
            try:
                # Create summary using improved frequency-based extraction
                word_freq = FreqDist(words)
                
                # Filter out very short words for scoring
                significant_words = [word for word in words if len(word) > 2]
                word_freq_filtered = FreqDist(significant_words)
                
                sent_scores = {}
                
                for i, sent in enumerate(sentences):
                    if not sent.strip():
                        sent_scores[i] = 0
                        continue
                        
                    try:
                        sent_words = word_tokenize(sent)
                    except:
                        sent_words = sent.split()
                    
                    # Score based on word frequency and sentence length
                    sent_score = 0
                    for word in sent_words:
                        if word in word_freq_filtered and len(word) > 2:
                            sent_score += word_freq_filtered[word]
                    
                    # Normalize by sentence length to avoid bias toward long sentences
                    sent_scores[i] = sent_score / len(sent_words) if len(sent_words) > 0 else 0
                
                # Select top sentences
                num_sent_for_summary = max(1, int(len(sentences) * summary_length / 100))
                top_sent_indices = sorted(
                    sorted(sent_scores.items(), key=lambda x: -x[1])[:num_sent_for_summary],
                    key=lambda x: x[0]
                )
                
                summary = ' '.join(sentences[idx].strip() for idx, _ in top_sent_indices if sentences[idx].strip())
                
                if summary.strip():
                    st.write("**Ringkasan Teks:**")
                    st.info(summary)
                    
                    # Summary statistics
                    compression_ratio = (1 - (len(summary) / len(preprocessed_text))) * 100
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Kompresi", f"{compression_ratio:.1f}%")
                    with col2:
                        st.metric("Kalimat dalam Ringkasan", f"{num_sent_for_summary} dari {sent_count}")
                else:
                    st.warning("⚠️ Tidak dapat membuat ringkasan yang bermakna dari teks ini.")
                    
            except Exception as e:
                st.error(f"❌ Gagal membuat ringkasan: {str(e)}")
                st.info("Coba dengan teks yang lebih panjang atau kurangi tingkat preprocessing.")
                
        else:
            st.info("📝 Teks terlalu pendek untuk membuat ringkasan ekstraktif (minimal 4 kalimat diperlukan).")
            
            # Suggest improvements
            with st.expander("💡 Saran untuk meningkatkan analisis"):
                st.write("""
                - Upload file dengan lebih banyak data teks
                - Pastikan preprocessing tidak terlalu agresif
                - Gabungkan beberapa ulasan untuk analisis yang lebih comprehensive
                """)
            
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan dalam analisis teks: {str(e)}")
        
        # Provide fallback basic statistics
        with st.expander("ℹ️ Informasi Debug"):
            st.text(f"Error: {str(e)}")
            st.text(f"Text length: {len(preprocessed_text)}")
            st.text(f"Text preview: {preprocessed_text[:100]}...")
            
        # Try basic statistics without NLTK
        try:
            basic_word_count = len(preprocessed_text.split())
            basic_char_count = len(preprocessed_text)
            
            st.write("**📊 Statistik Dasar (Mode Fallback):**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Jumlah Kata (perkiraan)", basic_word_count)
            with col2:
                st.metric("Jumlah Karakter", basic_char_count)
                
        except Exception as fallback_error:
            st.error(f"❌ Gagal menghitung statistik dasar: {str(fallback_error)}")

# ==============================================================================
# MAIN ANALYSIS TABS
# ==============================================================================

def render_analysis_tabs(df: pd.DataFrame, preprocessed_text: str) -> None:
    """Render all analysis tabs."""
    tabs = st.tabs([
        "📋 Tabel Hasil",
        "📝 Analisis Kata",  # Merged word frequency + N-gram
        "☁️ Word Cloud",
        "📝 Ringkasan Teks"
    ])

    with tabs[0]:
        render_results_table_tab(df)

    with tabs[1]:
        render_word_analysis_tab(df, preprocessed_text)

    with tabs[2]:
        render_wordcloud_tab(df)

    with tabs[3]:
        render_text_summary_tab(preprocessed_text)


def render_word_analysis_tab(df: pd.DataFrame, preprocessed_text: str) -> None:
    """Unified word analysis (frequency + N-gram) mirroring best pattern from dashboard summary (tanpa wordcloud)."""

    st.subheader("📝 Analisis Kata & Frasa")
    st.markdown(
        """
        *Temukan kata paling sering muncul dan kombinasi kata (N-Gram) untuk memahami pola umum dalam ulasan.*
        - Mode default menampilkan ringkasan frekuensi kata.
        - Aktifkan segmentasi per sentimen untuk membandingkan kata dominan antara ulasan positif dan negatif.
        - Bagian N-Gram tersedia di bawah dengan opsi pengaturan tambahan.
        """
    )

    if not preprocessed_text or not preprocessed_text.strip():
        st.warning("⚠️ Tidak ada teks yang tersedia untuk dianalisis.")
        return

    # =============================
    # 1. PENGATURAN UTAMA (Ringkas)
    # =============================
    col_main1, col_main2, col_main3 = st.columns([1, 1, 1])
    with col_main1:
        top_n_words = st.slider(
            "Top Kata:", 5, 50, 10, step=5,
            help="Jumlah kata teratas yang akan divisualisasikan"
        )
    with col_main2:
        segment_mode = st.selectbox(
            "Mode Frekuensi:", ["Gabungan", "Per Sentimen"], index=0,
            help="Pilih 'Per Sentimen' untuk melihat perbandingan POSITIF vs NEGATIF"
        )
    with col_main3:
        orientation = st.radio(
            "Orientasi Bar:", ["Horizontal", "Vertikal"], index=0,
            help="Ubah orientasi tampilan chart kata"
        )

    with st.expander("⚙️ Pengaturan Tambahan (Opsional)"):
        st.markdown("Gunakan pengaturan lanjutan di bawah ini jika diperlukan penyesuaian presentasi.")
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            min_token_len = st.number_input(
                "Minimal panjang token ikut dihitung", min_value=1, max_value=5, value=2, step=1,
                help="Token lebih pendek dari nilai ini akan diabaikan dalam perhitungan frekuensi"
            )
        with col_adv2:
            show_tables_expanded = st.checkbox(
                "Buka tabel frekuensi secara default", value=False,
                help="Jika aktif, tabel rincian langsung terbuka tanpa perlu diklik"
            )

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
    st.markdown("### 📊 Frekuensi Kata")

    if segment_mode == "Gabungan":
        if not overall_freq:
            st.info("📝 Tidak ada kata yang cukup untuk dianalisis.")
        else:
            top_items = list(overall_freq.items())[:top_n_words]
            freq_df = pd.DataFrame(top_items, columns=["Kata", "Frekuensi"])
            if orientation == "Horizontal":
                freq_df_plot = freq_df.sort_values('Frekuensi', ascending=True)
                fig = px.bar(
                    freq_df_plot,
                    x='Frekuensi', y='Kata', orientation='h',
                    title=f"Top {len(freq_df_plot)} Kata Teratas (Gabungan)",
                    color='Frekuensi', color_continuous_scale='Viridis'
                )
            else:
                fig = px.bar(
                    freq_df.head(top_n_words),
                    x='Kata', y='Frekuensi',
                    title=f"Top {len(freq_df)} Kata Teratas (Gabungan)",
                    color='Frekuensi', color_continuous_scale='Viridis'
                )
                fig.update_xaxes(tickangle=45)
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            coverage = (freq_df['Frekuensi'].sum() / max(sum(overall_freq.values()), 1)) * 100
            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                st.metric("Kata Ditampilkan", len(freq_df))
            with colm2:
                st.metric("Total Kata Unik", f"{len(overall_freq):,}")
            with colm3:
                st.metric("Cakupan Top Kata", f"{coverage:.1f}%")
            with st.expander("📋 Tabel Frekuensi (Gabungan)", expanded=show_tables_expanded):
                st.dataframe(freq_df, use_container_width=True, hide_index=True)
    else:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**😊 Positif**")
            if not pos_freq:
                st.info("Tidak ada data positif atau kolom sentimen tidak tersedia.")
            else:
                pos_items = list(pos_freq.items())[:top_n_words]
                pos_df = pd.DataFrame(pos_items, columns=["Kata", "Frekuensi"])
                pos_df_plot = pos_df.sort_values('Frekuensi', ascending=True)
                fig_pos = px.bar(
                    pos_df_plot,
                    x='Frekuensi', y='Kata', orientation='h',
                    title=f"Top {len(pos_df_plot)} Kata Positif",
                    color='Frekuensi', color_continuous_scale='Greens'
                )
                fig_pos.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_pos, use_container_width=True)
                with st.expander("📋 Tabel Kata Positif", expanded=show_tables_expanded):
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
        with colB:
            st.markdown("**😞 Negatif**")
            if not neg_freq:
                st.info("Tidak ada data negatif atau kolom sentimen tidak tersedia.")
            else:
                neg_items = list(neg_freq.items())[:top_n_words]
                neg_df = pd.DataFrame(neg_items, columns=["Kata", "Frekuensi"])
                neg_df_plot = neg_df.sort_values('Frekuensi', ascending=True)
                fig_neg = px.bar(
                    neg_df_plot,
                    x='Frekuensi', y='Kata', orientation='h',
                    title=f"Top {len(neg_df_plot)} Kata Negatif",
                    color='Frekuensi', color_continuous_scale='Reds'
                )
                fig_neg.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_neg, use_container_width=True)
                with st.expander("📋 Tabel Kata Negatif", expanded=show_tables_expanded):
                    st.dataframe(neg_df, use_container_width=True, hide_index=True)

    # =============================
    # 3. ANALISIS N-GRAM
    # =============================
    st.markdown("---")
    st.markdown("### 🔄 Analisis N-Gram")
    st.caption("N-Gram membantu menemukan frasa umum yang muncul bersama dan mengungkap konteks lebih spesifik.")

    with st.expander("⚙️ Pengaturan N-Gram", expanded=False):
        col_ng1, col_ng2, col_ng3 = st.columns(3)
        with col_ng1:
            ngram_type = st.radio("Tipe N-Gram:", ["Bigram", "Trigram"], horizontal=True)
        with col_ng2:
            top_n_ngrams = st.slider("Top N-Gram:", 5, 30, 10, step=5)
        with col_ng3:
            ngram_scope = st.selectbox(
                "Subset Data:", ["Gabungan", "Hanya Positif", "Hanya Negatif"], index=0,
                help="Pilih subset data untuk dihitung N-Gram"
            )
    # Defaults if expander collapsed
    if 'ngram_type' not in locals():
        ngram_type = "Bigram"
        top_n_ngrams = 10
        ngram_scope = "Gabungan"

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

    n_val = 2 if ngram_type == "Bigram" else 3
    ngram_data = get_ngrams(corpus_text, n_val, top_n=top_n_ngrams)

    if ngram_data:
        ngram_df = pd.DataFrame(list(ngram_data.items()), columns=['N-Gram', 'Frekuensi'])
        ngram_df_plot = ngram_df.sort_values('Frekuensi', ascending=True)
        fig_ng = px.bar(
            ngram_df_plot,
            x='Frekuensi', y='N-Gram', orientation='h',
            title=f"Top {len(ngram_df_plot)} {ngram_type} Teratas ({ngram_scope})",
            color='Frekuensi', color_continuous_scale='Plasma'
        )
        fig_ng.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_ng, use_container_width=True)
        with st.expander("📋 Tabel N-Gram"):
            st.dataframe(ngram_df, use_container_width=True, hide_index=True)
    else:
        st.info("📝 Tidak cukup data untuk membentuk N-Gram pada subset yang dipilih.")

    # =============================
    # 4. CATATAN AKHIR
    # =============================
    st.caption("Kata & N-Gram dihitung dari teks hasil preprocessing (token < panjang minimum diabaikan). Word Cloud tersedia pada tab terpisah.")

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
    
    # Page header
    st.title("📑 Analisis Data Teks GoRide")
    st.markdown("---")
    
    # Load model and data
    try:
        data = load_sample_data()
        if data.empty:
            st.error("❌ Data training tidak tersedia untuk analisis!")
            st.stop()
        
        # Load trained model - only get what we need
        preprocessing_options = DEFAULT_PREPROCESSING_OPTIONS.copy()
        pipeline, accuracy, precision, recall, f1, _, _, _, _, _ = get_or_train_model(
            data, preprocessing_options
        )
        
        # Display model info in sidebar
        with st.sidebar:
            st.info(f"""
            🤖 **Model Siap Digunakan**
            
            📊 **Performa Model:**
            - Akurasi: {accuracy:.2%}
            - Precision: {precision:.2%}
            - Recall: {recall:.2%}
            - F1-Score: {f1:.2%}
            
            📈 **Data Training:** {len(data):,} ulasan
            """)
            
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
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
        
        # Prepare preprocessed text for analysis with validation
        try:
            if 'review_text' in df.columns:
                all_text = " ".join(df['review_text'].astype(str).tolist())
            else:
                st.warning("⚠️ Kolom review_text tidak tersedia untuk analisis teks mendalam.")
                all_text = ""
            
            preprocess_options = st.session_state.get('preprocess_options', DEFAULT_PREPROCESSING_OPTIONS)
            
            if all_text.strip():
                preprocessed_all_text = preprocess_text(all_text, preprocess_options)
            else:
                preprocessed_all_text = ""
                
        except Exception as e:
            st.warning(f"⚠️ Gagal memproses teks untuk analisis mendalam: {str(e)}")
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
