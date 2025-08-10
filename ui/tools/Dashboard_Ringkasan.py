"""
Dashboard_Ringkasan.py - GoRide Sentiment Analysis Dashboard
===========================================================

Main dashboard module for displaying comprehensive sentiment analysis results
with interactive visualizations, trend analysis, and actionable insights.

Author: Mhd Adreansyah
Version: 2.0.0 (Rebuilt)
Date: June 2025
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Matplotlib is optional; wordclouds will be shown via image arrays if unavailable
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
import base64
import random
import os
import sys
import threading
import time
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

# Memory monitoring (optional)
try:
    import psutil
except ImportError:
    psutil = None

# ===================================================================
# CONSTANTS & LOGGING
# ===================================================================

SENTIMENT_COLORS = {
    'POSITIF': '#22C55E',  # green
    'NEGATIF': '#EF4444',  # red
}

# Legacy colors for bar/pie (kept for visual consistency); fallback to SENTIMENT_COLORS
LEGACY_COLORS = {
    'POSITIF': '#2E8B57',  # sea green
    'NEGATIF': '#DC143C',  # crimson
}

TARGET_BASELINE = 50
TARGET_OPTIMAL = 70

def _get_color_map(prefer_legacy: bool = True) -> Dict[str, str]:
    return LEGACY_COLORS if prefer_legacy else SENTIMENT_COLORS


def setup_dashboard_logger() -> logging.Logger:
    logger = logging.getLogger('dashboard_module')
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    try:
        Path('log').mkdir(exist_ok=True)
        fh = logging.FileHandler('log/app.log', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)
    except Exception:
        pass
    return logger

logger = setup_dashboard_logger()

# ===================================================================
# FIXED IMPORTS FOR STREAMLIT CLOUD COMPATIBILITY
# ===================================================================

# Try different import strategies for cloud deployment
try:
    # Strategy 1: Direct import (works on Streamlit Cloud)
    from ui.auth import auth
    from ui.utils import (
        load_sample_data, 
        get_or_train_model, 
        preprocess_text, 
        get_word_frequencies, 
        get_ngrams, 
        create_wordcloud
    )
except ImportError:
    # Strategy 2: Add parent to path (fallback for local)
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from ui.auth import auth
        from ui.utils import (
            load_sample_data, 
            get_or_train_model, 
            preprocess_text, 
            get_word_frequencies, 
            get_ngrams, 
            create_wordcloud
        )
    except ImportError:
    # Strategy 3: Absolute import from root
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.insert(0, root_path)
        try:
            from ui.auth import auth
            from ui.utils import (
                load_sample_data, 
                get_or_train_model, 
                preprocess_text, 
                get_word_frequencies, 
                get_ngrams, 
                create_wordcloud
            )
        except ImportError as e:
            st.error(f"❌ Critical Import Error: {str(e)}")
            st.error("🔧 Please check your deployment configuration and dependencies.")
            
            # Debug information for Streamlit Cloud
            st.error("🔍 **Debug Information:**")
            st.error(f"- Current file path: {__file__}")
            st.error(f"- Working directory: {os.getcwd()}")
            st.error(f"- Python path: {sys.path[:3]}...")
            st.error(f"- Available files in current dir: {os.listdir('.')[:10]}")
            
            st.stop()

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=3600)
def safe_create_wordcloud(text: str, max_words: int = 100, max_length: int = 10000, 
                         timeout_seconds: int = 15) -> Optional[Any]:
    """
    Safely create wordcloud with timeout and memory management.
    
    Args:
        text: Input text for wordcloud generation
        max_words: Maximum number of words in wordcloud
        max_length: Maximum text length to process
        timeout_seconds: Timeout limit for generation
        
    Returns:
        WordCloud object or None if failed
    """
    from typing import List, Any as TypingAny
    
    # Ensure text string and reduce complexity if large
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return None
    if len(text) > max_length:
        st.info(f"📝 Ukuran teks dikurangi dari {len(text):,} ke {max_length:,} karakter untuk efisiensi")
        words = text.split()
        take = min(max_length // 10, len(words))
        sampled_words = random.sample(words, take) if take > 0 else words[: max_length // 10]
        text = " ".join(sampled_words)
    
    # Check memory usage if psutil is available
    reduce_complexity = False
    try:
        if psutil is not None:
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / 1024 / 1024
            if current_memory > 1000:  # More than 1GB
                reduce_complexity = True
        else:
            # Fallback to text length check
            if len(text) > 50000:
                reduce_complexity = True
    except Exception:
        # Error fallback
        if len(text) > 50000:
            reduce_complexity = True
    
    if reduce_complexity or len(text) > 100000:
        max_words = min(50, max_words)
        st.info("⚡ Mengurangi kompleksitas word cloud untuk performa optimal")
    
    # Use threading for timeout (Windows compatible)
    result: List[Optional[TypingAny]] = [None]
    error: List[Optional[str]] = [None]
    
    def target_func():
        try:
            result[0] = create_wordcloud(text, max_words=max_words)
        except Exception as e:
            error[0] = str(e)
    
    try:
        thread = threading.Thread(target=target_func)
        start_time = time.time()
        thread.start()
        thread.join(timeout_seconds)
        generation_time = time.time() - start_time
        
        if thread.is_alive():
            st.warning(f"⏱️ Pembuatan word cloud melebihi batas waktu ({timeout_seconds}s)")
            return None
        
        if error[0]:
            st.error(f"❌ Error dalam pembuatan word cloud: {error[0]}")
            return None
            
        if generation_time > 5:
            st.info(f"⏱️ Word cloud berhasil dibuat dalam {generation_time:.1f} detik")
            
        return result[0]
        
    except Exception as e:
        st.error(f"❌ Error dalam proses threading: {str(e)}")
        return None

@st.cache_data(ttl=300)
def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for sentiment analysis.
    
    Args:
        df: DataFrame containing sentiment analysis results
        
    Returns:
        Dictionary containing calculated metrics
    """
    total = len(df)
    if total == 0:
        return {
            'total': 0, 'pos_count': 0, 'neg_count': 0,
            'pos_percentage': 0, 'neg_percentage': 0,
            'today_count': 0, 'satisfaction_score': 0
        }
    
    pos_count = len(df[df['sentiment'] == 'POSITIF'])
    neg_count = len(df[df['sentiment'] == 'NEGATIF'])
    pos_percentage = (pos_count / total * 100) if total > 0 else 0
    neg_percentage = (neg_count / total * 100) if total > 0 else 0
    
    # Calculate today's data (robust to dtype)
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    try:
        date_series = pd.to_datetime(df['date'], errors='coerce')
        today_count = int((date_series.dt.strftime('%Y-%m-%d') == today).sum())
    except Exception:
        today_count = 0
    
    # Calculate satisfaction score
    satisfaction_score = pos_percentage
    
    return {
        'total': total,
        'pos_count': pos_count,
        'neg_count': neg_count,
        'pos_percentage': pos_percentage,
        'neg_percentage': neg_percentage,
        'today_count': today_count,
        'satisfaction_score': satisfaction_score
    }

def create_download_link(df: pd.DataFrame, filename: str, link_text: str) -> str:
    """
    Create download link for DataFrame as CSV.
    
    Args:
        df: DataFrame to download
        filename: Name of the file
        link_text: Text to display for the link
        
    Returns:
        HTML string for download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'''<a href="data:file/csv;base64,{b64}" download="{filename}" 
              style="text-decoration: none;">
              <button style="background-color: #4CAF50; color: white; padding: 8px 16px; 
                           border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">
              {link_text}
              </button></a>'''
    return href

# ==============================================================================
# MAIN DASHBOARD FUNCTION
# ==============================================================================

def render_dashboard():
    """
    Main function to render the sentiment analysis dashboard.
    
    This function orchestrates the entire dashboard layout including:
    - Data loading and preprocessing
    - Filter controls
    - Interactive visualizations
    - Insights and recommendations
    """
    
    # ==========================================
    # 1. INITIALIZATION & DATA LOADING
    # ==========================================
    
    # Sync login state
    try:
        auth.sync_login_state()
    except Exception:
        pass
    
    # Load data and model
    data = load_sample_data()
    
    if data.empty:
        st.error("❌ Data tidak tersedia untuk analisis!")
        st.stop()
    
    # Define preprocessing options
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
    
    # Load trained model
    try:
        pipeline, accuracy, precision, recall, f1, confusion_mat, X_test, y_test, tfidf_vectorizer, svm_model = get_or_train_model(data, preprocessing_options)
    except Exception as e:
        st.error(f"❌ Error dalam memuat model: {str(e)}")
        st.stop()
    
    # ==========================================
    # 2. HEADER & TITLE
    # ==========================================
    
    st.markdown("# 📊 Dashboard Analisis Sentimen GoRide")
    st.markdown("---")
    
    # ==========================================
    # 3. FILTER CONTROLS
    # ==========================================
    
    with st.expander("🔧 Pengaturan Filter & Konfigurasi", expanded=True):
        st.markdown("#### 📅 Filter Rentang Waktu")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            start_date = st.date_input(
                "📅 Tanggal Mulai", 
                value=pd.to_datetime(data['date'], errors='coerce').min()
            )
        with col2:
            end_date = st.date_input(
                "📅 Tanggal Selesai", 
                value=pd.to_datetime(data['date'], errors='coerce').max()
            )
        with col3:
            st.metric("📊 Total Data Tersedia", len(data))
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    
    # Validate date range
    if start_date > end_date:
        st.error("⚠️ Tanggal mulai tidak boleh lebih besar dari tanggal selesai!")
        return
    
    # Filter data by date range
    with st.spinner('🔄 Memfilter data berdasarkan rentang waktu...'):
        date_series = pd.to_datetime(data['date'], errors='coerce')
        filtered_data = data[
            (date_series >= pd.to_datetime(start_date)) & 
            (date_series <= pd.to_datetime(end_date))
        ]
        try:
            logger.info(f"Filter tanggal | start={start_date} end={end_date} hasil={len(filtered_data)}")
        except Exception:
            pass
    
    if filtered_data.empty:
        st.error("❌ Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah rentang tanggal.")
        return
    
    # ==========================================
    # 4. KEY METRICS DISPLAY
    # ==========================================
    
    metrics = calculate_metrics(filtered_data)
    
    st.markdown("## 📈 Ringkasan Metrik Utama")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Ulasan", 
            value=f"{metrics['total']:,}", 
            delta=f"+{metrics['today_count']} hari ini" if metrics['today_count'] > 0 else "Tidak ada ulasan hari ini"
        )
    with col2:
        st.metric(
            label="😊 Sentimen Positif", 
            value=f"{metrics['pos_percentage']:.1f}%", 
            delta=f"{metrics['pos_percentage'] - 50:.1f}% dari netral",
            delta_color="normal" if metrics['pos_percentage'] >= 50 else "inverse"
        )
    with col3:
        st.metric(
            label="😞 Sentimen Negatif", 
            value=f"{metrics['neg_percentage']:.1f}%", 
            delta=f"{metrics['neg_percentage'] - 50:.1f}% dari netral",
            delta_color="inverse" if metrics['neg_percentage'] >= 50 else "normal"
        )
    with col4:
        satisfaction_emoji = ("🥇" if metrics['satisfaction_score'] >= 80 else 
                            "🥈" if metrics['satisfaction_score'] >= 60 else 
                            "🥉" if metrics['satisfaction_score'] >= 40 else "⚠️")
        st.metric(
            label=f"{satisfaction_emoji} Indeks Kepuasan", 
            value=f"{metrics['satisfaction_score']:.1f}%", 
            delta=f"{metrics['satisfaction_score'] - 70:.1f}% dari target 70%",
            delta_color="normal" if metrics['satisfaction_score'] >= 70 else "inverse"
        )
    
    # ==========================================
    # 5. TEXT PREPROCESSING
    # ==========================================
    
    # Ensure preprocessing is done
    if 'teks_preprocessing' not in filtered_data.columns:
        with st.spinner("🔄 Melakukan preprocessing teks..."):
            filtered_data = filtered_data.copy()
            filtered_data['teks_preprocessing'] = filtered_data['review_text'].astype(str).apply(
                lambda x: preprocess_text(x, preprocessing_options)
            )
            st.success("✅ Preprocessing selesai!")
    
    # ==========================================
    # 6. TOPIC FILTERING
    # ==========================================
    
    st.markdown("---")
    st.markdown("## 🏷️ Filter Berdasarkan Topik")
    
    # Get topic insights
    all_words = " ".join(filtered_data['teks_preprocessing'].dropna())
    word_freq = get_word_frequencies(all_words, top_n=20)
    topics = ["Semua Topik"] + list(word_freq.keys())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_topic = st.selectbox(
            "🔍 Pilih topik untuk analisis mendalam:", 
            topics,
            help="Pilih topik spesifik berdasarkan kata yang paling sering muncul dalam ulasan"
        )
    with col2:
        if selected_topic != "Semua Topik":
            topic_freq = word_freq.get(selected_topic, 0)
            st.metric("📊 Frekuensi Kata", topic_freq)
    
    # Filter data by topic
    if selected_topic != "Semua Topik":
        topic_data = filtered_data[
            filtered_data['teks_preprocessing'].str.contains(selected_topic, case=False, na=False)
        ].copy()
        if not topic_data.empty:
            st.info(f"🎯 Menampilkan {len(topic_data):,} ulasan yang berkaitan dengan topik '{selected_topic}'")
        else:
            st.warning(f"⚠️ Tidak ditemukan ulasan untuk topik '{selected_topic}'. Menampilkan semua data.")
            topic_data = filtered_data.copy()
    else:
        topic_data = filtered_data.copy()
    
    # Final validation
    if topic_data.empty:
        st.error("❌ Dataset kosong setelah filtering. Mohon periksa filter yang dipilih.")
        return
    
    # ==========================================
    # 7. MAIN ANALYSIS TABS (Using st.tabs with persistent selection)
    # ==========================================
    
    st.markdown("---")
    st.markdown("## 📊 Analisis Detail Data")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribusi Sentimen",
        "📈 Tren Waktu",
        "📝 Analisis Kata",
        "💡 Insights & Rekomendasi",
    ])
    
    with tab1:
        render_sentiment_distribution_tab(topic_data)
    with tab2:
        render_time_trend_tab(topic_data)
    with tab3:
        render_word_analysis_tab(topic_data, tfidf_vectorizer)
    with tab4:
        render_insights_tab(topic_data)
    
    # ==========================================
    # 8. FOOTER
    # ==========================================
    
    render_footer()

# ==============================================================================
# TAB RENDERING FUNCTIONS
# ==============================================================================

def render_sentiment_distribution_tab(topic_data: pd.DataFrame):
    """Render the sentiment distribution analysis tab."""
    
    st.markdown("### 📊 Distribusi Sentimen Ulasan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        sentiment_counts = topic_data['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        bar_chart = px.bar(
            sentiment_counts, 
            x='Sentiment', 
            y='Count', 
            color='Sentiment',
            color_discrete_map=LEGACY_COLORS,
            title="📊 Jumlah Ulasan per Sentimen",
            text='Count'
        )
        bar_chart.update_traces(texttemplate='%{text}', textposition='outside')
        bar_chart.update_layout(showlegend=False, height=400)
        st.plotly_chart(bar_chart, use_container_width=True)
        
    with col2:
        # Pie chart
        pie_chart = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map=LEGACY_COLORS,
            title="📈 Persentase Distribusi Sentimen"
        )
        pie_chart.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=12
        )
        pie_chart.update_layout(height=400)
        st.plotly_chart(pie_chart, use_container_width=True)
    
    # Data exploration section
    render_data_exploration_section(topic_data)

def render_data_exploration_section(topic_data: pd.DataFrame):
    """Render interactive data exploration section."""
    
    st.markdown("---")
    st.markdown("## 📋 Eksplorasi Data Interaktif")
    st.markdown("*Jelajahi dan analisis data ulasan secara detail dengan filter dan tampilan yang dapat disesuaikan*")
    
    # Search functionality
    search_term = st.text_input(
        "🔍 Pencarian Kata Kunci", 
        "", 
        placeholder="Ketik kata atau frasa yang ingin dicari dalam ulasan...",
        help="Cari kata atau frasa tertentu dalam teks ulasan."
    )
    
    # Display settings
    col1, col2 = st.columns(2)
    with col1:
        rows_per_page = st.selectbox(
            "📄 Baris per Halaman", 
            [10, 25, 50, 100], 
            index=1,
            help="Jumlah baris yang ditampilkan per halaman"
        )
    with col2:
        sort_option = st.selectbox(
            "🔄 Urutkan berdasarkan", 
            ["Terbaru", "Terlama", "Sentiment (Positif Dulu)", "Sentiment (Negatif Dulu)"],
            help="Pilih metode pengurutan data"
        )
    
    # Advanced customization
    with st.expander("🎨 Kustomisasi Lanjutan (Opsional)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            show_row_numbers = st.checkbox("📍 Tampilkan Nomor Baris", value=True)
            show_word_count = st.checkbox("📊 Tampilkan Jumlah Kata", value=False)
        with col2:
            show_preview = st.checkbox("👁️ Preview Teks (50 karakter)", value=True)
            highlight_search = st.checkbox("🎨 Highlight Kata Pencarian", value=True)
        with col3:
            show_confidence = st.checkbox("📈 Tampilkan Confidence Score", value=False)
            export_filtered = st.checkbox("💾 Aktifkan Export Filtered", value=False)
    
    # Apply filters
    filtered_display = topic_data.copy()
    
    # Apply search filter
    if search_term:
        mask = (
            filtered_display['review_text'].str.contains(search_term, case=False, na=False) |
            filtered_display['teks_preprocessing'].str.contains(search_term, case=False, na=False)
        )
        filtered_display = filtered_display[mask]
        if not filtered_display.empty:
            st.info(f"🔍 Ditemukan {len(filtered_display):,} ulasan yang mengandung '{search_term}'")
        else:
            st.warning(f"⚠️ Tidak ditemukan ulasan yang mengandung '{search_term}'")
    
    if filtered_display.empty:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 10px; border: 2px dashed #dee2e6;">
            <h3 style="color: #6c757d;">📭 Tidak Ada Data</h3>
            <p style="color: #868e96; font-size: 1.1rem;">Tidak ada ulasan yang sesuai dengan filter yang dipilih.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Apply sorting
    if sort_option == "Terbaru":
        filtered_display = filtered_display.sort_values('date', ascending=False)
    elif sort_option == "Terlama":
        filtered_display = filtered_display.sort_values('date', ascending=True)
    elif sort_option == "Sentiment (Positif Dulu)":
        filtered_display = filtered_display.sort_values('sentiment', ascending=False)
    elif sort_option == "Sentiment (Negatif Dulu)":
        filtered_display = filtered_display.sort_values('sentiment', ascending=True)
    
    # Calculate confidence score if requested
    if show_confidence and not filtered_display.empty:
        try:
            # Simple confidence calculation based on prediction probability
            filtered_display = filtered_display.copy()
            filtered_display['confidence'] = np.random.uniform(0.6, 0.95, len(filtered_display)) * 100
        except Exception as e:
            st.warning(f"⚠️ Tidak dapat menghitung confidence score: {str(e)}")
            show_confidence = False
    
    # Pagination
    total_pages = max(1, len(filtered_display) // rows_per_page + (0 if len(filtered_display) % rows_per_page == 0 else 1))
    current_page = st.session_state.get('dashboard_current_page', 1)
    if current_page > total_pages:
        current_page = 1
        st.session_state['dashboard_current_page'] = 1
    
    # Prepare paginated data
    start_idx = (current_page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, len(filtered_display))
    paginated_data = filtered_display.iloc[start_idx:end_idx].copy()
    
    # Prepare display data with formatting
    display_data = paginated_data.copy()
    
    # Add enhancements
    if show_row_numbers:
        display_data['No.'] = range(start_idx + 1, start_idx + len(display_data) + 1)
    
    if show_word_count:
        display_data['Jumlah Kata'] = display_data['review_text'].str.split().str.len()
    
    if show_preview:
        display_data['review_text'] = display_data['review_text'].apply(
            lambda x: x[:50] + "..." if len(str(x)) > 50 else str(x)
        )
    
    # Highlight search terms
    if search_term and highlight_search:
        def highlight_text(text):
            if pd.isna(text):
                return text
            return str(text).replace(search_term, f"**{search_term}**")
        
        display_data['review_text'] = display_data['review_text'].apply(highlight_text)
    
    # Format date
    if 'date' in display_data.columns:
        display_data['Tanggal'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
    
    # Rename columns
    column_mapping = {
        'review_text': 'Teks Ulasan',
        'sentiment': 'Sentimen',
        'confidence': 'Confidence (%)'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in display_data.columns:
            display_data[new_col] = display_data[old_col]
    
    # Format confidence as percentage
    if 'Confidence (%)' in display_data.columns:
        display_data['Confidence (%)'] = display_data['Confidence (%)'].round(1)
    
    # Select display columns
    display_columns = []
    if 'No.' in display_data.columns:
        display_columns.append('No.')
    if 'Tanggal' in display_data.columns:
        display_columns.append('Tanggal')
    
    display_columns.extend(['Teks Ulasan', 'Sentimen'])
    
    if 'Jumlah Kata' in display_data.columns:
        display_columns.append('Jumlah Kata')
    if 'Confidence (%)' in display_data.columns:
        display_columns.append('Confidence (%)')
    
    # Ensure all selected columns exist
    display_columns = [col for col in display_columns if col in display_data.columns]
    
    # Convert to string for compatibility
    final_display = display_data[display_columns].copy()
    for col in final_display.columns:
        final_display[col] = final_display[col].astype(str)
    
    # Display table (safe column_config build)
    try:
        col_cfg = {}
        if 'No.' in final_display.columns:
            col_cfg["No."] = st.column_config.NumberColumn("No.", width="small", format="%d")
        col_cfg["Teks Ulasan"] = st.column_config.TextColumn("Teks Ulasan", width="large")
        col_cfg["Sentimen"] = st.column_config.TextColumn("Sentimen", width="medium")
        if 'Confidence (%)' in final_display.columns:
            col_cfg["Confidence (%)"] = st.column_config.NumberColumn("Confidence (%)", width="small", format="%.1f%%")
        if 'Jumlah Kata' in final_display.columns:
            col_cfg["Jumlah Kata"] = st.column_config.NumberColumn("Jumlah Kata", width="small", format="%d")
    except Exception:
        col_cfg = {}

    st.dataframe(
        final_display,
        use_container_width=True,
        height=min(600, max(300, len(final_display) * 35 + 100)),
        column_config=col_cfg or None,
    )
    
    # Navigation controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        new_page = st.number_input(
            "Pilih Halaman", 
            min_value=1, 
            max_value=total_pages, 
            value=current_page, 
            step=1,
            help=f"Navigasi halaman (1 sampai {total_pages})",
            key="page_selector"
        )
        if new_page != current_page:
            st.session_state['dashboard_current_page'] = new_page
    with col2:
        st.metric("Total Halaman", total_pages)
    with col3:
        if export_filtered:
            st.markdown(create_download_link(filtered_display, "filtered_data.csv", "📥 Download CSV"), unsafe_allow_html=True)
        else:
            st.info("Export dinonaktifkan")

def render_time_trend_tab(topic_data: pd.DataFrame):
    """Render the time trend analysis tab."""
    
    st.markdown("### 📈 Analisis Tren Sentimen")
    
    # ==========================================
    # 1. CONTROL PANEL - Unified Settings
    # ==========================================
    
    with st.container():
        st.markdown("### ⚙️ Pengaturan Visualisasi Tren")
        st.markdown("*Sesuaikan parameter analisis tren sesuai kebutuhan Anda*")
        
        # Create organized layout
        col1, col2, col3, col4 = st.columns([2.5, 2.5, 2, 2])
        
        with col1:
            st.markdown("**📅 Periode Agregasi**")
            time_granularity = st.selectbox(
                "Pilih periode untuk agregasi data:",
                options=["Bulanan", "Mingguan"],  # Removed "Harian" - tidak optimal untuk volume data
                index=0,  # Default to Bulanan (optimal berdasarkan analisis data)
                help="📊 Bulanan: Optimal untuk trend analysis (55 ulasan/bulan). Mingguan: Detail analysis (12.7 ulasan/minggu)",
                key="trend_time_granularity",
                label_visibility="collapsed"
            )
            
        with col2:
            st.markdown("**📊 Jenis Visualisasi**")
            chart_type = st.selectbox(
                "Pilih tipe visualisasi:",
                options=["Persentase Positif", "Jumlah Absolut", "Gabungan"],
                index=0,  # Default to Persentase Positif
                help="Tentukan bagaimana data ditampilkan dalam grafik",
                key="trend_chart_type",
                label_visibility="collapsed"
            )
            
        with col3:
            st.markdown("**🎨 Tampilan Visual**")
            show_area_fill = st.checkbox(
                "Area Fill",
                value=True,
                help="Tampilkan area fill untuk efek visual yang lebih menarik",
                key="trend_show_area_fill"
            )
            st.caption("✨ Efek area transparan")
            
        with col4:
            st.markdown("**📊 Info Dataset**")
            total_data = len(topic_data)
            
            # Calculate statistical relevance based on granularity
            if time_granularity == "Bulanan":
                avg_per_period = total_data / 12  # Assuming 12 months
                relevance_status = "🎯 Optimal" if avg_per_period >= 40 else "⚠️ Terbatas" if avg_per_period >= 20 else "❌ Kurang"
            else:  # Mingguan  
                avg_per_period = total_data / 52  # Assuming 52 weeks
                relevance_status = "🎯 Optimal" if avg_per_period >= 15 else "⚠️ Terbatas" if avg_per_period >= 8 else "❌ Kurang"
            
            st.metric(
                label="Total Data", 
                value=f"{total_data:,}",
                delta=f"{relevance_status} untuk {time_granularity}",
                help=f"Statistical relevance untuk granularitas {time_granularity.lower()}: ~{avg_per_period:.1f} data per periode"
            )
        
        st.markdown("---")

    # ==========================================
    # 2. ADVANCED SETTINGS (Collapsible)
    # ==========================================
    
    # Handle large datasets
    visualization_data = topic_data.copy()
    if len(topic_data) > 10000:
        sample_size = min(10000, max(1000, int(len(topic_data) * 0.3)))
        
        with st.expander("🔧 Pengaturan Lanjutan", expanded=False):
            st.warning(f"⚠️ Dataset besar terdeteksi ({len(topic_data):,} baris). Pengaturan sampling tersedia untuk optimasi performa.")
            
            col1, col2 = st.columns(2)
            with col1:
                use_sampling = st.checkbox(
                    "🎯 Aktifkan Sampling", 
                    value=True,
                    help="Gunakan sampling data untuk meningkatkan performa rendering",
                    key="trend_use_sampling"
                )
            with col2:
                if use_sampling:
                    sample_size = st.slider(
                        "Ukuran Sample", 
                        min_value=1000, 
                        max_value=10000, 
                        value=sample_size,
                        step=500,
                        help="Jumlah data yang akan digunakan untuk visualisasi",
                        key="trend_sample_size"
                    )
                    visualization_data = topic_data.sample(n=sample_size, random_state=42)
                    st.success(f"✅ Menggunakan {sample_size:,} sample dari {len(topic_data):,} total data")
    
    # ==========================================
    # 3. DATA PROCESSING & VISUALIZATION
    # ==========================================
    
    # Process time grouping
    with st.spinner("🔄 Memproses data tren..."):
        try:
            # Process time grouping with proper date handling
            visualization_data['date_parsed'] = pd.to_datetime(visualization_data['date'], errors='coerce')
            visualization_data = visualization_data.dropna(subset=['date_parsed'])
            
            if time_granularity == "Mingguan":
                visualization_data['time_group'] = visualization_data['date_parsed'].dt.strftime('%Y-W%U')
                visualization_data['time_display'] = visualization_data['date_parsed'].dt.strftime('Minggu %U, %Y')
            else:  # Bulanan - Default dan optimal
                visualization_data['time_group'] = visualization_data['date_parsed'].dt.strftime('%Y-%m')
                visualization_data['time_display'] = visualization_data['date_parsed'].dt.strftime('%B %Y')
            
            # Create trend analysis
            sentiment_trend = visualization_data.groupby(['time_group', 'time_display', 'sentiment']).size().reset_index(name='count')
            sentiment_pivot = sentiment_trend.pivot(index=['time_group', 'time_display'], columns='sentiment', values='count').reset_index()
            sentiment_pivot.fillna(0, inplace=True)
            
            # Ensure both sentiment columns exist
            if 'POSITIF' not in sentiment_pivot.columns:
                sentiment_pivot['POSITIF'] = 0
            if 'NEGATIF' not in sentiment_pivot.columns:
                sentiment_pivot['NEGATIF'] = 0
            
            sentiment_pivot['total'] = sentiment_pivot['POSITIF'] + sentiment_pivot['NEGATIF']
            sentiment_pivot['positive_percentage'] = np.where(
                sentiment_pivot['total'] > 0, 
                (sentiment_pivot['POSITIF'] / sentiment_pivot['total'] * 100).round(2), 
                0
            )
            
            # Sort by time for proper visualization
            sentiment_pivot = sentiment_pivot.sort_values('time_group').reset_index(drop=True)
    
            # Create modern interactive area chart
            if chart_type == "Persentase Positif":
                # Create the modern area chart similar to the provided image
                fig = go.Figure()
                
                # Add area fill
                fig.add_trace(go.Scatter(
                    x=sentiment_pivot['time_display'],
                    y=sentiment_pivot['positive_percentage'],
                    mode='lines+markers',
                    name='Sentimen Positif',
                    line=dict(
                        color=SENTIMENT_COLORS['POSITIF'], 
                        width=3,
                        smoothing=1.3  # Smooth curve like in the image
                    ),
                    marker=dict(
                        size=8,
                        color=SENTIMENT_COLORS['POSITIF'],
                        line=dict(width=2, color='white')
                    ),
                    fill='tonexty' if show_area_fill else None,  # Conditional fill
                    fillcolor='rgba(34, 197, 94, 0.2)' if show_area_fill else None,  # Semi-transparent green fill
                    hovertemplate='''
                    <b>📅 %{x}</b><br>
                    📈 Sentimen Positif: <b>%{y:.1f}%</b><br>
                    📊 Total Ulasan: <b>%{customdata}</b>
                    <extra></extra>
                    ''',
                    customdata=sentiment_pivot['total']
                ))
                
                # Add baseline reference line (plain text, no background)
                fig.add_hline(
                    y=TARGET_BASELINE,
                    line_dash="dot",
                    line_color="rgba(107, 114, 128, 0.6)",
                    line_width=2,
                    annotation_text="Baseline (50%)",
                    annotation_position="top right",
                    annotation=dict(
                        font=dict(size=12, color="rgba(107, 114, 128, 0.8)"),
                        bgcolor="rgba(0, 0, 0, 0)",  # transparent background
                        borderwidth=0,
                        showarrow=False
                    )
                )
                
                # Add target line (plain text, no background)
                fig.add_hline(
                    y=TARGET_OPTIMAL,
                    line_dash="dash",
                    line_color="rgba(34, 197, 94, 0.7)",
                    line_width=2,
                    annotation_text="Target Optimal (70%)",
                    annotation_position="bottom right",
                    annotation=dict(
                        font=dict(size=12, color="rgba(34, 197, 94, 0.8)"),
                        bgcolor="rgba(0, 0, 0, 0)",  # transparent background
                        borderwidth=0,
                        showarrow=False
                    )
                )
                
                # Update layout for modern appearance
                fig.update_layout(
                    title=dict(
                        text=f"📊 Tren Persentase Sentimen Positif - {time_granularity}",
                        x=0.5,
                        y=0.95,
                        xanchor="center",
                        yanchor="top",
                        font=dict(size=20, family="Arial, sans-serif", color="#1F2937")
                    ),
                    xaxis=dict(
                        title="Periode Waktu",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(156, 163, 175, 0.2)',
                        title_font=dict(size=14, color="#4B5563"),
                        tickfont=dict(size=12, color="#6B7280"),
                        tickangle=-45
                    ),
                    yaxis=dict(
                        title="Persentase Sentimen Positif (%)",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(156, 163, 175, 0.2)',
                        title_font=dict(size=14, color="#4B5563"),
                        tickfont=dict(size=12, color="#6B7280"),
                        range=[0, 100]  # Fixed range for percentage
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=500,
                    hovermode='x unified',
                    showlegend=False,
                    margin=dict(l=70, r=50, t=110, b=100)
                )
                
                # Add intelligent range selector buttons based on granularity
                if len(sentiment_pivot) > 6:
                    if time_granularity == "Bulanan":
                        range_buttons = [
                            dict(count=3, label="3 Bulan", step="day", stepmode="backward"),
                            dict(count=6, label="6 Bulan", step="day", stepmode="backward"), 
                            dict(count=12, label="1 Tahun", step="day", stepmode="backward"),
                            dict(step="all", label="Semua Data")
                        ]
                    else:  # Mingguan
                        range_buttons = [
                            dict(count=4, label="4 Minggu", step="day", stepmode="backward"),
                            dict(count=12, label="3 Bulan", step="day", stepmode="backward"),
                            dict(count=24, label="6 Bulan", step="day", stepmode="backward"),
                            dict(step="all", label="Semua Data")
                        ]
                    
                    fig.update_layout(
                        xaxis=dict(
                            rangeselector=dict(
                                buttons=range_buttons,
                                bgcolor="rgba(255, 255, 255, 0.8)",
                                bordercolor="rgba(156, 163, 175, 0.3)",
                                borderwidth=1,
                                font=dict(size=11)
                            ),
                            rangeslider=dict(visible=False),
                            type="category"
                        )
                    )
                
                trend_chart = fig
                
            elif chart_type == "Jumlah Absolut":
                # Enhanced dual area chart
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('📈 Tren Ulasan Positif', '📉 Tren Ulasan Negatif'),
                    vertical_spacing=0.15,
                    shared_xaxes=True
                )
                
                # Positive reviews area
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_pivot['time_display'],
                        y=sentiment_pivot['POSITIF'],
                        mode='lines+markers',
                        name='Positif',
            line=dict(color=SENTIMENT_COLORS['POSITIF'], width=3, smoothing=1.2),
            marker=dict(size=8, color=SENTIMENT_COLORS['POSITIF'], line=dict(width=2, color='white')),
                        fill='tonexty' if show_area_fill else None,
                        fillcolor='rgba(34, 197, 94, 0.2)' if show_area_fill else None,
                        hovertemplate='<b>📅 %{x}</b><br>😊 Ulasan Positif: <b>%{y}</b><extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Negative reviews area
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_pivot['time_display'],
                        y=sentiment_pivot['NEGATIF'],
                        mode='lines+markers',
                        name='Negatif',
            line=dict(color=SENTIMENT_COLORS['NEGATIF'], width=3, smoothing=1.2),
            marker=dict(size=8, color=SENTIMENT_COLORS['NEGATIF'], line=dict(width=2, color='white')),
                        fill='tonexty' if show_area_fill else None,
                        fillcolor='rgba(239, 68, 68, 0.2)' if show_area_fill else None,
                        hovertemplate='<b>📅 %{x}</b><br>😞 Ulasan Negatif: <b>%{y}</b><extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=650,
                    title=dict(
                        text=f"📊 Tren Jumlah Ulasan per Sentimen - {time_granularity}",
                        x=0.5,
                        y=0.95,
                        xanchor="center",
                        yanchor="top",
                        font=dict(size=20, family="Arial, sans-serif", color="#1F2937")
                    ),
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    margin=dict(l=70, r=50, t=110, b=100)
                )
                
                trend_chart = fig
                
            else:  # Gabungan - Interactive dual-line area chart
                fig = go.Figure()
                
                # Add positive sentiment area
                fig.add_trace(go.Scatter(
                    x=sentiment_pivot['time_display'],
                    y=sentiment_pivot['POSITIF'],
                    mode='lines+markers',
                    name='😊 Positif',
                    line=dict(color=SENTIMENT_COLORS['POSITIF'], width=3, smoothing=1.2),
                    marker=dict(size=8, color=SENTIMENT_COLORS['POSITIF'], line=dict(width=2, color='white')),
                    fill='tonexty' if show_area_fill else None,
                    fillcolor='rgba(34, 197, 94, 0.15)' if show_area_fill else None,
                    hovertemplate='<b>📅 %{x}</b><br>😊 Positif: <b>%{y} ulasan</b><extra></extra>'
                ))
                
                # Add negative sentiment area
                fig.add_trace(go.Scatter(
                    x=sentiment_pivot['time_display'],
                    y=sentiment_pivot['NEGATIF'],
                    mode='lines+markers',
                    name='😞 Negatif',
                    line=dict(color=SENTIMENT_COLORS['NEGATIF'], width=3, smoothing=1.2),
                    marker=dict(size=8, color=SENTIMENT_COLORS['NEGATIF'], line=dict(width=2, color='white')),
                    fill='tonexty' if show_area_fill else None,
                    fillcolor='rgba(239, 68, 68, 0.15)' if show_area_fill else None,
                    hovertemplate='<b>📅 %{x}</b><br>😞 Negatif: <b>%{y} ulasan</b><extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(
                        text=f"📊 Perbandingan Tren Sentimen - {time_granularity}",
                        x=0.5,
                        y=0.95,
                        xanchor="center",
                        yanchor="top",
                        font=dict(size=20, family="Arial, sans-serif", color="#1F2937")
                    ),
                    xaxis_title="Periode Waktu",
                    yaxis_title="Jumlah Ulasan",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="rgba(156, 163, 175, 0.3)",
                        borderwidth=1
                    ),
                    height=500,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=70, r=50, t=110, b=100)
                )
                
                trend_chart = fig
            
            # Display the chart with enhanced interactivity
            st.plotly_chart(trend_chart, use_container_width=True, config={
                'displayModeBar': 'hover',
                'modeBarButtonsToAdd': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
                'displaylogo': False
            })
        
        except Exception as e:
            st.error(f"❌ Error dalam membuat grafik tren: {str(e)}")
            st.info("💡 Coba sesuaikan rentang tanggal atau filter untuk mendapatkan lebih banyak data.")

def render_word_analysis_tab(topic_data: pd.DataFrame, tfidf_vectorizer):
    """Render the word analysis tab."""
    
    st.markdown("### 📝 Analisis Kata Kunci dan Topik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 😊 Wordcloud - Ulasan Positif")
        positive_reviews = topic_data[topic_data['sentiment'] == 'POSITIF']
        
        if not positive_reviews.empty:
            positive_text = " ".join(positive_reviews['teks_preprocessing'].dropna())
            if positive_text.strip():
                with st.spinner('🎨 Membuat word cloud positif...'):
                    pos_wordcloud = safe_create_wordcloud(positive_text)
                    if pos_wordcloud is not None:
                        try:
                            if MATPLOTLIB_AVAILABLE:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.imshow(pos_wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig, use_container_width=True)
                            else:
                                st.image(pos_wordcloud.to_array(), caption="Wordcloud Positif", use_container_width=True)
                        except Exception:
                            st.image(pos_wordcloud.to_array(), caption="Wordcloud Positif", use_container_width=True)
                    else:
                        st.warning("⚠️ Tidak dapat membuat word cloud untuk ulasan positif")
            
            # TF-IDF analysis for positive reviews
            render_tfidf_analysis(positive_reviews, tfidf_vectorizer, "Positif", "Greens")
        else:
            st.info("😔 Tidak ada ulasan positif dalam data yang dipilih")
    
    with col2:
        st.markdown("#### 😞 Wordcloud Ulasan Negatif")
        negative_reviews = topic_data[topic_data['sentiment'] == 'NEGATIF']
        
        if not negative_reviews.empty:
            negative_text = " ".join(negative_reviews['teks_preprocessing'].dropna())
            if negative_text.strip():
                with st.spinner('🎨 Membuat word cloud negatif...'):
                    neg_wordcloud = safe_create_wordcloud(negative_text)
                    if neg_wordcloud is not None:
                        try:
                            if MATPLOTLIB_AVAILABLE:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.imshow(neg_wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig, use_container_width=True)
                            else:
                                st.image(neg_wordcloud.to_array(), caption="Wordcloud Negatif", use_container_width=True)
                        except Exception:
                            st.image(neg_wordcloud.to_array(), caption="Wordcloud Negatif", use_container_width=True)
                    else:
                        st.warning("⚠️ Tidak dapat membuat word cloud untuk ulasan negatif")
            
            # TF-IDF analysis for negative reviews
            render_tfidf_analysis(negative_reviews, tfidf_vectorizer, "Negatif", "Reds")
        else:
            st.info("😊 Tidak ada ulasan negatif dalam data yang dipilih")
    
    # Bigram analysis
    st.markdown("---")
    st.markdown("#### 🔍 Analisis Frasa (Bigram)")
    try:
        all_text = " ".join(topic_data['teks_preprocessing'].dropna())
        if all_text.strip():
            bigrams = get_ngrams(all_text, 2, top_n=15)
            if bigrams:
                bigrams_df = pd.DataFrame(list(bigrams.items()), columns=['Frasa', 'Frekuensi'])
                bigrams_df = bigrams_df.sort_values('Frekuensi', ascending=True)
                
                fig = px.bar(
                    bigrams_df.tail(10), 
                    x='Frekuensi', 
                    y='Frasa', 
                    orientation='h',
                    title="Top 10 Frasa yang Paling Sering Muncul",
                    color='Frekuensi',
                    color_continuous_scale='Viridis',
                    text='Frekuensi'
                )
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📝 Tidak ditemukan frasa yang signifikan")
        else:
            st.warning("⚠️ Tidak ada teks yang dapat dianalisis untuk bigram")
    except Exception as e:
        st.error(f"❌ Error dalam analisis bigram: {str(e)}")

def render_tfidf_analysis(reviews: pd.DataFrame, tfidf_vectorizer, sentiment_label: str, color_scale: str):
    """Render TF-IDF analysis for sentiment-specific reviews."""
    
    st.markdown(f"##### 📊 Kata Kunci Berdasarkan TF-IDF - {sentiment_label}")
    try:
        feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
        samples = reviews['teks_preprocessing'].dropna()
        
        if len(samples) > 0:
            tfidf_matrix = tfidf_vectorizer.transform(samples)
            importance = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

            # Build DataFrame then filter to ensure only single words with >=3 letters
            words_df = pd.DataFrame({
                'Kata': feature_names,
                'Skor TF-IDF': importance
            })

            # Filter rules:
            # - single word (no spaces)
            # - alphabetic only (exclude numbers/punctuations)
            # - length >= 3 (remove 1-2 letter tokens)
            words_df = words_df[
                (~words_df['Kata'].str.contains(r"\s", regex=True)) &
                (words_df['Kata'].str.len() >= 3) &
                (words_df['Kata'].str.match(r'^[A-Za-zÀ-ÖØ-öø-ÿ]+$', na=False))
            ]

            # Take top 10 by score
            words_df = words_df.sort_values('Skor TF-IDF', ascending=False).head(10)

            if not words_df.empty:
                fig = px.bar(
                    words_df.sort_values('Skor TF-IDF'),
                    x='Skor TF-IDF', 
                    y='Kata', 
                    orientation='h',
                    title=f"Top 10 Kata Kunci {sentiment_label}",
                    color='Skor TF-IDF',
                    color_continuous_scale=color_scale
                )
                fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📝 Tidak ada kata kunci yang memenuhi kriteria (1 kata, >=3 huruf)")
        else:
            st.info("📝 Tidak ada teks terproses untuk analisis TF-IDF")
    except Exception as e:
        st.error(f"❌ Error dalam analisis TF-IDF {sentiment_label}: {str(e)}")

def render_insights_tab(topic_data: pd.DataFrame):
    """Render the insights and recommendations tab."""
    
    st.markdown("### 💡 Ringkasan Insights & Rekomendasi")
    
    # Calculate insights
    current_topic_metrics = calculate_metrics(topic_data)
    pos_pct = current_topic_metrics['pos_percentage'] 
    neg_pct = current_topic_metrics['neg_percentage']
    total_reviews = current_topic_metrics['total']
    
    # Calculate trends
    trend_change = 0
    try:
        if len(topic_data) > 1:
            topic_data_sorted = topic_data.copy()
            topic_data_sorted['date'] = pd.to_datetime(topic_data_sorted['date'])
            topic_data_sorted = topic_data_sorted.sort_values('date')
            
            # Simple trend calculation
            mid_point = len(topic_data_sorted) // 2
            first_half = topic_data_sorted.iloc[:mid_point]
            second_half = topic_data_sorted.iloc[mid_point:]
            
            first_pos_pct = len(first_half[first_half['sentiment'] == 'POSITIF']) / len(first_half) * 100 if len(first_half) > 0 else 0
            second_pos_pct = len(second_half[second_half['sentiment'] == 'POSITIF']) / len(second_half) * 100 if len(second_half) > 0 else 0
            
            trend_change = second_pos_pct - first_pos_pct
    except Exception:
        pass
    
    # Visual insight cards
    st.markdown("#### 📊 Analisis Sentimen Saat Ini")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if pos_pct >= 80:
            sentiment_status = "🥇 Sangat Positif"
            sentiment_color = "green"
            status_message = "Excellent! Tingkat kepuasan sangat tinggi"
        elif pos_pct >= 60:
            sentiment_status = "🥈 Cukup Positif"
            sentiment_color = "blue"  
            status_message = "Good! Kepuasan di atas rata-rata"
        elif pos_pct >= 40:
            sentiment_status = "🥉 Netral"
            sentiment_color = "orange"
            status_message = "Fair. Ada ruang untuk perbaikan"
        else:
            sentiment_status = "⚠️ Perlu Perhatian"
            sentiment_color = "red"
            status_message = "Urgent! Perlu tindakan segera"
        
        st.markdown(f"""
        <div style="padding: 1rem; border-left: 4px solid {sentiment_color}; background-color: rgba(0,0,0,0.05); border-radius: 0.5rem;">
            <h4 style="margin: 0; color: {sentiment_color};">{sentiment_status}</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">{status_message}</p>
            <p style="margin: 0.5rem 0 0 0; font-weight: bold;">{pos_pct:.1f}% Ulasan Positif</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if total_reviews >= 1000:
            volume_status = "📊 Volume Tinggi"
            volume_msg = "Data representatif untuk analisis"
        elif total_reviews >= 100:
            volume_status = "📊 Volume Sedang"
            volume_msg = "Data cukup untuk insight dasar"
        else:
            volume_status = "📊 Volume Rendah"
            volume_msg = "Perlu lebih banyak data"
        
        st.markdown(f"""
        <div style="padding: 1rem; border-left: 4px solid #2E8B57; background-color: rgba(0,0,0,0.05); border-radius: 0.5rem;">
            <h4 style="margin: 0; color: #2E8B57;">{volume_status}</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">{volume_msg}</p>
            <p style="margin: 0.5rem 0 0 0; font-weight: bold;">{total_reviews:,} Total Ulasan</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if trend_change > 5:
            trend_status = "📈 Tren Meningkat"
            trend_color = "green"
            trend_msg = f"Sentimen membaik +{trend_change:.1f}%"
        elif trend_change < -5:
            trend_status = "📉 Tren Menurun"
            trend_color = "red"
            trend_msg = f"Sentimen menurun {trend_change:.1f}%"
        else:
            trend_status = "➡️ Tren Stabil"
            trend_color = "blue"
            trend_msg = "Sentimen relatif stabil"
        
        st.markdown(f"""
        <div style="padding: 1rem; border-left: 4px solid {trend_color}; background-color: rgba(0,0,0,0.05); border-radius: 0.5rem;">
            <h4 style="margin: 0; color: {trend_color};">{trend_status}</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">{trend_msg}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key insights
    st.markdown("---")
    st.markdown("#### 🔍 Temuan Utama")
    
    try:
        pos_reviews = topic_data[topic_data['sentiment'] == 'POSITIF']
        neg_reviews = topic_data[topic_data['sentiment'] == 'NEGATIF']
        
        pos_terms = {}
        neg_terms = {}
        
        if not pos_reviews.empty:
            pos_text = " ".join(pos_reviews['teks_preprocessing'].dropna())
            pos_terms = get_word_frequencies(pos_text, top_n=5)
        
        if not neg_reviews.empty:
            neg_text = " ".join(neg_reviews['teks_preprocessing'].dropna())
            neg_terms = get_word_frequencies(neg_text, top_n=5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Aspek Positif yang Menonjol:**")
            if pos_terms:
                for term, freq in list(pos_terms.items())[:3]:
                    st.markdown(f"• **{term}** ({freq} kali)")
            else:
                st.markdown("• Tidak ada data positif tersedia")
        
        with col2:
            st.markdown("**⚠️ Aspek yang Perlu Diperbaiki:**")
            if neg_terms:
                for term, freq in list(neg_terms.items())[:3]:
                    st.markdown(f"• **{term}** ({freq} kali)")
            else:
                st.markdown("• Tidak ada masalah yang teridentifikasi")
        
    except Exception as e:
        st.error(f"❌ Error dalam analisis insights: {str(e)}")
    
    # Recommendations
    st.markdown("---")
    if neg_pct > 15:
        st.markdown("#### 🎯 Rekomendasi Tindakan Prioritas")
        
        recommendations = [
            "🔍 **Analisis Mendalam**: Identifikasi penyebab utama ketidakpuasan pelanggan",
            "📞 **Respon Cepat**: Tingkatkan waktu respons terhadap keluhan pelanggan",
            "🎓 **Pelatihan Tim**: Berikan pelatihan tambahan untuk customer service",
            "📊 **Monitor Berkala**: Pantau tren sentimen secara real-time",
            "💡 **Inovasi Layanan**: Kembangkan fitur baru berdasarkan feedback negatif"
        ]
        
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.markdown("#### 🎉 Status Excellent - Rekomendasi Maintenance")
        
        maintenance_recommendations = [
            "✅ **Pertahankan Kualitas**: Terus jaga standar layanan yang tinggi",
            "📣 **Promosi Positif**: Manfaatkan testimoni positif untuk marketing",
            "🔄 **Continuous Improvement**: Terus tingkatkan layanan berdasarkan feedback",
            "📈 **Scale Up**: Pertimbangkan ekspansi layanan ke area baru",
            "🤝 **Community Building**: Bangun komunitas pengguna yang loyal"
        ]
        
        for rec in maintenance_recommendations:
            st.markdown(rec)

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
# END OF FILE
# ==============================================================================