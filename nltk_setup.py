"""
NLTK Setup for Streamlit Cloud Deployment
=========================================

This file ensures NLTK data is properly configured for Streamlit Cloud.
Import this at the very beginning of your app to avoid deployment issues.
"""

import nltk
import os
import sys
from pathlib import Path

def setup_nltk_for_streamlit_cloud():
    """
    Setup NLTK data for Streamlit Cloud deployment
    This function should be called before any NLTK operations
    """
    try:
        # Create nltk_data directory in home folder
        nltk_data_dir = Path.home() / 'nltk_data'
        nltk_data_dir.mkdir(exist_ok=True)
        
        # Add to NLTK path
        if str(nltk_data_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_data_dir))
        
        # Essential NLTK downloads for the app
        essential_data = ['punkt', 'stopwords']
        
        for data in essential_data:
            try:
                # Check if already exists
                if data == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                else:
                    nltk.data.find(f'corpora/{data}')
            except LookupError:
                # Download if not found
                try:
                    print(f"📥 Downloading {data}...")
                    nltk.download(data, download_dir=str(nltk_data_dir), quiet=True)
                    print(f"✅ {data} downloaded successfully")
                except Exception as e:
                    print(f"⚠️  Could not download {data}: {e}")
                    
    except Exception as e:
        print(f"⚠️  NLTK setup warning: {e}")
        # Continue without NLTK data if setup fails

# Run setup immediately when imported
if __name__ != "__main__":
    setup_nltk_for_streamlit_cloud()
