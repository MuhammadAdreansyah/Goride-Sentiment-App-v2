# 📁 SentimenGo Project Structure

## 🏠 Root Directory (Clean & Organized)

```
SentimenGo/
├── streamlit_app.py          # 🚀 Main application entry point
├── README.md                 # 📖 Project documentation
├── requirements.txt          # 📦 Python dependencies
├── .gitignore               # 🔒 Git ignore rules
├── secrets.toml.example     # 🔑 Configuration template
│
├── 📁 config/               # ⚙️ Configuration files
│   ├── client_secret_*.json
│   └── sentimentapp-*.json
│
├── 📁 data/                 # 📊 Dataset and preprocessing files
│   ├── ulasan_goride.csv
│   ├── ulasan_goride_preprocessed.csv
│   ├── kamus_slang_formal.txt
│   └── stopwordsID.txt
│
├── 📁 models/               # 🤖 Machine learning models
│   ├── svm_model_predict.pkl
│   ├── tfidf_vectorizer_predict.pkl
│   └── backup_*/            # Model backups
│
├── 📁 ui/                   # 🖥️ User interface modules
│   ├── auth/               # Authentication
│   ├── tools/              # Core application tools
│   └── utils.py            # Utility functions
│
├── 📁 notebooks/            # 📓 Jupyter notebooks for research
│   ├── 1CrawlingData.ipynb
│   ├── 2PreprocessingData.ipynb
│   └── 3SentimentAnalysis.ipynb
│
├── 📁 log/                  # 📝 Application logs
│   ├── app.log
│   └── analisis_data.log
│
├── 📁 docs/                 # 📚 ORGANIZED DOCUMENTATION
│   ├── README.md
│   ├── deployment/          # Deployment guides
│   ├── debugging/           # Troubleshooting guides  
│   └── testing/             # Testing documentation
│
├── 📁 scripts/              # 🛠️ ORGANIZED SCRIPTS
│   ├── README.md
│   ├── testing/             # Test scripts
│   ├── debugging/           # Debug scripts
│   └── maintenance/         # Maintenance scripts
│
└── 📁 .streamlit/           # Streamlit configuration
    └── config.toml
```

## ✅ What's in Root Now (Clean!)

### 🎯 Core Application Files
- `streamlit_app.py` - Main application entry point
- `README.md` - Project documentation  
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore configuration
- `secrets.toml.example` - Configuration template

### 📁 Essential Folders
- `config/` - Application configuration
- `data/` - Datasets and preprocessing files
- `models/` - ML models and backups
- `ui/` - User interface components
- `notebooks/` - Research and development notebooks
- `log/` - Application logs

### 🗂️ Organized Support Folders
- `docs/` - All documentation (deployment, debugging, testing)
- `scripts/` - All scripts (testing, debugging, maintenance)

## 🚀 Benefits of This Structure

### ✅ Clean Root Directory
- Only essential files in root
- Easy to navigate and understand
- Professional project structure

### ✅ Organized Documentation
- All docs categorized by purpose
- Easy to find specific information
- Better maintenance and updates

### ✅ Structured Scripts
- Testing scripts separate from debugging
- Maintenance scripts organized
- Clear purpose for each script

### ✅ Better Git Management
- Cleaner commit history
- Easier to track changes
- Better collaboration

## 🔄 Migration Summary

### Moved to `docs/`:
- `CRITICAL_FIX_SUMMARY.md` → `docs/deployment/`
- `CLOUD_DEPLOYMENT_FIXES.md` → `docs/deployment/`
- `DEPLOYMENT_READY.md` → `docs/deployment/`
- `STREAMLIT_CLOUD_TROUBLESHOOTING.md` → `docs/debugging/`
- `blackbox_testing_sentimengo.md` → `docs/testing/`
- `current_requirements.txt` → `docs/deployment/`

### Moved to `scripts/`:
- `test_cloud_deployment.py` → `scripts/testing/`
- `debug_streamlit.py` → `scripts/debugging/`
- `fix_streamlit_imports.py` → `scripts/debugging/`
- `fix_model_compatibility.py` → `scripts/maintenance/`

### Removed:
- `__pycache__/` - Python cache files (not needed in repo)

---
**Status**: ✅ **PROJECT STRUCTURE REORGANIZED**  
**Root Directory**: Clean and Professional ✨  
**Documentation**: Organized by Category 📚  
**Scripts**: Structured by Purpose 🛠️
