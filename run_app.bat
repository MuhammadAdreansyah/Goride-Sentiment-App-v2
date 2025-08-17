@echo off
echo 🚀 Starting GoRide Sentiment Analysis Application...
echo.
cd /d "D:\SentimenGo_App2"
D:\SentimenGo_App2\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501
pause
