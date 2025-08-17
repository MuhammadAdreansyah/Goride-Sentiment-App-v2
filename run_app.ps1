# GoRide Sentiment Analysis - PowerShell Launcher
Write-Host "🚀 Starting GoRide Sentiment Analysis Application..." -ForegroundColor Green
Write-Host ""
Set-Location "D:\SentimenGo_App2"
& "D:\SentimenGo_App2\.venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port 8501
Read-Host "Press Enter to exit"
