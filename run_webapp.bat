@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run streamlit_app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
