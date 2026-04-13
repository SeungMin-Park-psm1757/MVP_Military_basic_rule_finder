@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if exist "%PYTHON_EXE%" goto run_app

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_EXE=py -3"
    goto run_app
)

where python >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_EXE=python"
    goto run_app
)

echo [ERROR] Python executable was not found.
echo         Run setup_local_webapp.bat first, or install Python 3.10+.
pause
exit /b 1

:run_app
echo [INFO] Starting local LM Studio webapp...
echo [INFO] URL: http://127.0.0.1:8502
echo [INFO] If LM Studio is not ready, the app will fall back to evidence-only mode.
%PYTHON_EXE% -m streamlit run local\streamlit_app.py --server.headless true --server.address 127.0.0.1 --server.port 8502
exit /b %errorlevel%
