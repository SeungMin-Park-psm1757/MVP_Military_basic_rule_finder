@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
)

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [INFO] Installing dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto fail
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto fail

if not exist "local\.env" (
    if exist "local\.env.example" (
        copy /Y "local\.env.example" "local\.env" >nul
        echo [INFO] Created local\.env from template.
    )
)

echo [INFO] Building sample corpus...
"%VENV_PYTHON%" scripts\build_sample_corpus.py
if errorlevel 1 goto fail

echo [INFO] Ingesting sample corpus into Chroma...
"%VENV_PYTHON%" scripts\ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl
if errorlevel 1 goto fail

echo [INFO] Setup complete.
echo [INFO] Before asking questions, make sure LM Studio is running and exactly one LLM is loaded.
echo [INFO] If you want to force a specific model, set LM_STUDIO_MODEL in local\.env.
call run_local_webapp.bat
exit /b %errorlevel%

:fail
echo [ERROR] Setup failed.
pause
exit /b 1
