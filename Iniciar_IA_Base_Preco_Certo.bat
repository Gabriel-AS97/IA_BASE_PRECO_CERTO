@echo off
title Preco Certo IA - Inicializador
cd /d "%~dp0"

echo ========================================================
echo   Iniciando o Sistema Preco Certo IA...
echo ========================================================
echo.

echo [1/2] Verificando bibliotecas necessarias...
py -m pip install -q streamlit pandas scikit-learn joblib matplotlib seaborn 2>nul
if %errorlevel% neq 0 (
    python -m pip install -q streamlit pandas scikit-learn joblib matplotlib seaborn
)

echo [2/2] Abrindo a interface no navegador...
echo (Mantenha esta janela preta aberta enquanto usar o sistema)
echo.

py -m streamlit run interface_preco_certo.py
if %errorlevel% neq 0 (
    python -m streamlit run interface_preco_certo.py
)

pause