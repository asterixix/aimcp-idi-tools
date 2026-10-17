@echo off
rem Strona do kodowania IDI (weryfikacja propozycji LLM i wlasne kody). Instrukcja: kodowanie\README.md
chcp 65001 >nul
python "%~dp0scripts\kodowanie_llm.py" serwer %*
pause
