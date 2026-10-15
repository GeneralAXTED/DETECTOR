@echo off
title SMART CLASSROOM DETECTOR
cd /d "%~dp0"

echo ==============================================================
echo       SMART CLASSROOM DETECTOR - AI PROCTORING SYSTEM
echo ==============================================================
echo.
echo Tizim ishga tushmoqda...
echo Brauzerda ochish: http://localhost:8000
echo.

.\.venv\Scripts\python.exe main.py
pause
