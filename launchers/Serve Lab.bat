@echo off
cd /d "%~dp0.."
python -m desktop.app --open
if errorlevel 1 pause
