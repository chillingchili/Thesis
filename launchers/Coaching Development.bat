@echo off
cd /d "%~dp0.."
echo Coaching development: training clips only; Coach C evaluation stays frozen.
python scripts\build_coaching_development.py %*
if errorlevel 1 pause
