@echo off
cd /d "%~dp0.."
echo The Serve Study: http://127.0.0.1:7863
echo Press Ctrl+C to stop the local website.
python -m http.server 7863 --bind 127.0.0.1 --directory research-site
