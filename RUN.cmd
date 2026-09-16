@echo off
python -B "%~dp0run.py" %*
if errorlevel 1 pause
