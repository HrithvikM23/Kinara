@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\build_exe.ps1" %*
exit /b %ERRORLEVEL%
