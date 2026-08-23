@echo off
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_gui.ps1" %*

if errorlevel 1 (
    echo.
    echo GUI failed to start. Exit code: %errorlevel%
    pause
)

endlocal
