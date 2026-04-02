@echo off
setlocal
cd /d "%~dp0"

if not exist main.py (
  echo [ERROR] main.py not found in %cd%
  exit /b 1
)

python main.py %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] main.py exited with code %EXIT_CODE%
)

exit /b %EXIT_CODE%
