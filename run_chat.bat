@echo off
cd /d %~dp0

if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate

pip freeze > tmp_requirements.txt

for /f %%i in (requirements.txt) do (
    findstr /c:"%%i" tmp_requirements.txt >nul
    if errorlevel 1 (
        pip install %%i
    )
)

del tmp_requirements.txt

python -m main --mode chat

exit