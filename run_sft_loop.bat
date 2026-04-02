@echo off
setlocal

cd /d %~dp0

if "%OPENAI_API_KEY_LOCAL%"=="" (
  set OPENAI_API_KEY_LOCAL=dummy
)

set PYTHON_EXE=python

%PYTHON_EXE% src\train\run_sft_loop.py ^
  --project-root . ^
  --python-exe %PYTHON_EXE% ^
  --iterations 3 ^
  --base-model google/gemma-3-270m-it ^
  --prompts-file data\prompts\input_prompts.jsonl ^
  --teacher-config config\teachers.json ^
  --repeats-per-teacher 2 ^
  --sleep-sec 0.2 ^
  --train-epochs 2 ^
  --train-batch-size 8 ^
  --eval-batch-size 8 ^
  --grad-accum 4 ^
  --learning-rate 2e-4 ^
  --max-length 512 ^
  --lora-r 16 ^
  --lora-alpha 32 ^
  --lora-dropout 0.05 ^
  --merge-after-training

if errorlevel 1 (
  echo.
  echo [ERROR] run_sft_loop failed.
  exit /b 1
)

echo.
echo [INFO] run_sft_loop completed successfully.
exit /b 0