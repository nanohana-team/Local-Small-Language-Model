@echo off
setlocal

if "%RUN_ID%"=="" (
  echo [ERROR] RUN_ID is not set.
  echo Example:
  echo   set RUN_ID=20260403_120000
  echo   run_train_cycle.bat
  exit /b 1
)

python src\train\train_gemma3_270m_sft.py ^
  --model-name outputs\latest_learning_model ^
  --train-file data\conversations\%RUN_ID%\sft_train.jsonl ^
  --eval-file data\conversations\%RUN_ID%\sft_eval.jsonl ^
  --output-dir outputs\learning_lora ^
  --merged-output-dir outputs\learning_merged ^
  --max-length 512 ^
  --per-device-train-batch-size 8 ^
  --per-device-eval-batch-size 8 ^
  --gradient-accumulation-steps 4 ^
  --num-train-epochs 2 ^
  --learning-rate 2e-4 ^
  --lora-r 16 ^
  --lora-alpha 32 ^
  --lora-dropout 0.05 ^
  --merge-after-training

if errorlevel 1 exit /b 1

echo [INFO] Updating outputs\latest_learning_model ...
if exist outputs\latest_learning_model rmdir /s /q outputs\latest_learning_model
xcopy outputs\learning_merged outputs\latest_learning_model /e /i /y > nul

echo [INFO] Training cycle completed successfully.
endlocal
