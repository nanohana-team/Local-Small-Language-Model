@echo off
setlocal

if "%RUN_ID%"=="" (
  for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_ID=%%i
)

echo [INFO] RUN_ID=%RUN_ID%

python src\loop\run_conversation_loop.py ^
  --prompts-file data\prompts\input_prompts.jsonl ^
  --teachers-config config\teachers.json ^
  --learning-config config\learning.json ^
  --loop-config config\conversation_loop.json ^
  --output-file data\conversations\%RUN_ID%\raw_sessions.jsonl

if errorlevel 1 exit /b 1

python src\eval\evaluate_conversations_with_gemini.py ^
  --input-file data\conversations\%RUN_ID%\raw_sessions.jsonl ^
  --output-file data\conversations\%RUN_ID%\scored_sessions.jsonl ^
  --learning-name learning_gemma ^
  --schema-file config\gemini_eval_schema.json

if errorlevel 1 exit /b 1

python src\data\build_sft_dataset_from_sessions.py ^
  --input-file data\conversations\%RUN_ID%\scored_sessions.jsonl ^
  --train-output-file data\conversations\%RUN_ID%\sft_train.jsonl ^
  --eval-output-file data\conversations\%RUN_ID%\sft_eval.jsonl

if errorlevel 1 exit /b 1

python src\data\build_dpo_dataset_from_sessions.py ^
  --input-file data\conversations\%RUN_ID%\scored_sessions.jsonl ^
  --output-file data\conversations\%RUN_ID%\dpo_train.jsonl ^
  --learning-name learning_gemma

if errorlevel 1 exit /b 1

echo [INFO] Conversation loop completed successfully.
endlocal
