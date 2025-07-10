@echo off
REM ============================================================================
REM Infinite Loop Evaluation Script
REM ============================================================================
REM This batch file runs the evaluation script indefinitely using goto
REM It will restart the evaluation automatically if it crashes or completes
REM ============================================================================

echo.
echo ============================================================================
echo Starting Infinite Loop Evaluation
echo ============================================================================
echo.
echo This script will run the evaluation indefinitely.
echo Press Ctrl+C to stop the loop.
echo.
echo Current time: %date% %time%
echo ============================================================================
echo.

:LOOP_START
echo.
echo ============================================================================
echo Starting evaluation run at %date% %time%
echo ============================================================================
echo.

REM Run the PowerShell script with the evaluation script and input JSON
powershell.exe -ExecutionPolicy Bypass -File ".\scripts\run_evaluation_simple_fixed.ps1" ".\project\evaluation_DeCo_Diff2.py" ".\input_json\eval_input.json"

REM Check if the PowerShell script exited with an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================================
    echo Evaluation failed with error code: %ERRORLEVEL%
    echo Restarting in 10 seconds...
    echo ============================================================================
    echo.
    timeout /t 10 /nobreak >nul
) else (
    echo.
    echo ============================================================================
    echo Evaluation completed successfully
    echo Restarting in 5 seconds...
    echo ============================================================================
    echo.
    timeout /t 5 /nobreak >nul
)

echo.
echo ============================================================================
echo Restarting evaluation loop...
echo ============================================================================
echo.

REM Jump back to the start of the loop
goto LOOP_START 