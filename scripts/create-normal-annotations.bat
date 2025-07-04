@echo off
REM Batch script to create annotation JSON files for normal images
REM Usage: create-normal-annotations.bat <input_directory> <output_directory>

if "%~1"=="" (
    echo Usage: create-normal-annotations.bat ^<input_directory^> ^<output_directory^>
    echo.
    echo Example: create-normal-annotations.bat "C:\path\to\images" "C:\path\to\annotations"
    pause
    exit /b 1
)

if "%~2"=="" (
    echo Error: Output directory not specified
    echo Usage: create-normal-annotations.bat ^<input_directory^> ^<output_directory^>
    pause
    exit /b 1
)

if not exist "%~1" (
    echo Error: Input directory "%~1" does not exist
    pause
    exit /b 1
)

echo Creating normal annotations for images in: %~1
echo Output directory: %~2
echo.

cd /d "%~dp0.."
py -3.11 project/create_normal_annotations.py --input-dir "%~1" --output-dir "%~2"

echo.
echo Done! Press any key to exit.
pause 