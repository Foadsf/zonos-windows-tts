@echo off
setlocal enabledelayedexpansion

:: Zonos TTS Command Line Wrapper
:: Usage: tts.cmd <text_file> [options]

if "%~1"=="" (
    echo Usage: tts.cmd ^<text_file^> [options]
    echo.
    echo Options:
    echo   -o ^<output_file^>     Output audio file path
    echo   -s ^<speaker_file^>    Speaker reference audio file
    echo   -l ^<language^>        Language code ^(default: en-us^)
    echo.
    echo Examples:
    echo   tts.cmd "my_text.txt"
    echo   tts.cmd "my_text.txt" -o "output.wav"
    echo   tts.cmd "my_text.txt" -s "speaker.wav" -l "en-us"
    exit /b 1
)

:: Find conda installation directory
set "CONDA_ROOT="
set "CONDA_PATHS=C:\Users\%USERNAME%\AppData\Local\miniconda3"
set "CONDA_PATHS=!CONDA_PATHS!;C:\Users\%USERNAME%\miniconda3"
set "CONDA_PATHS=!CONDA_PATHS!;C:\ProgramData\miniconda3"
set "CONDA_PATHS=!CONDA_PATHS!;C:\miniconda3"
set "CONDA_PATHS=!CONDA_PATHS!;C:\Anaconda3"
set "CONDA_PATHS=!CONDA_PATHS!;C:\Users\%USERNAME%\Anaconda3"

for %%p in (!CONDA_PATHS!) do (
    if exist "%%p\Scripts\conda.exe" (
        set "CONDA_ROOT=%%p"
        goto :found_conda
    )
)

echo Error: Could not find conda installation.
exit /b 1

:found_conda
echo Using conda at: !CONDA_ROOT!

:: Check if conda environment exists
"!CONDA_ROOT!\Scripts\conda.exe" info --envs | findstr /C:"zonos_env" >nul 2>&1
if errorlevel 1 (
    echo Error: Conda environment 'zonos_env' not found.
    echo Please run install_zonos.ps1 first to set up the environment.
    exit /b 1
)

:: Get the script directory to find text_to_speech.py
set "SCRIPT_DIR=%~dp0"
set "PYTHON_SCRIPT=%SCRIPT_DIR%text_to_speech.py"

if not exist "!PYTHON_SCRIPT!" (
    echo Error: Python script not found at: !PYTHON_SCRIPT!
    exit /b 1
)

:: Check for eSpeak installation and add to PATH
set "ESPEAK_FOUND=0"
set "ESPEAK_PATH="

:: Check common eSpeak installation locations
if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" (
    set "ESPEAK_PATH=C:\Program Files\eSpeak NG"
    set "ESPEAK_FOUND=1"
    echo Found eSpeak NG at: !ESPEAK_PATH!
    goto :espeak_found
)

if exist "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" (
    set "ESPEAK_PATH=C:\Program Files (x86)\eSpeak NG"
    set "ESPEAK_FOUND=1"
    echo Found eSpeak NG at: !ESPEAK_PATH!
    goto :espeak_found
)

if exist "%USERPROFILE%\scoop\apps\espeak-ng\current\espeak-ng.exe" (
    set "ESPEAK_PATH=%USERPROFILE%\scoop\apps\espeak-ng\current"
    set "ESPEAK_FOUND=1"
    echo Found eSpeak NG at: !ESPEAK_PATH!
    goto :espeak_found
)

:: Check if espeak is already in PATH
where espeak-ng >nul 2>&1
if %errorlevel% equ 0 (
    set "ESPEAK_FOUND=1"
    echo eSpeak NG found in system PATH
    goto :espeak_found
)

where espeak >nul 2>&1
if %errorlevel% equ 0 (
    set "ESPEAK_FOUND=1"
    echo eSpeak found in system PATH
    goto :espeak_found
)

echo Error: eSpeak not found. eSpeak NG should be installed but not found in expected locations.
echo.
echo Expected locations checked:
echo   - C:\Program Files\eSpeak NG\espeak-ng.exe
echo   - C:\Program Files (x86)\eSpeak NG\espeak-ng.exe
echo   - %USERPROFILE%\scoop\apps\espeak-ng\current\espeak-ng.exe
echo.
echo Please ensure eSpeak NG is properly installed and accessible.
exit /b 1

:espeak_found

:: Build command with proper quoting
set "cmd_args="
set "current_arg="

:parse_args
if "%~1"=="" goto execute_cmd

set "current_arg=%~1"

:: Handle paths with spaces by ensuring quotes
if "!current_arg!" NEQ "!current_arg: =!" (
    set "cmd_args=!cmd_args! "!current_arg!""
) else (
    set "cmd_args=!cmd_args! !current_arg!"
)

shift
goto parse_args

:execute_cmd
echo Activating conda environment: zonos_env
echo Running TTS conversion...
echo.

:: Find the environment's Python executable
set "ENV_PYTHON=!CONDA_ROOT!\envs\zonos_env\python.exe"

if not exist "!ENV_PYTHON!" (
    echo Error: Python executable not found at: !ENV_PYTHON!
    exit /b 1
)

:: Set environment variables for conda environment and disable symlinks warning
set "CONDA_DEFAULT_ENV=zonos_env"
set "CONDA_PREFIX=!CONDA_ROOT!\envs\zonos_env"

:: Build PATH with eSpeak included
if defined ESPEAK_PATH (
    set "PATH=!ESPEAK_PATH!;!CONDA_ROOT!\envs\zonos_env;!CONDA_ROOT!\envs\zonos_env\Scripts;!CONDA_ROOT!\envs\zonos_env\Library\bin;!PATH!"
) else (
    set "PATH=!CONDA_ROOT!\envs\zonos_env;!CONDA_ROOT!\envs\zonos_env\Scripts;!CONDA_ROOT!\envs\zonos_env\Library\bin;!PATH!"
)

set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

:: Test eSpeak accessibility
if defined ESPEAK_PATH (
    echo Testing eSpeak accessibility...
    "!ESPEAK_PATH!\espeak-ng.exe" --version >nul 2>&1
    if errorlevel 1 (
        echo Warning: eSpeak found but not accessible - continuing anyway
    ) else (
        echo eSpeak NG is accessible and working
    )
)

:: Run Python directly from the environment
"!ENV_PYTHON!" "!PYTHON_SCRIPT!" !cmd_args!

if errorlevel 1 (
    echo.
    echo Error: TTS conversion failed.
    echo.
    echo Troubleshooting steps:
    echo 1. Ensure eSpeak NG is installed and accessible
    echo 2. Verify the input file exists and is readable
    echo 3. Check that the zonos_env environment has all required packages
    echo 4. Try running eSpeak directly: "!ESPEAK_PATH!\espeak-ng.exe" --version
    exit /b 1
)

echo.
echo TTS conversion completed successfully!
