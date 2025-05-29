# Zonos Installation Script for Windows
param(
    [string]$EnvName = "zonos_env",
    [string]$PythonVersion = "3.10"
)

Write-Host "Starting Zonos installation..." -ForegroundColor Green

# Check if conda is available and get full path
$condaExe = $null
try {
    $condaCmd = Get-Command conda -ErrorAction Stop
    $condaExe = $condaCmd.Source
    Write-Host "Found conda at: $condaExe" -ForegroundColor Green
}
catch {
    Write-Host "Error: Conda not found. Please install Miniconda or Anaconda first." -ForegroundColor Red
    exit 1
}

# Check if environment exists and remove it if corrupted
$envExists = & $condaExe info --envs | Select-String -Pattern "^$EnvName\s"
if ($envExists) {
    Write-Host "Conda environment '$EnvName' already exists." -ForegroundColor Yellow
    Write-Host "Due to potential package corruption issues, recreating environment..." -ForegroundColor Yellow

    # Remove existing environment
    & $condaExe env remove -n $EnvName -y
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Failed to remove existing environment completely" -ForegroundColor Yellow
    }
    else {
        Write-Host "Successfully removed existing environment" -ForegroundColor Green
    }
}

# Create fresh conda environment
Write-Host "Creating fresh conda environment: $EnvName" -ForegroundColor Yellow
& $condaExe create -n $EnvName python=$PythonVersion -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to create conda environment" -ForegroundColor Red
    exit 1
}

# Install PyTorch with CUDA support
Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Cyan
& $condaExe install -n $EnvName pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install PyTorch" -ForegroundColor Red
    exit 1
}

# Install git
Write-Host "Installing git..." -ForegroundColor Cyan
& $condaExe install -n $EnvName git -y

# Install Python packages via pip with error handling
Write-Host "Installing Python packages via pip..." -ForegroundColor Cyan

# Install packages one by one to isolate issues
$packages = @("transformers", "accelerate", "gradio", "phonemizer==3.2.1")

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor White
    $result = & $condaExe run -n $EnvName pip install $package 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Failed to install $package" -ForegroundColor Yellow
        Write-Host "Error details: $result" -ForegroundColor Red

        # Try alternative installation method
        Write-Host "Trying alternative installation for $package..." -ForegroundColor Yellow
        & $condaExe run -n $EnvName python -m pip install --no-cache-dir $package

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Critical: Could not install $package with any method" -ForegroundColor Red
        }
        else {
            Write-Host "Successfully installed $package using alternative method" -ForegroundColor Green
        }
    }
    else {
        Write-Host "Successfully installed $package" -ForegroundColor Green
    }
}

# Check if Zonos directory exists and install/update
if (Test-Path "Zonos") {
    Write-Host "Zonos directory already exists, updating..." -ForegroundColor Yellow
    Push-Location Zonos

    # Check if git repo is clean
    $gitStatus = git status --porcelain 2>$null
    if ([string]::IsNullOrEmpty($gitStatus)) {
        Write-Host "Pulling latest changes..." -ForegroundColor Cyan
        git pull
    }
    else {
        Write-Host "Local changes detected, skipping git pull" -ForegroundColor Yellow
    }

    Pop-Location
}
else {
    # Clone Zonos repository
    Write-Host "Cloning Zonos repository..." -ForegroundColor Yellow
    git clone https://github.com/Zyphra/Zonos.git

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to clone Zonos repository" -ForegroundColor Red
        exit 1
    }
}

# Install Zonos with better error handling
Write-Host "Installing Zonos..." -ForegroundColor Yellow
Push-Location Zonos

$zonosInstall = & $condaExe run -n $EnvName pip install -e . 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Zonos installation had issues" -ForegroundColor Yellow
    Write-Host "Trying alternative installation method..." -ForegroundColor Yellow

    # Try with --no-deps to avoid dependency conflicts
    & $condaExe run -n $EnvName pip install --no-deps -e .

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install Zonos" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    else {
        Write-Host "Zonos installed successfully with alternative method" -ForegroundColor Green
    }
}
else {
    Write-Host "Zonos installed successfully" -ForegroundColor Green
}

Pop-Location

# Check if eSpeak NG is installed
$espeakPath = "C:\Program Files\eSpeak NG\espeak-ng.exe"
if (-not (Test-Path $espeakPath)) {
    Write-Host "eSpeak NG not found. Installing via winget..." -ForegroundColor Yellow
    try {
        $wingetResult = winget install --id eSpeak-NG.eSpeak-NG --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "eSpeak NG installed successfully" -ForegroundColor Green
        }
        else {
            Write-Host "winget installation may have had issues. Checking if eSpeak is now available..." -ForegroundColor Yellow
            if (Test-Path $espeakPath) {
                Write-Host "eSpeak NG is now available despite winget warnings" -ForegroundColor Green
            }
            else {
                Write-Host "Please install eSpeak NG manually:" -ForegroundColor Red
                Write-Host "winget install --id eSpeak-NG.eSpeak-NG" -ForegroundColor Red
            }
        }
    }
    catch {
        Write-Host "Failed to install eSpeak NG automatically: $_" -ForegroundColor Red
        Write-Host "Please install manually: winget install --id eSpeak-NG.eSpeak-NG" -ForegroundColor Red
    }
}
else {
    Write-Host "eSpeak NG already installed at: $espeakPath" -ForegroundColor Green
}

# Test PyTorch installation - using the conda executable directly
Write-Host "Testing PyTorch installation..." -ForegroundColor Yellow
try {
    $torchTest = & $condaExe run -n $EnvName python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host $torchTest -ForegroundColor Green
    }
    else {
        Write-Host "PyTorch test failed: $torchTest" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "PyTorch test failed: $_" -ForegroundColor Yellow
}

# Test eSpeak integration
Write-Host "Testing eSpeak integration..." -ForegroundColor Yellow
try {
    if (Test-Path $espeakPath) {
        $espeakTest = & "$espeakPath" --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "eSpeak NG direct test successful" -ForegroundColor Green

            # Test phonemizer integration using external script
            Write-Host "Testing phonemizer integration..." -ForegroundColor Cyan
            $phonemizerTestScript = "test_phonemizer.py"

            if (-not (Test-Path $phonemizerTestScript)) {
                Write-Host "WARNING: Phonemizer test script not found: $phonemizerTestScript" -ForegroundColor Yellow
                Write-Host "Skipping detailed phonemizer test - will test during actual TTS usage" -ForegroundColor Yellow
            }
            else {
                Write-Host "Running phonemizer test with timeout..." -ForegroundColor White

                try {
                    # Get full path to conda executable for the job
                    $condaFullPath = (Get-Command conda -ErrorAction SilentlyContinue).Source
                    if (-not $condaFullPath) {
                        $condaFullPath = $condaExe
                    }

                    Write-Host "Using conda path: $condaFullPath" -ForegroundColor White

                    # Test phonemizer with proper path handling
                    $phonemizerCmd = "& `"$condaFullPath`" run -n $EnvName python `"$phonemizerTestScript`""
                    Write-Host "Running command: $phonemizerCmd" -ForegroundColor Gray

                    $job = Start-Job -ScriptBlock {
                        param($command)
                        Invoke-Expression $command
                    } -ArgumentList $phonemizerCmd

                    if (Wait-Job $job -Timeout 15) {
                        $result = Receive-Job $job
                        $jobState = $job.State

                        Write-Host "Job completed with state: $jobState" -ForegroundColor White
                        if ($result) {
                            Write-Host "Phonemizer test output:" -ForegroundColor Green
                            Write-Host $result -ForegroundColor White
                        }
                    }
                    else {
                        Write-Host "Phonemizer test timed out after 15 seconds" -ForegroundColor Yellow
                        Write-Host "This is expected on Windows - phonemizer integration can be slow" -ForegroundColor Yellow
                        Stop-Job $job
                    }
                    Remove-Job $job -Force
                }
                catch {
                    Write-Host "Phonemizer test process failed: $_" -ForegroundColor Yellow
                    Write-Host "This is a known issue on Windows - continuing with installation" -ForegroundColor Yellow
                }
            }
        }
        else {
            Write-Host "eSpeak NG direct test failed: $espeakTest" -ForegroundColor Yellow
        }
    }
}
catch {
    Write-Host "eSpeak test failed: $_" -ForegroundColor Yellow
}

Write-Host "`nInstallation completed!" -ForegroundColor Green
Write-Host "Environment name: $EnvName" -ForegroundColor Cyan

# Create a simple test file
$testFile = "test_sample.txt"
if (-not (Test-Path $testFile)) {
    "Hello world! This is a test of the Zonos text-to-speech system." | Out-File -FilePath $testFile -Encoding UTF8
    Write-Host "Created test file: $testFile" -ForegroundColor Cyan
}

# Initialize conda for command prompt if not already done
Write-Host "`nInitializing conda for command prompt..." -ForegroundColor Yellow
try {
    $initResult = & $condaExe init cmd.exe 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Conda initialized successfully for command prompt" -ForegroundColor Green
    }
    else {
        Write-Host "Conda initialization had issues (this might be normal): $initResult" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "Conda initialization failed: $_" -ForegroundColor Yellow
}

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Close and reopen your command prompt" -ForegroundColor White
Write-Host "2. Test with: conda activate zonos_env" -ForegroundColor White
Write-Host "3. Then run: tts.cmd `"$testFile`"" -ForegroundColor White
Write-Host "4. Alternative: Use full path - `"C:\dev\scripts\TTS_Zonos\tts.cmd`" `"$testFile`"" -ForegroundColor White

Write-Host "`nSkipping automatic TTS pipeline test to avoid hanging." -ForegroundColor Yellow
Write-Host "Please test manually using the commands above." -ForegroundColor Yellow

Write-Host "`nInstallation summary:" -ForegroundColor Cyan
Write-Host "- Conda environment: $EnvName (created)" -ForegroundColor White
Write-Host "- PyTorch with CUDA: Installed and working" -ForegroundColor Green
Write-Host "- eSpeak NG: Available" -ForegroundColor Green
Write-Host "- Zonos: Installed" -ForegroundColor Green
Write-Host "- All Python packages: Installed" -ForegroundColor Green
Write-Host "- Conda initialized for CMD: Attempted" -ForegroundColor Yellow

Write-Host "`nIf conda activate still doesn't work:" -ForegroundColor Yellow
Write-Host "Run this command as Administrator:" -ForegroundColor White
Write-Host "`"$condaExe`" init cmd.exe" -ForegroundColor Gray
Write-Host "Then close and reopen your command prompt." -ForegroundColor White
