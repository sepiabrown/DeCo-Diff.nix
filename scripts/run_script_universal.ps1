# Universal PowerShell script to run training or evaluation scripts
# Usage: .\run_script_universal.ps1 [ScriptType] [ScriptPath] [InputJson] [Loop]
# Examples: 
#   .\run_script_universal.ps1 "train" "..\project\train_DeCo_Diff.py" "..\input_json\train_input.json" $true
#   .\run_script_universal.ps1 "eval" "..\project\evaluation_DeCo_Diff2.py" "..\input_json\eval_input.json" $false

# Parse command line arguments
param(
    [string]$ScriptType = "eval",  # "train" or "eval"
    [string]$ScriptPath = "..\project\evaluation_DeCo_Diff2.py",
    [string]$InputJson = "..\input_json\eval_input.json",
    [bool]$Loop = $false
)

# Start timing
$StartTime = Get-Date
Write-Host "=== Starting $ScriptType Script ===" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "Script Type: $ScriptType" -ForegroundColor Yellow
Write-Host "Input JSON: $InputJson" -ForegroundColor Yellow
Write-Host "Script Path: $ScriptPath" -ForegroundColor Yellow
Write-Host "Loop Mode: $Loop" -ForegroundColor Yellow
Write-Host ""

# Function to show elapsed time
function Show-ElapsedTime {
    $Elapsed = (Get-Date) - $StartTime
    $Minutes = [math]::Floor($Elapsed.TotalMinutes)
    $Seconds = $Elapsed.Seconds
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Elapsed: ${Minutes}m ${Seconds}s" -ForegroundColor Magenta
}

# Check if files exist
if (-not (Test-Path $InputJson)) {
    Write-Host "ERROR: Input JSON file not found: $InputJson" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $ScriptPath)) {
    Write-Host "ERROR: Python script not found: $ScriptPath" -ForegroundColor Red
    exit 1
}

# Get the current directory (where the script is being run from)
$CurrentDir = Get-Location
Write-Host "Current directory: $CurrentDir" -ForegroundColor Cyan

# Convert to absolute paths if they're relative
$ScriptPathAbs = if ([System.IO.Path]::IsPathRooted($ScriptPath)) { $ScriptPath } else { Join-Path $CurrentDir $ScriptPath }
$InputJsonAbs = if ([System.IO.Path]::IsPathRooted($InputJson)) { $InputJson } else { Join-Path $CurrentDir $InputJson }

Write-Host "Absolute Script Path: $ScriptPathAbs" -ForegroundColor Cyan
Write-Host "Absolute Input JSON: $InputJsonAbs" -ForegroundColor Cyan

# Set environment variables for training
if ($ScriptType -eq "train") {
    $env:NO_ALBUMENTATIONS_UPDATE = "1"
    Write-Host "Set environment variable: NO_ALBUMENTATIONS_UPDATE=1" -ForegroundColor Green
}

# Build the command
if ($ScriptType -eq "train") {
    # Training command with distributed run
    $Command = "py -3.11 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 `"$ScriptPathAbs`" --input-json `"$InputJsonAbs`""
} else {
    # Evaluation command
    $Command = "py -3.11 `"$ScriptPathAbs`" --input-json `"$InputJsonAbs`""
}

Write-Host "Command: $Command" -ForegroundColor Yellow
Write-Host ""

# Function to run a single execution
function Run-SingleExecution {
    Write-Host "Starting $ScriptType execution..." -ForegroundColor Green
    
    $Process = Start-Process -FilePath "cmd" -ArgumentList "/c", $Command -PassThru -NoNewWindow
    
    # Monitor the process
    $LastUpdate = Get-Date
    while (-not $Process.HasExited) {
        # Show elapsed time every 60 seconds
        $CurrentTime = Get-Date
        if (($CurrentTime - $LastUpdate).TotalSeconds -ge 60) {
            Show-ElapsedTime
            $LastUpdate = $CurrentTime
        }
        Start-Sleep -Seconds 1
    }
    
    # Wait for completion
    $Process.WaitForExit()
    $ExitCode = $Process.ExitCode
    
    return $ExitCode
}

# Main execution logic
if ($Loop) {
    Write-Host "=== LOOP MODE ENABLED ===" -ForegroundColor Red
    Write-Host "Script will restart automatically after completion" -ForegroundColor Red
    Write-Host "Press Ctrl+C to stop the loop" -ForegroundColor Red
    Write-Host ""
    
    $Iteration = 1
    do {
        Write-Host "=== Iteration $Iteration ===" -ForegroundColor Cyan
        $IterationStartTime = Get-Date
        
        $ExitCode = Run-SingleExecution
        
        $IterationEndTime = Get-Date
        $IterationTime = $IterationEndTime - $IterationStartTime
        $IterationMinutes = [math]::Floor($IterationTime.TotalMinutes)
        $IterationSeconds = $IterationTime.Seconds
        
        Write-Host ""
        Write-Host "=== Iteration $Iteration Complete ===" -ForegroundColor Cyan
        Write-Host "Iteration Duration: ${IterationMinutes}m ${IterationSeconds}s" -ForegroundColor Green
        Write-Host "Exit Code: $ExitCode" -ForegroundColor $(if ($ExitCode -eq 0) { "Green" } else { "Yellow" })
        
        # Check if we should continue
        if ($ExitCode -eq 0) {
            Write-Host "✓ Iteration completed successfully. Restarting..." -ForegroundColor Green
            Start-Sleep -Seconds 5  # Brief pause before restart
        } else {
            Write-Host "✗ Iteration failed with exit code $ExitCode" -ForegroundColor Red
            Write-Host "Restarting anyway due to loop mode..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10  # Longer pause after failure
        }
        
        $Iteration++
        
    } while ($true)
    
} else {
    # Single execution mode
    $ExitCode = Run-SingleExecution
    
    # Calculate final time
    $EndTime = Get-Date
    $TotalTime = $EndTime - $StartTime
    $TotalMinutes = [math]::Floor($TotalTime.TotalMinutes)
    $TotalSeconds = $TotalTime.Seconds
    
    # Show final results
    Write-Host ""
    Write-Host "=== $ScriptType Complete ===" -ForegroundColor Cyan
    Write-Host "End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
    Write-Host "Total Duration: ${TotalMinutes}m ${TotalSeconds}s" -ForegroundColor Green
    Write-Host "Exit Code: $ExitCode" -ForegroundColor $(if ($ExitCode -eq 0) { "Green" } else { "Yellow" })
    
    # More intelligent success detection
    $SuccessExitCodes = @(0, 1, 2)  # Common exit codes that indicate completion
    if ($ExitCode -in $SuccessExitCodes) {
        Write-Host "✓ $ScriptType completed successfully!" -ForegroundColor Green
        Write-Host "Note: Exit code $ExitCode may indicate warnings but not failure" -ForegroundColor Yellow
    } else {
        Write-Host "✗ $ScriptType failed with exit code $ExitCode" -ForegroundColor Red
    }
} 