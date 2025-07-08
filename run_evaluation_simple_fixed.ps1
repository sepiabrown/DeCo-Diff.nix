# Simple PowerShell script to run evaluation with timing
# Usage: .\run_evaluation_simple_fixed.ps1 [ScriptPath] [InputJson]
# Example: .\run_evaluation_simple_fixed.ps1 "project\evaluation_DeCo_Diff2.py" ".\input_json\eval_input.json"

# Parse command line arguments
param(
    [string]$ScriptPath = "project\evaluation_DeCo_Diff2.py",
    [string]$InputJson = ".\input_json\eval_input.json"
)

# Start timing
$StartTime = Get-Date
Write-Host "=== Starting Evaluation ===" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "Input JSON: $InputJson" -ForegroundColor Yellow
Write-Host "Script Path: $ScriptPath" -ForegroundColor Yellow
Write-Host "Command: py -3.11 $ScriptPath --input-json $InputJson" -ForegroundColor Yellow
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

# Start the process
Write-Host "Starting evaluation..." -ForegroundColor Green
$Process = Start-Process -FilePath "py" -ArgumentList "-3.11", $ScriptPath, "--input-json", $InputJson -PassThru -NoNewWindow

# Monitor the process
$LastUpdate = Get-Date
while (-not $Process.HasExited) {
    # Show elapsed time every 30 seconds
    $CurrentTime = Get-Date
    if (($CurrentTime - $LastUpdate).TotalSeconds -ge 30) {
        Show-ElapsedTime
        $LastUpdate = $CurrentTime
    }
    Start-Sleep -Seconds 1
}

# Wait for completion
$Process.WaitForExit()
$ExitCode = $Process.ExitCode

# Calculate final time
$EndTime = Get-Date
$TotalTime = $EndTime - $StartTime
$TotalMinutes = [math]::Floor($TotalTime.TotalMinutes)
$TotalSeconds = $TotalTime.Seconds

# Show final results
Write-Host ""
Write-Host "=== Evaluation Complete ===" -ForegroundColor Cyan
Write-Host "End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "Total Duration: ${TotalMinutes}m ${TotalSeconds}s" -ForegroundColor Green
Write-Host "Exit Code: $ExitCode" -ForegroundColor $(if ($ExitCode -eq 0) { "Green" } else { "Red" })

if ($ExitCode -eq 0) {
    Write-Host "✓ Evaluation completed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Evaluation failed with exit code $ExitCode" -ForegroundColor Red
} 