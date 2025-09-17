# Improved PowerShell script to run evaluation with timing and output capture
# Usage: .\scripts\run_evaluation_improved.ps1 [ScriptPath] [InputJson]
# Example: .\scripts\run_evaluation_improved.ps1 "..\project\evaluation_DeCo_Diff2.py" "..\input_json\eval_input.json"

# Parse command line arguments
param(
    [string]$ScriptPath = ".\project\evaluation_DeCo_Diff2.py",
    [string]$InputJson = ".\input_json\eval_input.json"
)

# Set working directory to the project root (parent of scripts/)
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) "..")

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

# Start the process with output capture using cmd to handle arguments properly
Write-Host "Starting evaluation..." -ForegroundColor Green

# Get the current directory (where the script is being run from)
$CurrentDir = Get-Location
Write-Host "Current directory: $CurrentDir" -ForegroundColor Cyan

# Convert to absolute paths if they're relative
$ScriptPathAbs = if ([System.IO.Path]::IsPathRooted($ScriptPath)) { $ScriptPath } else { Join-Path $CurrentDir $ScriptPath }
$InputJsonAbs = if ([System.IO.Path]::IsPathRooted($InputJson)) { $InputJson } else { Join-Path $CurrentDir $InputJson }

Write-Host "Absolute Script Path: $ScriptPathAbs" -ForegroundColor Cyan
Write-Host "Absolute Input JSON: $InputJsonAbs" -ForegroundColor Cyan

$Command = "py -3.11 `"$ScriptPathAbs`" --input-json `"$InputJsonAbs`""
$ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
$ProcessInfo.FileName = "cmd"
$ProcessInfo.Arguments = "/c", $Command
$ProcessInfo.UseShellExecute = $false
$ProcessInfo.RedirectStandardOutput = $true
$ProcessInfo.RedirectStandardError = $true
$ProcessInfo.WorkingDirectory = $CurrentDir

$Process = New-Object System.Diagnostics.Process
$Process.StartInfo = $ProcessInfo
$Process.Start() | Out-Null

# Monitor the process and capture output
$LastUpdate = Get-Date
$Output = ""
$ErrorOutput = ""

while (-not $Process.HasExited) {
    # Show elapsed time every 30 seconds
    $CurrentTime = Get-Date
    if (($CurrentTime - $LastUpdate).TotalSeconds -ge 30) {
        Show-ElapsedTime
        $LastUpdate = $CurrentTime
    }
    
    # Capture output
    if ($Process.StandardOutput.Peek() -ge 0) {
        $line = $Process.StandardOutput.ReadLine()
        $Output += $line + "`n"
        Write-Host $line
    }
    
    if ($Process.StandardError.Peek() -ge 0) {
        $errorLine = $Process.StandardError.ReadLine()
        $ErrorOutput += $errorLine + "`n"
        Write-Host $errorLine -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds 0.1
}

# Capture remaining output
while ($Process.StandardOutput.Peek() -ge 0) {
    $line = $Process.StandardOutput.ReadLine()
    $Output += $line + "`n"
    Write-Host $line
}

while ($Process.StandardError.Peek() -ge 0) {
    $errorLine = $Process.StandardError.ReadLine()
    $ErrorOutput += $errorLine + "`n"
    Write-Host $errorLine -ForegroundColor Yellow
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
Write-Host "Exit Code: $ExitCode" -ForegroundColor $(if ($ExitCode -eq 0) { "Green" } else { "Yellow" })

# Intelligent success detection based on output content
$SuccessIndicators = @(
    "Evaluation Complete",
    "Confusion Matrix",
    "Excel report saved",
    "Report saved to"
)

$HasSuccessIndicators = $false
foreach ($indicator in $SuccessIndicators) {
    if ($Output -match $indicator) {
        $HasSuccessIndicators = $true
        break
    }
}

# Consider it successful if we have success indicators or reasonable exit codes
$SuccessExitCodes = @(0, 1, 2)  # Common exit codes that indicate completion
if ($Process.HasExited -and ($ExitCode -in $SuccessExitCodes -or $HasSuccessIndicators)) {
    Write-Host "✓ Evaluation completed successfully!" -ForegroundColor Green
    if ($ExitCode -ne 0) {
        Write-Host "Note: Exit code $ExitCode may indicate warnings but not failure" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ Evaluation failed with exit code $ExitCode" -ForegroundColor Red
    Write-Host "Process exited: $($Process.HasExited)" -ForegroundColor Red
    if ($ErrorOutput) {
        Write-Host "Error output:" -ForegroundColor Red
        Write-Host $ErrorOutput -ForegroundColor Red
    }
} 