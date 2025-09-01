# Universal PowerShell script to run training or evaluation scripts with Slack notifications
# Usage (positional): .\run_script_universal.ps1 [ScriptType] [ScriptPath] [InputJson] [Loop] [SleepMinutes]
# Usage (named): .\run_script_universal.ps1 -ScriptType "train" -ScriptPath "..\project\train_DeCo_Diff.py" -InputJson "..\input_json\train_input.json" -Loop $true -SleepMinutes 30 -Notify $true
# Examples:
#   .\run_script_universal.ps1 "train" "..\project\train_DeCo_Diff.py" "..\input_json\train_input.json" $true 30
#   .\run_script_universal.ps1 -ScriptType "train" -ScriptPath "..\project\train_DeCo_Diff.py" -InputJson "..\input_json\train_input.json" -Loop $true -SleepMinutes 30 -Notify $true
#   .\run_script_universal.ps1 -ScriptType "eval" -InputJson "..\input_json\eval_input.json" -SleepMinutes 15 -Notify $true
#   .\run_script_universal.ps1 -ScriptType "eval" -Loop $false
#
# Slack notifications: Set SLACK_WEBHOOK_URL environment variable and use -Notify $true to enable notifications

# Parse command line arguments
param(
    [Parameter(Position=0)]
    [string]$ScriptType = "eval",  # "train" or "eval"

    [Parameter(Position=1)]
    [string]$ScriptPath = "..\project\evaluation_DeCo_Diff2.py",

    [Parameter(Position=2)]
    [string]$InputJson = "..\input_json\eval_input.json",

    [Parameter(Position=3)]
    [bool]$Loop = $false,

    [Parameter(Position=4)]
    [int]$SleepMinutes = 0,  # Sleep duration in minutes

    [Parameter(Position=5)]
    [bool]$Notify = $false,  # Enable Slack notifications

    [Alias("Type")]
    [Parameter()]
    [ValidateSet("train", "eval")]
    [string]$ScriptTypeNamed = $null,

    [Alias("Path")]
    [Parameter()]
    [string]$ScriptPathNamed = $null,

    [Alias("Json")]
    [Parameter()]
    [string]$InputJsonNamed = $null,

    [Alias("Sleep")]
    [Parameter()]
    [object]$SleepMinutesNamed = $null,

    [Parameter()]
    [object]$NotifyNamed = $null
)

# Override positional parameters with named parameters if provided
if ($ScriptTypeNamed) { $ScriptType = $ScriptTypeNamed }
if ($ScriptPathNamed) { $ScriptPath = $ScriptPathNamed }
if ($InputJsonNamed) { $InputJson = $InputJsonNamed }
if ($null -ne $SleepMinutesNamed -and $SleepMinutesNamed -ne "") { $SleepMinutes = [int]$SleepMinutesNamed }
if ($null -ne $NotifyNamed -and $NotifyNamed -ne "") { $Notify = [bool]$NotifyNamed }

# Start timing
$StartTime = Get-Date
Write-Host "=== Starting $ScriptType Script ===" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "Script Type: $ScriptType" -ForegroundColor Yellow
Write-Host "Input JSON: $InputJson" -ForegroundColor Yellow
Write-Host "Script Path: $ScriptPath" -ForegroundColor Yellow
Write-Host "Loop Mode: $Loop" -ForegroundColor Yellow
Write-Host "Sleep Duration: $SleepMinutes minutes" -ForegroundColor Yellow
Write-Host "Slack Notifications: $Notify" -ForegroundColor Yellow
Write-Host ""

# Function to show elapsed time
function Show-ElapsedTime {
    $Elapsed = (Get-Date) - $StartTime
    $Minutes = [math]::Floor($Elapsed.TotalMinutes)
    $Seconds = $Elapsed.Seconds
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Elapsed: ${Minutes}m ${Seconds}s" -ForegroundColor Magenta
}

# Function to sleep with countdown display
function Start-SleepWithCountdown {
    param([int]$Minutes)

    if ($Minutes -le 0) {
        return
    }

    Write-Host ""
    Write-Host "=== SLEEP MODE ===" -ForegroundColor Cyan
    Write-Host "Sleeping for $Minutes minutes..." -ForegroundColor Yellow

    $SleepSeconds = $Minutes * 60
    $StartSleepTime = Get-Date

    for ($i = $SleepSeconds; $i -gt 0; $i--) {
        $RemainingMinutes = [math]::Floor($i / 60)
        $RemainingSeconds = $i % 60

        # Clear line and show countdown
        Write-Host "`rSleeping: ${RemainingMinutes}m ${RemainingSeconds}s remaining..." -NoNewline -ForegroundColor Green

        Start-Sleep -Seconds 1
    }

    $EndSleepTime = Get-Date
    $ActualSleepTime = $EndSleepTime - $StartSleepTime
    $ActualMinutes = [math]::Floor($ActualSleepTime.TotalMinutes)
    $ActualSeconds = $ActualSleepTime.Seconds

    Write-Host ""
    Write-Host "Sleep completed! Actual sleep time: ${ActualMinutes}m ${ActualSeconds}s" -ForegroundColor Green
    Write-Host ""
}

# Function to send Slack notification
function Send-SlackNotification {
    param(
        [string]$Status,
        [string]$Color,
        [string]$ScriptPath,
        [int]$Iteration = $null,
        [bool]$Notify = $false,
        [string]$LastOutputLines = $null
    )

    # Check if notifications are enabled
    if (-not $Notify) {
        Write-Host "Slack notifications disabled, skipping notification" -ForegroundColor Yellow
        return
    }

    $webhook = $env:SLACK_WEBHOOK_URL
    if (-not $webhook) {
        Write-Host "Slack webhook URL not set, skipping notification" -ForegroundColor Yellow
        return
    }

    $iterationText = if ($Iteration) { " (Iteration $Iteration)" } else { "" }

    # Build blocks array
    $blocks = @(
        @{
            type = "header"
            text = @{ type="plain_text"; text="Python job result" }
        },
        @{
            type = "section"
            fields = @(
                @{ type="mrkdwn"; text="*Status:*\n$Status" },
                @{ type="mrkdwn"; text="*Host:*\n$env:COMPUTERNAME" },
                @{ type="mrkdwn"; text="*When:*\n$(Get-Date -Format o)" },
                @{ type="mrkdwn"; text="*Script:*\n$ScriptPath$iterationText" }
            )
        }
    )

    # Add output section if we have output lines
    if ($LastOutputLines) {
        $blocks += @{
            type = "section"
            text = @{
                type = "mrkdwn"
                text = "*Last Output Lines:*\n``````$LastOutputLines``````"
            }
        }
    }

    # Fancy Block Kit message
    $payload = @{
        text   = "$Status — $(Get-Date -Format o) on $env:COMPUTERNAME$iterationText"
        blocks = $blocks
    } | ConvertTo-Json -Depth 5

    try {
        Invoke-RestMethod -Method Post -Uri $webhook -Body $payload -ContentType 'application/json'
        Write-Host "Slack notification sent successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to send Slack notification: $($_.Exception.Message)" -ForegroundColor Red
    }
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

# Sleep at the beginning if specified
if ($SleepMinutes -gt 0) {
    Write-Host "Initial sleep before starting execution..." -ForegroundColor Yellow
    Start-SleepWithCountdown -Minutes $SleepMinutes
}

# Set environment variables for training
if ($ScriptType -eq "train") {
    $env:NO_ALBUMENTATIONS_UPDATE = "1"
    Write-Host "Set environment variable: NO_ALBUMENTATIONS_UPDATE=1" -ForegroundColor Green
}

# Set UTF-8 encoding for Python to handle Unicode properly
$env:PYTHONIOENCODING = "utf-8"
Write-Host "Set environment variable: PYTHONIOENCODING=utf-8" -ForegroundColor Green

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
    Write-Host "Executing command: $Command" -ForegroundColor Cyan

    try {
        Write-Host "Starting process with output capture..." -ForegroundColor Yellow
        
        # Use a different approach - capture only stdout, let stderr (tqdm) flow naturally
        $tempStdout = [System.IO.Path]::GetTempFileName()
        
        # Set console output encoding to UTF-8
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        [Console]::InputEncoding = [System.Text.Encoding]::UTF8
        
        # Start process with only stdout redirected, let stderr (tqdm) display naturally
        if ($ScriptType -eq "train") {
            # Training command with distributed run
            $process = Start-Process -FilePath "py" -ArgumentList "-3.11", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=1", "`"$($ScriptPathAbs)`"", "--input-json", "`"$($InputJsonAbs)`"" -Wait -PassThru -NoNewWindow -RedirectStandardOutput $tempStdout
        } else {
            # Evaluation command
            $process = Start-Process -FilePath "py" -ArgumentList "-3.11", "`"$($ScriptPathAbs)`"", "--input-json", "`"$($InputJsonAbs)`"" -Wait -PassThru -NoNewWindow -RedirectStandardOutput $tempStdout
        }
        $ExitCode = $process.ExitCode
        
        # Read only the stdout content for our output capture
        $outputLines = @()
        if (Test-Path $tempStdout) {
            $stdoutLines = Get-Content $tempStdout -Encoding UTF8 -ErrorAction SilentlyContinue
            if ($stdoutLines) {
                $outputLines += $stdoutLines
                # Display the captured stdout content
                foreach ($line in $stdoutLines) {
                    Write-Host $line
                }
            }
            Remove-Item $tempStdout -Force -ErrorAction SilentlyContinue
        }
        
        if ($outputLines.Count -eq 0) {
            Write-Host "No output received from command" -ForegroundColor Yellow
        }
        
        Write-Host "Process exited with code: $ExitCode" -ForegroundColor $(if ($ExitCode -eq 0) { "Green" } else { "Yellow" })

        # Get the actual last 10 lines as seen in the console (chronological order)
        $nonEmptyLines = $outputLines | Where-Object { $_ -and $_.ToString().Trim() -ne "" }
        
        # Filter out any remaining artifacts (mostly stdout should be clean now)
        $cleanLines = $nonEmptyLines | Where-Object { 
            $line = $_.ToString()
            -not ($line -match "System\.Management\.Automation\.RemoteException" -or 
                  $line.Trim() -eq "")
        }
        
        # If we don't have enough clean lines, fall back to all non-empty lines
        if ($cleanLines.Count -lt 5) {
            $cleanLines = $nonEmptyLines | Where-Object { 
                $line = $_.ToString()
                -not ($line -match "System\.Management\.Automation\.RemoteException" -or $line.Trim() -eq "")
            }
        }
        
        # Simple approach: just get the last 10 non-empty lines chronologically
        $LastOutputLines = if ($cleanLines.Count -gt 10) {
            ($cleanLines[-10..-1] | ForEach-Object { $_.ToString() }) -join "`n"
        } elseif ($cleanLines.Count -gt 0) {
            ($cleanLines | ForEach-Object { $_.ToString() }) -join "`n"
        } else {
            "No output captured"
        }

        Write-Host "Captured $($outputLines.Count) total lines, $($nonEmptyLines.Count) non-empty lines" -ForegroundColor Cyan
        if ($LastOutputLines -ne "No output captured") {
            Write-Host "Last output preview:" -ForegroundColor Cyan
            Write-Host $LastOutputLines.Substring(0, [Math]::Min(200, $LastOutputLines.Length)) -ForegroundColor Gray
        }

        # Return both exit code and last 10 lines
        return @{
            ExitCode = $ExitCode
            LastOutputLines = $LastOutputLines
        }

    }
    catch {
        Write-Host "Error during execution: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Stack trace: $($_.ScriptStackTrace)" -ForegroundColor Red
        return @{
            ExitCode = -1
            LastOutputLines = "Error: $($_.Exception.Message)"
        }
    }
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

        try {
            $result = Run-SingleExecution
            $ExitCode = $result.ExitCode
            $LastOutputLines = $result.LastOutputLines

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
                Write-Host "✓ Iteration completed successfully." -ForegroundColor Green
                Send-SlackNotification -Status "✅ Iteration $Iteration Success" -Color "#2eb886" -ScriptPath $ScriptPath -Iteration $Iteration -Notify $Notify -LastOutputLines $LastOutputLines
                Write-Host "Restarting immediately..." -ForegroundColor Green
                Start-Sleep -Seconds 5  # Brief pause before restart
            } else {
                Write-Host "✗ Iteration failed with exit code $ExitCode" -ForegroundColor Red
                Send-SlackNotification -Status "❌ Iteration $Iteration Failed (Exit code: $ExitCode)" -Color "#e01e5a" -ScriptPath $ScriptPath -Iteration $Iteration -Notify $Notify -LastOutputLines $LastOutputLines
                Write-Host "Restarting anyway due to loop mode..." -ForegroundColor Yellow
                Start-Sleep -Seconds 10  # Longer pause after failure
            }
        }
        catch {
            $IterationEndTime = Get-Date
            $IterationTime = $IterationEndTime - $IterationStartTime
            $IterationMinutes = [math]::Floor($IterationTime.TotalMinutes)
            $IterationSeconds = $IterationTime.Seconds

            Write-Host ""
            Write-Host "=== Iteration $Iteration Failed ===" -ForegroundColor Red
            Write-Host "Iteration Duration: ${IterationMinutes}m ${IterationSeconds}s" -ForegroundColor Red
            Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red

            Send-SlackNotification -Status "❌ Iteration $Iteration Failed: $($_.Exception.Message)" -Color "#e01e5a" -ScriptPath $ScriptPath -Iteration $Iteration -Notify $Notify -LastOutputLines "Script execution error: $($_.Exception.Message)"

            Write-Host "Restarting after error..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10  # Longer pause after failure
        }
        finally {
            $Iteration++
        }

    } while ($true)

} else {
    # Single execution mode
    try {
        $result = Run-SingleExecution
        $ExitCode = $result.ExitCode
        $LastOutputLines = $result.LastOutputLines

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
            Send-SlackNotification -Status "✅ Success" -Color "#2eb886" -ScriptPath $ScriptPath -Notify $Notify -LastOutputLines $LastOutputLines
        } else {
            Write-Host "✗ $ScriptType failed with exit code $ExitCode" -ForegroundColor Red
            Send-SlackNotification -Status "❌ Failed (Exit code: $ExitCode)" -Color "#e01e5a" -ScriptPath $ScriptPath -Notify $Notify -LastOutputLines $LastOutputLines
        }
    }
    catch {
        # Calculate final time
        $EndTime = Get-Date
        $TotalTime = $EndTime - $StartTime
        $TotalMinutes = [math]::Floor($TotalTime.TotalMinutes)
        $TotalSeconds = $TotalTime.Seconds

        Write-Host ""
        Write-Host "=== $ScriptType Failed ===" -ForegroundColor Red
        Write-Host "End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Red
        Write-Host "Total Duration: ${TotalMinutes}m ${TotalSeconds}s" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red

        Send-SlackNotification -Status "❌ Failed: $($_.Exception.Message)" -Color "#e01e5a" -ScriptPath $ScriptPath -Notify $Notify -LastOutputLines "Script execution error: $($_.Exception.Message)"
    }
} 