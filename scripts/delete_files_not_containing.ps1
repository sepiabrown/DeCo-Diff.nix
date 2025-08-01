# PowerShell script to delete files that don't contain a specified string in their name
# Usage: .\delete_files_not_containing.ps1 -Directory "C:\path\to\directory" -String "required_string" [-WhatIf] [-Confirm]

param(
    [Parameter(Mandatory=$true)]
    [string]$Directory,
    
    [Parameter(Mandatory=$true)]
    [string]$String,
    
    [switch]$WhatIf,
    
    [switch]$Confirm
)

# Function to display help
function Show-Help {
    Write-Host @"
PowerShell script to delete files that don't contain a specified string in their name.

Usage:
    .\delete_files_not_containing.ps1 -Directory "C:\path\to\directory" -String "required_string" [-WhatIf] [-Confirm]

Parameters:
    -Directory    : Path to the directory containing files to check
    -String       : String that must be present in the filename (case-insensitive)
    -WhatIf       : Shows what would be deleted without actually deleting (dry run)
    -Confirm      : Prompts for confirmation before each deletion

Examples:
    .\delete_files_not_containing.ps1 -Directory "C:\temp\files" -String "backup" -WhatIf
    .\delete_files_not_containing.ps1 -Directory "C:\temp\files" -String "important" -Confirm
    .\delete_files_not_containing.ps1 -Directory "C:\temp\files" -String "2024"

"@
}

# Check if help is requested
if ($args -contains "-h" -or $args -contains "--help" -or $args -contains "-help") {
    Show-Help
    exit 0
}

# Validate directory exists
if (-not (Test-Path -Path $Directory -PathType Container)) {
    Write-Error "Directory '$Directory' does not exist or is not accessible."
    exit 1
}

# Get all files in the directory
try {
    $files = Get-ChildItem -Path $Directory -File
} catch {
    Write-Error "Failed to access directory '$Directory': $($_.Exception.Message)"
    exit 1
}

if ($files.Count -eq 0) {
    Write-Host "No files found in directory '$Directory'."
    exit 0
}

# Filter files that don't contain the specified string
$filesToDelete = $files | Where-Object { $_.Name -notmatch [regex]::Escape($String) }

if ($filesToDelete.Count -eq 0) {
    Write-Host "All files in '$Directory' contain the string '$String'. No files to delete."
    exit 0
}

# Display summary
Write-Host "Directory: $Directory"
Write-Host "Required string: '$String'"
Write-Host "Total files in directory: $($files.Count)"
Write-Host "Files to delete: $($filesToDelete.Count)"
Write-Host "Files to keep: $($files.Count - $filesToDelete.Count)"
Write-Host ""

# Show files that will be deleted
Write-Host "Files that will be deleted:"
$filesToDelete | ForEach-Object { Write-Host "  - $($_.Name)" }
Write-Host ""

# If WhatIf is specified, show what would be deleted and exit
if ($WhatIf) {
    Write-Host "WhatIf mode: No files will be deleted. Use without -WhatIf to actually delete files."
    exit 0
}

# Ask for confirmation if not using -Confirm parameter
if (-not $Confirm) {
    $response = Read-Host "Do you want to proceed with deleting these files? (y/N)"
    if ($response -notmatch "^[Yy]$") {
        Write-Host "Operation cancelled by user."
        exit 0
    }
}

# Delete files
$deletedCount = 0
$errorCount = 0

foreach ($file in $filesToDelete) {
    try {
        if ($Confirm) {
            $response = Read-Host "Delete file '$($file.Name)'? (y/N)"
            if ($response -notmatch "^[Yy]$") {
                Write-Host "Skipping file: $($file.Name)"
                continue
            }
        }
        
        Remove-Item -Path $file.FullName -Force
        Write-Host "Deleted: $($file.Name)"
        $deletedCount++
    } catch {
        Write-Error "Failed to delete file '$($file.Name)': $($_.Exception.Message)"
        $errorCount++
    }
}

# Display final summary
Write-Host ""
Write-Host "Operation completed:"
Write-Host "  Files successfully deleted: $deletedCount"
Write-Host "  Files with errors: $errorCount"
Write-Host "  Files remaining in directory: $($files.Count - $deletedCount)"

if ($errorCount -gt 0) {
    Write-Host "Some files could not be deleted. Check the error messages above."
    exit 1
} else {
    Write-Host "All specified files deleted successfully."
    exit 0
} 