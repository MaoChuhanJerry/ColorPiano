<#
.SYNOPSIS
    Run ColorPiano Pro, creating nothing and changing nothing.

.DESCRIPTION
    Uses the project's .venv if it exists, otherwise whatever "python" is on
    PATH.  Any extra arguments are passed straight through:

        powershell -ExecutionPolicy Bypass -File scripts\run.ps1 --cvd deuteranopia
        powershell -ExecutionPolicy Bypass -File scripts\run.ps1 --source synthetic
        powershell -ExecutionPolicy Bypass -File scripts\run.ps1 --selftest --pipeline
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
    Write-Host "using $venvPython" -ForegroundColor DarkGray
} else {
    $python = "python"
    Write-Host "no .venv found, using 'python' from PATH" -ForegroundColor Yellow
    Write-Host "run scripts\setup.ps1 first for an isolated environment" -ForegroundColor Yellow
}

& $python "main.py" @Args
exit $LASTEXITCODE
