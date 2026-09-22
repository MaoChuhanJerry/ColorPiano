<#
.SYNOPSIS
    Create the virtual environment and install ColorPiano Pro's dependencies.

.DESCRIPTION
    Run from anywhere:

        powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

    The environment is created at <project>\.venv.  Keeping it inside the
    project directory and on a plain-ASCII path matters: MediaPipe's native
    layer cannot open its own model file when the path contains non-ASCII
    characters, which is what silently disabled hand gestures in the original
    project (its folder was named "... - 副本").  This script checks for that
    and tells you before you waste time on it.

.PARAMETER Python
    Interpreter to build the environment with.  Defaults to "python".

.PARAMETER NoMediapipe
    Skip the optional hand-gesture package.

.PARAMETER Mirror
    Use a PyPI mirror (Tsinghua by default) -- much faster from mainland China.

.PARAMETER SkipChecks
    Do not run the self test afterwards.
#>
[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$NoMediapipe,
    [switch]$Mirror,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"

function Write-Step($message) {
    Write-Host ""
    Write-Host "=== $message" -ForegroundColor Cyan
}

function Write-Ok($message)   { Write-Host "  [ok]   $message" -ForegroundColor Green }
function Write-Warn($message) { Write-Host "  [warn] $message" -ForegroundColor Yellow }
function Write-Bad($message)  { Write-Host "  [fail] $message" -ForegroundColor Red }

Write-Host ""
Write-Host "ColorPiano Pro -- setup" -ForegroundColor White

# --------------------------------------------------------------------------- #
# Path sanity
# --------------------------------------------------------------------------- #
Write-Step "Checking the install path"

$isAscii = $true
foreach ($ch in $projectRoot.ToCharArray()) {
    if ([int]$ch -gt 127) { $isAscii = $false; break }
}
if ($isAscii) {
    Write-Ok "path is ASCII-only: $projectRoot"
} else {
    Write-Warn "this path contains non-ASCII characters:"
    Write-Warn "  $projectRoot"
    Write-Warn "MediaPipe cannot load its hand model from such a path.  The"
    Write-Warn "program works around it by copying mediapipe to a temporary"
    Write-Warn "ASCII directory, but a clean path (e.g. C:\dev\colorpiano)"
    Write-Warn "is more reliable and much faster to start."
}

# --------------------------------------------------------------------------- #
# Interpreter
# --------------------------------------------------------------------------- #
Write-Step "Looking for Python"

& $Python --version
if (-not $?) {
    Write-Bad "'$Python' did not run. Install Python 3.9+ and put it on PATH,"
    Write-Bad "or pass an interpreter explicitly: -Python C:\Python310\python.exe"
    exit 1
}

$version = (& $Python -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
$major, $minor = $version.Split(".")
if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 9)) {
    Write-Bad "Python $version found, but 3.9 or newer is required."
    exit 1
}
Write-Ok "Python $version"

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
Write-Step "Creating the virtual environment"

if (Test-Path $venvPath) {
    Write-Ok "$venvPath already exists, reusing it"
} else {
    & $Python -m venv $venvPath
    if (-not $?) { Write-Bad "venv creation failed"; exit 1 }
    Write-Ok "created $venvPath"
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Bad "expected interpreter not found: $venvPython"
    exit 1
}

Write-Step "Installing dependencies"

& $venvPython -m pip install --upgrade pip --quiet
if ($?) { Write-Ok "pip upgraded" }

$requirements = Join-Path $projectRoot "requirements.txt"
$pipArgs = @("install", "-r", $requirements)
if ($Mirror) {
    $pipArgs += @("-i", "https://pypi.tuna.tsinghua.edu.cn/simple")
    Write-Host "  using the Tsinghua mirror"
}

& $venvPython -m pip @pipArgs
if (-not $?) {
    Write-Bad "dependency install failed."
    Write-Bad "Retry with the mirror:  .\scripts\setup.ps1 -Mirror"
    exit 1
}
Write-Ok "dependencies installed"

if ($NoMediapipe) {
    Write-Warn "hand gestures skipped as requested (the rest works normally)"
} else {
    $hasMediapipe = (& $venvPython -c "import importlib.util as u; print(u.find_spec('mediapipe') is not None)").Trim()
    if ($hasMediapipe -ne "True") {
        Write-Warn "mediapipe is not installed; hand gestures will be off."
        Write-Warn "Everything else works.  Add it later with:"
        Write-Warn "  $venvPython -m pip install mediapipe"
        Write-Warn "  $venvPython -m tools.fetch_model"
    } else {
        # Recent mediapipe wheels removed the self-contained API and ship no
        # model file, so hand tracking needs an extra 8 MB download.  Doing it
        # here means "gestures work" right after setup, with no surprise later.
        Write-Host "  checking the hand model..."
        Push-Location $projectRoot
        try {
            & $venvPython "tools\fetch_model.py"
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "hand gestures ready"
            } else {
                Write-Warn "the hand model could not be fetched; gestures will be off"
                Write-Warn "Everything else works.  Retry later with:"
                Write-Warn "  $venvPython -m tools.fetch_model"
                Write-Warn "or download it by hand (see tools\fetch_model.py)."
            }
        } finally {
            Pop-Location
        }
    }
}

# --------------------------------------------------------------------------- #
# Verify
# --------------------------------------------------------------------------- #
if (-not $SkipChecks) {
    Write-Step "Running the self test"
    Push-Location $projectRoot
    try {
        & $venvPython "main.py" --selftest --skip-hardware
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "self test passed"
        } else {
            Write-Warn "the self test reported a problem (see above)"
        }
    } finally {
        Pop-Location
    }
}

Write-Step "Done"
Write-Host "  Activate the environment:  .\.venv\Scripts\Activate.ps1"
Write-Host "  Run the instrument:        python main.py"
Write-Host ""
Write-Host "  For a colour-blind user, start with:" -ForegroundColor White
Write-Host "    python main.py --palette cvd_safe --cvd deuteranopia"
Write-Host "  (replace deuteranopia with your own type; press 'h' at run time"
Write-Host "   for the full key list)"
Write-Host ""
Write-Host "  If activation is blocked by policy:"
Write-Host "    Set-ExecutionPolicy -Scope Process RemoteSigned"
Write-Host ""
