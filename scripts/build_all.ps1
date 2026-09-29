$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $root
try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) { py -3.11 -m venv .venv }
    $python = '.venv\Scripts\python.exe'
    $env:PYTHONPATH = Join-Path $root 'core'
    & $python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Core test suite failed.' }
    & $python -m compileall -q core tests scripts
    if ($LASTEXITCODE -ne 0) { throw 'Python compilation failed.' }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency check failed.' }
    & (Join-Path $PSScriptRoot 'build_android.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Android build failed.' }
    & (Join-Path $PSScriptRoot 'build_windows.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Windows build failed.' }
    Write-Output 'Windows and Android artifacts were built locally. Linux and macOS builds run in .github/workflows/build.yml on Linux and macOS runners.'
} finally { Pop-Location }
