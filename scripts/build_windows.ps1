$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path (Join-Path $root '.venv\Scripts\python.exe'))) { py -3.11 -m venv (Join-Path $root '.venv') }
$python = Join-Path $root '.venv\Scripts\python.exe'
$out = Join-Path $root 'dist\windows'
$build = Join-Path $root 'build\windows'
New-Item -ItemType Directory -Force -Path $out,$build | Out-Null

Push-Location $root
try {
    & $python -m pip install -r requirements-build.txt
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
    & $python scripts\generate_icons.py
    if ($LASTEXITCODE -ne 0) { throw 'Icon generation failed.' }
    $env:PYTHONPATH = Join-Path $root 'core'
    & $python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed.' }
    & $python -m compileall -q core tests scripts
    if ($LASTEXITCODE -ne 0) { throw 'Python compilation failed.' }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }

    & $python -m PyInstaller --noconfirm --clean --onefile --windowed `
        --name MotionVision --icon assets\icon\MotionVision.ico `
        --version-file windows\version-info.txt --paths core `
        --collect-all mediapipe --collect-all cv2 `
        --hidden-import mediapipe.tasks.python.vision.face_landmarker `
        --hidden-import mediapipe.tasks.python.vision.hand_landmarker `
        --hidden-import mediapipe.tasks.python.vision.pose_landmarker `
        --add-data "models;models" --add-data "config\version.json;config" `
        --distpath $out --workpath $build core\app.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
    $exe = Join-Path $out 'MotionVision.exe'
    if (-not (Test-Path $exe) -or (Get-Item $exe).Length -lt 10MB) { throw 'Portable EXE is missing or unexpectedly small.' }
    & $python scripts\verify_windows_icon.py $exe
    if ($LASTEXITCODE -ne 0) { throw 'The portable executable is missing embedded icon resources.' }
    & $exe --version
    if ($LASTEXITCODE -ne 0) { throw 'Packaged executable failed its version check.' }

    $iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if (-not $iscc) {
        $isccPath = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
        $localIscc = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Programs') -Filter ISCC.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($localIscc) { $compiler = $localIscc.FullName }
        elseif (Test-Path $isccPath) { $compiler = $isccPath }
        else {
            winget install --id JRSoftware.InnoSetup --exact --scope user --silent --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -ne 0) { throw 'User-scope Inno Setup installation failed.' }
            $localIscc = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Programs') -Filter ISCC.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($localIscc) { $compiler = $localIscc.FullName }
            elseif (Test-Path $isccPath) { $compiler = $isccPath }
            else { throw 'Inno Setup 6 is required to produce the installer.' }
        }
    } else { $compiler = $iscc.Source }
    $version = (& $python -c "import json; print(json.load(open('config/version.json', encoding='utf-8'))['version'])").Trim()
    & $compiler "/DAppVersion=$version" windows\MotionVision.iss
    if ($LASTEXITCODE -ne 0) { throw 'Inno Setup compiler failed.' }
    foreach ($artifact in @($exe,(Join-Path $out 'MotionVision-Setup.exe'))) {
        if (-not (Test-Path $artifact) -or (Get-Item $artifact).Length -eq 0) { throw "Missing artifact: $artifact" }
        $file = Get-Item $artifact
        Write-Output ("ARTIFACT {0} {1} bytes" -f $file.FullName,$file.Length)
    }
} finally { Pop-Location }
