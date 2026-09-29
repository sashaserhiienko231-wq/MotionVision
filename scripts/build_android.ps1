$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $env:ANDROID_SDK_ROOT) { $env:ANDROID_SDK_ROOT = Join-Path $env:LOCALAPPDATA 'Android\Sdk' }
$env:ANDROID_HOME = $env:ANDROID_SDK_ROOT
if (-not (Test-Path (Join-Path $env:ANDROID_SDK_ROOT 'platforms\android-36'))) { throw 'Android SDK platform 36 is required.' }
$android = Join-Path $root 'android'
$out = Join-Path $root 'dist\android'
New-Item -ItemType Directory -Force -Path $out | Out-Null
Push-Location $android
try {
    $sdk = $env:ANDROID_SDK_ROOT.Replace('\','\\')
    "sdk.dir=$sdk" | Set-Content -Encoding ascii local.properties
    if (-not (Test-Path '.\gradlew.bat')) {
        gradle wrapper --gradle-version 8.11.1 --distribution-type bin
        if ($LASTEXITCODE -ne 0) { throw 'Gradle wrapper generation failed.' }
    }
    .\gradlew.bat --no-daemon testDebugUnitTest lintVitalRelease assembleRelease bundleRelease
    if ($LASTEXITCODE -ne 0) { throw 'Android build or unit tests failed.' }
    Copy-Item app\build\outputs\apk\release\app-release.apk (Join-Path $out 'MotionVision.apk') -Force
    Copy-Item app\build\outputs\bundle\release\app-release.aab (Join-Path $out 'MotionVision.aab') -Force
    foreach ($name in @('MotionVision.apk','MotionVision.aab')) {
        $artifact = Get-Item (Join-Path $out $name)
        if ($artifact.Length -eq 0) { throw "Empty Android artifact: $name" }
        Write-Output ("ARTIFACT {0} {1} bytes" -f $artifact.FullName,$artifact.Length)
    }
    $buildTools = Get-ChildItem (Join-Path $env:ANDROID_SDK_ROOT 'build-tools') -Directory |
        Where-Object Name -like '36.*' | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
    if (-not $buildTools) { throw 'Android Build Tools 36.x is required for APK verification.' }
    & (Join-Path $buildTools 'apksigner.bat') verify (Join-Path $out 'MotionVision.apk')
    if ($LASTEXITCODE -ne 0) { throw 'APK signature verification failed.' }
    $badging = (& (Join-Path $buildTools 'aapt.exe') dump badging (Join-Path $out 'MotionVision.apk')) -join "`n"
    if ($LASTEXITCODE -ne 0 -or ($badging -notmatch "name='com\.motionvision\.app'") -or ($badging -notmatch 'application-icon')) {
        throw 'Android package ID or embedded launcher icon verification failed.'
    }
} finally { Pop-Location }
