# Motion Vision

**Version 1.0.0** · package ID `com.motionvision.app`

Motion Vision is a local-first camera application with four platform projects. The desktop applications reuse the existing Python computer-vision pipeline and OpenCV HUD. Android has a separate native CameraX screen and MediaPipe Tasks pipeline; it does not embed or launch the desktop GUI.

## Current feature coverage

| Capability | Windows / Linux / macOS desktop | Android |
|---|---|---|
| Camera preview and actual camera mode reporting | Yes | Yes, CameraX reports delivered analysis dimensions |
| Face, hand and pose landmarks | MediaPipe Tasks; multi-face and multi-person | MediaPipe Tasks; up to 8 faces/hands and 4 poses |
| Gesture history / temporal smoothing | Full desktop recognizer and HUD | Landmark geometry with 5-frame voting for static hand gestures |
| Stable person IDs | Face tracker | Face-center tracker with a short missing-frame grace period |
| SFace face matching and local face database | Yes; opt-in enrollment and local descriptors | Not implemented in 1.0.0 |
| Recording | Local video and desktop Telegram upload | Local MP4 recording; no Telegram upload |
| HUD drag/pinch, desktop window controls, Telegram bot | Yes | Native Android controls; Telegram/HUD dragging not implemented |
| Camera permission | Requested by the OS backend | Only `CAMERA` is declared/requested |

The Android app uses the front or rear camera, portrait or landscape layout, local face/hand/pose inference, face tracks, hand side labels, static gesture classification, CPU fallback if the MediaPipe GPU delegate cannot initialize, and local recording. Its full parity gaps are listed in [known limitations](docs/KNOWN_LIMITATIONS.md); the Android build is a distinct app, not a claim that every desktop feature is present on mobile.

Camera resolution and FPS are requests, not promises. Desktop HUD and Android overlay show observed frame dimensions/rates. No upscaling is used to report a higher camera mode.

## Privacy

Face matching can be turned off on desktop with `FACE_RECOGNITION_ENABLED=false`. Face landmarks and tracking for the camera overlay remain active, while SFace embedding and database matching are skipped; saved entries stay on disk so matching can be re-enabled later. Face embeddings are stored locally but are not encrypted by the application. Use face recognition only with the people’s consent and in compliance with applicable laws.

## Project layout

```text
MotionVision/
├── core/                 Shared desktop Python vision and Telegram logic
├── android/              Native Android app (CameraX + MediaPipe Tasks)
├── windows/              Inno Setup definition and Windows metadata
├── linux/                Desktop entry and Linux packaging inputs
├── macos/                macOS packaging notes
├── assets/icon/          SVG source, PNGs, ICO and ICNS
├── assets/platform/      Linux hicolor icon theme
├── models/               Local MediaPipe tasks and SFace model
├── config/               Version and safe environment template
├── tests/                Desktop tests; Android unit tests are under android/app
├── scripts/              Per-platform and aggregate build commands
├── .github/workflows/    Windows, Linux, Android and macOS hosted builds
└── dist/                 Created by build scripts; never contains placeholders
```

## Desktop run (Python 3.11)

From this directory on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config\.env.example .env
python core\app.py
```

Choose a camera request and start the app with `python core\app.py --camera 0 --resolution 1280x720 --fps 30`. `--list-cameras` probes cameras and `--reset-window` restores safe window geometry. The desktop app's existing `R`, `F11`, `Ctrl+Shift+R`, and `F3` controls remain available. `python core\app.py --version` prints the product version.

When packaged, desktop settings, recordings and face descriptors live in the per-user data folder. Optional Telegram values go in `.env` in that folder. The example is `config/.env.example`; the token is never included in an artifact.

## Android

Android 10+ (API 29) is the minimum. The app requests only camera permission and processes frames locally. `INTERNET`, audio and notification permissions are not requested. From Android Studio open `android/`, or use `scripts/build_android.ps1` on Windows / `scripts/build_android.sh` on Linux/macOS with JDK 17 and Android SDK Platform 36.

The APK is signed with the machine/runner's Android debug certificate so it can be installed directly for testing. The AAB is also signed with that debug certificate by default and is **not a production Play Store upload**. Configure a private release keystore before publishing. Build outputs are copied to `dist/android/MotionVision.apk` and `dist/android/MotionVision.aab` only after Gradle succeeds.

## Build outputs

| Platform | Build command | Artifact paths |
|---|---|---|
| Windows x64 | `powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1` | `dist/windows/MotionVision.exe`, `MotionVision-Setup.exe` |
| Android | `powershell -ExecutionPolicy Bypass -File scripts/build_android.ps1` | `dist/android/MotionVision.apk`, `MotionVision.aab` |
| Linux x86_64 | `bash scripts/build_linux.sh` on Ubuntu 22.04+ | `dist/linux/MotionVision.AppImage`, `MotionVision.deb` |
| macOS native architecture | `bash scripts/build_macos.sh` on `macos-latest` | `dist/macos/Motion Vision.app`, `MotionVision.dmg` |

Each desktop build script installs its Python dependencies, regenerates icon derivatives, runs the desktop tests and compilation checks, then verifies the named output files. Android runs shared Python checks and its Kotlin unit tests in CI. `scripts/build_all.ps1` builds Windows and Android on a Windows development machine. `scripts/build_all.sh` builds Linux and Android on Linux. The GitHub Actions workflow builds Windows x64, Linux x86_64, macOS on the native `macos-latest` architecture, and Android.

`.github/workflows/build.yml` runs the four release jobs on pushes and pull requests and uploads version/platform/architecture named artifacts. `.github/workflows/release.yml` manually starts the same native build matrix through `workflow_dispatch`.

`dist/` is deliberately ignored and starts empty. A checked-in workflow is a build configuration, not evidence that a hosted build has run. Workflow artifacts only appear after this project is pushed to a GitHub repository and the jobs succeed.

## Tests

The results from the local Windows and Android builds, measured camera run, artifact status, and source inventory are in [the build report](docs/BUILD_REPORT.md).

```powershell
$env:PYTHONPATH = (Resolve-Path core).Path
python -m unittest discover -s tests -v
python -m compileall -q core tests scripts
python -m pip check
```

Desktop tests use synthetic frames/landmarks, local MediaPipe model smoke tests, mocked Telegram transport, and temporary recording files. They do not require Telegram credentials. The Android Gradle build runs `testDebugUnitTest`, including tracking-ID continuity, temporal vote and malformed-landmark tests. Device camera, Android GPU delegates, recording playback, macOS camera permission prompts, and OS window-manager behavior still require testing on the target devices.
