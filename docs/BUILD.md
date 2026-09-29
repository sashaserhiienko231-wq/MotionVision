# Build notes

## Windows

Requires 64-bit Python 3.11, a writable build directory, and Inno Setup 6. The script installs Python requirements, runs all desktop tests, creates the portable `MotionVision.exe`, checks `--version`, then compiles `MotionVision-Setup.exe`.

## Linux

Use Ubuntu 22.04 or newer x86_64 with Python 3.11, `dpkg-deb`, and `curl`. The script builds a one-file executable, Debian package and AppImage. Linux needs access to the V4L2 camera device (commonly `/dev/video0`) and standard OpenCV GUI libraries. Ubuntu desktop sessions usually grant camera access to the logged-in user; on systems that require group access, add that user to the system's `video` group and sign in again.

## Android

Requires JDK 17, Android SDK Platform 36 and Gradle 8.11.1 (the checked-in wrapper pins the version). The project sets API 29 as minimum, targets API 35 and uses package ID `com.motionvision.app`. Build tools install no emulator by themselves.

## macOS

Requires Python 3.11 with a MediaPipe wheel for the host architecture, Xcode command-line tools and `hdiutil`. The GitHub job uses `macos-latest`; the build script picks that runner's actual architecture, embeds the camera-use text and icon in the app bundle, then creates and verifies a compressed `.dmg`. Signing, Apple Developer provisioning and notarization require developer credentials not included with this project.

## All platforms

Run a platform script on its native OS. `scripts/build_all.ps1` builds Windows and Android locally; `scripts/build_all.sh` builds Linux and Android locally. A Windows/Linux process cannot create a valid macOS `.app` or `.dmg`. Push this root directory to GitHub to run `.github/workflows/build.yml`; successful jobs upload their actual files as version/platform/architecture named workflow artifacts. `.github/workflows/release.yml` manually starts the same native matrix through `workflow_dispatch`.
