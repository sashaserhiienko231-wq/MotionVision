# Architecture and data flow

## Desktop

`core/app.py` owns the capture/render loop and composes `camera.py`, `actions.py`, `face_identity.py`, `sface.py`, `hud_drag.py`, `windowing.py`, and `telegram_bot.py`. MediaPipe Tasks and SFace work on local frames. The desktop video file is written from captured camera frames without HUD drawing. Telegram configuration is optional and secrets are loaded from `.env` or the process environment.

For frozen builds, `paths.py` locates immutable models in the PyInstaller bundle and puts mutable settings, recordings and face data under the current user's application data directory. Bundled task models are not downloaded again.

## Android

`MainActivity` uses CameraX Preview, ImageAnalysis and VideoCapture use cases. The analyzer runs on a dedicated single-thread executor with `STRATEGY_KEEP_ONLY_LATEST`. `VisionPipeline` runs the face, hand and pose Tasks on the rotated camera bitmap, tries the MediaPipe GPU delegate at startup, and retries CPU if task initialization fails. `VisionOverlayView` draws camera-space boxes, hand/body landmarks and observed FPS/resolution. `PersonTracker` associates current face boxes with nearby recent centers; it is lightweight camera tracking, not identity recognition.

The Android manifest declares only `CAMERA`. Model inference is local and the app has no network permission or Telegram client in this release.

## Build and artifact checks

Build scripts stop on the first failed test, compilation, dependency check or packaging command. They check output files and print their byte sizes. macOS is built only by `build_macos.sh` on an actual macOS host; the GitHub workflow assigns Apple-silicon macOS runners and has a separate optional Intel compatibility job.
