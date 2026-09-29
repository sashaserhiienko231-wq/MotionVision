# Known limitations in Motion Vision 1.0.0

- Android has real CameraX preview/recording and MediaPipe face, hand and pose landmark detection, but does not yet have the desktop SFace embedding database, face enrollment/matching, Telegram bot, gesture HUD dragging, Telegram alerts, or the desktop gesture set's complete motion-based wave/raised-hand logic.
- Android classifies static hand shapes from landmark geometry and applies a short temporal vote. This is not the desktop classifier's complete smoothing, per-person state, cooldown and event history pipeline.
- Android input is converted from YUV to JPEG/Bitmap before MediaPipe inference. This is a compatibility-oriented conversion and costs CPU time; delivered FPS depends on the phone, simultaneous CameraX use cases, lighting and number of people. No Android device was available in this build environment for a measured camera/GPU benchmark.
- Desktop local face embeddings use OpenCV SFace. Face matching can be affected by pose, lighting and image quality and is not a security/access-control feature.
- Desktop face embeddings are stored locally but are not encrypted by the application. Protect the user data folder with the operating system account and disk encryption.
- Telegram send/receive uses the desktop Bot API implementation. Without user-provided credentials only mock tests are run; no live Telegram message was sent.
- macOS build scripts and a `macos-latest` GitHub Actions job are configured, but no artifact is ready until that hosted job actually completes. The local machine is Windows. macOS packages are currently unsigned/not notarized; Gatekeeper may require an explicit user override.
- The macOS job builds only for the architecture reported by `macos-latest`. The CI artifact name records that runner architecture; no separate Intel compatibility job is configured.
- The Android AAB defaults to debug-key signing for build/test convenience and cannot be treated as a production Play upload. A private release signing key must be configured for publishing.
- Windows setup and desktop packages are not code-signed. Ubuntu/Debian packages target x86_64 and require working camera device access for the user running the app.
- Cross-platform CI is configured but cannot upload artifacts until this project is committed to a GitHub repository and the workflow is run. No repository URL/remote was supplied.
