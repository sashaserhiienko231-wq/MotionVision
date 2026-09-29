#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="$ROOT/dist/macos"
BUILD="$ROOT/build/macos"
mkdir -p "$OUT" "$BUILD"
cd "$ROOT"
VERSION="$("$PYTHON" -c 'import json; print(json.load(open("config/version.json", encoding="utf-8"))["version"])')"
"$PYTHON" -m pip install -r requirements-build.txt
"$PYTHON" scripts/generate_icons.py
PYTHONPATH="$ROOT/core" "$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m compileall -q core tests scripts
"$PYTHON" -m pip check
ARCH="$(uname -m)"
case "$ARCH" in arm64|x86_64) ;; *) echo "Unsupported macOS architecture: $ARCH" >&2; exit 2 ;; esac
"$PYTHON" -m PyInstaller --noconfirm --clean --windowed \
  --name "Motion Vision" --osx-bundle-identifier com.motionvision.app \
  --target-architecture "$ARCH" --icon assets/icon/MotionVision.icns --paths core \
  --collect-all mediapipe --collect-all cv2 \
  --hidden-import mediapipe.tasks.python.vision.face_landmarker \
  --hidden-import mediapipe.tasks.python.vision.hand_landmarker \
  --hidden-import mediapipe.tasks.python.vision.pose_landmarker \
  --add-data "models:models" --add-data "config/version.json:config" \
  --distpath "$OUT" --workpath "$BUILD" core/app.py
APP="$OUT/Motion Vision.app"
PLIST="$APP/Contents/Info.plist"
ICON_IN_BUNDLE="$(find "$APP/Contents/Resources" -maxdepth 1 -type f -iname '*.icns' -size +0c -print -quit)"
test -n "$ICON_IN_BUNDLE"
set_plist_string() {
  key="$1"
  value="$2"
  if /usr/libexec/PlistBuddy -c "Print :$key" "$PLIST" >/dev/null 2>&1; then
    /usr/libexec/PlistBuddy -c "Set :$key \"$value\"" "$PLIST"
  else
    /usr/libexec/PlistBuddy -c "Add :$key string \"$value\"" "$PLIST"
  fi
}
verify_plist_string() {
  key="$1"
  expected="$2"
  actual="$(/usr/libexec/PlistBuddy -c "Print :$key" "$PLIST")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Unexpected Info.plist value for $key: $actual" >&2
    return 1
  fi
}
set_plist_string CFBundleName "Motion Vision"
set_plist_string CFBundleShortVersionString "$VERSION"
set_plist_string CFBundleVersion "$VERSION"
set_plist_string CFBundleIdentifier "com.motionvision.app"
set_plist_string NSCameraUsageDescription "Motion Vision processes camera frames locally to detect faces, hands, poses and gestures."
set_plist_string CFBundleIconFile "$(basename "$ICON_IN_BUNDLE")"
plutil -lint "$PLIST"
verify_plist_string CFBundleName "Motion Vision"
verify_plist_string CFBundleShortVersionString "$VERSION"
verify_plist_string CFBundleVersion "$VERSION"
verify_plist_string CFBundleIdentifier "com.motionvision.app"
verify_plist_string NSCameraUsageDescription "Motion Vision processes camera frames locally to detect faces, hands, poses and gestures."
verify_plist_string CFBundleIconFile "$(basename "$ICON_IN_BUNDLE")"
hdiutil create -volname "Motion Vision" -srcfolder "$APP" -ov -format UDZO "$OUT/MotionVision.dmg"
test -d "$APP" && test -s "$OUT/MotionVision.dmg"
hdiutil verify "$OUT/MotionVision.dmg"
du -sh "$APP" "$OUT/MotionVision.dmg"
echo "ARCHITECTURE $ARCH"
