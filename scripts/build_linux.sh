#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="$ROOT/dist/linux"
BUILD="$ROOT/build/linux"
mkdir -p "$OUT" "$BUILD"
cd "$ROOT"
ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" ]]; then
  echo "Linux packaging currently targets x86_64; found $ARCH" >&2
  exit 2
fi
VERSION="$("$PYTHON" -c 'import json; print(json.load(open("config/version.json", encoding="utf-8"))["version"])')"
"$PYTHON" -m pip install -r requirements-build.txt
"$PYTHON" scripts/generate_icons.py
PYTHONPATH="$ROOT/core" "$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m compileall -q core tests scripts
"$PYTHON" -m pip check
"$PYTHON" -m PyInstaller --noconfirm --clean --onefile --windowed \
  --name MotionVision --paths core --collect-all mediapipe --collect-all cv2 \
  --hidden-import mediapipe.tasks.python.vision.face_landmarker \
  --hidden-import mediapipe.tasks.python.vision.hand_landmarker \
  --hidden-import mediapipe.tasks.python.vision.pose_landmarker \
  --add-data "models:models" --add-data "config/version.json:config" \
  --distpath "$OUT" --workpath "$BUILD" core/app.py

APPDIR="$BUILD/AppDir"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor"
cp "$OUT/MotionVision" "$APPDIR/usr/bin/MotionVision"
sed "s/^X-AppVersion=.*/X-AppVersion=$VERSION/" linux/MotionVision.desktop > "$BUILD/com.motionvision.app.desktop"
cp "$BUILD/com.motionvision.app.desktop" "$APPDIR/usr/share/applications/com.motionvision.app.desktop"
cp -R assets/platform/linux/hicolor/. "$APPDIR/usr/share/icons/hicolor/"
cp "$APPDIR/usr/share/applications/com.motionvision.app.desktop" "$APPDIR/com.motionvision.app.desktop"
cp assets/platform/linux/hicolor/512x512/apps/com.motionvision.app.png "$APPDIR/com.motionvision.app.png"
printf '#!/bin/sh\nHERE="$(dirname "$(readlink -f "$0")")"\nexec "$HERE/usr/bin/MotionVision" "$@"\n' > "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun" "$APPDIR/usr/bin/MotionVision"

DEBROOT="$BUILD/debroot"
mkdir -p "$DEBROOT/opt/motionvision" "$DEBROOT/usr/share/applications" "$DEBROOT/usr/share/icons/hicolor" "$DEBROOT/usr/share/doc/motion-vision" "$DEBROOT/DEBIAN"
cp "$OUT/MotionVision" "$DEBROOT/opt/motionvision/MotionVision"
sed -e 's/^Exec=MotionVision$/Exec=motionvision/' -e "s/^X-AppVersion=.*/X-AppVersion=$VERSION/" linux/MotionVision.desktop > "$DEBROOT/usr/share/applications/com.motionvision.app.desktop"
cp -R assets/platform/linux/hicolor/. "$DEBROOT/usr/share/icons/hicolor/"
test -s "$DEBROOT/usr/share/icons/hicolor/512x512/apps/com.motionvision.app.png"
cp config/.env.example "$DEBROOT/usr/share/doc/motion-vision/.env.example"
printf '#!/bin/sh\nexec /opt/motionvision/MotionVision "$@"\n' > "$DEBROOT/usr/bin/motionvision"
chmod +x "$DEBROOT/usr/bin/motionvision"
cat > "$DEBROOT/DEBIAN/control" <<EOF
Package: motion-vision
Version: $VERSION
Section: video
Priority: optional
Architecture: amd64
Maintainer: Motion Vision
Description: Local camera face, hand, pose and gesture tracking
Depends: libgl1, libglib2.0-0, libx11-6, libxcb1
EOF
dpkg-deb --build --root-owner-group "$DEBROOT" "$OUT/MotionVision.deb"

if [[ -z "${APPIMAGETOOL:-}" ]]; then
  APPIMAGETOOL="$BUILD/appimagetool-x86_64.AppImage"
  curl -fL --retry 3 -o "$APPIMAGETOOL" https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$APPIMAGETOOL"
fi
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$OUT/MotionVision.AppImage"
chmod +x "$OUT/MotionVision.AppImage"
test -s "$OUT/MotionVision" && test -s "$OUT/MotionVision.deb" && test -s "$OUT/MotionVision.AppImage"
dpkg-deb --field "$OUT/MotionVision.deb" Version Architecture
dpkg-deb --contents "$OUT/MotionVision.deb" | grep 'com.motionvision.app.png' >/dev/null
stat -c 'ARTIFACT %n %s bytes' "$OUT/MotionVision" "$OUT/MotionVision.deb" "$OUT/MotionVision.AppImage"
