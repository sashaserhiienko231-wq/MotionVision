#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
if [[ -z "$SDK" || ! -d "$SDK/platforms/android-36" ]]; then
  echo "Android SDK platform 36 is required (set ANDROID_SDK_ROOT)." >&2
  exit 2
fi
export ANDROID_SDK_ROOT="$SDK" ANDROID_HOME="$SDK"
OUT="$ROOT/dist/android"
mkdir -p "$OUT"
cd "$ROOT/android"
printf 'sdk.dir=%s\n' "$SDK" > local.properties
if [[ ! -x ./gradlew ]]; then gradle wrapper --gradle-version 8.11.1 --distribution-type bin; fi
./gradlew --no-daemon testDebugUnitTest lintVitalRelease assembleRelease bundleRelease
cp app/build/outputs/apk/release/app-release.apk "$OUT/MotionVision.apk"
cp app/build/outputs/bundle/release/app-release.aab "$OUT/MotionVision.aab"
test -s "$OUT/MotionVision.apk" && test -s "$OUT/MotionVision.aab"
TOOLS="$(find "$SDK/build-tools" -mindepth 1 -maxdepth 1 -type d -name '36.*' | sort -V | tail -n 1)"
test -n "$TOOLS" && test -x "$TOOLS/apksigner" && test -x "$TOOLS/aapt"
"$TOOLS/apksigner" verify "$OUT/MotionVision.apk"
BADGING="$("$TOOLS/aapt" dump badging "$OUT/MotionVision.apk")"
[[ "$BADGING" == *"name='com.motionvision.app'"* ]]
[[ "$BADGING" == *application-icon* ]]
stat -c 'ARTIFACT %n %s bytes' "$OUT/MotionVision.apk" "$OUT/MotionVision.aab" 2>/dev/null || ls -lh "$OUT/MotionVision.apk" "$OUT/MotionVision.aab"
