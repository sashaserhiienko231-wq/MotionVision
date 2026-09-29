# MediaPipe task models and CameraX recorder use reflection in some runtime paths.
-keep class com.google.mediapipe.** { *; }
-keep class androidx.camera.** { *; }
