package com.motionvision.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.ByteArrayOutputStream
import kotlin.math.max

class VisionPipeline(context: Context) : AutoCloseable {
    private data class Detectors(val hand: HandLandmarker, val face: FaceLandmarker, val pose: PoseLandmarker)
    private val hand: HandLandmarker
    private val face: FaceLandmarker
    private val pose: PoseLandmarker
    private val tracker = PersonTracker()
    private val voters = mutableMapOf<String, GestureVoter>()
    private var lastTimestamp = 0L
    private var cameraCount = 0
    private var detectCount = 0
    private var rateWindow = SystemClock.elapsedRealtime()
    private var cameraFps = 0f
    private var detectionFps = 0f
    private var frameNumber = 0L
    var delegateName: String = "CPU"
        private set

    private var lastGpuDetectors: Detectors? = null

    init {
        val detectors = try {
            createDetectors(context, true)
        } catch (_: Exception) {
            createDetectors(context, false)
        } catch (_: LinkageError) {
            createDetectors(context, false)
        }
        hand = detectors.hand
        face = detectors.face
        pose = detectors.pose
        delegateName = if (detectors === lastGpuDetectors) "GPU" else "CPU"
    }

    private fun createDetectors(context: Context, gpu: Boolean): Detectors {
        var handTask: HandLandmarker? = null
        var faceTask: FaceLandmarker? = null
        try {
            handTask = createHand(context, gpu)
            faceTask = createFace(context, gpu)
            val poseTask = createPose(context, gpu)
            return Detectors(handTask, faceTask, poseTask).also {
                if (gpu) lastGpuDetectors = it
            }
        } catch (error: Throwable) {
            handTask?.close()
            faceTask?.close()
            throw error
        }
    }

    private fun options(model: String, gpu: Boolean) = BaseOptions.builder()
        .setModelAssetPath(model)
        .apply { if (gpu) setDelegate(Delegate.GPU) }
        .build()

    private fun createHand(context: Context, gpu: Boolean) = HandLandmarker.createFromOptions(
        context,
        HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(options("hand_landmarker.task", gpu))
            .setRunningMode(RunningMode.VIDEO).setNumHands(8)
            .setMinHandDetectionConfidence(0.5f).setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f).build(),
    )

    private fun createFace(context: Context, gpu: Boolean) = FaceLandmarker.createFromOptions(
        context,
        FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(options("face_landmarker.task", gpu))
            .setRunningMode(RunningMode.VIDEO).setNumFaces(8)
            .setMinFaceDetectionConfidence(0.5f).setMinFacePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f).build(),
    )

    private fun createPose(context: Context, gpu: Boolean) = PoseLandmarker.createFromOptions(
        context,
        PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(options("pose_landmarker_lite.task", gpu))
            .setRunningMode(RunningMode.VIDEO).setNumPoses(4)
            .setMinPoseDetectionConfidence(0.5f).setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f).build(),
    )

    fun process(image: ImageProxy): MotionSnapshot? {
        cameraCount++
        updateRates()
        val bitmap = imageToBitmap(image) ?: return null
        val started = SystemClock.elapsedRealtimeNanos()
        val timestamp = max(lastTimestamp + 1, SystemClock.elapsedRealtime())
        lastTimestamp = timestamp
        val mpImage = BitmapImageBuilder(bitmap).build()
        try {
            val faces = face.detectForVideo(mpImage, timestamp).faceLandmarks().mapNotNull { points ->
                if (points.size < 4) return@mapNotNull null
                val xs = points.map { it.x() }; val ys = points.map { it.y() }
                FaceBox(xs.min(), ys.min(), xs.max(), ys.max())
            }
            val tracked = tracker.update(faces)
            val hands = hand.detectForVideo(mpImage, timestamp).let { result ->
                result.landmarks().mapIndexed { index, points ->
                    val side = result.handedness().getOrNull(index)?.firstOrNull()?.categoryName() ?: "Hand"
                    val key = side.lowercase()
                    val points2d = points.map { Point2D(it.x(), it.y()) }
                    val candidate = GestureClassifier.classify(points2d)
                    val gesture = voters.getOrPut(key) { GestureVoter() }.update(candidate)
                    OverlayHand(side, gesture, points2d)
                }
            }
            val poses = pose.detectForVideo(mpImage, timestamp).landmarks().map { points ->
                points.map { Point2D(it.x(), it.y()) }
            }
            detectCount++
            frameNumber++
            updateRates()
            return MotionSnapshot(
                width = bitmap.width, height = bitmap.height, faces = tracked, hands = hands, poses = poses,
                cameraFps = cameraFps, detectionFps = detectionFps,
                inferenceMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000f,
                frameNumber = frameNumber,
            )
        } finally {
            mpImage.close()
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    private fun updateRates() {
        val now = SystemClock.elapsedRealtime()
        val elapsed = now - rateWindow
        if (elapsed >= 1000) {
            cameraFps = cameraCount * 1000f / elapsed
            detectionFps = detectCount * 1000f / elapsed
            cameraCount = 0; detectCount = 0; rateWindow = now
        }
    }

    private fun imageToBitmap(image: ImageProxy): Bitmap? {
        val width = image.width; val height = image.height
        if (image.planes.size < 3) return null
        val y = image.planes[0]; val u = image.planes[1]; val v = image.planes[2]
        val nv21 = ByteArray(width * height + width * height / 2)
        val yBuffer = y.buffer.duplicate(); val uBuffer = u.buffer.duplicate(); val vBuffer = v.buffer.duplicate()
        var output = 0
        for (row in 0 until height) {
            val rowStart = row * y.rowStride
            for (column in 0 until width) nv21[output++] = yBuffer.get(rowStart + column * y.pixelStride)
        }
        val chromaHeight = height / 2; val chromaWidth = width / 2
        for (row in 0 until chromaHeight) {
            val uRow = row * u.rowStride; val vRow = row * v.rowStride
            for (column in 0 until chromaWidth) {
                val atU = uRow + column * u.pixelStride; val atV = vRow + column * v.pixelStride
                nv21[output++] = vBuffer.get(atV); nv21[output++] = uBuffer.get(atU)
            }
        }
        val bytes = ByteArrayOutputStream()
        YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
            .compressToJpeg(Rect(0, 0, width, height), 78, bytes)
        val decoded = BitmapFactory.decodeByteArray(bytes.toByteArray(), 0, bytes.size()) ?: return null
        val degrees = image.imageInfo.rotationDegrees
        if (degrees == 0) return decoded
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        val rotated = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
        if (rotated !== decoded) decoded.recycle()
        return rotated
    }

    override fun close() {
        hand.close(); face.close(); pose.close()
    }
}
