package com.motionvision.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.Environment
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: VisionOverlayView
    private lateinit var statusText: TextView
    private lateinit var cameraButton: Button
    private lateinit var recordButton: Button
    private lateinit var modeButton: Button
    private lateinit var videoCapture: VideoCapture<Recorder>
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: VisionPipeline? = null
    @Volatile private var recording: Recording? = null
    @Volatile private var usingFrontCamera = true
    private var fullscreen = false
    @Volatile private var modeIndex = 1
    private var sourceFrames = 0L
    private val modes = listOf("QUALITY", "BALANCED", "PERFORMANCE")
    private val permissionRequest = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) openCamera() else statusText.text = "Camera permission is needed for live detection."
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        cameraExecutor = Executors.newSingleThreadExecutor()
        buildNativeUi()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera()
        } else {
            permissionRequest.launch(Manifest.permission.CAMERA)
        }
    }

    private fun buildNativeUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(11, 18, 32))
        }
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), dp(6), dp(12), dp(6)); setBackgroundColor(Color.rgb(11, 18, 32))
        }
        val brand = TextView(this).apply {
            text = "◉  Motion Vision  ${BuildConfig.VERSION_NAME}"; textSize = 18f; setTextColor(Color.WHITE)
            typeface = android.graphics.Typeface.create("sans-serif-medium", 0)
        }
        header.addView(brand, LinearLayout.LayoutParams(0, dp(48), 1f))
        cameraButton = button("Switch camera") { usingFrontCamera = !usingFrontCamera; bindUseCases() }
        header.addView(cameraButton)
        val fullscreenButton = button("Fullscreen") {
            fullscreen = !fullscreen
            WindowInsetsControllerCompat(window, root).apply {
                if (fullscreen) hide(WindowInsetsCompat.Type.systemBars()) else show(WindowInsetsCompat.Type.systemBars())
                systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        }
        header.addView(fullscreenButton)
        root.addView(header, LinearLayout.LayoutParams(-1, dp(56)))

        val stage = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        overlay = VisionOverlayView(this)
        stage.addView(previewView, FrameLayout.LayoutParams(-1, -1))
        stage.addView(overlay, FrameLayout.LayoutParams(-1, -1))
        statusText = TextView(this).apply {
            text = "Starting camera…"; textSize = 13f; setTextColor(Color.WHITE)
            setPadding(dp(12), dp(6), dp(12), dp(6)); setBackgroundColor(0xA8152033.toInt())
        }
        stage.addView(statusText, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.START).apply {
            setMargins(dp(12), 0, dp(12), dp(12))
        })
        root.addView(stage, LinearLayout.LayoutParams(-1, 0, 1f))

        val footer = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER
            setPadding(dp(8), dp(6), dp(8), dp(8)); setBackgroundColor(Color.rgb(11, 18, 32))
        }
        recordButton = button("●  Record") { toggleRecording() }
        recordButton.isEnabled = false
        modeButton = button("Mode: ${modes[modeIndex]}") {
            modeIndex = (modeIndex + 1) % modes.size
            modeButton.text = "Mode: ${modes[modeIndex]}"
            overlay.snapshot = overlay.snapshot.copy(mode = modes[modeIndex])
        }
        footer.addView(recordButton, LinearLayout.LayoutParams(0, dp(48), 1f))
        footer.addView(modeButton, LinearLayout.LayoutParams(0, dp(48), 1f))
        root.addView(footer, LinearLayout.LayoutParams(-1, dp(64)))
        setContentView(root)
    }

    private fun button(label: String, action: () -> Unit) = Button(this).apply {
        text = label; textSize = 12f; isAllCaps = false; setOnClickListener { action() }
    }

    private fun openCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                cameraProvider = future.get()
                bindUseCases()
            } catch (error: Exception) {
                statusText.text = "Camera unavailable: ${error.javaClass.simpleName}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindUseCases() {
        val provider = cameraProvider ?: return
        val selector = if (usingFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
        try {
            val chosen = if (provider.hasCamera(selector)) selector else {
                usingFrontCamera = !usingFrontCamera
                statusText.text = "Requested lens unavailable; using the other camera."
                if (usingFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            }
            if (!provider.hasCamera(chosen)) {
                statusText.text = "No camera is available on this device."; return
            }
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val resolution = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(
                        android.util.Size(1280, 720),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER,
                    ),
                )
                .build()
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(resolution)
                .build()
            val recorder = Recorder.Builder()
                .setQualitySelector(QualitySelector.from(Quality.HD, FallbackStrategy.lowerQualityOrHigherThan(Quality.SD)))
                .build()
            videoCapture = VideoCapture.withOutput(recorder)
            analysis.setAnalyzer(cameraExecutor) { image ->
                sourceFrames++
                val skip = when (modeIndex) { 0 -> 0L; 1 -> 0L; else -> 1L }
                if (skip > 0 && sourceFrames % (skip + 1) != 0L) { image.close(); return@setAnalyzer }
                try {
                    if (detector == null) detector = VisionPipeline(applicationContext)
                    val result = detector?.process(image)
                    if (result != null) runOnUiThread {
                        overlay.mirrored = usingFrontCamera
                        overlay.snapshot = result.copy(mode = modes[modeIndex], recording = recording != null)
                        statusText.text = "${detector?.delegateName ?: "CPU"} · ${result.width}×${result.height} · local processing"
                    }
                } catch (error: Exception) {
                    runOnUiThread { statusText.text = "Vision pipeline: ${error.javaClass.simpleName}" }
                } finally {
                    image.close()
                }
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, chosen, preview, analysis, videoCapture)
            runOnUiThread { recordButton.isEnabled = true }
        } catch (error: Exception) {
            statusText.text = "Camera setup failed: ${error.javaClass.simpleName}"
        }
    }

    private fun toggleRecording() {
        val active = recording
        if (active != null) {
            active.stop(); recording = null
            recordButton.text = "●  Record"
            overlay.snapshot = overlay.snapshot.copy(recording = false)
            return
        }
        val directory = File(getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir, "recordings")
        if (!directory.exists() && !directory.mkdirs()) {
            statusText.text = "Cannot create local recording folder."; return
        }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(directory, "MotionVision_$stamp.mp4")
        recording = videoCapture.output.prepareRecording(this, FileOutputOptions.Builder(file).build())
            .start(ContextCompat.getMainExecutor(this)) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> {
                        recordButton.text = "■  Stop recording"
                        statusText.text = "Recording locally · ${file.name}"
                    }
                    is VideoRecordEvent.Finalize -> {
                        recording = null
                        recordButton.text = "●  Record"
                        statusText.text = if (event.hasError()) "Recording failed (${event.error})." else "Saved local video: ${file.name}"
                    }
                }
            }
        overlay.snapshot = overlay.snapshot.copy(recording = true)
    }

    override fun onDestroy() {
        recording?.stop(); recording = null
        cameraExecutor.execute { detector?.close(); detector = null; cameraExecutor.shutdown() }
        super.onDestroy()
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
}
