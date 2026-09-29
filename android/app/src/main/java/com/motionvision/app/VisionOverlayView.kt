package com.motionvision.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

data class OverlayHand(val side: String, val gesture: String?, val points: List<Point2D>)
data class MotionSnapshot(
    val width: Int = 1,
    val height: Int = 1,
    val faces: List<TrackedFace> = emptyList(),
    val hands: List<OverlayHand> = emptyList(),
    val poses: List<List<Point2D>> = emptyList(),
    val cameraFps: Float = 0f,
    val detectionFps: Float = 0f,
    val inferenceMs: Float = 0f,
    val frameNumber: Long = 0,
    val mode: String = "BALANCED",
    val recording: Boolean = false,
)

class VisionOverlayView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    @Volatile var snapshot = MotionSnapshot()
        set(value) { field = value; postInvalidateOnAnimation() }
    var mirrored = true
    private val outline = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(72, 223, 205); style = Paint.Style.STROKE; strokeWidth = dp(2f)
    }
    private val landmark = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(115, 167, 255); style = Paint.Style.FILL
    }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = dp(13f); typeface = android.graphics.Typeface.create("sans-serif-medium", 0)
    }
    private val card = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xB9152033.toInt() }
    private val poseLinks = listOf(
        11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16, 11 to 23, 12 to 24,
        23 to 24, 23 to 25, 25 to 27, 27 to 29, 29 to 31, 24 to 26, 26 to 28,
        28 to 30, 30 to 32, 27 to 31, 28 to 32,
    )
    private val handLinks = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8,
        5 to 9, 9 to 10, 10 to 11, 11 to 12, 9 to 13, 13 to 14, 14 to 15,
        15 to 16, 13 to 17, 17 to 18, 18 to 19, 19 to 20, 0 to 17,
    )

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val s = snapshot
        if (width <= 0 || height <= 0 || s.width <= 0 || s.height <= 0) return
        val scale = maxOf(width.toFloat() / s.width, height.toFloat() / s.height)
        val offsetX = (width - s.width * scale) / 2f
        val offsetY = (height - s.height * scale) / 2f
        fun screen(point: Point2D): Point2D {
            val x = if (mirrored) 1f - point.x else point.x
            return Point2D(offsetX + x * s.width * scale, offsetY + point.y * s.height * scale)
        }
        outline.color = Color.rgb(72, 223, 205)
        s.poses.forEach { pose ->
            poseLinks.forEach { (a, b) -> if (a < pose.size && b < pose.size) {
                val p = screen(pose[a]); val q = screen(pose[b]); canvas.drawLine(p.x, p.y, q.x, q.y, outline)
            } }
        }
        s.faces.forEach { face ->
            val left = if (mirrored) 1f - face.box.right else face.box.left
            val right = if (mirrored) 1f - face.box.left else face.box.right
            val rect = RectF(offsetX + left * s.width * scale, offsetY + face.box.top * s.height * scale,
                offsetX + right * s.width * scale, offsetY + face.box.bottom * s.height * scale)
            canvas.drawRoundRect(rect, dp(10f), dp(10f), outline)
            canvas.drawText("PERSON ${face.id}", rect.left, rect.top - dp(7f), text)
        }
        s.hands.forEach { hand ->
            outline.color = if (hand.side.equals("left", true)) Color.rgb(72, 223, 205) else Color.rgb(255, 150, 118)
            handLinks.forEach { (a, b) -> if (a < hand.points.size && b < hand.points.size) {
                val p = screen(hand.points[a]); val q = screen(hand.points[b]); canvas.drawLine(p.x, p.y, q.x, q.y, outline)
            } }
            hand.points.forEach { p -> val at = screen(p); canvas.drawCircle(at.x, at.y, dp(2.2f), landmark) }
            hand.points.firstOrNull()?.let { p ->
                val at = screen(p); canvas.drawText("${hand.side.uppercase()} · ${hand.gesture ?: "hand"}", at.x + dp(8f), at.y - dp(8f), text)
            }
        }
        val rows = listOf(
            "Motion Vision  ${BuildConfig.VERSION_NAME}  ·  ${s.mode}",
            "Camera ${s.cameraFps.toInt()} FPS   Detection ${s.detectionFps.toInt()} FPS",
            "Infer ${"%.1f".format(s.inferenceMs)} ms   ${s.width} × ${s.height}   Faces ${s.faces.size}   Hands ${s.hands.size}",
            if (s.recording) "● RECORDING" else "R  Start local recording",
        )
        val x = dp(12f); val y = dp(16f); val line = dp(20f)
        val cardWidth = minOf(width - x * 2, dp(310f))
        canvas.drawRoundRect(RectF(x, y, x + cardWidth, y + line * rows.size + dp(12f)), dp(10f), dp(10f), card)
        rows.forEachIndexed { index, row -> canvas.drawText(row, x + dp(11f), y + dp(19f) + index * line, text) }
    }

    private fun dp(value: Float) = value * resources.displayMetrics.density
}
