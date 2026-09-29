package com.motionvision.app

import kotlin.math.abs
import kotlin.math.hypot

data class Point2D(val x: Float, val y: Float)
data class FaceBox(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val centerX get() = (left + right) / 2f
    val centerY get() = (top + bottom) / 2f
}
data class TrackedFace(val id: Int, val box: FaceBox)

/** Greedy nearest-center tracker with a short missing-frame grace period. */
class PersonTracker(private val maxMissedFrames: Int = 12) {
    private data class Track(val id: Int, var x: Float, var y: Float, var missed: Int)
    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    fun update(boxes: List<FaceBox>): List<TrackedFace> {
        tracks.forEach { it.missed++ }
        val available = tracks.toMutableSet()
        val matched = mutableListOf<TrackedFace>()
        boxes.forEach { box ->
            val closest = available.minByOrNull { hypot(it.x - box.centerX, it.y - box.centerY) }
            val match = closest?.takeIf {
                hypot(it.x - box.centerX, it.y - box.centerY) < 0.22f
            }
            val track = if (match != null) {
                available.remove(match)
                match.x = box.centerX
                match.y = box.centerY
                match.missed = 0
                match
            } else {
                Track(nextId++, box.centerX, box.centerY, 0).also(tracks::add)
            }
            matched += TrackedFace(track.id, box)
        }
        tracks.removeAll { it.missed > maxMissedFrames }
        return matched
    }
}

/** Finger-shape classifier; uses landmark distances and multi-frame voting. */
object GestureClassifier {
    private val fingerChains = listOf(5 to 8, 9 to 12, 13 to 16, 17 to 20)

    fun classify(points: List<Point2D>): String? {
        if (points.size < 21 || points.any { !it.x.isFinite() || !it.y.isFinite() }) return null
        val wrist = points[0]
        val extended = fingerChains.map { (base, tip) ->
            distance(points[tip], wrist) > distance(points[base], wrist) * 1.22f
        }
        val thumbExtended = distance(points[4], points[5]) > distance(points[3], points[5]) * 1.16f
        val count = extended.count { it }
        if (count == 4 && thumbExtended) return "open_palm"
        if (count == 0 && thumbExtended) {
            if (points[4].y < wrist.y - 0.12f) return "thumbs_up"
            if (points[4].y > wrist.y + 0.12f) return "thumbs_down"
            val dx = points[4].x - points[2].x
            if (abs(dx) > 0.12f) return if (dx < 0) "thumb_side_left" else "thumb_side_right"
        }
        if (count == 0 && !thumbExtended) {
            return "fist"
        }
        if (extended == listOf(true, true, false, false)) return "peace_sign"
        if (extended == listOf(false, true, false, false)) return "middle_finger"
        if (extended == listOf(true, false, false, false)) {
            val dx = points[8].x - wrist.x
            val dy = points[8].y - wrist.y
            return if (abs(dx) > abs(dy)) {
                if (dx < 0) "pointing_left" else "pointing_right"
            } else {
                "pointing_up"
            }
        }
        return null
    }

    private fun distance(a: Point2D, b: Point2D) = hypot(a.x - b.x, a.y - b.y)
}

class GestureVoter(private val windowSize: Int = 5, private val votesRequired: Int = 3) {
    private val history = ArrayDeque<String?>()
    private var stable: String? = null

    fun update(candidate: String?): String? {
        history.addLast(candidate)
        while (history.size > windowSize) history.removeFirst()
        val winner = history.filterNotNull().groupingBy { it }.eachCount().maxByOrNull { it.value }
        if (winner != null && winner.value >= votesRequired) stable = winner.key
        else if (candidate == null && history.count { it == null } >= votesRequired) stable = null
        return stable
    }
}
