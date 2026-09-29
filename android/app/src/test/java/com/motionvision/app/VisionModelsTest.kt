package com.motionvision.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class VisionModelsTest {
    @Test fun personIdsSurviveDetectionOrderChangesAndShortGaps() {
        val tracker = PersonTracker()
        val first = tracker.update(listOf(FaceBox(0.1f, 0.1f, 0.2f, 0.2f), FaceBox(0.7f, 0.1f, 0.8f, 0.2f)))
        val second = tracker.update(listOf(FaceBox(0.71f, 0.1f, 0.81f, 0.2f), FaceBox(0.11f, 0.1f, 0.21f, 0.2f)))
        assertEquals(first[1].id, second[0].id)
        assertEquals(first[0].id, second[1].id)
        tracker.update(emptyList())
        assertEquals(first[0].id, tracker.update(listOf(FaceBox(0.12f, 0.1f, 0.22f, 0.2f))).single().id)
    }

    @Test fun gestureVoterFiltersSingleFrameNoiseAndDebouncesStableState() {
        val voter = GestureVoter()
        assertNull(voter.update("fist"))
        assertNull(voter.update("open_palm"))
        assertNull(voter.update("open_palm"))
        assertEquals("open_palm", voter.update("open_palm"))
        assertEquals("open_palm", voter.update("fist"))
        assertEquals("open_palm", voter.update("fist"))
        assertEquals("fist", voter.update("fist"))
    }

    @Test fun geometricClassifierRecognizesAnOpenPalm() {
        val points = MutableList(21) { Point2D(0.5f, 0.6f) }
        points[0] = Point2D(0.5f, 0.82f)
        points[1] = Point2D(0.46f, 0.72f); points[2] = Point2D(0.40f, 0.62f)
        points[3] = Point2D(0.34f, 0.53f); points[4] = Point2D(0.25f, 0.43f)
        points[5] = Point2D(0.41f, 0.55f); points[6] = Point2D(0.37f, 0.37f)
        points[7] = Point2D(0.35f, 0.26f); points[8] = Point2D(0.34f, 0.15f)
        points[9] = Point2D(0.48f, 0.52f); points[12] = Point2D(0.48f, 0.09f)
        points[13] = Point2D(0.55f, 0.54f); points[16] = Point2D(0.58f, 0.17f)
        points[17] = Point2D(0.61f, 0.59f); points[20] = Point2D(0.68f, 0.27f)
        assertEquals("open_palm", GestureClassifier.classify(points))
    }

    @Test fun malformedLandmarksAreIgnored() {
        assertNull(GestureClassifier.classify(emptyList()))
        assertNull(GestureClassifier.classify(List(20) { Point2D(0f, 0f) }))
    }
}
