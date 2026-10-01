package com.thesis.pickleballserve.landing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core

@RunWith(AndroidJUnit4::class)
class RealtimeTrackerTest {
    private fun frame(index: Int, visible: Boolean = true): Bitmap {
        val bitmap = Bitmap.createBitmap(320, 180, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.rgb(30, 65, 90))
        if (visible) {
            val paint = Paint().apply { color = Color.YELLOW }
            canvas.drawCircle(40f + 3 * index, 80f, 4f, paint)
            paint.color = Color.rgb(170, 180, 10)
            canvas.drawCircle(39f + 3 * index, 79f, 1f, paint)
        }
        return bitmap
    }
    private val detection = YoloRunner.FrameDet(
        YoloRunner.BallDet(40.0, 80.0, 0.95, 34.0, 74.0, 46.0, 86.0), null)

    @Test fun delayedDetectionIsTrackedToCurrentFrameAndStopsWhenBallDisappears() {
        assertTrue(OpenCVLoader.initLocal())
        Core.setNumThreads(1)
        RealtimeTracker().use { tracker ->
            for (i in 0..20) frame(i).let { tracker.advance(i, it); it.recycle() }
            val observations = tracker.correct(0, detection)
            assertEquals(21, observations.size)
            assertEquals(100.0, tracker.currentBall!!.cx, 1.0)
            for (i in 21..30) {
                frame(i).let { tracker.advance(i, it); it.recycle() }
                assertEquals(40.0 + 3 * i, tracker.currentBall!!.cx, 1.0)
            }
            frame(31, visible = false).let { tracker.advance(31, it); it.recycle() }
            assertNull("Missing ball must not be extrapolated as a detection", tracker.currentBall)
        }
    }

    @Test fun detectionOlderThanHistoryIsDiscarded() {
        assertTrue(OpenCVLoader.initLocal())
        RealtimeTracker().use { tracker ->
            for (i in 0..100) frame(i, visible = false).let { tracker.advance(i, it); it.recycle() }
            assertTrue(tracker.correct(0, detection).isEmpty())
            assertNull(tracker.currentBall)
        }
    }
}
