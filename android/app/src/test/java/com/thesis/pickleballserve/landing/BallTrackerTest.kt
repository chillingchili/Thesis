package com.thesis.pickleballserve.landing

import org.junit.Assert.assertTrue
import org.junit.Test

class BallTrackerTest {
    @Test fun historicalCorrectionDoesNotRewindStreamingFiltersOrReuseStaleWorldCoordinates() {
        val tracker = BallTracker(30.0)
        assertTrue(tracker.add(20, 200.0, 100.0, 10.0, 20.0))
        val identity = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        tracker.correct(2, 5.0, 5.0, identity)
        assertTrue(tracker.add(21, 202.0, 105.0, 10.5, 20.5))
        tracker.correct(2, 7.0, 8.0, null)
        assertTrue(tracker.worldX[2].isNaN())
        assertTrue(tracker.worldY[2].isNaN())
    }
    @Test fun missingDetectionsStillExtendTimeline() {
        val tracker = BallTracker(30.0, initialN = 4)
        tracker.advance(12000)
        assertTrue(tracker.ballY.size > 12000)
        assertTrue(tracker.ballY[12000].isNaN())
        assertTrue(tracker.worldX[12000].isNaN())
    }
}
