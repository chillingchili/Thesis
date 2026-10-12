package com.thesis.pickleballserve

import org.junit.Assert.*
import org.junit.Test

class ServePresentationTest {
    @Test fun uncertainPredictionIsStillIdentifiedWithoutClaimingConfidence() {
        assertTrue(ServePresentation.classification("drive", .5999f).contains("uncertain", true))
        assertTrue(ServePresentation.classification("drive", .5999f).contains("Drive"))
        assertEquals("Drive", ServePresentation.classification("drive", .60f))
    }
    @Test fun invalidClassificationNeverLooksReliable() {
        for (value in listOf(Float.NaN, Float.POSITIVE_INFINITY, -.1f, 1.1f))
            assertEquals("Unavailable", ServePresentation.classification("lob", value))
        assertEquals("Unavailable", ServePresentation.classification("unknown", .9f))
    }
    @Test fun missingMetricsCannotBecomePositiveResults() {
        assertEquals("Unavailable", ServePresentation.hip(Float.NaN, true))
        assertEquals("Unavailable", ServePresentation.hip(.2f, null))
        assertEquals("Unavailable", ServePresentation.paddle(null, FeedbackEngine.PaddleState.OPTIMAL))
        assertEquals("Unavailable", ServePresentation.paddle(180.0, FeedbackEngine.PaddleState.OPTIMAL))
        assertTrue(ServePresentation.hip(.12f, false).contains("Below reference"))
        assertTrue(ServePresentation.paddle(45.0, FeedbackEngine.PaddleState.OUTSIDE_BASELINE).contains("Outside reference"))
    }
    @Test fun resetInvalidatesOldResultsButAcceptsNewSession() {
        val session = SessionEpoch()
        val old = session.current
        val current = session.invalidate()
        assertFalse(session.accepts(old))
        assertTrue(session.accepts(current))
        session.invalidate()
        assertFalse(session.accepts(current))
    }
}
