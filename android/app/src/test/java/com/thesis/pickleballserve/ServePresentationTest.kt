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
    @Test fun serveTypeCardMatchesFigureStates() {
        val sure = ServePresentation.serveTypeCard("topspin", .82f, floatArrayOf(.1f, .08f, .82f))
        assertEquals("Topspin", sure.value)
        assertEquals("82% confident", sure.detail)
        assertTrue(sure.reliable)
        val unsure = ServePresentation.serveTypeCard("lob", .54f, floatArrayOf(.31f, .54f, .15f))
        assertEquals("Not sure", unsure.value)
        assertEquals("Lob 54% · Drive 31% · Topspin 15%", unsure.detail)
        assertFalse(unsure.reliable)
        assertFalse(ServePresentation.serveTypeCard(null, null, null).reliable)
        assertFalse(ServePresentation.serveTypeCard("drive", Float.NaN, null).reliable)
    }
    @Test fun metricCardsNeverInventAPass() {
        assertEquals("Not measured", ServePresentation.hipCard(null, true).value)
        assertNull(ServePresentation.hipCard(Float.NaN, true).chip)
        assertEquals("Needs work", ServePresentation.hipCard(.1f, false).chip)
        assertEquals("Within range", ServePresentation.hipCard(.3f, true).chip)
        val hidden = ServePresentation.paddleCard(null, FeedbackEngine.PaddleState.OPTIMAL, 13.4, 165.7)
        assertEquals("Not detected", hidden.value)
        assertEquals(ServePresentation.Tone.WARN, hidden.tone)
        val out = ServePresentation.paddleCard(171.0, FeedbackEngine.PaddleState.OUTSIDE_BASELINE, 13.4, 165.7)
        assertEquals("171°", out.value)
        assertEquals("Outside reference", out.chip)
        assertTrue(out.note.contains("13–166°"))
    }
    @Test fun partialWhenAnythingIsMissing() {
        val sure = ServePresentation.serveTypeCard("drive", .8f, null)
        val unsure = ServePresentation.serveTypeCard("drive", .5f, null)
        val hip = ServePresentation.hipCard(.2f, true)
        val noHip = ServePresentation.hipCard(null, null)
        val paddle = ServePresentation.paddleCard(90.0, FeedbackEngine.PaddleState.OPTIMAL, 13.4, 165.7)
        val noPaddle = ServePresentation.paddleCard(null, null, 13.4, 165.7)
        assertFalse(ServePresentation.isPartial(sure, hip, paddle))
        assertTrue(ServePresentation.isPartial(unsure, hip, paddle))
        assertTrue(ServePresentation.isPartial(sure, noHip, paddle))
        assertTrue(ServePresentation.isPartial(sure, hip, noPaddle))
        assertEquals("Serve 8 · partial result", ServePresentation.resultStatus(8, true, 900))
        assertEquals("Serve 7 · result ready", ServePresentation.resultStatus(7, false, null))
        assertTrue(ServePresentation.resultStatus(7, false, 1600).startsWith("Serve 7 · analyzed in 1"))
        assertEquals("• A\n• B", ServePresentation.coachingText(listOf("A", " ", "B")))
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
