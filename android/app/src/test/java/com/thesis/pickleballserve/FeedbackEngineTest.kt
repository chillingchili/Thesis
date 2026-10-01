package com.thesis.pickleballserve

import java.io.File
import org.junit.Assert.*
import org.junit.Test

class FeedbackEngineTest {
    private val engine = FeedbackEngine.fromJson(File("src/main/assets/feedback_rules.json").readText())
    private val good = FeedbackEngine.Input("drive", 0.8f, true, FeedbackEngine.PaddleState.OPTIMAL)

    @Test fun confidenceGateWithholdsAllCuesEvenWhenBothStreamsFailTheirThresholds() {
        val result = engine.evaluate(good.copy(gruConfidence = 0.5999f, shiftSufficient = false,
            paddle = FeedbackEngine.PaddleState.OUTSIDE_BASELINE))
        assertEquals(FeedbackEngine.Status.LOW_CONFIDENCE, result.status)
        assertTrue(result.ruleIds.isEmpty())
        assertFalse(result.text.contains("Transfer your weight"))
    }

    @Test fun exactConfidenceBoundaryIsAccepted() {
        assertEquals(FeedbackEngine.Status.POSITIVE, engine.evaluate(good.copy(gruConfidence = 0.60f)).status)
    }

    @Test fun missingOrInvalidClassificationNeverReusesAPreviousResult() {
        engine.evaluate(good)
        for (confidence in listOf(null, Float.NaN, Float.POSITIVE_INFINITY, -0.1f, 1.1f)) {
            val result = engine.evaluate(good.copy(gruConfidence = confidence))
            assertEquals(FeedbackEngine.Status.INSUFFICIENT_DATA, result.status)
            assertTrue(result.ruleIds.isEmpty())
        }
        assertEquals(FeedbackEngine.Status.INSUFFICIENT_DATA, engine.evaluate(good.copy(serveType = null)).status)
        assertEquals(FeedbackEngine.Status.INSUFFICIENT_DATA, engine.evaluate(good.copy(serveType = "slice")).status)
    }

    @Test fun missingPaddlePreservesWeightCueAndNamesMissingSignal() {
        val result = engine.evaluate(good.copy(shiftSufficient = false, paddle = null))
        assertEquals(listOf("C4_WEIGHT_TRANSFER"), result.ruleIds)
        assertEquals(listOf("Paddle orientation unavailable."), result.notifications)
    }

    @Test fun missingWeightPreservesPaddleCueWithoutPositiveConfirmation() {
        val result = engine.evaluate(good.copy(shiftSufficient = null, paddle = FeedbackEngine.PaddleState.OUTSIDE_BASELINE))
        assertEquals(listOf("C6_PADDLE_REVIEW"), result.ruleIds)
        assertTrue(result.notifications.contains("Weight shift unavailable."))
        assertEquals(FeedbackEngine.Status.PARTIAL, engine.evaluate(good.copy(shiftSufficient = null)).status)
    }

    @Test fun matchingRulesComposeInStableOrder() {
        for (subtype in listOf("drive", "lob", "topspin")) {
            val input = good.copy(serveType = subtype, shiftSufficient = false,
                paddle = FeedbackEngine.PaddleState.OUTSIDE_BASELINE)
            val result = engine.evaluate(input)
            assertEquals(listOf("C4_WEIGHT_TRANSFER", "C6_PADDLE_REVIEW"), result.ruleIds)
            assertEquals(result, engine.evaluate(input))
        }
    }

    @Test fun encouragementIsLimitedToAssessedMechanicsAndRequiresBothCoreSignals() {
        val result = engine.evaluate(good)
        assertEquals(FeedbackEngine.Status.POSITIVE, result.status)
        assertTrue(result.text.contains("measured weight shift and paddle angle"))
        assertFalse(result.text.contains("good form", ignoreCase = true))
        assertEquals(FeedbackEngine.Status.PARTIAL, engine.evaluate(good.copy(paddle = null)).status)
    }

    @Test fun optionalLandingIsOmittedUnlessActiveAndNeverInventsACause() {
        assertTrue(engine.evaluate(good.copy(placementZone = "Fault-Long")).notifications.isEmpty())
        for (zone in listOf("Fault-Long", "Fault-Wide", "Fault-Short", "garbage", null)) {
            val result = engine.evaluate(good.copy(placementActive = true, placementZone = zone))
            assertEquals(FeedbackEngine.Status.PARTIAL, result.status)
            assertTrue(result.ruleIds.isEmpty())
        }
        assertEquals(FeedbackEngine.Status.POSITIVE,
            engine.evaluate(good.copy(placementActive = true, placementZone = "Deep-Left")).status)
    }

    @Test fun rawPaddleAngleDoesNotProduceUnverifiedDirectionalAdvice() {
        assertEquals(FeedbackEngine.PaddleState.OUTSIDE_BASELINE, FeedbackEngine.paddleState(5.0, 10.0, 160.0))
        assertEquals(FeedbackEngine.PaddleState.OUTSIDE_BASELINE, FeedbackEngine.paddleState(170.0, 10.0, 160.0))
        assertEquals(FeedbackEngine.PaddleState.TOO_CLOSED, FeedbackEngine.paddleState(5.0, 10.0, 160.0, true))
        assertEquals(FeedbackEngine.PaddleState.TOO_OPEN, FeedbackEngine.paddleState(170.0, 10.0, 160.0, true))
        for (angle in listOf(10.0, 160.0)) {
            assertEquals(FeedbackEngine.PaddleState.OPTIMAL, FeedbackEngine.paddleState(angle, 10.0, 160.0))
        }
        for (angle in listOf(null, Double.NaN, Double.POSITIVE_INFINITY, -1.0, 180.0)) {
            assertNull(FeedbackEngine.paddleState(angle, 10.0, 160.0))
        }
        assertNull(FeedbackEngine.paddleState(90.0, 160.0, 10.0))
    }

    @Test fun directionalCuesAreRestrictedToSupportedSubtypes() {
        assertEquals(listOf("C6_TOO_CLOSED"), engine.evaluate(good.copy(serveType = "topspin",
            paddle = FeedbackEngine.PaddleState.TOO_CLOSED)).ruleIds)
        assertTrue(engine.evaluate(good.copy(paddle = FeedbackEngine.PaddleState.TOO_CLOSED)).ruleIds.isEmpty())
        assertEquals("pending_coach_c", engine.evaluate(good).validationStatus)
    }

    @Test(expected = IllegalArgumentException::class)
    fun unknownConditionInRuleTableFailsLoading() {
        FeedbackEngine.fromJson(File("src/main/assets/feedback_rules.json").readText()
            .replace("shift_insufficient", "made_up_detector"))
    }
}
