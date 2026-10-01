package com.thesis.pickleballserve

import java.io.File
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class HybridPredictionTest {
    @Test fun fusedConfidenceControlsFeedbackInsteadOfTheGruConfidence() {
        val gru = floatArrayOf(.8f,.1f,.1f)
        val finalProbs = HybridPrediction.combine(gru, floatArrayOf(0f,1f,0f))
        assertArrayEquals(floatArrayOf(.4f,.55f,.05f), finalProbs, 1e-6f)
        val engine = FeedbackEngine.fromJson(File("src/main/assets/feedback_rules.json").readText())
        val feedback = engine.evaluate(FeedbackEngine.Input("lob", finalProbs[1], false, FeedbackEngine.PaddleState.OPTIMAL))
        assertEquals(FeedbackEngine.Status.LOW_CONFIDENCE, feedback.status)
        assertTrue(feedback.ruleIds.isEmpty())
        assertArrayEquals(floatArrayOf(.8f,.1f,.1f), gru, 0f)
        assertArrayEquals(gru, HybridPrediction.combine(gru, null), 0f)
    }

    @Test fun timingAndKnnMatchPythonFixturesUsingActualBundledTrainingBank() {
        val fixture = JSONObject(File("src/test/resources/fixtures/hybrid.json").readText())
        val knn = KnnClassifier.fromBytes(File("src/main/assets/knn_meta.json").readText(),
            File("src/main/assets/knn_train.bin").readBytes())
        val cases = fixture.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            fun values(name: String): FloatArray {
                val array = case.getJSONArray(name)
                return FloatArray(array.length()) { array.getDouble(it).toFloat() }
            }
            val features = TimingFeatures.fromFlatWindow(values("window"), case.getInt("valid_frames"))
            assertArrayEquals(case.getString("clip_id"), values("features"), features, 1e-5f)
            assertArrayEquals(case.getString("clip_id"), values("knn_probabilities"), knn.predictProba(features), 1e-5f)
        }
    }
}
