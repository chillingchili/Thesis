package com.thesis.pickleballserve

import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Run on a connected phone to check its actual TFLite runtime against Python. */
class ServeResearchModelTest {
    private fun values(a: JSONArray)=FloatArray(a.length()) { a.getDouble(it).toFloat() }
    @Test fun selectedModelBankAndCalibrationMatchPython() {
        val instrumentation=InstrumentationRegistry.getInstrumentation()
        val context=instrumentation.targetContext
        val fixture=JSONObject(instrumentation.context.assets.open("serve_refinement.json").bufferedReader().use { it.readText() })
        val manifest=JSONObject(context.assets.open("serve_research/manifest.json").bufferedReader().use { it.readText() })
        assertEquals(fixture.getString("condition"),manifest.getString("condition"))
        val policy=ProbabilityPolicy.fromJson(manifest.getJSONObject("policy"))
        val bank=KnnClassifier.fromBytes(context.assets.open("serve_research/knn_meta.json").bufferedReader().use { it.readText() },context.assets.open("serve_research/knn_train.bin").use { it.readBytes() })
        val bytes=context.assets.open("serve_research/model.tflite").use { it.readBytes() }
        val buffer=ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes);rewind() }
        Interpreter(buffer,Interpreter.Options().setNumThreads(2)).use { model ->
            val cases=fixture.getJSONArray("cases");val dims=fixture.getInt("n_features")
            for (i in 0 until cases.length()) {
                val case=cases.getJSONObject(i);val flat=values(case.getJSONArray("input"))
                val tensor=Array(1) { Array(128) { t -> FloatArray(dims) { f -> flat[t*dims+f] } } }
                val gp=FloatArray(3);model.run(tensor,arrayOf(gp))
                val kp=bank.predictProba(values(case.getJSONArray("timing")))
                assertArrayEquals(values(case.getJSONArray("gru")),gp,3e-4f)
                assertArrayEquals(values(case.getJSONArray("knn")),kp,3e-4f)
                assertArrayEquals(values(case.getJSONArray("calibrated")),policy.apply(gp,kp),3e-4f)
            }
        }
    }
}
