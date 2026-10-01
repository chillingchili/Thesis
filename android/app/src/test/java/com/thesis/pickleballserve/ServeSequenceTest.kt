package com.thesis.pickleballserve

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ServeSequenceTest {
    private fun values(array: JSONArray)=FloatArray(array.length()) { array.getDouble(it).toFloat() }
    private fun fixture()=JSONObject(File("src/test/resources/fixtures/serve_sequence.json").readText())
    @Test fun completeTimelinesAndMissingPosesMatchPythonForAllFeatureContracts() {
        val cases=fixture().getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val case=cases.getJSONObject(i);val json=case.getJSONArray("frames")
            val frames=(0 until json.length()).map { j ->
                val f=json.getJSONObject(j)
                ServeSequence.Frame(f.getDouble("timestamp_ms"),if(f.isNull("angles")) null else values(f.getJSONArray("angles")),
                    if(f.isNull("xy")) null else values(f.getJSONArray("xy")),if(f.isNull("visibility")) null else values(f.getJSONArray("visibility")))
            }
            val actual=ServeSequence.build(frames,case.getDouble("aspect"))
            assertEquals(case.getBoolean("accepted"),actual.accepted)
            for ((name,feature) in listOf("angles" to actual.angles,"skeleton" to actual.skeleton,"motion" to actual.motion,
                                        "lean" to actual.features("lean_thesis_aug"))) {
                assertArrayEquals(case.getString("id")+" "+name,values(case.getJSONArray(name)),feature.flatMap { it.asIterable() }.toFloatArray(),2e-5f)
            }
        }
    }
    @Test fun calibratedProbabilitiesMatchPythonAndKeepDisabledAcceptance() {
        val f=fixture();val policy=ProbabilityPolicy.fromJson(f.getJSONObject("policy"))
        assertNull(policy.threshold)
        val g=f.getJSONArray("policy_gru");val k=f.getJSONArray("policy_knn");val expected=f.getJSONArray("policy_expected")
        for (i in 0 until g.length()) assertArrayEquals(values(expected.getJSONArray(i)),policy.apply(values(g.getJSONArray(i)),values(k.getJSONArray(i))),1e-6f)
    }
    @Test(expected=IllegalArgumentException::class) fun duplicateTimestampsAreRejected() {
        ServeSequence.build(listOf(ServeSequence.Frame(0.0,null,null,null),ServeSequence.Frame(0.0,null,null,null)),1.0)
    }
    @Test fun insufficientTrackingDoesNotProduceAnAcceptedServe() {
        val result=ServeSequence.build((0 until 60).map { ServeSequence.Frame(it*33.0,null,null,null) },1.0)
        assertFalse(result.accepted);assertEquals(0,result.validSourceFrames)
    }
}
