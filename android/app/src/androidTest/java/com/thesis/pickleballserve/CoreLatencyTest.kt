package com.thesis.pickleballserve

import android.content.Intent
import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.json.JSONObject

class CoreLatencyTest {
    @Test fun recordCompleteCoreBenchmark() {
        val context=InstrumentationRegistry.getInstrumentation().targetContext
        ActivityScenario.launch<LatencyBenchmarkActivity>(Intent(context,LatencyBenchmarkActivity::class.java).putExtra("autostart",true)).use { scenario ->
            val deadline=SystemClock.elapsedRealtime()+1_200_000
            var done=false
            while(!done && SystemClock.elapsedRealtime()<deadline) {
                scenario.onActivity { done=it.finished }
                SystemClock.sleep(500)
            }
            assertTrue("Benchmark exceeded 20 minutes; partial records retained",done)
            scenario.onActivity {
                assertNull(it.failure)
                val report=JSONObject(requireNotNull(it.resultFile).readText())
                assertTrue(report.getBoolean("completed"))
                assertEquals(10,report.getJSONArray("runs").length())
                val runs=report.getJSONArray("runs")
                assertTrue("Every measured attempt failed; inspect retained errors", (1 until runs.length()).any { index -> runs.getJSONObject(index).getBoolean("success") })
                assertFalse(report.getBoolean("live_two_second_target_verified"))
            }
        }
    }
}
