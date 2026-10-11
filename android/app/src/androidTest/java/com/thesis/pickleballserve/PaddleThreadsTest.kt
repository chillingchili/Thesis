package com.thesis.pickleballserve

import android.media.MediaMetadataRetriever
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import kotlin.math.abs

class PaddleThreadsTest {
    @Test fun compareThreadsAndPreserveDetections() {
        if (android.os.Build.VERSION.SDK_INT < 28) return
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val frames = listOf("C001", "C002", "C007").flatMap { id ->
            val reader = MediaMetadataRetriever()
            try {
                reader.setDataSource(File(context.filesDir,"latency_input/$id.mp4").absolutePath)
                val count = reader.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)!!.toInt()
                listOf(1,2,3).map { requireNotNull(reader.getFrameAtIndex(count * it / 4)) }
            } finally { reader.release() }
        }
        val threadCounts = (InstrumentationRegistry.getArguments().getString("threads") ?: "4,6,8").split(",").map { it.toInt() }
        require(threadCounts.size == 3 && 4 in threadCounts)
        val models = threadCounts.associateWith { PaddleDetector.open(context,it) }
        val rows = JSONArray()
        val report = JSONObject().put("protocol","Nine frames, three clips; one warm-up and three measured rounds; rotating thread order; decode excluded")
            .put("runs",rows)
        val file = File(context.filesDir,"latency_results/paddle-threads.json")
        try {
            val baseline = frames.map { models.getValue(4).detectAll(it) }
            for (round in 0..3) {
                val orders = (0..2).map { offset -> (0..2).map { threadCounts[(it+offset)%3] } }
                for (threads in orders[round % 3]) {
                    for ((index,bitmap) in frames.withIndex()) {
                        val start=SystemClock.elapsedRealtimeNanos()
                        val actual=models.getValue(threads).detectAll(bitmap)
                        val elapsed=(SystemClock.elapsedRealtimeNanos()-start)/1e6
                        val expected=baseline[index]
                        assertEquals("Detection count changed at frame $index, threads $threads",expected.size,actual.size)
                        var maxBox=0f; var maxConfidence=0f
                        for (i in expected.indices) {
                            maxConfidence=maxOf(maxConfidence,abs(expected[i].conf-actual[i].conf))
                            for (j in 0..3) maxBox=maxOf(maxBox,abs(expected[i].box[j]-actual[i].box[j]))
                        }
                        rows.put(JSONObject().put("threads",threads).put("round",round).put("warmup",round==0)
                            .put("frame",index).put("ms",elapsed).put("detections",actual.size)
                            .put("max_box_delta_px",maxBox).put("max_confidence_delta",maxConfidence))
                        file.writeText(report.toString(2))
                        assertTrue("Box difference exceeds 0.1 pixel",maxBox<=.1f)
                        assertTrue("Confidence difference exceeds 0.0001",maxConfidence<=.0001f)
                    }
                }
            }
            report.put("completed",true); file.writeText(report.toString(2))
        } finally { models.values.forEach { it.close() }; frames.forEach { it.recycle() } }
    }
}
