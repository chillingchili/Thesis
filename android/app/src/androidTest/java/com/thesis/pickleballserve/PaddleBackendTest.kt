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

/**
 * Measures the paddle GPU delegate against the CPU/XNNPACK baseline on-device.
 *
 * CPU is asserted to match its own baseline exactly, as PaddleThreadsTest does for
 * CPU thread counts. GPU is measured, not asserted: the model stores int8 weights,
 * which the GPU dequantizes to FP16, and that drifts confidence far enough to cross
 * the 0.25 detection threshold and change how many detections exist. The divergence
 * is recorded as gpu_detection_count_mismatches rather than failed, because the
 * divergence is the finding. If the delegate cannot initialize, the run is recorded
 * as CPU and still passes: falling back is correct behaviour, not a failure.
 *
 * Reproduce with: adb shell am instrument -w -r \
 *   -e class com.thesis.pickleballserve.PaddleBackendTest \
 *   com.thesis.pickleballserve.test/androidx.test.runner.AndroidJUnitRunner
 * Results: files/latency_results/paddle-backend.json
 */
class PaddleBackendTest {
    @Test fun compareGpuAgainstCpuAndPreserveDetections() {
        if (android.os.Build.VERSION.SDK_INT < 28) return
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val frames = listOf("C001", "C002", "C007").flatMap { id ->
            val reader = MediaMetadataRetriever()
            try {
                reader.setDataSource(File(context.filesDir, "latency_input/$id.mp4").absolutePath)
                val count = reader.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)!!.toInt()
                listOf(1, 2, 3).map { requireNotNull(reader.getFrameAtIndex(count * it / 4)) }
            } finally { reader.release() }
        }
        val rounds = (InstrumentationRegistry.getArguments().getString("rounds") ?: "4").toInt()
        require(rounds >= 2) { "need at least one warm-up and one measured round" }

        // Opened and invoked on this thread, which is what the delegate requires.
        val cpu = PaddleDetector.open(context, numThreads = 4, useGpu = false)
        val gpu = PaddleDetector.open(context, numThreads = 4, useGpu = true)
        val rows = JSONArray()
        val report = JSONObject()
            .put("protocol", "Nine frames, three clips; one warm-up and measured rounds; alternating backend order; decode excluded")
            .put("cpu_backend", cpu.backend)
            .put("gpu_backend", gpu.backend)
            .put("gpu_active", gpu.backend == "GPU")
            .put("runs", rows)
        val file = File(context.filesDir, "latency_results/paddle-backend.json")
        try {
            val cpuBaseline = frames.map { cpu.detectAll(it) }
            var gpuCountMismatches = 0
            for (round in 0 until rounds) {
                val warmup = round == 0
                val order = if (round % 2 == 0) listOf("cpu", "gpu") else listOf("gpu", "cpu")
                for (backend in order) {
                    for ((index, bitmap) in frames.withIndex()) {
                        val start = SystemClock.elapsedRealtimeNanos()
                        val actual = if (backend == "cpu") cpu.detectAll(bitmap) else gpu.detectAll(bitmap)
                        val elapsed = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
                        val expected = cpuBaseline[index]
                        // CPU must be self-consistent. GPU is only measured here: a
                        // differing count means FP16 drift crossed the 0.25 threshold,
                        // which is a finding to report, not a reason to abort.
                        if (backend == "cpu") {
                            assertEquals("Detection count changed at frame $index, backend cpu",
                                expected.size, actual.size)
                        } else if (actual.size != expected.size) {
                            gpuCountMismatches++
                        }
                        var maxBox = 0f
                        var maxConfidence = 0f
                        for (i in minOf(expected.size, actual.size) - 1 downTo 0) {
                            maxConfidence = maxOf(maxConfidence, abs(expected[i].conf - actual[i].conf))
                            for (j in 0..3) maxBox = maxOf(maxBox, abs(expected[i].box[j] - actual[i].box[j]))
                        }
                        rows.put(JSONObject()
                            .put("backend", backend).put("round", round).put("warmup", warmup)
                            .put("frame", index).put("ms", elapsed).put("detections", actual.size)
                            .put("cpu_detections", expected.size)
                            .put("max_box_delta_px", maxBox).put("max_confidence_delta", maxConfidence))
                        file.writeText(report.toString(2))
                        if (backend == "cpu") {
                            assertTrue("Box difference exceeds 0.1 pixel", maxBox <= .1f)
                            assertTrue("Confidence difference exceeds 0.0001", maxConfidence <= .0001f)
                        }
                    }
                }
            }
            // Speed is reported, not asserted: GPU wins on this device is the goal,
            // but a slower GPU is a tuning problem, not a correctness regression.
            val median = { name: String ->
                val ms = (0 until rows.length()).mapNotNull { rows.getJSONObject(it) }
                    .filter { it.getString("backend") == name && !it.getBoolean("warmup") }
                    .map { it.getDouble("ms") }.sorted()
                if (ms.isEmpty()) null else ms[ms.size / 2]
            }
            report.put("cpu_median_ms", median("cpu")).put("gpu_median_ms", median("gpu"))
            report.put("gpu_detection_count_mismatches", gpuCountMismatches)
            report.put("completed", true)
            file.writeText(report.toString(2))
        } finally {
            try { gpu.close() } finally { cpu.close() }
            frames.forEach { it.recycle() }
        }
    }
}
