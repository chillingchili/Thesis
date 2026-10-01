package com.thesis.pickleballserve.landing

import android.graphics.Bitmap
import android.graphics.Color
import android.content.Intent
import android.widget.TextView
import android.os.SystemClock
import android.os.Handler
import android.os.Looper
import android.view.FrameMetrics
import android.view.Window
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import com.thesis.pickleballserve.R

@RunWith(AndroidJUnit4::class)
class LandingPipelineTest {
    @Test fun note12RealtimeServeWithRegionDetections() = benchmarkServe()
    @Test fun note12Realtime1080p60Source() = benchmarkServe("serve-1080p60.mp4")

    private fun benchmarkServe(asset: String = "serve-replay.mp4") {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val target = instrumentation.targetContext
        val file = File(target.cacheDir, "serve-realtime-benchmark.mp4")
        instrumentation.context.assets.open(asset).use { source ->
            file.outputStream().use { source.copyTo(it) }
        }
        try {
            val intent = Intent(target, LandingActivity::class.java)
                .putExtra("video", file.absolutePath).putExtra("recordTrace", true)
            ActivityScenario.launch<LandingActivity>(intent).use { scenario ->
                val renderTimes = ArrayList<Double>()
                val listener = Window.OnFrameMetricsAvailableListener { _, metrics, _ ->
                    renderTimes.add(metrics.getMetric(FrameMetrics.TOTAL_DURATION) / 1_000_000.0)
                }
                scenario.onActivity { it.window.addOnFrameMetricsAvailableListener(listener, Handler(Looper.getMainLooper())) }
                val deadline = SystemClock.elapsedRealtime() + 40_000
                var metrics: LandingActivity.ReplayMetrics? = null
                while (metrics == null && SystemClock.elapsedRealtime() < deadline) {
                    scenario.onActivity {
                        val status = it.findViewById<TextView>(R.id.landingStatus).text.toString()
                        assertFalse(status, status.contains("ERR:"))
                        metrics = it.replayMetrics
                    }
                    SystemClock.sleep(100)
                }
                val result = requireNotNull(metrics) { "10-second serve did not finish within 40 seconds" }
                var renderP95 = 0.0
                scenario.onActivity {
                    it.window.removeOnFrameMetricsAvailableListener(listener)
                    val times = renderTimes.sorted()
                    renderP95 = times.getOrElse((times.size * .95).toInt()) { Double.POSITIVE_INFINITY }
                }
                Log.i("LandingBenchmark", "asset=$asset $result")
                Log.i("LandingBenchmark", "rendered=${renderTimes.size} renderP95Ms=$renderP95")
                assertEquals(if (asset == "serve-1080p60.mp4") 301 else 300, result.frames)
                assertTrue("Playback runs slower than real time: $result", result.elapsedMs <= result.sourceMs * 1.10 + 200)
                assertTrue("Preview dropped more than 5% of frames: $result", result.displayed >= 285)
                assertTrue("Tracking exceeds frame budget: $result", result.processingP95Ms < 25)
                assertTrue("Rendering exceeds the 30fps frame budget: $renderP95 ms", renderP95 < 33.4)
                assertTrue("Court tracking missing for most of the clip: $result", result.courtFrames >= 210)
                assertTrue("The real serve must have tracked ball observations: $result", result.ballFrames >= 10)
                run {
                    // Reference centers on visible ball frames, in the 640px tracking coordinate system.
                    val reference = mapOf(84 to (479.0 to 167.5), 88 to (481.0 to 165.0),
                        91 to (481.0 to 175.5), 93 to (481.0 to 187.5), 105 to (362.0 to 140.5),
                        112 to (315.5 to 118.0), 169 to (422.0 to 220.0), 176 to (487.0 to 244.5))
                    var matched = 0
                    scenario.onActivity { activity ->
                        for ((frame, expected) in reference) {
                            val point = activity.replayBallSamples[frame]
                            if (point != null && kotlin.math.hypot(point.first - expected.first, point.second - expected.second) < 8) matched++
                            Log.i("LandingBenchmark", "accuracy frame=$frame expected=$expected actual=$point")
                        }
                    }
                    assertTrue("Correct ball must be tracked during the serve, matched=$matched/8", matched >= 5)
                }
            }
        } finally { file.delete() }
    }

    @Test fun replayScreenProcessesWholeClipWithoutModelOrBitmapErrors() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val target = instrumentation.targetContext
        val file = File(target.cacheDir, "landing-screen-regression.mp4")
        instrumentation.context.assets.open("replay.mp4").use { source ->
            file.outputStream().use { source.copyTo(it) }
        }
        try {
            val intent = Intent(target, LandingActivity::class.java).putExtra("video", file.absolutePath)
            ActivityScenario.launch<LandingActivity>(intent).use { scenario ->
                val deadline = SystemClock.elapsedRealtime() + 600_000L
                var status = ""
                do {
                    scenario.onActivity { status = it.findViewById<TextView>(R.id.landingStatus).text.toString() }
                    assertFalse("Replay failed: $status", status.contains("ERR:"))
                    if (status.contains("replay total=60")) break
                    SystemClock.sleep(250)
                } while (SystemClock.elapsedRealtime() < deadline)
                assertTrue("Replay must finish all 60 frames: $status", status.contains("replay total=60"))
            }
        } finally { file.delete() }
    }

    @Test fun rotatedVideoIsDecodedWithoutApplyingRotationTwice() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val target = instrumentation.targetContext
        val file = File(target.cacheDir, "landing-rotation-regression.mp4")
        instrumentation.context.assets.open("replay-rotated.mp4").use { source ->
            file.outputStream().use { source.copyTo(it) }
        }
        try {
            FastVideoDecoder(file.absolutePath, null, target).use { decoder ->
                assertTrue("Fixture must carry a quarter-turn rotation", decoder.rotation % 180 != 0)
                val frame = requireNotNull(decoder.nextFrame())
                assertTrue("Decoder must retain raw landscape pixels", frame.width > frame.height)
                YoloInput().use { input ->
                    val layout = input.prepare(frame, decoder.rotation)
                    assertTrue("One rotation produces portrait letterboxing", layout.left > 0 && layout.top == 0.0)
                }
            }
        } finally { file.delete() }
    }

    @Test fun preprocessingRefreshesMutatedBitmapAndRotatesOnce() {
        val bitmap = Bitmap.createBitmap(320, 160, Bitmap.Config.ARGB_8888)
        try {
            YoloInput().use { input ->
                bitmap.eraseColor(Color.RED)
                val first = input.prepare(bitmap, 0)
                assertEquals(160.0, first.top, 0.0)
                val center = 320 * 640 + 320
                val area = 640 * 640
                assertEquals(1f, input.buffer.getFloat(center * 4), 0.001f)
                bitmap.eraseColor(Color.BLUE)
                input.prepare(bitmap, 0)
                assertEquals(0f, input.buffer.getFloat(center * 4), 0.001f)
                assertEquals(1f, input.buffer.getFloat((2 * area + center) * 4), 0.001f)
                val rotated = input.prepare(bitmap, 90)
                assertEquals(160.0, rotated.left, 0.0)
                assertEquals(0.0, rotated.top, 0.0)
                assertEquals(114f / 255f, input.buffer.getFloat(0), 0.001f)
            }
        } finally { bitmap.recycle() }
    }

    @Test fun decoderDrainsEveryFrameAndPreviewOwnsItsPixels() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val target = instrumentation.targetContext
        val file = File(target.cacheDir, "landing-regression.mp4")
        instrumentation.context.assets.open("replay.mp4").use { source ->
            file.outputStream().use { source.copyTo(it) }
        }
        try {
            // Fixture is 60 H.264 frames at 30000/1001 FPS, independently verified with ffprobe.
            val expected = 60
            FastVideoDecoder(file.absolutePath, null, target).use { decoder ->
                val first = requireNotNull(decoder.nextFrame())
                val snapshot = first.copy(Bitmap.Config.ARGB_8888, false)!!
                try {
                    val before = snapshot.generationId
                    var count = 1
                    while (decoder.nextFrame() != null) {
                        assertFalse(first.isRecycled)
                        count++
                    }
                    assertEquals(expected, count)
                    assertEquals(before, snapshot.generationId)
                    assertFalse(snapshot.isRecycled)
                } finally { snapshot.recycle() }
            }
            FastVideoDecoder(file.absolutePath, null, target, isCancelled = { true }).use {
                assertNull(it.nextFrame())
            }
        } finally { file.delete() }
    }

    @Test fun modelsLoadRunAndCloseOnSameWorker() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val target = instrumentation.targetContext
        val file = File(target.cacheDir, "landing-model-benchmark.mp4")
        instrumentation.context.assets.open("replay.mp4").use { source ->
            file.outputStream().use { source.copyTo(it) }
        }
        try {
            // Exercise both original and realtime assets without incurring multi-minute GPU compilation.
            assertNotNull(Class.forName("org.tensorflow.lite.gpu.GpuDelegateFactory\$Options"))
            for (realtime in listOf(false, true)) {
                YoloRunner.open(target, useGpu = false, realtimeMode = realtime).use { runner ->
                    FastVideoDecoder(file.absolutePath, null, target).use { decoder ->
                        repeat(3) { index ->
                            val frame = requireNotNull(decoder.nextFrame())
                            val start = SystemClock.elapsedRealtime()
                            val detections = runner.detectFrame(frame, decoder.rotation)
                            val elapsed = SystemClock.elapsedRealtime() - start
                            Log.i("LandingBenchmark", "realtime=$realtime sample=$index elapsed=${elapsed}ms " +
                                "${runner.backend} ball=${detections.ball?.conf} court=${detections.court?.boxConf}")
                            detections.ball?.let { assertTrue(it.cx.isFinite() && it.cy.isFinite()) }
                            detections.court?.let { assertTrue(it.kptsXy.all(Double::isFinite)) }
                        }
                        val wrongThread = java.util.concurrent.Executors.newSingleThreadExecutor()
                        try {
                            val frame = requireNotNull(decoder.nextFrame())
                            val rejected = wrongThread.submit<Boolean> {
                                try { runner.detectFrame(frame); false } catch (_: IllegalStateException) { true }
                            }.get()
                            assertTrue("Cross-thread inference must be rejected before native GPU calls", rejected)
                        } finally { wrongThread.shutdown() }
                    }
                }
            }
        } finally { file.delete() }
    }
}
