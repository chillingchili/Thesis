package com.thesis.pickleballserve

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.PointF
import android.media.MediaMetadataRetriever
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/** Full recorded-file benchmark, explicitly NOT camera-end-to-display latency. */
class CoreLatencyBenchmark(private val context: Context) {
    private fun now() = SystemClock.elapsedRealtimeNanos()
    private fun ms(start: Long) = (now() - start) / 1_000_000.0
    private fun digest(file: File): String = file.inputStream().use { input ->
        val hash = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(65536)
        while (true) { val n = input.read(buffer); if (n < 0) break; hash.update(buffer, 0, n) }
        hash.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
    }
    private fun environment(): JSONObject {
        val b = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val power = context.getSystemService(PowerManager::class.java)
        return JSONObject().put("battery_percent", b?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1))
            .put("battery_temperature_c", (b?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1) / 10.0)
            .put("plugged", b?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1))
            .put("power_save", power.isPowerSaveMode)
            .put("thermal_status", if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else -1)
    }

    fun run(files: List<File>, cancelled: () -> Boolean, progress: (String) -> Unit): File {
        if (Build.VERSION.SDK_INT < 28) throw IllegalArgumentException("Requires Android 9 or newer")
        require(files.size == 3 && files.all { it.isFile }) { "Prepare C001, C002 and C007 in latency_input first" }
        val output = File(context.filesDir, "latency_results").apply { mkdirs() }
        val destination = File(output, "core-${System.currentTimeMillis()}.json")
        val memory = ActivityManager.MemoryInfo().also { context.getSystemService(ActivityManager::class.java).getMemoryInfo(it) }
        val report = JSONObject().put("benchmark_version", "core-recorded-v2")
            .put("metric", "Recorded file open to computed coaching result; no camera or display timing")
            .put("live_two_second_target_verified", false)
            .put("debuggable", context.applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0)
            .put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL)
            .put("soc", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE)
            .put("android", Build.VERSION.RELEASE).put("ram_bytes", memory.totalMem)
            .put("start_environment", environment())
            .put("classifier", "original GRU five-fold ensemble; no kNN")
            .put("window", "Android last 128 usable frames; full clip for hip/pose; paddle YOLO only on contact +/- 5 usable frames (even stride, same samples as full pass)")
            .put("pose_mode", "Lite VIDEO; one pose; CPU default; 0.5 tracking/detection/presence")
            .put("pose_timestamps", "Uniform frame-index timestamps derived from clip duration; not live camera timestamps")
            .put("landing_enabled", false)
            .put("protocol", "One complete C001 warm-up excluded; three rounds of C001,C002,C007, no dropped frames or response filtering")
            .put("clips", JSONArray(files.map { JSONObject().put("name", it.name).put("sha256", digest(it)) }))
        val hashes = JSONObject()
        for (name in listOf("pose_landmarker_lite.task", "manifest.json", "feedback_rules.json", "shift_config.json", "paddle_config.json", "paddle_w8a32.tflite") + (1..5).map { "gru_fold$it.tflite" }) {
            val hash = MessageDigest.getInstance("SHA-256")
            context.assets.open(name).use { input ->
                val buffer = ByteArray(65536)
                while (true) { val n=input.read(buffer); if(n<0) break; hash.update(buffer,0,n) }
            }
            hashes.put(name, hash.digest().joinToString("") { "%02x".format(it.toInt() and 255) })
        }
        report.put("asset_sha256", hashes)
        val rows = JSONArray(); report.put("runs", rows)
        fun save() { destination.writeText(report.toString(2)) }
        save()
        var gru: GruClassifier? = null
        var paddle: PaddleStream? = null
        var pose: PoseLandmarker? = null
        try {
            progress("Loading models…")
            val loadStart = now()
            val classifier = GruClassifier.load(context, true).also { gru = it }
            // The backend is part of the measurement, not an implementation detail, so
            // it is set explicitly and recorded in the export. Paddle runs CPU/XNNPACK:
            // the GPU delegate was measured on this phone and was both too small a gain
            // (~17%) and a detection-changing precision loss. See PaddleStream.load.
            val equipment = requireNotNull(PaddleStream.load(context, useGpu = false)) { "Paddle model failed to load" }.also { paddle = it }
            val shift = requireNotNull(ShiftDetector.load(context)) { "Hip configuration unavailable" }
            val feedback = FeedbackEngine.fromJson(context.assets.open("feedback_rules.json").bufferedReader().use { it.readText() })
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build())
                .setRunningMode(RunningMode.VIDEO).setNumPoses(1)
                .setMinPoseDetectionConfidence(.5f).setMinPosePresenceConfidence(.5f).setMinTrackingConfidence(.5f).build()
            report.put("model_load_ms", ms(loadStart))
            report.put("paddle_backend", equipment.backend)
            var completed = 0
            for (round in 0..3) for (file in if (round == 0) files.take(1) else files) {
                check(!cancelled()) { "Cancelled" }
                val row = JSONObject().put("clip", file.name).put("round", round).put("warmup", round == 0)
                    .put("environment_before", environment())
                val resetStart = now()
                pose?.close(); pose = PoseLandmarker.createFromOptions(context, options)
                row.put("pose_reset_load_ms", ms(resetStart))
                val tracker = requireNotNull(pose)
                equipment.clear(); shift.clear()
                val buffer = AngleBuffer(128, 10)
                val reader = MediaMetadataRetriever()
                var decodeMs = 0.0; var poseMs = 0.0; var hipMs = 0.0; var paddleMs = 0.0
                var decoded = 0; var usable = 0
                // Two-pass paddle: pass 1 records wrist positions per usable frame
                // (no YOLO); pass 2 runs detection only on the contact window.
                val wristXs = ArrayList<Float>()
                val wristYs = ArrayList<Float>()
                val usableSources = ArrayList<Int>()
                val start = now()
                try {
                    reader.setDataSource(file.absolutePath)
                    val count = requireNotNull(reader.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull())
                    val duration = requireNotNull(reader.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull())
                    row.put("source_frames", count).put("source_duration_ms", duration)
                    require(count > 1 && duration in 1..12000) { "Expected a complete clip up to 12 seconds" }
                    for (i in 0 until count) {
                        check(!cancelled()) { "Cancelled" }
                        var mark = now()
                        val bitmap = requireNotNull(reader.getFrameAtIndex(i, MediaMetadataRetriever.BitmapParams().apply { preferredConfig = Bitmap.Config.ARGB_8888 }))
                        decodeMs += ms(mark); decoded++
                        var image: com.google.mediapipe.framework.image.MPImage? = null
                        try {
                            if (i == 0) row.put("decoded_width", bitmap.width).put("decoded_height", bitmap.height)
                            mark = now()
                            image = BitmapImageBuilder(bitmap).build()
                            val result = tracker.detectForVideo(image, i.toLong() * duration / count)
                            poseMs += ms(mark)
                            val landmarks = result.landmarks().firstOrNull() ?: continue
                            val points = Array(33) { PointF(landmarks[it].x(), landmarks[it].y()) }
                            val angles = JointAngles.fromLandmarks(points) ?: continue
                            if (angles.any { !it.isFinite() }) continue
                            buffer.add(angles); usable++
                            mark = now()
                            shift.add((points[23].x + points[24].x) / 2f, kotlin.math.abs(points[0].y - (points[27].y + points[28].y) / 2f))
                            hipMs += ms(mark)
                            // Pass 1: record wrist only; YOLO runs in pass 2 on the contact window.
                            wristXs.add(points[16].x); wristYs.add(points[16].y); usableSources.add(i)
                        } finally { image?.close(); if (!bitmap.isRecycled) bitmap.recycle() }
                        if (i % 30 == 0) progress("${if(round==0) "Warm-up" else "Run ${completed+1}/9"}: ${file.name}, frame ${i+1}/$count")
                    }
                    require(buffer.validFrames() >= 32) { "Too few usable pose frames: $usable" }
                    // Pass 2: contact from wrist speeds (same math as live-path
                    // bookkeeping: no entry for frame 0, 0 when wrist unknown),
                    // then YOLO only on contact +/- 5 usable frames at even
                    // stride — the exact frames the live path samples, so the
                    // window median and all outputs are unchanged.
                    val speeds = ArrayList<Double>(wristXs.size)
                    for (k in 1 until wristXs.size) {
                        speeds.add(if (wristXs[k].isNaN() || wristYs[k].isNaN() ||
                            wristXs[k - 1].isNaN() || wristYs[k - 1].isNaN()) 0.0 else {
                            val dx = wristXs[k] - wristXs[k - 1]; val dy = wristYs[k] - wristYs[k - 1]
                            kotlin.math.sqrt((dx * dx + dy * dy).toDouble())
                        })
                    }
                    val contact = PaddleStream.contactFrame(speeds, wristXs.size)
                    val windowParams = MediaMetadataRetriever.BitmapParams().apply { preferredConfig = Bitmap.Config.ARGB_8888 }
                    for (u in (contact - 5)..(contact + 5)) {
                        if (u < 0 || u >= wristXs.size || u % 2 != 0) continue
                        val windowFrame = reader.getFrameAtIndex(usableSources[u], windowParams) ?: continue
                        try {
                            val paddleStart = now()
                            equipment.addWindow(u, wristXs[u], wristYs[u], windowFrame)
                            paddleMs += ms(paddleStart)
                        } finally { if (!windowFrame.isRecycled) windowFrame.recycle() }
                    }
                    var mark = now()
                    val probabilities = classifier.infer(buffer.toInputTensor(), buffer.validFrames())
                    row.put("gru_ms", ms(mark))
                    val prediction = classifier.predict(probabilities)
                    mark = now()
                    val angle = equipment.computeForFeedback()
                    val state = FeedbackEngine.paddleState(angle, equipment.low, equipment.high, equipment.directionValidated)
                    val result = feedback.evaluate(FeedbackEngine.Input(prediction.first, prediction.second, shift.passed(), state))
                    row.put("final_measurements_and_feedback_ms", ms(mark))
                    row.put("total_processing_ms", ms(start)).put("success", true)
                        .put("predicted_serve", prediction.first).put("confidence", prediction.second)
                        .put("feedback_status", result.status.name).put("feedback", result.text)
                        .put("paddle_available", angle != null).put("hip_available", shift.compute() != null)
                } catch (e: Exception) {
                    row.put("success", false).put("error", e.message).put("error_stack", e.stackTraceToString()).put("total_processing_ms", ms(start))
                } finally {
                    reader.release()
                    row.put("decoded_frames", decoded).put("usable_frames", usable)
                        .put("decode_ms", decodeMs).put("pose_ms", poseMs).put("hip_accumulation_ms", hipMs)
                        .put("paddle_ms", paddleMs).put("environment_after", environment())
                    rows.put(row); save()
                }
                if (round != 0) completed++
            }
            report.put("completed", true).put("end_environment", environment()); save()
        } catch (e: Exception) {
            report.put("completed", false).put("error", e.message); save(); throw e
        } finally { pose?.close(); paddle?.close(); gru?.close() }
        return destination
    }
}
