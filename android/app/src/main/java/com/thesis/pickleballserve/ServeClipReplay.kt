package com.thesis.pickleballserve

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Build
import android.os.SystemClock
import android.util.Log
import java.io.File
import kotlin.concurrent.thread
import kotlin.math.min

/**
 * Feeds a recorded serve into the live pose pipeline in place of the camera.
 *
 * Only the frame source changes. Everything downstream of PosePipeline.analyze —
 * the motion gate, the async paddle worker, the GRU, feedback and the LiveLatency
 * instrumentation — is the same code the camera drives, so a replay exercises the
 * live path rather than a reimplementation of it.
 *
 * Frames are decoded up front, in index order. getFrameAtIndex seeks back to a
 * keyframe whenever the index jumps, so decoding on demand makes one late frame
 * force a seek that makes the next frame late, and the clip collapses to about one
 * frame per second. A live camera hands frames over with no seeking at all, so
 * decoding ahead is what keeps the source faithful.
 *
 * Playback is paced against wall clock, submitting only the frame due at each tick,
 * so the gate's lowHoldMs and maxActiveMs see real elapsed time and falling behind
 * skips frames rather than queueing them, matching ImageAnalysis KEEP_ONLY_LATEST.
 *
 * The frame callback also reports the source frame index (or -1 for synthetic
 * idle frames) so callers can map analyzed frames back to the recording, e.g. to
 * re-decode a window at full resolution for a second measurement pass.
 *
 * Frames are fitted into the live ImageAnalysis box without distorting aspect,
 * because joint angles are direction-based and a non-uniform scale would silently
 * change them.
 */
class ServeClipReplay(
    private val file: File,
    private val onFrame: (Bitmap, Boolean, Int) -> Unit,
    private val onDone: (framesSent: Int) -> Unit = {},
    private val analysisWidth: Int = 640,
    private val analysisHeight: Int = 480,
    private val cameraFps: Int = 30,
    private val loop: Boolean = false,
    // Longer than the motion gate's refractory window, so a looped clip starts a
    // genuinely separate serve instead of colliding with the previous one's tail.
    private val gapMs: Long = 3000,
) {
    companion object {
        private const val TAG = "ServeClipReplay"
    }

    @Volatile
    private var running = false
    private var worker: Thread? = null

    fun start() {
        if (running) return
        running = true
        worker = thread(name = "serve-clip-replay") { run() }
    }

    fun stop() {
        running = false
    }

    private fun run() {
        val retriever = MediaMetadataRetriever()
        var sent = 0
        try {
            if (Build.VERSION.SDK_INT < 28) {
                throw IllegalArgumentException("Clip replay requires Android 9 or newer")
            }
            retriever.setDataSource(file.absolutePath)
            val count = requireNotNull(
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()
            ) { "No frame count" }
            val durationMs = requireNotNull(
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull()
            ) { "No duration" }
            require(count > 1 && durationMs in 1..12000) { "Expected a complete clip up to 12 seconds" }
            val params = MediaMetadataRetriever.BitmapParams().apply {
                preferredConfig = Bitmap.Config.ARGB_8888
            }

            // A 30fps camera never delivers 60 distinct frames per second, so decode
            // only the ones it would have produced. That also halves the start-up wait
            // and the memory held for the cached clip.
            val step = Math.round(count * 1000.0 / durationMs / cameraFps).toInt().coerceAtLeast(1)
            val decodeStart = SystemClock.uptimeMillis()
            val frames = ArrayList<Bitmap>((count + step - 1) / step)
            // Decode contiguous batches once, preserving exact source-frame selection.
            for (start in 0 until count step 16) {
                if (!running) break
                val batch = retriever.getFramesAtIndex(start, minOf(16, count-start), params)
                for ((offset, raw) in batch.withIndex()) {
                    if ((start+offset) % step == 0) frames.add(fitInto(raw)) else raw.recycle()
                }
            }
            val decodeMs = SystemClock.uptimeMillis() - decodeStart
            require(frames.size >= 2) { "Could not decode frames" }
            Log.i(TAG, "Decoded clip=${file.name} frames=${frames.size}/${count} " +
                    "in ${decodeMs}ms (${decodeMs / frames.size}ms/frame) " +
                    "size=${frames[0].width}x${frames[0].height}")

            var round = 0
            do {
                idle(frames.first(), 1500L)
                Log.i(TAG, "replay_start clip=${file.name} round=${++round} time=${SystemClock.uptimeMillis()}")
                sent += playOnce(frames, durationMs, step, count)
                Log.i(TAG, "source_end clip=${file.name} round=$round time=${SystemClock.uptimeMillis()}")
                idle(frames.last(), gapMs)
            } while (running && loop)
            Log.i(TAG, "Replay finished clip=${file.name} frames_sent=$sent")
        } catch (e: Throwable) {
            Log.e(TAG, "Replay failed", e)
        } finally {
            runCatching { retriever.release() }
            onDone(sent)
        }
    }

    /** One pass at camera pace. Returns the number of frames submitted. */
    private fun playOnce(frames: List<Bitmap>, durationMs: Long, step: Int, count: Int): Int {
        val started = SystemClock.uptimeMillis()
        val intervalMs = 1000L / cameraFps
        var next = started
        var lastFrame = -1
        var sent = 0
        while (running) {
            val wait = next - SystemClock.uptimeMillis()
            if (wait > 0) SystemClock.sleep(wait)
            val elapsed = SystemClock.uptimeMillis() - started
            if (elapsed > durationMs) break
            val index = ((elapsed * frames.size) / durationMs).toInt().coerceIn(0, frames.lastIndex)
            if (index != lastFrame) {
                onFrame(frames[index], true, (index * step).coerceAtMost(count - 1))
                sent++
                lastFrame = index
            }
            next += intervalMs
        }
        return sent
    }

    /** Repeated idle frames exercise end detection, never supply classification evidence. */
    private fun idle(bitmap: Bitmap, durationMs: Long) {
        val until = SystemClock.uptimeMillis() + durationMs
        while (running && SystemClock.uptimeMillis() < until) {
            onFrame(bitmap, false, -1)
            SystemClock.sleep(1000L / cameraFps)
        }
    }

    /** Scales down to the analysis box, preserving aspect. Leaves smaller frames alone. */
    private fun fitInto(src: Bitmap): Bitmap {
        if (src.width <= analysisWidth && src.height <= analysisHeight) return src
        val scale = min(
            analysisWidth.toFloat() / src.width,
            analysisHeight.toFloat() / src.height,
        )
        val width = (src.width * scale).toInt().coerceAtLeast(1)
        val height = (src.height * scale).toInt().coerceAtLeast(1)
        val out = Bitmap.createScaledBitmap(src, width, height, true)
        if (out !== src) src.recycle()
        return out
    }
}
