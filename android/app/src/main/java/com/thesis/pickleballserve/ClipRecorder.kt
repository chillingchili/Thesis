package com.thesis.pickleballserve

import android.graphics.Bitmap
import android.graphics.PointF
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

/** One review-clip frame: a small JPEG plus the pose at that frame (33 normalized x,y pairs). */
class ClipFrame(val tMs: Long, val jpeg: ByteArray, val pose: FloatArray?)

/** A serve's review clip, held in memory only. [peakMs] marks peak motion (on the frame clock). */
class ServeClip(val frames: List<ClipFrame>, val peakMs: Long?) {
    val startMs: Long get() = frames.first().tMs
    val durationMs: Long get() = frames.last().tMs - frames.first().tMs
}

/**
 * Keeps a short pre-roll of downscaled frames while idle and every frame while a
 * serve is captured, so the review screen can replay it with the pose drawn on.
 * Display-only: it never feeds inference. Encoding runs on its own thread so the
 * pose callback only pays for one small scale.
 */
class ClipRecorder(
    private val preRollMs: Long = 1000L,
    private val maxFrames: Int = 240,
    private val width: Int = 320,
) {
    private val lock = Any()
    private val frames = ArrayDeque<ClipFrame>()
    private var recording = false
    private val encoder = Executors.newSingleThreadExecutor()

    fun offer(bitmap: Bitmap, landmarks: Array<PointF>?, tMs: Long) {
        if (bitmap.isRecycled || bitmap.width <= 0) return
        val h = (bitmap.height.toLong() * width / bitmap.width).toInt().coerceAtLeast(1)
        val small = try { Bitmap.createScaledBitmap(bitmap, width, h, true) } catch (_: Throwable) { return }
        val pose = landmarks?.takeIf { it.size >= 33 }?.let { pts ->
            FloatArray(66) { i -> if (i % 2 == 0) pts[i / 2].x else pts[i / 2].y }
        }
        try {
            encoder.execute {
                val out = ByteArrayOutputStream(16_384)
                small.compress(Bitmap.CompressFormat.JPEG, 70, out)
                if (small !== bitmap) small.recycle()
                synchronized(lock) {
                    frames.addLast(ClipFrame(tMs, out.toByteArray(), pose))
                    if (!recording) while (frames.size > 1 && tMs - frames.first().tMs > preRollMs) frames.removeFirst()
                    while (frames.size > maxFrames) frames.removeFirst()
                }
            }
        } catch (_: java.util.concurrent.RejectedExecutionException) { }
    }

    /** A serve started: keep everything from the pre-roll on. */
    fun begin() { synchronized(lock) { recording = true } }

    /**
     * The serve ended: hand over what is encoded so far without waiting, so the
     * measured end-to-feedback latency is unchanged (a few tail frames may be missing).
     */
    fun finish(peakMs: Long?): ServeClip? {
        synchronized(lock) {
            recording = false
            if (frames.size < 2) { frames.clear(); return null }
            val clip = ServeClip(frames.toList(), peakMs)
            frames.clear()
            return clip
        }
    }

    fun reset() { synchronized(lock) { recording = false; frames.clear() } }
    fun close() { encoder.shutdownNow(); reset() }
}
