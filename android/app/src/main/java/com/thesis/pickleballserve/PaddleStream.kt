package com.thesis.pickleballserve

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat

/**
 * Paddle-orientation stream (thesis Section 4.3.2) — Kotlin port of the
 * per-clip scalar logic in scripts/extract_paddle_angles.py:
 *   contact = peak right-wrist speed (trim_after_contact median-filter
 *             heuristic, same guard), scalar = median angle over
 *             contact +/- 5 frames (full series if the window is empty),
 *   verdict = too_closed / optimal / too_open against the band from
 *             assets/paddle_config.json.
 * Samples are added frame-by-frame during the gated serve window only.
 */
class PaddleStream private constructor(
    private val detector: PaddleDetector,
    private val bandLow: Double,
    private val bandHigh: Double,
    val directionValidated: Boolean,
) : AutoCloseable {

    companion object {
        private const val TAG = "PaddleStream"
        private const val WINDOW = 5
        private const val DETECT_STRIDE = 2
        private const val MIN_CROP = 8

        /**
         * useGpu: enable the paddle GPU delegate when supported; see PaddleDetector.open.
         *
         * Defaults off. Measured on the Xiaomi 23021RAAEG (see PaddleBackendTest) the
         * GPU ran 562ms -> 466ms per inference, only ~17% faster, while its FP16
         * dequantization of the int8 weights changed the detection count on frames
         * with borderline confidences. That is a coaching-correctness risk for a gain
         * too small to matter. Keep this flag so a re-exported model can re-test it.
         */
        fun load(context: Context, useGpu: Boolean = false): PaddleStream? = try {
            if (!OpenCVLoader.initLocal()) {
                Log.e(TAG, "OpenCV failed to load — paddle stream disabled")
                null
            } else {
                val json = context.assets.open("paddle_config.json")
                    .bufferedReader().use { it.readText() }
                val cfg = JSONObject(json)
                val low = cfg.getDouble("band_low")
                val high = cfg.getDouble("band_high")
                require(low.isFinite() && high.isFinite() && low >= 0 && high < 180 && low <= high)
                val det = PaddleDetector.open(context, useGpu = useGpu)
                Log.i(TAG, "loaded band=[$low, $high] backend=${det.backend} rule=${cfg.getString("threshold_rule")}")
                PaddleStream(det, low, high, cfg.optBoolean("direction_validated", false))
            }
        } catch (e: Throwable) {
            Log.e(TAG, "paddle stream disabled", e)
            null
        }

        internal fun bitmapGray(bitmap: Bitmap): Mat {
            val rgba = Mat()
            return try {
                Utils.bitmapToMat(bitmap, rgba)
                PaddleGeometry.grayFromRgba(rgba)
            } finally { rgba.release() }
        }

        internal fun cropGray(bitmap: Bitmap, rect: org.opencv.core.Rect): Mat {
            // GaussianBlur(3x3) reads one pixel outside an OpenCV ROI. Preserve that halo.
            val left = (rect.x - 1).coerceAtLeast(0)
            val top = (rect.y - 1).coerceAtLeast(0)
            val right = (rect.x + rect.width + 1).coerceAtMost(bitmap.width)
            val bottom = (rect.y + rect.height + 1).coerceAtMost(bitmap.height)
            val crop = Bitmap.createBitmap(bitmap, left, top, right-left, bottom-top)
            val gray = try { bitmapGray(crop) }
                finally { if (crop !== bitmap) crop.recycle() }
            return try { Mat(gray, org.opencv.core.Rect(rect.x-left,rect.y-top,rect.width,rect.height)) }
                finally { gray.release() } // Returned ROI retains the backing allocation.
        }

        fun median(values: List<Double>): Double? {
            if (values.isEmpty()) return null
            val s = values.sorted()
            val n = s.size
            return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2.0
        }

        /** Python estimate_contact_frame() over per-frame wrist speeds (len T-1). */
        fun contactFrame(speeds: List<Double>, t: Int): Int {
            if (t < 3) return (0.7 * t).toInt()
            val n = speeds.size
            if (n == 0) return (0.7 * t).toInt()
            var mf = speeds
            if (n >= 5) {
                val padded = DoubleArray(n + 4)
                for (i in padded.indices) padded[i] = speeds[(i - 2).coerceIn(0, n - 1)]
                mf = (0 until n).map { i ->
                    padded.sliceArray(i until i + 5).sortedArray()[2]
                }
            }
            var arg = 0
            for (i in 1 until mf.size) if (mf[i] > mf[arg]) arg = i
            var contact = arg + 1
            if (contact < 0.2 * t) contact = (0.7 * t).toInt()
            return contact
        }

        /** Window median scalar with full-series fallback; null when no angle. */
        fun scalarFor(angles: List<Pair<Int, Double>>, contact: Int): Double? {
            if (angles.isEmpty()) return null
            val win = angles.filter { kotlin.math.abs(it.first - contact) <= WINDOW }
                .map { it.second }
            val use = if (win.isNotEmpty()) win else angles.map { it.second }
            return median(use)
        }

        fun contactScalarFor(angles: List<Pair<Int, Double>>, contact: Int): Double? =
            median(angles.filter { kotlin.math.abs(it.first - contact) <= WINDOW && it.second.isFinite() }
                .map { it.second })

        fun verdictFor(scalar: Double?, low: Double, high: Double): String = when {
            scalar == null -> "PADDLE: --"
            scalar < low -> "PADDLE: TOO CLOSED"
            scalar > high -> "PADDLE: TOO OPEN"
            else -> "PADDLE: OPTIMAL"
        }
    }

    private val lock = Any()
    private val angles = ArrayList<Pair<Int, Double>>()
    private val speeds = ArrayList<Float>()
    private var lastWristX = Float.NaN
    private var lastWristY = Float.NaN
    private var frameIdx = 0
    private var strideCount = 0
    // Bumped by clear(); an in-flight detection that started before a reset
    // discards its result instead of contaminating the next serve.
    private var generation = 0L

    fun clear() {
        synchronized(lock) {
            generation++
            angles.clear()
            speeds.clear()
            lastWristX = Float.NaN
            lastWristY = Float.NaN
            frameIdx = 0
            strideCount = 0
        }
    }

    /**
     * wristX/wristY: normalized right-wrist coords (NaN if pose unknown);
     * bitmap: frame for detection (null = speed bookkeeping only).
     *
     * Detection is the expensive step (~600ms on this phone) and runs outside
     * the lock so clear() from another thread can never block on it.
     */
    internal data class Sample(val index: Int, val generation: Long, val wristX: Float, val wristY: Float)

    fun add(wristX: Float, wristY: Float, bitmap: Bitmap?) {
        val sample = prepare(wristX, wristY, bitmap != null)
        if (bitmap != null && sample != null) detect(sample, bitmap)
    }

    /** Ordered, cheap bookkeeping runs on the pose thread. */
    internal fun prepare(wristX: Float, wristY: Float, hasBitmap: Boolean): Sample? {
        var idx = 0
        var gen = 0L
        var detect = false
        synchronized(lock) {
            gen = generation
            idx = frameIdx++
            if (idx > 0) {
                val speed = if (wristX.isNaN() || wristY.isNaN() ||
                    lastWristX.isNaN() || lastWristY.isNaN()
                ) 0f else {
                    val dx = wristX - lastWristX
                    val dy = wristY - lastWristY
                    kotlin.math.sqrt(dx * dx + dy * dy)
                }
                speeds.add(speed)
            }
            lastWristX = wristX
            lastWristY = wristY

            detect = hasBitmap && strideCount++ % DETECT_STRIDE == 0
        }
        return if (detect) Sample(idx, gen, wristX, wristY) else null
    }

    /** Expensive detection may finish later; generation prevents cross-serve writes. */
    internal fun detect(sample: Sample, frame: Bitmap) {
        val idx = sample.index
        val gen = sample.generation
        val wristX = sample.wristX
        val wristY = sample.wristY
        val dets = detector.detectAll(frame)
        val maxPx = PaddleDetector.WRIST_MAX_PX * frame.width / PaddleDetector.REF_WIDTH_PX
        val det = PaddleDetector.pick(dets, wristX * frame.width, wristY * frame.height, maxPx)
            ?: return
        val rect = PaddleGeometry.expandBox(det.box, frame.width, frame.height)
        if (rect.width < MIN_CROP || rect.height < MIN_CROP) return
        // Preserve source pixels; avoid converting an entire camera frame for one small crop.
        val gray = cropGray(frame, rect)
        val res = try { PaddleGeometry.cropAngle(gray) } finally { gray.release() }
        if (res.status == "ok" && res.angle != null) {
            synchronized(lock) {
                if (gen == generation) angles.add(idx to res.angle)
            }
        }
    }


    /** Python estimate_contact_frame() on the collected wrist speeds. */
    fun contactFrame(): Int = synchronized(lock) {
        contactFrame(speeds.map { it.toDouble() }, frameIdx)
    }

    /** Window median scalar (full-series fallback), or null when no angle. */
    fun compute(): Double? = synchronized(lock) { scalarFor(angles, contactFrame()) }

    /** Coaching must use contact-window evidence; never the full-series fallback. */
    fun computeForFeedback(): Double? = synchronized(lock) {
        contactScalarFor(angles, contactFrame())
    }

    fun verdict(scalar: Double?): String = verdictFor(scalar, bandLow, bandHigh)

    val low: Double get() = bandLow
    val high: Double get() = bandHigh
    val backend: String get() = detector.backend

    override fun close() {
        detector.close()
    }
}
