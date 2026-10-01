package com.thesis.pickleballserve

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
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

        fun load(context: Context): PaddleStream? = try {
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
                val det = PaddleDetector.open(context)
                Log.i(TAG, "loaded band=[$low, $high] rule=${cfg.getString("threshold_rule")}")
                PaddleStream(det, low, high, cfg.optBoolean("direction_validated", false))
            }
        } catch (e: Throwable) {
            Log.e(TAG, "paddle stream disabled", e)
            null
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

    private val angles = ArrayList<Pair<Int, Double>>()
    private val speeds = ArrayList<Float>()
    private var lastWristX = Float.NaN
    private var lastWristY = Float.NaN
    private var frameIdx = 0
    private var strideCount = 0
    private var pxBuffer = IntArray(0)

    fun clear() {
        angles.clear()
        speeds.clear()
        lastWristX = Float.NaN
        lastWristY = Float.NaN
        frameIdx = 0
        strideCount = 0
    }

    /**
     * wristX/wristY: normalized right-wrist coords (NaN if pose unknown);
     * bitmap: frame for detection (null = speed bookkeeping only).
     */
    fun add(wristX: Float, wristY: Float, bitmap: Bitmap?) {
        val idx = frameIdx++
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

        if (bitmap == null || strideCount++ % DETECT_STRIDE != 0) return
        val dets = detector.detectAll(bitmap)
        val maxPx = PaddleDetector.WRIST_MAX_PX * bitmap.width / PaddleDetector.REF_WIDTH_PX
        val det = PaddleDetector.pick(dets, wristX * bitmap.width, wristY * bitmap.height, maxPx)
            ?: return
        val rect = PaddleGeometry.expandBox(det.box, bitmap.width, bitmap.height)
        if (rect.width < MIN_CROP || rect.height < MIN_CROP) return
        val n = bitmap.width * bitmap.height
        if (pxBuffer.size != n) pxBuffer = IntArray(n)
        bitmap.getPixels(pxBuffer, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val rgba = Mat(bitmap.height, bitmap.width, CvType.CV_8UC4)
        rgba.put(0, 0, pxBuffer)
        val gray = PaddleGeometry.grayFromRgba(rgba)
        rgba.release()
        val crop = Mat(gray, rect)
        val res = PaddleGeometry.cropAngle(crop)
        crop.release()
        gray.release()
        if (res.status == "ok" && res.angle != null) {
            angles.add(idx to res.angle)
        }
    }

    /** Python estimate_contact_frame() on the collected wrist speeds. */
    fun contactFrame(): Int = contactFrame(speeds.map { it.toDouble() }, frameIdx)

    /** Window median scalar (full-series fallback), or null when no angle. */
    fun compute(): Double? = scalarFor(angles, contactFrame())

    /** Coaching must use contact-window evidence; never the full-series fallback. */
    fun computeForFeedback(): Double? = contactScalarFor(angles, contactFrame())

    fun verdict(scalar: Double?): String = verdictFor(scalar, bandLow, bandHigh)

    val low: Double get() = bandLow
    val high: Double get() = bandHigh

    override fun close() {
        detector.close()
    }
}
