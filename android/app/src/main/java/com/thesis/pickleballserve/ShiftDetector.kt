package com.thesis.pickleballserve

import android.content.Context
import android.util.Log
import org.json.JSONObject
import kotlin.math.floor

/**
 * Deterministic body-weight-shift detector.
 *
 * Metric (ported from scripts/derive_shift_threshold.py — keep identical):
 *   hip_x  = (x23 + x24) / 2 per frame, then 5-frame moving average (valid)
 *   body_h = median over frames of |y_nose - mean(y_ankle27, y_ankle28)|
 *   shift  = (p95(hip_x) - p5(hip_x)) / body_h
 *   pass   = shift >= threshold   (threshold = p10 of CoachA good-form clips)
 *
 * Samples are added frame-by-frame during the gated serve window only,
 * cleared together with the angle buffer.
 */
class ShiftDetector private constructor(private val threshold: Float) {

    private val hipX = ArrayList<Float>(256)
    private val bodyH = ArrayList<Float>(256)

    fun clear() {
        hipX.clear()
        bodyH.clear()
    }

    fun add(hipCenterX: Float, bodyHeight: Float) {
        if (bodyHeight < 1e-6f) return
        hipX.add(hipCenterX)
        bodyH.add(bodyHeight)
    }

    /** Shift value over collected samples, or null if too few / degenerate. */
    fun compute(): Float? {
        if (hipX.size < 10) return null
        // 5-frame moving average (valid), matches np.convolve(..., mode="valid")
        val w = SMOOTH_WINDOW
        val n = hipX.size
        if (n < w) return null
        val smoothed = FloatArray(n - w + 1)
        var run = 0f
        for (i in 0 until w) run += hipX[i]
        smoothed[0] = run / w
        for (i in w until n) {
            run += hipX[i] - hipX[i - w]
            smoothed[i - w + 1] = run / w
        }
        val rng = percentile(smoothed, P_HIGH) - percentile(smoothed, P_LOW)
        val sortedH = bodyH.toFloatArray().also { java.util.Arrays.sort(it) }
        val medH = percentile(sortedH, 50f)
        if (medH < 1e-6f) return null
        return rng / medH
    }

    fun passed(): Boolean? = compute()?.let { it >= threshold }

    val thr: Float get() = threshold

    companion object {
        private const val TAG = "ShiftDetector"
        private const val SMOOTH_WINDOW = 5
        private const val P_LOW = 5f
        private const val P_HIGH = 95f
        private const val MIN_SAMPLES = 10

        /** Load threshold from assets/shift_config.json; null if absent. */
        fun load(context: Context): ShiftDetector? = try {
            val json = context.assets.open("shift_config.json")
                .bufferedReader().use { it.readText() }
            val cfg = JSONObject(json)
            val thr = cfg.getDouble("threshold").toFloat()
            Log.i(TAG, "loaded threshold=$thr rule=${cfg.getString("threshold_rule")}")
            ShiftDetector(thr)
        } catch (e: Throwable) {
            Log.e(TAG, "shift_config.json missing/invalid — detector disabled", e)
            null
        }

        /** numpy-compatible linear-interpolation percentile on a sorted copy. */
        fun percentile(values: FloatArray, p: Float): Float {
            val sorted = if (values.isEmpty()) return 0f else
                if (isSorted(values)) values else values.copyOf().also { java.util.Arrays.sort(it) }
            if (sorted.size == 1) return sorted[0]
            val pos = (p / 100f) * (sorted.size - 1)
            val lo = floor(pos.toDouble()).toInt()
            val hi = lo + 1
            return if (hi >= sorted.size) sorted[lo]
            else sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo])
        }

        private fun isSorted(a: FloatArray): Boolean {
            for (i in 1 until a.size) if (a[i] < a[i - 1]) return false
            return true
        }
    }
}
