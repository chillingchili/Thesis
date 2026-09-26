package com.thesis.pickleballserve

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max

/**
 * Port of scripts/extract_timing_features.py timing_features() to Kotlin.
 *
 * Input: joint-angle sequence as sin/cos, shape (frames, 5, 2) flattened row-major
 *        from AngleBuffer: each frame is [sin0,cos0, sin1,cos1, ..., sin4,cos4].
 * Output: 32-dim FloatArray:
 *   per angle k in 0..4: [t_peak, t_trough, t_maxvel, amp, mean, std]  (6 x 5 = 30)
 *   + [lag_pk(shoulder-elbow), lag_tr(shoulder-elbow)]                 (2)
 *
 * Times normalized to [0,1] over valid length L; trailing zero-pad excluded.
 */
object TimingFeatures {

    const val N_FEATURES = 32
    private const val N_ANGLES = 5
    private const val MIN_LEN = 4

    /**
     * @param flatWindow (seqLen * 10) flat array from AngleBuffer (frame-major: 10 floats/frame)
     * @param validFrames number of real frames (rest is trailing zero-pad)
     */
    fun fromFlatWindow(flatWindow: FloatArray, validFrames: Int): FloatArray {
        if (validFrames < MIN_LEN) return FloatArray(N_FEATURES)

        // Reconstruct angles: for each valid frame, 5 angles as atan2(sin, cos)
        val L = validFrames
        val ang = Array(L) { FloatArray(N_ANGLES) }
        for (t in 0 until L) {
            val off = t * N_ANGLES * 2
            for (k in 0 until N_ANGLES) {
                val s = flatWindow[off + k * 2]
                val c = flatWindow[off + k * 2 + 1]
                ang[t][k] = atan2(s, c)
            }
        }

        val feats = FloatArray(N_FEATURES)
        val peaks = FloatArray(N_ANGLES)
        val troughs = FloatArray(N_ANGLES)
        val denom = max(L - 1, 1).toFloat()
        var fi = 0

        for (k in 0 until N_ANGLES) {
            // argmax / argmin of angle series
            var iPk = 0
            var iTr = 0
            var amax = ang[0][k]
            var amin = ang[0][k]
            for (t in 1 until L) {
                if (ang[t][k] > amax) { amax = ang[t][k]; iPk = t }
                if (ang[t][k] < amin) { amin = ang[t][k]; iTr = t }
            }
            // velocity = diff with prepend 0
            var iVel = 0
            var vmax = 0f
            var prev = 0f
            for (t in 0 until L) {
                val v = ang[t][k] - prev
                prev = ang[t][k]
                val av = abs(v)
                if (av > vmax) { vmax = av; iVel = t }
            }

            // mean / std
            var sum = 0.0
            for (t in 0 until L) sum += ang[t][k]
            val mean = (sum / L).toFloat()
            var sq = 0.0
            for (t in 0 until L) {
                val d = ang[t][k] - mean
                sq += (d * d).toDouble()
            }
            val std = kotlin.math.sqrt((sq / L).toFloat())

            peaks[k] = iPk / denom
            troughs[k] = iTr / denom

            feats[fi++] = iPk / denom
            feats[fi++] = iTr / denom
            feats[fi++] = iVel / denom
            feats[fi++] = amax - amin
            feats[fi++] = mean
            feats[fi++] = std
        }

        // cross-angle lags: shoulder (idx 1) - elbow (idx 0)
        feats[fi++] = peaks[1] - peaks[0]
        feats[fi] = troughs[1] - troughs[0]
        return feats
    }
}
