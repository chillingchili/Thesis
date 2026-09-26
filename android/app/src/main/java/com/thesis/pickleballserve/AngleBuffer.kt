package com.thesis.pickleballserve

/**
 * Ring buffer of joint-angle feature vectors for GRU inference.
 *
 * Matches scripts/train_gru.py pad_or_truncate():
 *   - Feature vector per frame: 10 floats (5 angles x sin/cos).
 *   - Fixed window length: 128 frames.
 *   - Short sequences: zero-padded at the END (trailing pad, Masking ignores).
 *   - Long sequences: keeps the FIRST 128 frames (truncate tail).
 *
 * For live streaming we keep a sliding window of the most recent 128 valid
 * frames; when the buffer has fewer than 128, the tail is zero-padded — same
 * as offline short clips.
 */
class AngleBuffer(private val seqLen: Int = 128, private val featDim: Int = 10) {

    // Store as flat array [frame0_f0..f9, frame1_f0..f9, ...]
    private val buf = FloatArray(seqLen * featDim)
    private var count = 0  // number of valid frames currently held (0..seqLen)

    /** Append one frame's 10-dim feature. Overwrites oldest when full. */
    fun add(features: FloatArray) {
        require(features.size == featDim) { "expected $featDim features, got ${features.size}" }
        if (count < seqLen) {
            val off = count * featDim
            System.arraycopy(features, 0, buf, off, featDim)
            count++
        } else {
            // Shift left by one frame, write new at end (sliding window)
            System.arraycopy(buf, featDim, buf, 0, (seqLen - 1) * featDim)
            System.arraycopy(features, 0, buf, (seqLen - 1) * featDim, featDim)
        }
    }

    /** Number of valid (non-pad) frames currently in the window. */
    fun validFrames(): Int = count

    /** True when the window has been filled at least once (full 128 valid frames). */
    fun isFull(): Boolean = count >= seqLen

    fun clear() {
        java.util.Arrays.fill(buf, 0f)
        count = 0
    }

    /**
     * Build a (1, 128, 10) input tensor for TFLite.
     * Valid frames are at the START; remaining slots are zero (trailing pad),
     * matching pad_or_truncate() in train_gru.py / testrun.py.
     */
    fun toInputTensor(): Array<Array<FloatArray>> {
        val out = Array(1) { Array(seqLen) { FloatArray(featDim) } }
        for (t in 0 until count) {
            val src = t * featDim
            for (f in 0 until featDim) {
                out[0][t][f] = buf[src + f]
            }
        }
        // frames count..seqLen remain zeros (trailing pad)
        return out
    }
}
