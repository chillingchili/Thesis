package com.thesis.pickleballserve

import kotlin.math.exp

/** Normalize angular displacement to the 80 ms CPU reference interval. */
class MotionRate {
    private var previousMs: Long? = null
    private var smoothed = 0f
    fun reset() { previousMs = null; smoothed = 0f }
    fun update(displacement: Float, timestampMs: Long): Float {
        val previous = previousMs
        previousMs = timestampMs
        if (previous == null || timestampMs <= previous || timestampMs-previous > 500L) {
            smoothed = 0f
            return 0f
        }
        val elapsed = timestampMs-previous
        val rate = displacement * 80f / elapsed
        val alpha = (1.0-exp(-elapsed/200.0)).toFloat()
        smoothed += alpha*(rate-smoothed)
        return smoothed
    }
}
