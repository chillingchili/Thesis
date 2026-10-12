package com.thesis.pickleballserve

/** Allow settling relative to the observed peak, without ending during its immediate tail. */
object ServeEndPolicy {
    fun quiet(motion: Float, idleThreshold: Float, peak: Float, sincePeakMs: Long): Boolean =
        motion < maxOf(idleThreshold, if (sincePeakMs >= 300) peak * 0.20f else 0f)
}
