package com.thesis.pickleballserve

import org.junit.Assert.*
import org.junit.Test

class MotionRateTest {
    @Test fun constantAngularVelocityIsIndependentOfFrameInterval() {
        fun sample(step: Long): Float {
            val motion = MotionRate()
            var result = motion.update(0f, 0L)
            for (t in step..1600L step step) result = motion.update(step*0.001f, t)
            return result
        }
        assertEquals(sample(40), sample(80), 0.00001f)
    }
    @Test fun longTrackingGapsAndResetDoNotInventMotion() {
        val motion = MotionRate()
        motion.update(0f, 0L)
        assertTrue(motion.update(0.1f, 80L)>0f)
        assertEquals(0f, motion.update(1f, 1000L), 0f)
        motion.reset()
        assertEquals(0f, motion.update(1f, 1080L), 0f)
    }
}
