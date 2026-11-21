package com.thesis.pickleballserve

import org.junit.Assert.assertEquals
import org.junit.Test

class FramingCheckTest {
    private fun check(noseY: Float, ankleY: Float, hipX: Float) = FramingCheck.evaluate(noseY, ankleY, hipX)

    @Test fun framingStates() {
        assertEquals(FramingCheck.Result.NO_PLAYER, FramingCheck.evaluate(null))
        assertEquals(FramingCheck.Result.NO_PLAYER, check(Float.NaN, .9f, .5f))
        assertEquals(FramingCheck.Result.OK, check(.12f, .9f, .5f))
        assertEquals(FramingCheck.Result.HEAD_CUT, check(-.02f, .9f, .5f))
        assertEquals(FramingCheck.Result.FEET_CUT, check(.1f, 1.04f, .5f))
        assertEquals(FramingCheck.Result.TOO_FAR, check(.4f, .7f, .5f))
        assertEquals(FramingCheck.Result.TOO_CLOSE, check(.02f, .98f, .5f))
        assertEquals(FramingCheck.Result.OFF_CENTER, check(.12f, .9f, .85f))
    }
}
