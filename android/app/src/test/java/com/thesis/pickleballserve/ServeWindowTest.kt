package com.thesis.pickleballserve

import org.junit.Assert.*
import org.junit.Test

class ServeWindowTest {
    private fun frame(value: Int) = FloatArray(10) { value.toFloat() }
    @Test fun preRollExpiresAndDoesNotDuplicateTrigger() {
        val window = ServeWindow()
        for (i in 0..40) window.add(i * 50L, frame(i))
        assertEquals(21, window.size)
        window.begin()
        window.add(2050, frame(41))
        val input = window.input()
        assertEquals(22, input.validFrames)
        assertEquals(20f, input.tensor[0][0][0], 0f)
        assertEquals(41f, input.tensor[0][21][0], 0f)
        assertEquals(0f, input.tensor[0][22][0], 0f)
    }
    @Test fun completeWindowKeepsSetupAndFollowThroughWithUniqueRealFrames() {
        val window = ServeWindow()
        window.begin()
        for (i in 0..300) window.add(i * i.toLong(), frame(i))
        val input = window.input()
        assertEquals(128, input.validFrames)
        assertEquals(0f, input.tensor[0].first()[0], 0f)
        assertEquals(300f, input.tensor[0].last()[0], 0f)
        assertEquals(128, input.tensor[0].map { it[0] }.toSet().size)
        assertTrue(input.tensor[0][64][0] > 190f) // elapsed-time midpoint, not frame-count midpoint
    }
    @Test fun shortWindowsStayShortAndCopiesDoNotMutateHistory() {
        val window = ServeWindow()
        val features = frame(7)
        window.add(10, features)
        features[0] = 99f
        window.add(10, frame(8))
        window.add(9, frame(9))
        assertEquals(1, window.input().validFrames)
        val input = window.input()
        assertEquals(7f, input.tensor[0][0][0], 0f)
        input.tensor[0][0][0] = 44f
        assertEquals(7f, window.input().tensor[0][0][0], 0f)
        window.clear()
        assertEquals(0, window.size)
        assertFalse(window.active)
    }
}
