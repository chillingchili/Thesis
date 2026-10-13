package com.thesis.pickleballserve

import org.junit.Assert.*
import org.junit.Test

class SessionLogTest {
    @Test fun sessionNumbersIncrementAndClearResets() {
        SessionStore.clear()
        assertEquals(1, SessionStore.nextNumber())
        assertEquals(2, SessionStore.nextNumber())
        SessionStore.add(ServeRecord(1, "20:00", "Drive", .8f, "hip", "paddle", "FEEDBACK", "cue"))
        assertEquals(1, SessionStore.size())
        assertEquals("Drive", SessionStore.snapshot().single().serveType)
        SessionStore.clear()
        assertEquals(0, SessionStore.size())
        assertEquals(1, SessionStore.nextNumber())
        SessionStore.clear()
    }

    @Test fun landingAmendmentKeepsOtherFields() {
        SessionStore.clear()
        SessionStore.add(ServeRecord(1, "20:00", "Drive", .8f, "hip", "paddle", "FEEDBACK", "cue"))
        assertTrue(SessionStore.update(1) { it.copy(landing = "Deep-Right", feedback = "cue\nBall landing: Deep-Right.") })
        val amended = SessionStore.snapshot().single()
        assertEquals("Deep-Right", amended.landing)
        assertEquals("Drive", amended.serveType)
        assertFalse(SessionStore.update(99) { it })
        SessionStore.clear()
    }
}
