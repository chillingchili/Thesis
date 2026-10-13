package com.thesis.pickleballserve.landing

import org.junit.Assert.*
import org.junit.Test

class LandingResultTest {
    @Test fun roundTripKeepsZoneAndCoordinates() {
        val original = LandingResult("Deep-Right", 8.2, 32.5)
        val parsed = LandingResult.parse(original.encode())
        assertEquals(original, parsed)
    }

    @Test fun payloadStaysShortForLowDensityQr() {
        val payload = LandingResult("Fault-Long", 21.5, 45.0).encode()
        assertTrue("payload too long for a robust code: $payload", payload.length <= 120)
    }

    @Test fun missingCoordinatesSurviveAsNull() {
        val parsed = LandingResult.parse(LandingResult("Short-Left").encode())
        assertEquals(LandingResult("Short-Left", null, null), parsed)
    }

    @Test fun garbageNeverParses() {
        for (text in listOf("", "   ", "hello", "Deep-Right", "{not json", "[1,2]", "x".repeat(600))) {
            assertNull("should reject: $text", LandingResult.parse(text))
        }
    }

    @Test fun unknownZoneOrVersionIsRejected() {
        assertNull(LandingResult.parse("""{"v":1,"zone":"Center"}"""))
        assertNull(LandingResult.parse("""{"v":1}"""))
        assertNull(LandingResult.parse("""{"v":2,"zone":"Deep-Left"}"""))
        assertNull(LandingResult.parse("""{"v":1,"zone":"deep-left"}"""))
    }

    @Test fun nonFiniteCoordinatesAreDroppedNotRejected() {
        // NaN serializes as a non-numeric literal; the zone must still parse.
        val parsed = LandingResult.parse("""{"v":1,"zone":"Deep-Left","x":null}""")
        assertEquals(LandingResult("Deep-Left", null, null), parsed)
    }

    @Test fun allFeedbackZonesAreAccepted() {
        for (zone in LandingResult.ZONES) {
            assertEquals(zone, LandingResult.parse(LandingResult(zone).encode())?.zone)
        }
    }
}
