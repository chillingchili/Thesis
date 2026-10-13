package com.thesis.pickleballserve.landing

import org.junit.Assert.*
import org.junit.Test

class QrCodeTest {
    @Test fun encodeDecodeRoundTrip() {
        val payload = LandingResult("Short-Right", 12.0, 20.0).encode()
        val matrix = QrCode.encodeMatrix(payload)
        val text = QrCode.decodePixels(QrCode.matrixToPixels(matrix), matrix.width, matrix.height)
        assertEquals(payload, text)
    }

    @Test fun blankImageDecodesToNullWithoutThrowing() {
        val size = 264
        assertNull(QrCode.decodePixels(IntArray(size * size) { 0xFFFFFFFF.toInt() }, size, size))
    }

    @Test fun decodedPayloadParsesAsLandingResult() {
        val payload = LandingResult("Fault-Wide").encode()
        val matrix = QrCode.encodeMatrix(payload)
        val text = QrCode.decodePixels(QrCode.matrixToPixels(matrix), matrix.width, matrix.height)
        assertEquals(LandingResult("Fault-Wide", null, null), LandingResult.parse(text!!))
    }
}
