package com.thesis.pickleballserve.landing

import org.json.JSONObject

/**
 * Dual-phone transfer payload for the auxiliary ball-landing module.
 *
 * Phone B (court view) encodes the bounce result as a QR code; phone A
 * (side view) scans it and fuses the zone through [com.thesis.pickleballserve.FeedbackEngine]
 * with `placementActive = true`. The payload stays under ~120 chars so the
 * QR remains a low-density, sunlight-readable code.
 *
 * Pure Kotlin + org.json: no Android dependency, JVM-testable.
 */
data class LandingResult(
    val zone: String,
    val xFt: Double? = null,
    val yFt: Double? = null,
) {
    fun encode(): String {
        val json = JSONObject()
            .put("v", VERSION)
            .put("zone", zone)
        if (xFt != null && xFt.isFinite()) json.put("x", xFt)
        if (yFt != null && yFt.isFinite()) json.put("y", yFt)
        return json.toString()
    }

    companion object {
        const val VERSION = 1

        /** Must stay in sync with FeedbackEngine's placement zones. */
        val ZONES = setOf(
            "Deep-Left", "Deep-Right", "Short-Left", "Short-Right",
            "Fault-Long", "Fault-Wide", "Fault-Short",
        )

        /** Returns the parsed result, or null when the text is not a landing code. */
        fun parse(text: String): LandingResult? {
            if (text.isBlank() || text.length > 512) return null
            val json = try {
                JSONObject(text)
            } catch (_: Exception) {
                return null
            }
            if (json.optInt("v", VERSION) != VERSION) return null
            val zone = json.optString("zone", "")
            if (zone !in ZONES) return null
            return LandingResult(zone, optCoord(json, "x"), optCoord(json, "y"))
        }

        private fun optCoord(json: JSONObject, key: String): Double? {
            if (!json.has(key) || json.isNull(key)) return null
            val value = json.optDouble(key, Double.NaN)
            return if (value.isFinite()) value else null
        }
    }
}
