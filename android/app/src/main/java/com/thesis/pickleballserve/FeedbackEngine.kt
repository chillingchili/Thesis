package com.thesis.pickleballserve

import org.json.JSONObject
import java.util.Locale

/** Decision-level fusion only. No inference, threshold fitting or generated advice. */
class FeedbackEngine private constructor(
    val version: String,
    val validationStatus: String,
    private val rules: List<Rule>,
) {
    enum class PaddleState { OPTIMAL, OUTSIDE_BASELINE, TOO_OPEN, TOO_CLOSED }
    enum class Status { CORRECTIVE, POSITIVE, PARTIAL, LOW_CONFIDENCE, INSUFFICIENT_DATA }

    data class Input(
        val serveType: String?,
        val gruConfidence: Float?,
        val shiftSufficient: Boolean?,
        val paddle: PaddleState?,
        val placementActive: Boolean = false,
        val placementZone: String? = null,
    )

    data class Rule(val id: String, val condition: String, val serveTypes: Set<String>,
                    val text: String, val sourceClips: List<String>)
    data class Result(val status: Status, val ruleIds: List<String>, val messages: List<String>,
                      val notifications: List<String>, val version: String,
                      val validationStatus: String) {
        val text: String get() = (messages + notifications).joinToString("\n")
    }

    fun evaluate(input: Input): Result {
        val notes = mutableListOf<String>()
        if (input.shiftSufficient == null) notes += "Weight shift unavailable."
        if (input.paddle == null) notes += "Paddle orientation unavailable."
        val zone = input.placementZone?.takeIf { it in ZONES }
        if (input.placementActive) {
            notes += if (zone == null) "Ball landing unavailable." else "Ball landing: $zone."
        }
        fun result(status: Status, messages: List<String>, ids: List<String> = emptyList()) =
            Result(status, ids, messages, notes.toList(), version, validationStatus)

        val subtype = input.serveType?.lowercase(Locale.ROOT)
        val confidence = input.gruConfidence
        if (subtype !in SERVE_TYPES || confidence == null || !confidence.isFinite() ||
            confidence !in 0f..1f) {
            return result(Status.INSUFFICIENT_DATA,
                listOf("Serve classification unavailable. Record another serve for coaching feedback."))
        }
        if (confidence < MIN_CONFIDENCE) {
            return result(Status.LOW_CONFIDENCE,
                listOf("Serve classification confidence is below 60%. Record another serve for coaching feedback."))
        }
        val conditions = buildSet {
            if (input.shiftSufficient == false) add("shift_insufficient")
            when (input.paddle) {
                PaddleState.OUTSIDE_BASELINE -> add("paddle_outside_baseline")
                PaddleState.TOO_OPEN -> add("paddle_too_open")
                PaddleState.TOO_CLOSED -> add("paddle_too_closed")
                else -> Unit
            }
        }
        val matches = rules.filter { subtype in it.serveTypes && it.condition in conditions }
        if (matches.isNotEmpty()) {
            return result(Status.CORRECTIVE, matches.map { it.text }, matches.map { it.id })
        }
        if (input.shiftSufficient == true && input.paddle == PaddleState.OPTIMAL &&
            (!input.placementActive || (zone != null && !zone.startsWith("Fault-")))) {
            return result(Status.POSITIVE,
                listOf("Good work: your measured weight shift and paddle angle are within the reference ranges. Keep practicing."),
                listOf("ASSESSED_MECHANICS_OK"))
        }
        return result(Status.PARTIAL,
            listOf("No supported corrective cue for the available measurements. Overall serve form has not been assessed."))
    }

    companion object {
        const val MIN_CONFIDENCE = 0.60f
        private val SERVE_TYPES = setOf("drive", "lob", "topspin")
        private val CONDITIONS = setOf("shift_insufficient", "paddle_outside_baseline",
            "paddle_too_open", "paddle_too_closed")
        private val ZONES = setOf("Deep-Left", "Deep-Right", "Short-Left", "Short-Right",
            "Fault-Long", "Fault-Wide", "Fault-Short")

        fun fromJson(text: String): FeedbackEngine {
            val json = JSONObject(text)
            val array = json.getJSONArray("rules")
            val rules = (0 until array.length()).map { i ->
                val row = array.getJSONObject(i)
                fun strings(key: String): List<String> = row.getJSONArray(key).let { a ->
                    (0 until a.length()).map { a.getString(it) }
                }
                Rule(row.getString("id"), row.getString("condition"),
                    strings("serve_types").toSet(), row.getString("text"), strings("source_clips"))
            }
            require(rules.isNotEmpty() && rules.map { it.id }.distinct().size == rules.size)
            require(rules.all { it.id.isNotBlank() && it.condition in CONDITIONS &&
                it.serveTypes.isNotEmpty() && SERVE_TYPES.containsAll(it.serveTypes) &&
                it.text.isNotBlank() && it.sourceClips.isNotEmpty() })
            return FeedbackEngine(json.getString("version"), json.getString("validation_status"), rules)
        }

        /** Never interpret invalid scalars as an acceptable angle. */
        fun paddleState(angle: Double?, low: Double, high: Double,
                        directionValidated: Boolean = false): PaddleState? {
            if (angle == null || !angle.isFinite() || !low.isFinite() || !high.isFinite() ||
                angle !in 0.0..<180.0 || low < 0 || high >= 180 || low > high) return null
            return when {
                angle in low..high -> PaddleState.OPTIMAL
                !directionValidated -> PaddleState.OUTSIDE_BASELINE
                angle < low -> PaddleState.TOO_CLOSED
                else -> PaddleState.TOO_OPEN
            }
        }
    }
}
