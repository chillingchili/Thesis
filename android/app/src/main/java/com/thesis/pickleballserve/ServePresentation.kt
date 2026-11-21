package com.thesis.pickleballserve

import java.util.Locale
import kotlin.math.roundToInt

/** Display-only formatting; never changes inference or feedback policy. */
object ServePresentation {
    fun classification(label: String?, confidence: Float?): String {
        val name = when (label?.lowercase(Locale.ROOT)) {
            "drive" -> "Drive"
            "lob" -> "Lob"
            "topspin" -> "Topspin"
            else -> return "Unavailable"
        }
        if (confidence == null || !confidence.isFinite() || confidence !in 0f..1f) return "Unavailable"
        return if (confidence < FeedbackEngine.MIN_CONFIDENCE) "$name · uncertain" else name
    }
    fun hip(value: Float?, passed: Boolean?): String {
        if (value == null || !value.isFinite() || value < 0 || passed == null) return "Unavailable"
        return String.format(Locale.getDefault(), "%s · %.3f shift",
            if (passed) "Within reference" else "Below reference", value)
    }
    fun paddle(angle: Double?, state: FeedbackEngine.PaddleState?): String {
        if (angle == null || !angle.isFinite() || angle !in 0.0..<180.0 || state == null) return "Unavailable"
        return String.format(Locale.getDefault(), "%s · %.1f°",
            if (state == FeedbackEngine.PaddleState.OPTIMAL) "Within reference" else "Outside reference",
            angle)
    }

    // --- Live result cards (Figure 7.2). Display wording only. ---

    enum class Tone { GOOD, WARN, NEUTRAL }

    /** One measurement card: big value, optional status chip, one-line note. */
    data class Metric(val value: String, val chip: String?, val tone: Tone, val note: String)

    /** Serve type card: big value plus a detail line; [reliable] gates the class bars. */
    data class ServeType(val value: String, val detail: String, val reliable: Boolean)

    private fun validConfidence(c: Float?) = c != null && c.isFinite() && c in 0f..1f

    fun serveTypeCard(label: String?, confidence: Float?, probs: FloatArray?): ServeType {
        val name = when (label?.lowercase(Locale.ROOT)) {
            "drive" -> "Drive"; "lob" -> "Lob"; "topspin" -> "Topspin"; else -> null
        }
        if (name == null || !validConfidence(confidence))
            return ServeType("Not measured", "Not enough of the serve was captured.", false)
        if (confidence!! < FeedbackEngine.MIN_CONFIDENCE) {
            val detail = if (probs != null && probs.size == 3 && probs.all { it.isFinite() }) {
                listOf("Drive" to probs[0], "Lob" to probs[1], "Topspin" to probs[2])
                    .sortedByDescending { it.second }
                    .joinToString(" · ") { "${it.first} ${pct(it.second)}%" }
            } else "Low confidence"
            return ServeType("Not sure", detail, false)
        }
        return ServeType(name, "${pct(confidence)}% confident", true)
    }

    fun hipCard(value: Float?, passed: Boolean?): Metric {
        if (value == null || !value.isFinite() || value < 0 || passed == null)
            return Metric("Not measured", null, Tone.NEUTRAL,
                "Hip motion could not be measured on this serve.")
        return if (passed) Metric("Good", "Within range", Tone.GOOD,
            "Hip motion was within the coach reference band.")
        else Metric("Limited", "Needs work", Tone.WARN,
            "Measured hip motion was below the coach reference band.")
    }

    fun paddleCard(angle: Double?, state: FeedbackEngine.PaddleState?, low: Double?, high: Double?): Metric {
        if (angle == null || !angle.isFinite() || angle !in 0.0..<180.0 || state == null)
            return Metric("Not detected", null, Tone.WARN,
                "The paddle was hidden at contact. Keep the paddle face visible to the camera.")
        val band = if (low != null && high != null && low.isFinite() && high.isFinite())
            String.format(Locale.getDefault(), "Coach reference band: %.0f–%.0f° at contact.", low, high)
        else "Compared with the coach reference band at contact."
        val value = String.format(Locale.getDefault(), "%.0f°", angle)
        return when (state) {
            FeedbackEngine.PaddleState.OPTIMAL -> Metric(value, "Within range", Tone.GOOD, band)
            FeedbackEngine.PaddleState.OUTSIDE_BASELINE -> Metric(value, "Outside reference", Tone.WARN, band)
            FeedbackEngine.PaddleState.TOO_OPEN -> Metric(value, "Too open", Tone.WARN, band)
            FeedbackEngine.PaddleState.TOO_CLOSED -> Metric(value, "Too closed", Tone.WARN, band)
        }
    }

    /** Partial when the serve type is unreliable or any measurement is missing. */
    fun isPartial(serve: ServeType, hip: Metric, paddle: Metric): Boolean =
        !serve.reliable || hip.chip == null || paddle.chip == null

    /** Status pill after a serve: "Serve 7 · analyzed in 1.6 s" or "Serve 8 · partial result". */
    fun resultStatus(number: Int, partial: Boolean, analysisMs: Long?): String = when {
        partial -> "Serve $number · partial result"
        analysisMs != null && analysisMs >= 0 ->
            String.format(Locale.getDefault(), "Serve %d · analyzed in %.1f s", number, analysisMs / 1000.0)
        else -> "Serve $number · result ready"
    }

    /** Coaching cues as bullets; engine notes follow as plain lines. */
    fun coachingText(messages: List<String>): String =
        messages.filter { it.isNotBlank() }.joinToString("\n") { "• $it" }

    /** Whole-percent confidence, rounded; the one formatter for live, review and speech. */
    fun pct(v: Float) = (v * 100f).roundToInt()
}
