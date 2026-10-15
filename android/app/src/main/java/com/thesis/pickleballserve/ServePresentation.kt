package com.thesis.pickleballserve

import java.util.Locale

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
}
