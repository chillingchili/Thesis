package com.thesis.pickleballserve

import android.graphics.PointF

/**
 * Setup-screen framing check from one pose (MediaPipe normalized coordinates of
 * the analysis frame, so it judges what the model sees, not the preview crop).
 * Display guidance only; it never gates or changes serve analysis.
 */
object FramingCheck {
    /** Outline drawn over the preview, in normalized analysis-frame coordinates. */
    const val OUTLINE_LEFT = 0.30f
    const val OUTLINE_RIGHT = 0.70f
    const val OUTLINE_TOP = 0.05f
    const val OUTLINE_BOTTOM = 0.97f

    private const val NOSE = 0
    private const val L_HIP = 23
    private const val R_HIP = 24
    private const val L_ANKLE = 27
    private const val R_ANKLE = 28
    private const val EDGE = 0.01f
    private const val MIN_BODY = 0.40f
    private const val MAX_BODY = 0.93f

    enum class Result(val message: String, val ok: Boolean) {
        NO_PLAYER("No player found. Step into view", false),
        HEAD_CUT("Head is cut off. Step back or tilt the phone", false),
        FEET_CUT("Feet are cut off. Step back or tilt the phone", false),
        TOO_FAR("Too far. Move closer to the phone", false),
        TOO_CLOSE("Too close. Step back from the phone", false),
        OFF_CENTER("Step into the outline", false),
        OK("Full body in outline", true),
    }

    fun evaluate(landmarks: Array<PointF>?): Result {
        if (landmarks == null || landmarks.size < 33) return Result.NO_PLAYER
        return evaluate(
            noseY = landmarks[NOSE].y,
            ankleY = maxOf(landmarks[L_ANKLE].y, landmarks[R_ANKLE].y),
            hipX = (landmarks[L_HIP].x + landmarks[R_HIP].x) / 2f,
        )
    }

    /** Pure rule on the three values that matter (unit-testable without Android). */
    fun evaluate(noseY: Float, ankleY: Float, hipX: Float): Result {
        if (!noseY.isFinite() || !ankleY.isFinite() || !hipX.isFinite()) return Result.NO_PLAYER
        if (noseY < EDGE) return Result.HEAD_CUT
        if (ankleY > 1f - EDGE) return Result.FEET_CUT
        // Nose-to-ankle understates full height a little; the bounds allow for it.
        val body = ankleY - noseY
        if (body < MIN_BODY) return Result.TOO_FAR
        if (body > MAX_BODY) return Result.TOO_CLOSE
        if (hipX < OUTLINE_LEFT || hipX > OUTLINE_RIGHT) return Result.OFF_CENTER
        return Result.OK
    }
}
