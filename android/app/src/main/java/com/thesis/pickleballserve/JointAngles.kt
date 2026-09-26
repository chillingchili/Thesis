package com.thesis.pickleballserve

import android.graphics.PointF
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Port of scripts/extract_joint_angles.py clip_angles() to Kotlin.
 *
 * BlazePose landmark indices (MediaPipe Pose Landmarker uses same ordering):
 *   LEFT_SHOULDER=11, RIGHT_SHOULDER=12, RIGHT_ELBOW=14, RIGHT_WRIST=16,
 *   RIGHT_INDEX=20, LEFT_HIP=23, RIGHT_HIP=24
 *
 * Input: 33 landmarks as PointF array (x,y in image/normalized coords, y grows down).
 * Output: FloatArray(10) = [sin_elbow, cos_elbow, sin_shoulder, cos_shoulder,
 *                            sin_wrist, cos_wrist, sin_torso, cos_torso,
 *                            sin_twist, cos_twist]
 */
object JointAngles {

    private const val LEFT_SHOULDER = 11
    private const val RIGHT_SHOULDER = 12
    private const val RIGHT_ELBOW = 14
    private const val RIGHT_WRIST = 16
    private const val RIGHT_INDEX = 20
    private const val LEFT_HIP = 23
    private const val RIGHT_HIP = 24

    data class Vec2(val x: Float, val y: Float) {
        operator fun minus(o: Vec2) = Vec2(x - o.x, y - o.y)
        operator fun plus(o: Vec2) = Vec2(x + o.x, y + o.y)
        fun scale(s: Float) = Vec2(x * s, y * s)
        fun dot(o: Vec2) = x * o.x + y * o.y
        fun length() = sqrt(x * x + y * y)
    }

    private fun unit(v: Vec2, eps: Float = 1e-8f): Vec2 {
        val n = v.length()
        val d = if (n < eps) eps else n
        return Vec2(v.x / d, v.y / d)
    }

    /** Angle at vertex b formed by points a-b-c (radians, interior angle). */
    private fun interiorAngle(a: Vec2, b: Vec2, c: Vec2): Float {
        val v1 = unit(a - b)
        val v2 = unit(c - b)
        val cosTheta = (v1.dot(v2)).coerceIn(-1f, 1f)
        return atan2(sqrt(1f - cosTheta * cosTheta), cosTheta) // acos via atan2 for stability
    }

    /** Signed angle between v and 'up' in image coords (y grows down). */
    private fun signedAngleFromVertical(v: Vec2): Float = atan2(v.x, -v.y)

    /** Signed angle from v1 to v2 (2D cross/dot). */
    private fun signedAngleBetween(v1: Vec2, v2: Vec2): Float {
        val cross = v1.x * v2.y - v1.y * v2.x
        val dot = v1.dot(v2)
        return atan2(cross, dot)
    }

    private fun toPt(p: PointF) = Vec2(p.x, p.y)
    private fun mid(a: Vec2, b: Vec2) = Vec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f)

    /**
     * Compute 5 joint angles for one frame, encoded as (sin, cos) pairs.
     * Mirrors Python: angles = stack([elbow, shoulder, wrist, torso, twist]);
     *                 return stack([sin, cos], axis=-1) -> (5, 2) flattened row-major.
     */
    fun fromLandmarks(landmarks: Array<PointF>): FloatArray? {
        if (landmarks.size < 25) return null
        // Basic visibility check: shoulders + hips present (non-zero)
        val rs = toPt(landmarks[RIGHT_SHOULDER])
        val ls = toPt(landmarks[LEFT_SHOULDER])
        val rh = toPt(landmarks[RIGHT_HIP])
        val lh = toPt(landmarks[LEFT_HIP])
        val re = toPt(landmarks[RIGHT_ELBOW])
        val rw = toPt(landmarks[RIGHT_WRIST])
        val ri = toPt(landmarks[RIGHT_INDEX])

        if (rs.length() < 1e-6f && ls.length() < 1e-6f) return null

        val midShoulder = mid(ls, rs)
        val midHip = mid(lh, rh)

        val elbow = interiorAngle(rs, re, rw)
        val shoulder = interiorAngle(rh, rs, re)
        val wrist = interiorAngle(re, rw, ri)
        val torso = signedAngleFromVertical(midShoulder - midHip)
        val twist = signedAngleBetween(rs - ls, rh - lh)

        val angles = floatArrayOf(elbow, shoulder, wrist, torso, twist)
        val out = FloatArray(10)
        for (i in 0 until 5) {
            out[i * 2] = sin(angles[i])
            out[i * 2 + 1] = cos(angles[i])
        }
        return out
    }
}
