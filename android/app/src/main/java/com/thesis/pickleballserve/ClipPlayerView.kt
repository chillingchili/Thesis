package com.thesis.pickleballserve

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

/**
 * Draws one frame of a [ServeClip], fit-centered, with the tracked pose on top.
 * The skeleton is colored by body side (the player's left and right) so a limb can
 * be identified at a glance; the trunk lines and head stay neutral.
 * Frame stepping is driven by the review screen; this view only renders.
 */
class ClipPlayerView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    var clip: ServeClip? = null
        set(value) { field = value; index = 0; decoded = null; decodedIndex = -1; invalidate() }

    var index: Int = 0
        set(value) { field = value.coerceIn(0, ((clip?.frames?.size ?: 1) - 1).coerceAtLeast(0)); invalidate() }

    var showPose: Boolean = true
        set(value) { field = value; invalidate() }

    private enum class Side { LEFT, RIGHT, CENTER }

    private var decoded: Bitmap? = null
    private var decodedIndex = -1
    private val dst = RectF()
    private val imagePaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val strokeW = 3f * resources.displayMetrics.density
    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = strokeW
        strokeCap = Paint.Cap.ROUND
        style = Paint.Style.STROKE
    }
    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val colorOf = mapOf(
        Side.LEFT to ContextCompat.getColor(context, R.color.pose_left),
        Side.RIGHT to ContextCompat.getColor(context, R.color.pose_right),
        Side.CENTER to ContextCompat.getColor(context, R.color.pose_center),
    )

    override fun onDraw(canvas: Canvas) {
        val frames = clip?.frames ?: return
        if (frames.isEmpty()) return
        val f = frames[index]
        if (decodedIndex != index) {
            decoded = BitmapFactory.decodeByteArray(f.jpeg, 0, f.jpeg.size)
            decodedIndex = index
        }
        val bmp = decoded ?: return
        val scale = minOf(width / bmp.width.toFloat(), height / bmp.height.toFloat())
        val w = bmp.width * scale; val h = bmp.height * scale
        dst.set((width - w) / 2f, (height - h) / 2f, (width + w) / 2f, (height + h) / 2f)
        canvas.drawBitmap(bmp, null, dst, imagePaint)
        val pose = f.pose ?: return
        if (!showPose) return
        fun x(i: Int) = dst.left + pose[i * 2] * dst.width()
        fun y(i: Int) = dst.top + pose[i * 2 + 1] * dst.height()
        for ((a, b, side) in BONES) {
            bonePaint.color = colorOf.getValue(side)
            canvas.drawLine(x(a), y(a), x(b), y(b), bonePaint)
        }
        val r = 3.5f * resources.displayMetrics.density
        for (j in JOINTS) {
            jointPaint.color = colorOf.getValue(sideOf(j))
            canvas.drawCircle(x(j), y(j), r, jointPaint)
        }
    }

    private companion object {
        /** BlazePose labels the subject's own left and right: odd landmark indices are left, even are right. */
        fun sideOf(i: Int) = when { i == 0 -> Side.CENTER; i % 2 == 1 -> Side.LEFT; else -> Side.RIGHT }

        // Shoulders, arms, torso, legs. Trunk cross-lines (shoulders, hips) are neutral.
        val BONES = listOf(
            Triple(11, 12, Side.CENTER), Triple(23, 24, Side.CENTER),
            Triple(11, 13, Side.LEFT), Triple(13, 15, Side.LEFT), Triple(11, 23, Side.LEFT),
            Triple(23, 25, Side.LEFT), Triple(25, 27, Side.LEFT),
            Triple(12, 14, Side.RIGHT), Triple(14, 16, Side.RIGHT), Triple(12, 24, Side.RIGHT),
            Triple(24, 26, Side.RIGHT), Triple(26, 28, Side.RIGHT),
        )
        val JOINTS = intArrayOf(0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    }
}
