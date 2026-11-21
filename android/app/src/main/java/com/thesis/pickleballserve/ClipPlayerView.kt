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

    private var decoded: Bitmap? = null
    private var decodedIndex = -1
    private val dst = RectF()
    private val imagePaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.dot_warn)
        strokeWidth = 3f * resources.displayMetrics.density
        strokeCap = Paint.Cap.ROUND
        style = Paint.Style.STROKE
    }
    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.dot_warn)
        style = Paint.Style.FILL
    }

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
        for ((a, b) in BONES) canvas.drawLine(x(a), y(a), x(b), y(b), bonePaint)
        val r = 3.5f * resources.displayMetrics.density
        for (j in JOINTS) canvas.drawCircle(x(j), y(j), r, jointPaint)
    }

    private companion object {
        // BlazePose indices: shoulders, arms, torso, legs.
        val BONES = listOf(11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16,
            11 to 23, 12 to 24, 23 to 24, 23 to 25, 25 to 27, 24 to 26, 26 to 28)
        val JOINTS = intArrayOf(0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    }
}
