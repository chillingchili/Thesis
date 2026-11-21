package com.thesis.pickleballserve

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Body outline over the camera. The preview uses fitCenter, so the outline is
 * placed inside the letterboxed video rect using the analysis frame's aspect
 * ratio; FramingCheck uses the same normalized outline bounds.
 */
class FramingGuideView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    /** Analysis frame width / height; updated from the first real frame. */
    var sourceAspect = 4f / 3f
        set(value) { if (value.isFinite() && value > 0f && value != field) { field = value; invalidate() } }

    /** Solid green when the framing check passes; dashed white otherwise. */
    var ok = false
        set(value) { if (value != field) { field = value; invalidate() } }

    private val density = resources.displayMetrics.density
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f * density
    }
    /** Fixed on-camera colors (same in both themes): bright pass green over a dark keyline. */
    private val framingOk = androidx.core.content.ContextCompat.getColor(context, R.color.framing_ok)
    private val keyline = androidx.core.content.ContextCompat.getColor(context, R.color.framing_keyline)
    private val dash = DashPathEffect(floatArrayOf(8f * density, 6f * density), 0f)
    private val path = Path()
    private val rect = RectF()
    private val arc = RectF()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width == 0 || height == 0) return
        // Fit-center video rect inside this view.
        val viewAspect = width.toFloat() / height
        val vw: Float; val vh: Float
        if (viewAspect > sourceAspect) { vh = height.toFloat(); vw = vh * sourceAspect }
        else { vw = width.toFloat(); vh = vw / sourceAspect }
        val ox = (width - vw) / 2f
        val oy = (height - vh) / 2f
        rect.set(
            ox + FramingCheck.OUTLINE_LEFT * vw, oy + FramingCheck.OUTLINE_TOP * vh,
            ox + FramingCheck.OUTLINE_RIGHT * vw, oy + FramingCheck.OUTLINE_BOTTOM * vh,
        )
        val r = minOf(rect.width() / 2f, rect.height() / 2f)
        path.reset()
        path.moveTo(rect.left, rect.bottom)
        path.lineTo(rect.left, rect.top + r)
        arc.set(rect.left, rect.top, rect.left + 2 * r, rect.top + 2 * r)
        path.arcTo(arc, 180f, 90f)
        path.lineTo(rect.right - r, rect.top)
        arc.set(rect.right - 2 * r, rect.top, rect.right, rect.top + 2 * r)
        path.arcTo(arc, 270f, 90f)
        path.lineTo(rect.right, rect.bottom)
        path.close()
        // Dark keyline under the stroke keeps the outline visible on any court color.
        paint.pathEffect = if (ok) null else dash
        paint.color = keyline
        paint.strokeWidth = 4.5f * density
        canvas.drawPath(path, paint)
        paint.color = if (ok) framingOk else Color.WHITE
        paint.strokeWidth = 2.5f * density
        canvas.drawPath(path, paint)
    }
}
