package com.thesis.pickleballserve

import android.content.Context
import android.util.AttributeSet
import android.view.View
import android.widget.FrameLayout

/**
 * FrameLayout that keeps a fixed width:height ratio (default 4:3, the analysis
 * stream's aspect) inside whatever space its parent offers. Height-led: it takes
 * the full available height and derives the width, shrinking both when the
 * width runs out. The preview inside is then never letterboxed.
 */
class AspectFrameLayout @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : FrameLayout(context, attrs) {

    var ratio: Float = 4f / 3f
        set(value) { field = value; requestLayout() }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val maxW = View.MeasureSpec.getSize(widthMeasureSpec)
        val maxH = View.MeasureSpec.getSize(heightMeasureSpec)
        if (maxW == 0 || maxH == 0) { super.onMeasure(widthMeasureSpec, heightMeasureSpec); return }
        var h = maxH
        var w = (h * ratio).toInt()
        if (w > maxW) { w = maxW; h = (w / ratio).toInt() }
        super.onMeasure(
            View.MeasureSpec.makeMeasureSpec(w, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(h, View.MeasureSpec.EXACTLY),
        )
    }
}
