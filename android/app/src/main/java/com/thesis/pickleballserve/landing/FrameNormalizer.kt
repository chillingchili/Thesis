package com.thesis.pickleballserve.landing

import android.graphics.*

/** Borrowed, oriented image, longest side at most 640; reused by the frame worker. */
internal class FrameNormalizer : AutoCloseable {
    private var bitmap: Bitmap? = null
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()
    fun prepare(source: Bitmap, rotation: Int): Bitmap {
        if (rotation == 0 && maxOf(source.width, source.height) <= 640) return source
        val w = if (rotation % 180 == 0) source.width else source.height
        val h = if (rotation % 180 == 0) source.height else source.width
        val scale = minOf(1.0, 640.0 / maxOf(w, h))
        val outW = (w * scale).toInt().coerceAtLeast(1)
        val outH = (h * scale).toInt().coerceAtLeast(1)
        if (bitmap?.width != outW || bitmap?.height != outH) {
            bitmap?.recycle()
            bitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        }
        matrix.reset()
        matrix.postTranslate(-source.width / 2f, -source.height / 2f)
        matrix.postRotate(rotation.toFloat())
        matrix.postScale(outW.toFloat() / w, outH.toFloat() / h)
        matrix.postTranslate(outW / 2f, outH / 2f)
        Canvas(bitmap!!).drawBitmap(source, matrix, paint)
        return bitmap!!
    }
    override fun close() { bitmap?.recycle(); bitmap = null }
}
