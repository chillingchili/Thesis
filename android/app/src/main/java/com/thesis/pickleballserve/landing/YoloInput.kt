package com.thesis.pickleballserve.landing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Reusable storage, refreshed explicitly for every frame (including reused decoder bitmaps). */
internal class YoloInput(private val size: Int = 640) : AutoCloseable {
    data class Layout(val scale: Double, val left: Double, val top: Double)
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(3 * size * size * 4).order(ByteOrder.nativeOrder())
    private val floats = buffer.asFloatBuffer()
    private val pixels = IntArray(size * size)
    private val values = FloatArray(3 * size * size)
    private val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(bitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()

    fun prepare(source: Bitmap, rotation: Int): Layout {
        val w = if (rotation % 180 != 0) source.height else source.width
        val h = if (rotation % 180 != 0) source.width else source.height
        val scale = minOf(size.toDouble() / w, size.toDouble() / h)
        val nw = Math.round(w * scale).toInt()
        val nh = Math.round(h * scale).toInt()
        val left = Math.round((size - nw) / 2.0 - 0.1).toInt()
        val top = Math.round((size - nh) / 2.0 - 0.1).toInt()
        canvas.drawColor(android.graphics.Color.rgb(114, 114, 114))
        matrix.reset()
        matrix.postTranslate(-source.width / 2f, -source.height / 2f)
        matrix.postRotate(rotation.toFloat())
        matrix.postScale(nw.toFloat() / w, nh.toFloat() / h)
        matrix.postTranslate(left + nw / 2f, top + nh / 2f)
        canvas.drawBitmap(source, matrix, paint)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)
        val area = size * size
        val inv255 = 1f / 255f
        for (i in 0 until area) {
            val c = pixels[i]
            values[i] = ((c shr 16) and 255) * inv255
            values[area + i] = ((c shr 8) and 255) * inv255
            values[2 * area + i] = (c and 255) * inv255
        }
        floats.rewind()
        floats.put(values)
        buffer.rewind()
        return Layout(scale, left.toDouble(), top.toDouble())
    }

    override fun close() { bitmap.recycle() }
}
