package com.thesis.pickleballserve.landing

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class LandingOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    var frameW = 1
    var frameH = 1
    var ball: YoloRunner.BallDet? = null
    var kpts: DoubleArray? = null
    var kptValid: BooleanArray = BooleanArray(14)
    var trailFt: DoubleArray = DoubleArray(0)
    var hit: BounceDetector.Hit? = null
    var hitZone: String? = null

    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }
    private val thin = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 18f
    }
    private val textBold = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 24f
        isFakeBoldText = true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val fw = frameW.toFloat()
        val fh = frameH.toFloat()
        val s = minOf(width / fw, height / fh)
        val ox = (width - fw * s) / 2f
        val oy = (height - fh * s) / 2f

        drawSkeleton(canvas, s, ox, oy)
        drawBall(canvas, s, ox, oy)
        drawBev(canvas)

        hit?.let { h ->
            textBold.color = Color.RED
            canvas.drawText("LANDING", 20f, 44f, textBold)
            textBold.color = Color.WHITE
            canvas.drawText(
                "${h.frame}  ${"%.2f".format(h.xFt)}, ${"%.2f".format(h.yFt)}  $hitZone",
                20f, 74f, textBold,
            )
        }
    }

    private fun drawSkeleton(canvas: Canvas, s: Float, ox: Float, oy: Float) {
        val k = kpts ?: return
        val green = Color.rgb(0, 255, 0)
        val gray = Color.rgb(60, 60, 60)
        thin.color = green
        for ((a, b) in Zone.SKELETON_EDGES) {
            if (!kptValid[a] || !kptValid[b]) continue
            canvas.drawLine(
                ox + (k[2 * a] * s).toFloat(), oy + (k[2 * a + 1] * s).toFloat(),
                ox + (k[2 * b] * s).toFloat(), oy + (k[2 * b + 1] * s).toFloat(),
                thin,
            )
        }
        if (k.size >= 18 && kptValid[4] && kptValid[5] && kptValid[8]) {
            val midX = (k[8] + k[10]) / 2
            val midY = (k[9] + k[11]) / 2
            canvas.drawLine(
                ox + (k[16] * s).toFloat(), oy + (k[17] * s).toFloat(),
                ox + (midX * s).toFloat(), oy + (midY * s).toFloat(),
                thin,
            )
        }
        fill.style = Paint.Style.FILL
        for (p in 0..13) {
            fill.color = if (kptValid.getOrElse(p) { true }) green else gray
            canvas.drawCircle(
                ox + (k[2 * p] * s).toFloat(),
                oy + (k[2 * p + 1] * s).toFloat(),
                5f, fill,
            )
        }
    }

    private fun drawBall(canvas: Canvas, s: Float, ox: Float, oy: Float) {
        val b = ball ?: return
        stroke.color = Color.GREEN
        canvas.drawRect(
            ox + (b.x0 * s).toFloat(), oy + (b.y0 * s).toFloat(),
            ox + (b.x1 * s).toFloat(), oy + (b.y1 * s).toFloat(),
            stroke,
        )
        fill.color = Color.RED
        canvas.drawCircle(ox + (b.cx * s).toFloat(), oy + (b.cy * s).toFloat(), 6f, fill)
        text.color = Color.GREEN
        canvas.drawText(
            "Ball ${"%.2f".format(b.conf)}",
            ox + (b.x0 * s).toFloat(),
            oy + (b.y0 * s).toFloat() - 8f,
            text,
        )
    }

    private fun drawBev(canvas: Canvas) {
        val sc = minOf(1f, (height - 16f) / Zone.BEV_H)
        val x0 = width - Zone.BEV_W * sc - 8f
        val y0 = (height - Zone.BEV_H * sc) / 2f
        canvas.save()
        canvas.translate(x0, y0)
        canvas.scale(sc, sc)

        fill.color = Color.rgb(28, 28, 28)
        canvas.drawRect(0f, 0f, Zone.BEV_W, Zone.BEV_H, fill)

        fill.color = Color.rgb(48, 62, 44)
        canvas.drawRect(
            bx(0f), by(15f), bx(20f), by(29f), fill,
        )

        stroke.color = Color.rgb(240, 240, 240)
        stroke.strokeWidth = 2f
        canvas.drawRect(bx(0f), by(0f), bx(20f), by(44f), stroke)
        canvas.drawLine(bx(0f), by(15f), bx(20f), by(15f), stroke)
        canvas.drawLine(bx(0f), by(29f), bx(20f), by(29f), stroke)
        canvas.drawLine(bx(10f), by(0f), bx(10f), by(15f), stroke)
        canvas.drawLine(bx(10f), by(29f), bx(10f), by(44f), stroke)

        stroke.color = Color.rgb(220, 60, 60)
        stroke.strokeWidth = 4f
        canvas.drawLine(bx(0f), by(22f), bx(20f), by(22f), stroke)

        thin.color = Color.rgb(150, 150, 150)
        thin.strokeWidth = 1f
        for (y in listOf(7.5f, 36.5f)) {
            var t = 0f
            while (t < 20f) {
                canvas.drawLine(bx(t), by(y), bx(minOf(t + 0.5f, 20f)), by(y), thin)
                t += 1f
            }
        }

        thin.color = Color.rgb(255, 255, 0)
        thin.strokeWidth = 2f
        var i = 0
        while (i + 3 < trailFt.size) {
            canvas.drawLine(
                bx(trailFt[i].toFloat()), by(trailFt[i + 1].toFloat()),
                bx(trailFt[i + 2].toFloat()), by(trailFt[i + 3].toFloat()),
                thin,
            )
            i += 2
        }

        val h = hit
        val zone = hitZone
        if (h != null && zone != null) {
            if (zone.startsWith("Deep") || zone.startsWith("Short")) {
                val rx0 = if (h.xFt < 10) 0f else 10f
                val rx1 = rx0 + 10f
                val ry0: Float
                val ry1: Float
                if (h.yFt < 15) {
                    ry0 = if (h.yFt < 7.5) 0f else 7.5f
                    ry1 = ry0 + 7.5f
                } else {
                    ry0 = if (h.yFt < 36.5) 29f else 36.5f
                    ry1 = ry0 + 7.5f
                }
                fill.color = Color.argb(90, 255, 140, 0)
                canvas.drawRect(bx(rx0), by(ry0), bx(rx1), by(ry1), fill)
                stroke.color = Color.rgb(255, 140, 0)
                stroke.strokeWidth = 3f
                canvas.drawRect(bx(rx0), by(ry0), bx(rx1), by(ry1), stroke)
                text.color = Color.WHITE
                text.textSize = 20f
                canvas.drawText(zone, bx(rx0) + 6f, by(ry0) + 26f, text)
            }
            fill.color = Color.RED
            canvas.drawCircle(bx(h.xFt.toFloat()), by(h.yFt.toFloat()), 8f, fill)
        }

        text.color = Color.WHITE
        text.textSize = 18f
        canvas.drawText("BIRD EYE", Zone.BEV_MARGIN, 24f, text)
        canvas.restore()
    }

    private fun bx(xFt: Float): Float = Zone.BEV_MARGIN + xFt * Zone.BEV_SCALE
    private fun by(yFt: Float): Float = Zone.BEV_MARGIN + yFt * Zone.BEV_SCALE
}
