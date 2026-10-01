package com.thesis.pickleballserve

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Rect
import org.opencv.core.RotatedRect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Paddle crop angle geometry — Kotlin port of contour_rect / edge_rect /
 * crop_angle / long_axis_angle / expand_box in scripts/extract_paddle_angles.py
 * (thesis Section 4.3.2). Keep both identical: Otsu contour on both polarities
 * with fill/cover scoring (25-90% crop fill, cover >= 0.5), Canny edge
 * minAreaRect fallback (20-96% fill), 90-degree axis ambiguity resolved by
 * aspect ratio, exact 0/90 rejected as undeterminable.
 */
object PaddleGeometry {

    data class AngleResult(val angle: Double?, val status: String)

    private const val MIN_CROP = 8
    private const val CONTOUR_MIN_AREA = 16.0
    private const val FILL_MIN = 0.25
    private const val FILL_MAX = 0.90
    private const val COVER_MIN = 0.5
    private const val EDGE_MIN_PTS = 30
    private const val EDGE_FILL_MIN = 0.20
    private const val EDGE_FILL_MAX = 0.96

    fun mod180(x: Double): Double {
        val m = x % 180.0
        return if (m < 0) m + 180.0 else m
    }

    fun longAxisAngle(w: Double, h: Double, ang: Double): Double {
        var a = if (w >= h) mod180(ang) else mod180(ang - 90.0)
        if (a >= 180.0 - 1e-9 || abs(a) < 1e-9) a = 0.0
        return a
    }

    /**
     * python: max(0, int(x)) / min(shape, int(x)) with int() truncation;
     * numpy float32 scalar arithmetic, so pad math stays in float32 too.
     */
    fun expandBox(box: FloatArray, width: Int, height: Int, pad: Float = 0.12f): Rect {
        var x1 = box[0]
        var y1 = box[1]
        var x2 = box[2]
        var y2 = box[3]
        val bw = x2 - x1
        val bh = y2 - y1
        x1 -= bw * pad
        y1 -= bh * pad
        x2 += bw * pad
        y2 += bh * pad
        val ex1 = max(0, x1.toInt())
        val ey1 = max(0, y1.toInt())
        val ex2 = min(width, x2.toInt())
        val ey2 = min(height, y2.toInt())
        return Rect(ex1, ey1, max(0, ex2 - ex1), max(0, ey2 - ey1))
    }

    fun cropAngle(crop: Mat): AngleResult {
        if (crop.rows() < MIN_CROP || crop.cols() < MIN_CROP) return AngleResult(null, "no_contour")
        val contour = contourRect(crop)
        if (contour != null) {
            val r = contour.rect
            val angle = longAxisAngle(r.size.width.toDouble(), r.size.height.toDouble(),
                r.angle.toDouble())
            if (angle != 0.0 && angle != 90.0) return AngleResult(angle, "ok")
        }
        val edge = edgeRect(crop)
        if (edge != null) {
            val r = edge.rect
            val angle = longAxisAngle(r.size.width.toDouble(), r.size.height.toDouble(),
                r.angle.toDouble())
            if (angle != 0.0 && angle != 90.0) return AngleResult(angle, "ok")
            return AngleResult(null, "deg")
        }
        if (contour != null) return AngleResult(null, "deg")
        return AngleResult(null, "no_contour")
    }

    private data class ScoredRect(val score: Double, val rect: RotatedRect)

    private fun contourRect(crop: Mat): ScoredRect? {
        val h = crop.rows()
        val w = crop.cols()
        val th1 = Mat()
        val th2 = Mat()
        Imgproc.threshold(crop, th1, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        Imgproc.threshold(crop, th2, 0.0, 255.0, Imgproc.THRESH_BINARY_INV or Imgproc.THRESH_OTSU)
        var best: ScoredRect? = null
        var bestScore = 0.0
        val thList = arrayOf(th1, th2)
        val hierarchy = Mat()
        for (thI in thList.indices) {
            val contours = ArrayList<MatOfPoint>()
            Imgproc.findContours(thList[thI], contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            for (c in contours) {
                val area = Imgproc.contourArea(c)
                if (area < CONTOUR_MIN_AREA) {
                    c.release()
                    continue
                }
                val pts = MatOfPoint2f()
                c.convertTo(pts, CvType.CV_32F)
                c.release()
                val rect = Imgproc.minAreaRect(pts)
                pts.release()
                val rectArea = rect.size.width.toDouble() * rect.size.height.toDouble()
                val fillCrop = rectArea / (h * w)
                if (fillCrop < FILL_MIN || fillCrop > FILL_MAX) continue
                val cover = area / rectArea
                if (cover < COVER_MIN) continue
                val dx = rect.center.x - w / 2.0
                val dy = rect.center.y - h / 2.0
                val centered = 1.0 - min(1.0, sqrt(dx * dx + dy * dy) / (max(w, h) / 2.0))
                val score = cover * (0.7 + 0.3 * centered)
                if (score > bestScore) {
                    bestScore = score
                    best = ScoredRect(score, rect)
                }
            }
        }
        th1.release()
        th2.release()
        hierarchy.release()
        return best
    }

    private data class EdgeRect(val fill: Double, val rect: RotatedRect)

    private fun edgeRect(crop: Mat): EdgeRect? {
        val h = crop.rows()
        val w = crop.cols()
        val blur = Mat()
        Imgproc.GaussianBlur(crop, blur, Size(3.0, 3.0), 0.0)
        val edges = Mat()
        Imgproc.Canny(blur, edges, 50.0, 150.0)
        blur.release()
        val pts = MatOfPoint2f()
        Core.findNonZero(edges, pts)
        edges.release()
        if (pts.rows() < EDGE_MIN_PTS) {
            pts.release()
            return null
        }
        val rect = Imgproc.minAreaRect(pts)
        pts.release()
        val rectArea = rect.size.width.toDouble() * rect.size.height.toDouble()
        val fillCrop = rectArea / (h * w)
        if (fillCrop < EDGE_FILL_MIN || fillCrop > EDGE_FILL_MAX) return null
        return EdgeRect(fillCrop, rect)
    }

    fun grayFromRgba(rgba: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_BGRA2GRAY)
        return gray
    }
}
