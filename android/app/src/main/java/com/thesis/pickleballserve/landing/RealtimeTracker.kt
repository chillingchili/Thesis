package com.thesis.pickleballserve.landing

import android.graphics.Bitmap
import android.graphics.Rect as AndroidRect
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import org.opencv.calib3d.Calib3d
import kotlin.math.hypot
import kotlin.math.roundToInt

/** Frame-worker owned. Keeps bounded history to align delayed neural detections with playback. */
internal class RealtimeTracker(private val fps: Double = 30.0) : AutoCloseable {
    data class Frame(val index: Int, val rgb: Mat, val gray: Mat, val moving: List<Point>, var homography: DoubleArray? = null) : AutoCloseable {
        override fun close() { rgb.release(); gray.release() }
    }
    private val history = ArrayDeque<Frame>()
    private var ball: YoloRunner.BallDet? = null
    private var ballTemplate: Mat? = null
    private var vx = 0.0
    private var vy = 0.0
    private var lastConfirmed = -1
    private var court: YoloRunner.CourtDet? = null
    private var courtConfirmed = -1
    private var stationaryFrames = 0
    val currentBall get() = ball
    val currentCourt get() = court
    data class Observation(val index: Int, val ball: YoloRunner.BallDet, val homography: DoubleArray?)

    fun saveHomography(value: DoubleArray?) { history.lastOrNull()?.homography = value?.copyOf() }

    /** A motion proposal is only a crop hint. It is never published without model confirmation. */
    fun suggestedRegion(): AndroidRect? {
        val frame = history.lastOrNull() ?: return null
        val b = ball
        val point = if (b != null && hypot(vx, vy) > 0.5) Point(b.cx + vx, b.cy + vy)
            else frame.moving.firstOrNull() ?: return null
        val w = minOf(160, frame.rgb.cols()); val h = minOf(160, frame.rgb.rows())
        val left = (point.x.roundToInt() - w / 2).coerceIn(0, frame.rgb.cols() - w)
        val top = (point.y.roundToInt() - h / 2).coerceIn(0, frame.rgb.rows() - h)
        return AndroidRect(left, top, left + w, top + h)
    }

    fun advance(index: Int, bitmap: Bitmap) {
        val rgba = Mat()
        val rgb = Mat()
        val gray = Mat()
        try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)
        } finally { rgba.release() }
        val next = Frame(index, rgb, gray, movingCandidates(rgb, gray, history.lastOrNull()?.gray))
        history.lastOrNull()?.let { previous ->
            // Missing camera frames are not interpolated into fake ball observations.
            if (index - previous.index <= 2) ball = followBall(next)
            else { ball = null; clearTemplate() }
            court = court?.let { followCourt(previous.gray, gray, it) }
        }
        history.addLast(next)
        while (history.size > (fps * 3).toInt().coerceIn(30, 180)) history.removeFirst().close()
        if (index - lastConfirmed > fps * 2) { ball = null; clearTemplate() }
        if (index - courtConfirmed > fps * 6) court = null
    }

    /** Returns observations at their ORIGINAL frame indices, never at the inference completion time. */
    fun correct(index: Int, detections: YoloRunner.FrameDet): List<Observation> {
        val frames = history.filter { it.index >= index }
        if (frames.firstOrNull()?.index != index) return emptyList()
        detections.court?.let { detected ->
            court = followCourt(frames.first().gray, frames.last().gray, detected)
            courtConfirmed = index
        }
        val detected = detections.ball ?: return emptyList()
        ball = detected
        lastConfirmed = index
        vx = 0.0; vy = 0.0
        stationaryFrames = 0
        clearTemplate()
        ballTemplate = cropTemplate(frames.first().rgb, detected)
        if (ballTemplate == null) { ball = null; return emptyList() }
        val observations = ArrayList<Observation>()
        observations.add(Observation(index, detected, frames.first().homography))
        for (i in 1 until frames.size) {
            if (frames[i].index - frames[i - 1].index > 2) { ball = null; break }
            ball = followBall(frames[i])
            val tracked = ball ?: break
            observations.add(Observation(frames[i].index, tracked, frames[i].homography))
        }
        if (observations.size >= 5 && observations.all {
                hypot(it.ball.cx - detected.cx, it.ball.cy - detected.cy) < 2.0
            }) {
            ball = null
            clearTemplate()
            return emptyList()
        }
        return observations
    }

    private fun movingCandidates(rgb: Mat, gray: Mat, previous: Mat?): List<Point> {
        if (previous == null || previous.size() != gray.size()) return emptyList()
        val hsv = Mat(); val color = Mat(); val difference = Mat(); val hierarchy = Mat()
        val contours = ArrayList<MatOfPoint>()
        try {
            Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
            // Common yellow/green pickleballs. Other colors still use the full-frame detector.
            Core.inRange(hsv, Scalar(15.0, 80.0, 120.0), Scalar(75.0, 255.0, 255.0), color)
            Core.absdiff(gray, previous, difference)
            Imgproc.threshold(difference, difference, 15.0, 255.0, Imgproc.THRESH_BINARY)
            Core.bitwise_and(color, difference, color)
            Imgproc.findContours(color, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            return contours.mapNotNull { contour ->
                val bounds = Imgproc.boundingRect(contour)
                val area = Imgproc.contourArea(contour)
                if (area !in 1.0..100.0 || bounds.width !in 2..18 || bounds.height !in 2..18) null
                else area to Point(bounds.x + bounds.width / 2.0, bounds.y + bounds.height / 2.0)
            }.sortedByDescending { it.first }.take(8).map { it.second }
        } finally {
            contours.forEach { it.release() }
            hsv.release(); color.release(); difference.release(); hierarchy.release()
        }
    }

    private fun cropTemplate(rgb: Mat, b: YoloRunner.BallDet): Mat? {
        val size = maxOf(b.x1 - b.x0, b.y1 - b.y0).roundToInt().coerceIn(7, 21) or 1
        val x = b.cx.roundToInt() - size / 2
        val y = b.cy.roundToInt() - size / 2
        if (x < 0 || y < 0 || x + size > rgb.cols() || y + size > rgb.rows()) return null
        val roi = rgb.submat(Rect(x, y, size, size))
        val mean = MatOfDouble(); val deviation = MatOfDouble()
        return try {
            Core.meanStdDev(roi, mean, deviation)
            if (deviation.toArray().maxOrNull()!! < 8.0) null else roi.clone()
        } finally { roi.release(); mean.release(); deviation.release() }
    }

    private fun followBall(frame: Frame): YoloRunner.BallDet? {
        val b = ball ?: return null
        val template = ballTemplate ?: return null
        // Local search only; a poor match means lost, not a guessed/extrapolated observation.
        val radius = 28
        val px = (b.cx + vx).roundToInt()
        val py = (b.cy + vy).roundToInt()
        val moving = frame.moving.minByOrNull { hypot(it.x - px, it.y - py) }
            ?.takeIf { hypot(it.x - px, it.y - py) < 20 && hypot(it.x - b.cx, it.y - b.cy) < 45 }
        if (moving != null) return movedBall(b, moving.x, moving.y, b.conf)
        val half = template.cols() / 2
        val x = maxOf(0, px - radius - half)
        val y = maxOf(0, py - radius - half)
        val right = minOf(frame.rgb.cols(), px + radius + half + 1)
        val bottom = minOf(frame.rgb.rows(), py + radius + half + 1)
        if (right - x < template.cols() || bottom - y < template.rows()) return null
        val search = frame.rgb.submat(Rect(x, y, right - x, bottom - y))
        val scores = Mat()
        try {
            Imgproc.matchTemplate(search, template, scores, Imgproc.TM_CCOEFF_NORMED)
            val peak = Core.minMaxLoc(scores)
            if (!peak.maxVal.isFinite() || peak.maxVal < 0.70) return null
            val cx = x + peak.maxLoc.x + half
            val cy = y + peak.maxLoc.y + half
            if (hypot(cx - b.cx, cy - b.cy) > 55) return null
            return movedBall(b, cx, cy, minOf(b.conf, peak.maxVal))
        } finally { search.release(); scores.release() }
    }

    private fun movedBall(b: YoloRunner.BallDet, cx: Double, cy: Double, confidence: Double): YoloRunner.BallDet? {
        vx = cx - b.cx; vy = cy - b.cy
        stationaryFrames = if (hypot(vx, vy) < 0.7) stationaryFrames + 1 else 0
        if (stationaryFrames > 8) return null
        return b.copy(cx = cx, cy = cy, conf = confidence,
            x0 = b.x0 + vx, y0 = b.y0 + vy, x1 = b.x1 + vx, y1 = b.y1 + vy)
    }

    private fun followCourt(previous: Mat, next: Mat, c: YoloRunner.CourtDet): YoloRunner.CourtDet? {
        // Estimate camera motion from many background features. Tracking the predicted court
        // points directly can attach them to a passing player and deform a stationary court.
        val smallPrevious = Mat(); val smallNext = Mat()
        val scale = minOf(1.0, 320.0 / maxOf(previous.cols(), previous.rows()))
        val size = Size((previous.cols() * scale).toInt().toDouble(), (previous.rows() * scale).toInt().toDouble())
        Imgproc.resize(previous, smallPrevious, size, 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.resize(next, smallNext, size, 0.0, 0.0, Imgproc.INTER_AREA)
        val features = MatOfPoint()
        Imgproc.goodFeaturesToTrack(smallPrevious, features, 60, 0.01, 8.0)
        val before = MatOfPoint2f(*features.toArray())
        features.release()
        if (before.rows() < 10) { before.release(); smallPrevious.release(); smallNext.release(); return null }
        val after = MatOfPoint2f()
        val back = MatOfPoint2f()
        val status = MatOfByte(); val reverseStatus = MatOfByte()
        val error = MatOfFloat(); val reverseError = MatOfFloat()
        try {
            Video.calcOpticalFlowPyrLK(smallPrevious, smallNext, before, after, status, error, Size(15.0, 15.0), 2)
            Video.calcOpticalFlowPyrLK(smallNext, smallPrevious, after, back, reverseStatus, reverseError, Size(15.0, 15.0), 2)
            val origin = before.toArray(); val forward = after.toArray(); val backward = back.toArray()
            val ok = status.toArray(); val reverseOk = reverseStatus.toArray(); val errors = error.toArray()
            val good = origin.indices.filter { i -> ok[i].toInt() != 0 && reverseOk[i].toInt() != 0 && errors[i] < 25 &&
                hypot(backward[i].x - origin[i].x, backward[i].y - origin[i].y) <= 1.5 }
            if (good.size < 10) return null
            val from = MatOfPoint2f(*good.map { origin[it] }.toTypedArray())
            val to = MatOfPoint2f(*good.map { forward[it] }.toTypedArray())
            val inliers = Mat()
            try {
                val transform = Calib3d.estimateAffinePartial2D(from, to, inliers, Calib3d.RANSAC, 1.5)
                try {
                    if (transform.empty() || Core.countNonZero(inliers) < maxOf(10, good.size * 3 / 5)) return null
                    val a = DoubleArray(6).also { transform.get(0, 0, it) }
                    val zoom = hypot(a[0], a[1])
                    if (zoom !in 0.9..1.1) return null
                    val xy = DoubleArray(28)
                    for (k in 0..13) {
                        val x = c.kptsXy[k * 2]; val y = c.kptsXy[k * 2 + 1]
                        xy[k * 2] = a[0] * x + a[1] * y + a[2] / scale
                        xy[k * 2 + 1] = a[3] * x + a[4] * y + a[5] / scale
                    }
                    return c.copy(kptsXy = xy)
                } finally { transform.release() }
            } finally { from.release(); to.release(); inliers.release() }
        } finally {
            before.release(); after.release(); back.release(); status.release(); reverseStatus.release()
            error.release(); reverseError.release()
            smallPrevious.release(); smallNext.release()
        }
    }

    private fun clearTemplate() { ballTemplate?.release(); ballTemplate = null }
    override fun close() { history.forEach { it.close() }; history.clear(); clearTemplate(); ball = null; court = null }
}
