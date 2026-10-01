package com.thesis.pickleballserve.landing

import kotlin.math.abs

class BallTracker(private val fps: Double, initialN: Int = 8192) {
    var ballX = FloatArray(initialN) { Float.NaN }
        private set
    var ballY = FloatArray(initialN) { Float.NaN }
        private set
    var worldX = FloatArray(initialN) { Float.NaN }
        private set
    var worldY = FloatArray(initialN) { Float.NaN }
        private set

    private var prevBx = Double.NaN
    private var prevBy = Double.NaN
    private var staticN = 0
    private var prevWx = Double.NaN
    private var prevWy = Double.NaN
    private var lastWpI = -10

    /** Grow even when detections are missing, so a long replay retains its full timeline. */
    fun advance(frameIdx: Int) { ensure(frameIdx + 1) }

    /** Retrospective correction must not rewind the streaming velocity/static-object filters. */
    fun correct(frameIdx: Int, cx: Double, cy: Double, homography: DoubleArray?) {
        ensure(frameIdx + 1)
        ballX[frameIdx] = cx.toFloat()
        ballY[frameIdx] = cy.toFloat()
        worldX[frameIdx] = Float.NaN
        worldY[frameIdx] = Float.NaN
        if (homography == null) return
        val w = homography[6] * cx + homography[7] * cy + homography[8]
        if (abs(w) < 1e-12) return
        worldX[frameIdx] = ((homography[0] * cx + homography[1] * cy + homography[2]) / w).toFloat()
        worldY[frameIdx] = ((homography[3] * cx + homography[4] * cy + homography[5]) / w).toFloat()
    }

    fun add(frameIdx: Int, cx: Double, cy: Double, wx: Double, wy: Double): Boolean {
        ensure(frameIdx + 1)
        if (!prevBx.isNaN() && abs(cx - prevBx) < 2 && abs(cy - prevBy) < 2) {
            staticN++
        } else {
            staticN = 0
        }
        prevBx = cx
        prevBy = cy
        if (staticN > fps) return false
        if (!prevWx.isNaN() && frameIdx - lastWpI <= 2 && (abs(wx - prevWx) > 6 || abs(wy - prevWy) > 6)) {
            return false
        }
        ballX[frameIdx] = cx.toFloat()
        ballY[frameIdx] = cy.toFloat()
        worldX[frameIdx] = wx.toFloat()
        worldY[frameIdx] = wy.toFloat()
        prevWx = wx
        prevWy = wy
        lastWpI = frameIdx
        return true
    }

    private fun ensure(n: Int) {
        if (n <= ballX.size) return
        var s = ballX.size
        while (s < n) s *= 2
        ballX = grow(ballX, s)
        ballY = grow(ballY, s)
        worldX = grow(worldX, s)
        worldY = grow(worldY, s)
    }

    private fun grow(src: FloatArray, n: Int): FloatArray {
        val out = FloatArray(n) { Float.NaN }
        src.copyInto(out)
        return out
    }
}
