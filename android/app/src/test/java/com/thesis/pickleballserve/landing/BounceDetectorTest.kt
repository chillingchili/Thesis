package com.thesis.pickleballserve.landing

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class BounceDetectorTest {

    private fun build(
        n: Int,
        pts: Map<Int, Pair<Double, Double>>,
        wpts: Map<Int, Pair<Double, Double>>,
    ): Triple<DoubleArray, DoubleArray, DoubleArray> {
        val cy = DoubleArray(n) { Double.NaN }
        val wx = DoubleArray(n) { Double.NaN }
        val wy = DoubleArray(n) { Double.NaN }
        for ((f, p) in pts) {
            cy[f] = p.second
            wx[f] = wpts[f]!!.first
            wy[f] = wpts[f]!!.second
        }
        return Triple(cy, wx, wy)
    }

    private fun fixture(name: String): JSONObject {
        val text = javaClass.classLoader!!.getResource("fixtures/$name")!!.readText()
        return JSONObject(text)
    }

    private fun arrays(obj: JSONObject): Triple<DoubleArray, DoubleArray, DoubleArray> {
        val frames = obj.getInt("frames")
        val ball = obj.getJSONArray("ball_xy")
        val world = obj.getJSONArray("world_xy")
        val cy = DoubleArray(frames)
        val wx = DoubleArray(frames)
        val wy = DoubleArray(frames)
        for (i in 0 until frames) {
            cy[i] = if (ball.getJSONArray(i).isNull(1)) Double.NaN else ball.getJSONArray(i).getDouble(1)
            wx[i] = if (world.getJSONArray(i).isNull(0)) Double.NaN else world.getJSONArray(i).getDouble(0)
            wy[i] = if (world.getJSONArray(i).isNull(1)) Double.NaN else world.getJSONArray(i).getDouble(1)
        }
        return Triple(cy, wx, wy)
    }

    @Test
    fun caseA_staticRampRejected_realBounceFound() {
        val pts = HashMap<Int, Pair<Double, Double>>()
        val wpts = HashMap<Int, Pair<Double, Double>>()
        for (f in 0..40) {
            pts[f] = 337.0 to 294.0
            wpts[f] = -75.0 to -80.0
        }
        for (f in 161..188) {
            pts[f] = 1420.0 to (470.0 + (f - 161) * 3.9)
            wpts[f] = 19.0 to 30.0
        }
        for (f in 191..201) {
            pts[f] = 337.0 to 301.6
            wpts[f] = -75.0 to -80.0
        }
        for (f in 204..234) {
            val t = (f - 204).toDouble()
            pts[f] = (1300.0 - 8 * t) to (533.0 - 6 * t + 0.1 * t * t)
            wpts[f] = 17.0 to 31.0
        }
        val (cy, wx, wy) = build(900, pts, wpts)
        val r = BounceDetector.detect(cy, wx, wy, minI = 29, fps = 59.94)
        assertNotNull(r)
        assertTrue("frame ${r!!.frame}", r.frame in 185..200)
    }

    @Test
    fun caseB_fallThenStaleClusterAcrossGap_rejected() {
        val pts = HashMap<Int, Pair<Double, Double>>()
        val wpts = HashMap<Int, Pair<Double, Double>>()
        for (f in 0..29) {
            pts[f] = 337.0 to 294.0
            wpts[f] = -75.0 to -80.0
        }
        for (f in 100..120) {
            pts[f] = 700.0 to (400.0 + (f - 100) * 12.0)
            wpts[f] = 5.0 to 20.0
        }
        for (f in 150..160) {
            pts[f] = 694.0 to 220.0
            wpts[f] = 6.0 to 8.0
        }
        val (cy, wx, wy) = build(900, pts, wpts)
        assertNull(BounceDetector.detect(cy, wx, wy, minI = 29, fps = 59.94))
    }

    @Test
    fun caseC_liveWindowBounce_found() {
        val pts = HashMap<Int, Pair<Double, Double>>()
        val wpts = HashMap<Int, Pair<Double, Double>>()
        for (f in 0..34) {
            pts[f] = 337.0 to 294.0
            wpts[f] = -75.0 to -80.0
        }
        for (f in 90..114) {
            val t = (f - 90).toDouble()
            pts[f] = 700.0 to (300.0 + (12 - abs(t - 12)) * (12 - abs(t - 12)) * 1.5)
            wpts[f] = 8.0 to 25.0
        }
        for (f in 120..149) {
            pts[f] = 337.0 to 301.6
            wpts[f] = -75.0 to -80.0
        }
        val (cy, wx, wy) = build(150, pts, wpts)
        val r = BounceDetector.detect(cy, wx, wy, fps = 59.94)
        assertNotNull(r)
        assertTrue("frame ${r!!.frame}", r.frame in 90..130)
    }

    @Test
    fun caseD_allStaticWindow_rejected() {
        val pts = HashMap<Int, Pair<Double, Double>>()
        val wpts = HashMap<Int, Pair<Double, Double>>()
        for (f in 0..59) {
            pts[f] = 337.0 to 294.0
            wpts[f] = -75.0 to -80.0
        }
        val (cy, wx, wy) = build(150, pts, wpts)
        assertNull(BounceDetector.detect(cy, wx, wy, fps = 59.94))
    }

    @Test
    fun fixtureTest15_parity() {
        val obj = fixture("test15.json")
        val (cy, wx, wy) = arrays(obj)
        val landing = obj.getJSONObject("landing")
        val r = BounceDetector.detect(
            cy, wx, wy,
            minI = obj.getInt("min_i"),
            fps = obj.getDouble("fps"),
        )
        assertNotNull(r)
        assertEquals(landing.getInt("frame"), r!!.frame)
        assertEquals(landing.getDouble("x_ft"), r.xFt, 0.01)
        assertEquals(landing.getDouble("y_ft"), r.yFt, 0.01)
        assertEquals(landing.getString("zone"), Zone.classify(r.xFt, r.yFt))
    }

    @Test
    fun fixtureClip40_parity() {
        val obj = fixture("clip40.json")
        val (cy, wx, wy) = arrays(obj)
        val landing = obj.getJSONObject("landing")
        val r = BounceDetector.detect(
            cy, wx, wy,
            minI = obj.getInt("min_i"),
            fps = obj.getDouble("fps"),
        )
        assertNotNull(r)
        assertEquals(landing.getInt("frame"), r!!.frame)
        assertEquals(landing.getDouble("x_ft"), r.xFt, 0.01)
        assertEquals(landing.getDouble("y_ft"), r.yFt, 0.01)
        assertEquals(landing.getString("zone"), Zone.classify(r.xFt, r.yFt))
    }

    @Test
    fun courtTracker_recoversKnownHomography() {
        val h = doubleArrayOf(
            0.002, 0.0005, 10.0,
            0.0003, -0.0025, 500.0,
            1e-6, -2e-6, 1.0,
        )
        val hInv = invert3(h)
        val px = DoubleArray(28)
        for (p in 0 until 14) {
            val x = Zone.KPT_COURT_FT[2 * p]
            val y = Zone.KPT_COURT_FT[2 * p + 1]
            val u = hInv[0] * x + hInv[1] * y + hInv[2]
            val v = hInv[3] * x + hInv[4] * y + hInv[5]
            val w = hInv[6] * x + hInv[7] * y + hInv[8]
            px[2 * p] = u / w
            px[2 * p + 1] = v / w
        }
        val tracker = CourtTracker()
        assertTrue(tracker.update(1.0, px, DoubleArray(14) { 1.0 }))
        assertNotNull(tracker.homography)
        for (p in 0 until 14) {
            val t = tracker.transform(px[2 * p], px[2 * p + 1])
            assertNotNull(t)
            val ex = Zone.KPT_COURT_FT[2 * p]
            val ey = Zone.KPT_COURT_FT[2 * p + 1]
            assertTrue("kpt $p dx=${t!![0] - ex}", abs(t[0] - ex) < 0.01)
            assertTrue("kpt $p dy=${t[1] - ey}", abs(t[1] - ey) < 0.01)
        }
    }

    private fun invert3(m: DoubleArray): DoubleArray {
        val a = m[0]; val b = m[1]; val c = m[2]
        val d = m[3]; val e = m[4]; val f = m[5]
        val g = m[6]; val h = m[7]; val i = m[8]
        val det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        val invDet = 1.0 / det
        return doubleArrayOf(
            (e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet,
            (f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet,
            (d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet,
        )
    }
}
