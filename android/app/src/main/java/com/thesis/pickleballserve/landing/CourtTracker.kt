package com.thesis.pickleballserve.landing

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.ln
import kotlin.math.sqrt

class CourtTracker(
    private val boxConfThr: Double = 0.5,
    private val kptConfThr: Double = 0.4,
    private val ransacThresh: Double = 3.0,
    private val estimator: (DoubleArray, DoubleArray, Double) -> DoubleArray? = { src, dst, threshold ->
        ransacHomography(src, dst, threshold)
    },
) {
    var emaKpts: DoubleArray? = null
        private set
    var validMask: BooleanArray = BooleanArray(14)
        private set
    var homography: DoubleArray? = null
        private set
    var courtFrames: Int = 0
        private set

    fun update(boxConf: Double, kptsXy: DoubleArray, kptConf: DoubleArray): Boolean {
        if (boxConf < boxConfThr) return false
        require(kptsXy.size == 28 && kptConf.size == 14)
        validMask = BooleanArray(14) { kptConf[it] >= kptConfThr }
        val ema = emaKpts
        emaKpts = if (ema == null) {
            kptsXy.copyOf()
        } else {
            DoubleArray(28) { 0.5 * ema[it] + 0.5 * kptsXy[it] }
        }
        refreshHomography()
        courtFrames++
        return true
    }

    fun track(kptsXy: DoubleArray, valid: BooleanArray) {
        emaKpts = kptsXy.copyOf()
        validMask = valid.copyOf()
        refreshHomography()
    }

    private fun refreshHomography() {
        val xy = emaKpts ?: return
        val points = validMask.indices.filter { validMask[it] }
        val src = DoubleArray(points.size * 2) { xy[points[it / 2] * 2 + it % 2] }
        val dst = DoubleArray(points.size * 2) { Zone.KPT_COURT_FT[points[it / 2] * 2 + it % 2] }
        homography = estimator(src, dst, ransacThresh)
    }

    fun transform(x: Double, y: Double): DoubleArray? {
        val h = homography ?: return null
        val w = h[6] * x + h[7] * y + h[8]
        if (abs(w) < 1e-12) return null
        return doubleArrayOf((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
    }

    companion object {
        fun ransacHomography(
            src: DoubleArray,
            dst: DoubleArray,
            thresh: Double,
            maxIters: Int = 200, // Reduced from 2000
            confidence: Double = 0.995,
            seed: Long = 42L,
        ): DoubleArray? {
            val n = src.size / 2
            if (n < 4) return null
            val rng = java.util.Random(seed)
            var bestMask = BooleanArray(n)
            var bestCount = -1
            var limit = maxIters
            var iter = 0
            while (iter < limit) {
                val pick = IntArray(4)
                var filled = 0
                var attempts = 0
                while (filled < 4 && attempts < 64) {
                    val c = rng.nextInt(n)
                    attempts++
                    if (pick.take(filled).none { it == c }) {
                        pick[filled] = c
                        filled++
                    }
                }
                if (filled < 4) {
                    iter++
                    continue
                }
                val h = dlt(src, dst, pick)
                if (h != null) {
                    val mask = BooleanArray(n)
                    var count = 0
                    for (p in 0 until n) {
                        if (reprojErr(h, src[2 * p], src[2 * p + 1], dst[2 * p], dst[2 * p + 1]) < thresh) {
                            mask[p] = true
                            count++
                        }
                    }
                    if (count > bestCount) {
                        bestCount = count
                        bestMask = mask
                        val w = count.toDouble() / n
                        val denom = 1.0 - w * w * w * w
                        if (denom > 1e-12) {
                            limit = minOf(maxIters.toDouble(), ceil(ln(1.0 - confidence) / ln(denom)).toInt().toDouble())
                                .toInt()
                        }
                    }
                }
                iter++
            }
            if (bestCount < 4) return null
            val inliers = IntArray(bestCount)
            var t = 0
            for (p in 0 until n) if (bestMask[p]) inliers[t++] = p
            return dlt(src, dst, inliers) ?: run {
                val fallback = IntArray(n) { it }
                dlt(src, dst, fallback)
            }
        }

        private fun reprojErr(h: DoubleArray, u: Double, v: Double, x: Double, y: Double): Double {
            val w = h[6] * u + h[7] * v + h[8]
            if (abs(w) < 1e-12) return Double.MAX_VALUE
            val px = (h[0] * u + h[1] * v + h[2]) / w
            val py = (h[3] * u + h[4] * v + h[5]) / w
            return sqrt((px - x) * (px - x) + (py - y) * (py - y))
        }

        private fun dlt(src: DoubleArray, dst: DoubleArray, idx: IntArray): DoubleArray? {
            if (idx.size < 4) return null
            val ata = Array(9) { DoubleArray(9) }
            val row = DoubleArray(9)
            for (p in idx) {
                val u = src[2 * p]
                val v = src[2 * p + 1]
                val x = dst[2 * p]
                val y = dst[2 * p + 1]
                row[0] = u; row[1] = v; row[2] = 1.0
                row[3] = 0.0; row[4] = 0.0; row[5] = 0.0
                row[6] = -x * u; row[7] = -x * v; row[8] = -x
                addOuter(ata, row)
                row[0] = 0.0; row[1] = 0.0; row[2] = 0.0
                row[3] = u; row[4] = v; row[5] = 1.0
                row[6] = -y * u; row[7] = -y * v; row[8] = -y
                addOuter(ata, row)
            }
            val evec = smallestEigenvector(ata) ?: return null
            val h8 = evec[8]
            if (abs(h8) < 1e-12) return null
            val h = DoubleArray(9) { evec[it] / h8 }
            for (v in h) if (!v.isFinite()) return null
            return h
        }

        private fun addOuter(ata: Array<DoubleArray>, row: DoubleArray) {
            for (r in 0 until 9) {
                val rv = row[r]
                if (rv == 0.0) continue
                val rowR = ata[r]
                for (c in 0 until 9) rowR[c] += rv * row[c]
            }
        }

        private fun smallestEigenvector(a0: Array<DoubleArray>): DoubleArray? {
            val n = 9
            val a = Array(n) { r -> a0[r].copyOf() }
            val v = Array(n) { r -> DoubleArray(n) { c -> if (r == c) 1.0 else 0.0 } }
            for (sweep in 0 until 50) {
                var off = 0.0
                for (p in 0 until n) for (q in p + 1 until n) off += a[p][q] * a[p][q]
                if (off < 1e-24) break
                for (p in 0 until n) {
                    for (q in p + 1 until n) {
                        val apq = a[p][q]
                        if (abs(apq) < 1e-24) continue
                        val app = a[p][p]
                        val aqq = a[q][q]
                        val theta = (aqq - app) / (2.0 * apq)
                        val t = (if (theta >= 0) 1.0 else -1.0) / (abs(theta) + sqrt(theta * theta + 1.0))
                        val c = 1.0 / sqrt(t * t + 1.0)
                        val s = t * c
                        for (k in 0 until n) {
                            val akp = a[k][p]
                            val akq = a[k][q]
                            a[k][p] = c * akp - s * akq
                            a[k][q] = s * akp + c * akq
                        }
                        for (k in 0 until n) {
                            val apk = a[p][k]
                            val aqk = a[q][k]
                            a[p][k] = c * apk - s * aqk
                            a[q][k] = s * apk + c * aqk
                        }
                        for (k in 0 until n) {
                            val vkp = v[k][p]
                            val vkq = v[k][q]
                            v[k][p] = c * vkp - s * vkq
                            v[k][q] = s * vkp + c * vkq
                        }
                    }
                }
            }
            var minI = 0
            for (i in 1 until n) if (a[i][i] < a[minI][minI]) minI = i
            if (a[minI][minI] < -1e-9) return null
            val evec = DoubleArray(n) { v[it][minI] }
            if (evec.all { abs(it) < 1e-14 }) return null
            return evec
        }
    }
}
