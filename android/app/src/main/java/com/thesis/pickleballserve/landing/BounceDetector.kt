package com.thesis.pickleballserve.landing

import kotlin.math.abs

object BounceDetector {

    data class Hit(val frame: Int, val xFt: Double, val yFt: Double)

    fun detect(
        cy: DoubleArray,
        wx: DoubleArray,
        wy: DoubleArray,
        minI: Int = 0,
        fps: Double = 30.0,
    ): Hit? {
        val n = cy.size
        if (n == 0) return null
        val valid = BooleanArray(n) { !cy[it].isNaN() }
        val runs = ArrayList<IntArray>()
        var a = -1
        var b = -1
        for (f in 0 until n) {
            if (!valid[f]) continue
            if (a < 0) {
                a = f
                b = f
            } else if (f == b + 1) {
                b = f
            } else {
                runs.add(intArrayOf(a, b))
                a = f
                b = f
            }
        }
        if (a >= 0) runs.add(intArrayOf(a, b))
        for (run in runs) {
            val ra = run[0]
            val rb = run[1]
            var j = ra
            while (j <= rb) {
                var k = j
                while (k + 1 <= rb && abs(cy[k + 1] - cy[j]) <= 0.6) k++
                if (k - j + 1 >= 8) {
                    for (t in j..k) valid[t] = false
                    j = k + 1
                } else {
                    j++
                }
            }
        }
        val validCount = valid.count { it }
        if (validCount < 20) return null
        val vidx = IntArray(validCount)
        run {
            var t = 0
            for (i in 0 until n) if (valid[i]) vidx[t++] = i
        }
        val s = fps / 30.0
        val win = maxOf(3, Math.round(5 * s).toInt()) or 1
        val preN = maxOf(4, Math.round(8 * s).toInt())
        val postN = maxOf(4, Math.round(6 * s).toInt())
        val thr = 0.6 / s
        val start = maxOf(12, Math.round(12 * s).toInt(), minI)
        val end = n - maxOf(6, Math.round(8 * s).toInt())
        if (start >= end) return null

        val cyS = interp(cy, valid)
        val vy = gradient(convolveSame(cyS, win))

        val vwx = BooleanArray(n) { !wx[it].isNaN() && valid[it] }
        val vwy = BooleanArray(n) { !wy[it].isNaN() && valid[it] }
        if (!vwx.any() || !vwy.any()) return null
        val wxS = interp(wx, vwx)
        val wyS = interp(wy, vwy)

        var i = start
        while (i < end) {
            var preSum = 0.0
            for (t in i - preN until i) preSum += vy[t]
            var postSum = 0.0
            for (t in i until i + postN) postSum += vy[t]
            if (preSum / preN >= thr && postSum / postN <= -thr) {
                var beforeLast = -1
                var afterFirst = -1
                var nearBefore = 0
                var nearAfter = 0
                var anchor = false
                for (vi in vidx) {
                    when {
                        vi < i -> {
                            beforeLast = vi
                            if (vi >= i - 2 * preN) {
                                nearBefore++
                                if (vi <= i - preN) anchor = true
                            }
                        }
                        vi > i -> {
                            if (afterFirst < 0) afterFirst = vi
                            if (vi <= i + 2 * postN) nearAfter++
                        }
                    }
                }
                if (beforeLast >= 0 && afterFirst >= 0 &&
                    i - beforeLast <= 2 * preN && afterFirst - i <= 2 * postN &&
                    nearBefore >= 3 && nearAfter >= 2 && anchor
                ) {
                    val wxi = wxS[i]
                    val wyi = wyS[i]
                    if (wxi >= -4 && wxi <= 24 && wyi >= -4 && wyi <= 48) {
                        return Hit(i, wxi, wyi)
                    }
                }
            }
            i++
        }
        return null
    }

    private fun interp(vals: DoubleArray, mask: BooleanArray): DoubleArray {
        val xs = IntArray(mask.count { it })
        val ys = DoubleArray(xs.size)
        run {
            var t = 0
            for (i in vals.indices) if (mask[i]) {
                xs[t] = i
                ys[t] = vals[i]
                t++
            }
        }
        val out = DoubleArray(vals.size)
        if (xs.isEmpty()) return out
        var seg = 0
        for (i in vals.indices) {
            out[i] = when {
                i <= xs[0] -> ys[0]
                i >= xs[xs.size - 1] -> ys[ys.size - 1]
                else -> {
                    while (seg < xs.size - 2 && i >= xs[seg + 1]) seg++
                    val x0 = xs[seg].toDouble()
                    val x1 = xs[seg + 1].toDouble()
                    val t = (i - x0) / (x1 - x0)
                    ys[seg] + t * (ys[seg + 1] - ys[seg])
                }
            }
        }
        return out
    }

    private fun convolveSame(x: DoubleArray, win: Int): DoubleArray {
        val n = x.size
        val c = (win - 1) / 2
        val out = DoubleArray(n)
        val k = 1.0 / win
        for (i in 0 until n) {
            var sum = 0.0
            for (j in 0 until win) {
                val idx = i + c - j
                if (idx in 0 until n) sum += x[idx]
            }
            out[i] = sum * k
        }
        return out
    }

    private fun gradient(y: DoubleArray): DoubleArray {
        val n = y.size
        val out = DoubleArray(n)
        if (n == 1) return out
        out[0] = y[1] - y[0]
        for (i in 1 until n - 1) out[i] = (y[i + 1] - y[i - 1]) / 2.0
        out[n - 1] = y[n - 1] - y[n - 2]
        return out
    }
}
