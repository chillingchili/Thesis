package com.thesis.pickleballserve

import kotlin.math.sqrt

/** Complete selected serve, elapsed-time resampling. Independent of the legacy ring buffer. */
object ServeSequence {
    val required = intArrayOf(11, 12, 14, 16, 20, 23, 24)
    data class Frame(val timestampMs: Double, val angles: FloatArray?,
                     val xy: FloatArray?, val visibility: FloatArray?)
    data class Result(val angles: Array<FloatArray>, val skeleton: Array<FloatArray>,
                      val motion: Array<FloatArray>, val sourceFrames: Int,
                      val validSourceFrames: Int, val validWindowFrames: Int, val accepted: Boolean) {
        fun features(condition: String): Array<FloatArray> = when {
            condition == "angles_thesis_aug" -> angles
            condition == "skeleton_thesis_aug" -> skeleton
            condition == "skeleton_motion_thesis_aug" -> motion
            condition.startsWith("lean") -> Array(128) { t ->
                FloatArray(45).also { out ->
                    angles[t].copyInto(out)
                    required.forEachIndexed { i, j ->
                        out[10 + 2*i] = skeleton[t][2*j]
                        out[11 + 2*i] = skeleton[t][2*j+1]
                        out[24+i] = skeleton[t][66+j]
                        out[31+2*i] = motion[t][99+2*j]
                        out[32+2*i] = motion[t][100+2*j]
                    }
                }
            }
            else -> error("Unknown feature contract: $condition")
        }
    }

    fun build(frames: List<Frame>, aspect: Double): Result {
        require(frames.size >= 2 && aspect.isFinite() && aspect > 0)
        require(frames.all { it.timestampMs.isFinite() })
        require(frames.zipWithNext().all { (a,b) -> b.timestampMs > a.timestampMs })
        require(frames.last().timestampMs - frames.first().timestampMs <= 12000.0)
        val n = frames.size
        val coordinates = Array(n) { DoubleArray(66) }
        val masks = Array(n) { BooleanArray(33) }
        val angleSource = Array(n) { FloatArray(10) }
        val angleValid = BooleanArray(n)
        val scales = ArrayList<Double>()
        for (i in frames.indices) {
            val frame = frames[i]
            val xy = frame.xy ?: continue
            val visibility = frame.visibility ?: continue
            require(xy.size == 66 && visibility.size == 33)
            for (j in 0 until 33) masks[i][j] = visibility[j] >= .5f && xy[2*j].isFinite() && xy[2*j+1].isFinite()
            val a = frame.angles
            if (a != null && a.size == 10 && a.all { it.isFinite() } && required.all { masks[i][it] }) {
                angleSource[i] = a.copyOf(); angleValid[i] = true
            }
            val anchors = intArrayOf(11,12,23,24).all { masks[i][it] }
            if (!anchors) { masks[i].fill(false); continue }
            val hx = (xy[46] + xy[48]).toDouble()*.5*aspect
            val hy = (xy[47] + xy[49]).toDouble()*.5
            val sx = (xy[22] + xy[24]).toDouble()*.5*aspect
            val sy = (xy[23] + xy[25]).toDouble()*.5
            val torso = sqrt((sx-hx)*(sx-hx)+(sy-hy)*(sy-hy))
            if (torso > 1e-6) scales.add(torso)
            for (j in 0 until 33) {
                coordinates[i][2*j] = xy[2*j]*aspect-hx
                coordinates[i][2*j+1] = xy[2*j+1]-hy
            }
        }
        scales.sort()
        val scale = if (scales.isEmpty()) 1.0 else if (scales.size%2 == 1) scales[scales.size/2]
                    else (scales[scales.size/2-1]+scales[scales.size/2])*.5
        val angles = Array(128) { FloatArray(10) }
        val skeleton = Array(128) { FloatArray(99) }
        var right = 0
        for (t in 0 until 128) {
            val target = frames.first().timestampMs+(frames.last().timestampMs-frames.first().timestampMs)*t/127.0
            while (right < n-1 && frames[right].timestampMs < target) right++
            val left = if (kotlin.math.abs(target-frames[right].timestampMs)<1e-6) right else maxOf(0,right-1)
            val span = frames[right].timestampMs-frames[left].timestampMs
            val alpha = if (span > 0) (target-frames[left].timestampMs)/span else 0.0
            if (angleValid[left] && angleValid[right]) {
                var usable = true
                for (k in 0 until 5) {
                    val sin = angleSource[left][2*k]*(1-alpha)+angleSource[right][2*k]*alpha
                    val cos = angleSource[left][2*k+1]*(1-alpha)+angleSource[right][2*k+1]*alpha
                    val norm = sqrt(sin*sin+cos*cos)
                    if (norm <= 1e-6) { usable=false; break }
                    angles[t][2*k]=(sin/norm).toFloat(); angles[t][2*k+1]=(cos/norm).toFloat()
                }
                if (!usable) angles[t].fill(0f)
            }
            if (scales.isNotEmpty()) for (j in 0 until 33) if (masks[left][j] && masks[right][j]) {
                skeleton[t][66+j]=1f
                for (c in 0..1) skeleton[t][2*j+c]=((coordinates[left][2*j+c]*(1-alpha)+coordinates[right][2*j+c]*alpha)/scale).toFloat()
            }
        }
        val motion = Array(128) { t -> FloatArray(165).also { out ->
            skeleton[t].copyInto(out)
            if (t > 0) for (j in 0 until 33) if (skeleton[t][66+j] > 0 && skeleton[t-1][66+j] > 0)
                for (c in 0..1) out[99+2*j+c]=skeleton[t][2*j+c]-skeleton[t-1][2*j+c]
        } }
        val validSource=angleValid.count { it }
        val validWindow=angles.count { a -> a.any { it != 0f } }
        return Result(angles,skeleton,motion,n,validSource,validWindow,
                      validSource >= 32 && validSource.toDouble()/n >= .8 && validWindow/128.0 >= .8)
    }
}
