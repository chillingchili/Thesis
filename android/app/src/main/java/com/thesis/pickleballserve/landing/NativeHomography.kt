package com.thesis.pickleballserve.landing

import org.opencv.calib3d.Calib3d
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point

/** OpenCV's normalized native solver avoids repeated Kotlin matrix/eigenvector allocations. */
internal object NativeHomography {
    fun estimate(src: DoubleArray, dst: DoubleArray, threshold: Double): DoubleArray? {
        if (src.size < 8 || src.size != dst.size) return null
        val from = MatOfPoint2f(*Array(src.size / 2) { Point(src[it * 2], src[it * 2 + 1]) })
        val to = MatOfPoint2f(*Array(dst.size / 2) { Point(dst[it * 2], dst[it * 2 + 1]) })
        try {
            val matrix = Calib3d.findHomography(from, to, Calib3d.RANSAC, threshold)
            try {
                if (matrix.empty()) return null
                return DoubleArray(9).also { matrix.get(0, 0, it) }.takeIf { it.all(Double::isFinite) }
            } finally { matrix.release() }
        } finally { from.release(); to.release() }
    }
}
