package com.thesis.pickleballserve

/** Same selected classifier feeds the final display and coaching confidence gate. */
object HybridPrediction {
    fun combine(gru: FloatArray, knn: FloatArray?): FloatArray {
        require(gru.size == 3 && (knn == null || knn.size == 3))
        return if (knn == null) gru.copyOf() else FloatArray(3) { (gru[it] + knn[it]) * 0.5f }
    }
}
