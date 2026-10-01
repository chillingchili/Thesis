package com.thesis.pickleballserve

import kotlin.math.exp
import kotlin.math.ln
import org.json.JSONObject

/** A policy belongs to its fitted model and feature contract, never to arbitrary old assets. */
data class ProbabilityPolicy(val gruWeight: Double, val temperature: Double, val threshold: Double?) {
    init { require(gruWeight in 0.0..1.0 && temperature.isFinite() && temperature > 0)
           require(threshold == null || threshold in 0.0..1.0) }
    fun apply(gru: FloatArray, knn: FloatArray): FloatArray {
        require(gru.size == 3 && knn.size == 3)
        val logits = DoubleArray(3) { i -> ln((gruWeight*gru[i]+(1-gruWeight)*knn[i]).coerceIn(1e-7,1.0))/temperature }
        val top = logits.maxOrNull()!!
        val e = DoubleArray(3) { exp(logits[it]-top) }; val sum=e.sum()
        return FloatArray(3) { (e[it]/sum).toFloat() }
    }
    companion object {
        fun fromJson(json: JSONObject) = ProbabilityPolicy(json.getDouble("gru_weight"), json.getDouble("temperature"),
            if (json.isNull("confidence_threshold")) null else json.getDouble("confidence_threshold"))
    }
}
