package com.thesis.pickleballserve

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * On-device kNN5 (euclidean, distance-weighted) on timing features.
 *
 * Trains = bundled binary from scripts/export_knn_assets.py:
 *   int32 n, int32 f, float32 X[n,f], int32 y[n]
 *
 * Matches sklearn KNeighborsClassifier(n_neighbors=5, weights='distance')
 * with no feature scaling (best holdout mean 56.1% per RESULTS_SUMMARY).
 *
 * predict() returns FloatArray(3) softmax-like probs (normalized distance weights).
 */
class KnnClassifier private constructor(
    private val X: FloatArray,   // n * f row-major
    private val y: IntArray,     // class indices 0..2
    private val n: Int,
    private val f: Int,
    private val k: Int,
    val classes: Array<String>
) {
    companion object {
        private const val TAG = "KnnClassifier"

        fun load(context: Context): KnnClassifier {
            val metaJson = context.assets.open("knn_meta.json")
                .bufferedReader().use { it.readText() }
            val bytes = context.assets.open("knn_train.bin").use { it.readBytes() }
            val model = fromBytes(metaJson, bytes)
            Log.i(TAG, "Loaded kNN n=${model.n} f=${model.f} k=${model.k} classes=${model.classes.joinToString()}")
            return model
        }

        internal fun fromBytes(metaJson: String, bytes: ByteArray): KnnClassifier {
            val meta = JSONObject(metaJson)
            val n = meta.getInt("n_samples")
            val f = meta.getInt("n_features")
            val k = meta.getInt("k")
            val classesArr = meta.getJSONArray("classes")
            val classes = Array(classesArr.length()) { classesArr.getString(it) }

            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val nFile = buf.int
            val fFile = buf.int
            require(nFile == n && fFile == f) {
                "meta/bin mismatch meta=($n,$f) bin=($nFile,$fFile)"
            }
            val xSize = n * f
            val X = FloatArray(xSize)
            for (i in 0 until xSize) X[i] = buf.float
            val y = IntArray(n)
            for (i in 0 until n) y[i] = buf.int

            return KnnClassifier(X, y, n, f, k, classes)
        }
    }

    /**
     * @param feats 32-dim timing features
     * @return FloatArray(3) distance-weighted class probabilities
     */
    fun predictProba(feats: FloatArray): FloatArray {
        require(feats.size == f) { "expected $f features, got ${feats.size}" }

        // Compute squared distances to all train points
        val dists = FloatArray(n)
        for (i in 0 until n) {
            val off = i * f
            var s = 0f
            for (j in 0 until f) {
                val d = feats[j] - X[off + j]
                s += d * d
            }
            dists[i] = sqrt(s)
        }

        // Partial select top-k smallest (k is small, simple selection is fine)
        val idx = Array(n) { it }
        // Sort by distance ascending — n=438, fine to full sort
        idx.sortBy { dists[it] }

        val probs = FloatArray(classes.size)
        var wSum = 0f
        for (ti in 0 until k) {
            val i = idx[ti]
            val d = dists[i]
            // sklearn: weight = 1/d ; if d==0 -> infinite weight for that point only
            val w = if (d < 1e-12f) 1e12f else 1f / d
            val c = y[i]
            if (c in probs.indices) probs[c] += w
            wSum += w
        }
        if (wSum > 0f) {
            for (i in probs.indices) probs[i] /= wSum
        } else {
            // fallback uniform
            val u = 1f / probs.size
            for (i in probs.indices) probs[i] = u
        }
        return probs
    }

    fun predict(probs: FloatArray): Pair<String, Float> {
        var idx = 0
        for (i in 1 until probs.size) if (probs[i] > probs[idx]) idx = i
        return classes[idx] to probs[idx]
    }

    fun argmax(probs: FloatArray): Int {
        var idx = 0
        for (i in 1 until probs.size) if (probs[i] > probs[idx]) idx = i
        return idx
    }
}
