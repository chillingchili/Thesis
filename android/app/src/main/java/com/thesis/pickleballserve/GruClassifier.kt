package com.thesis.pickleballserve

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.TensorFlowLite
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * TFLite GRU classifier for pickleball serve subtype (drive / lob / topspin).
 *
 * Supports:
 *   - single mode: one .tflite (gru_single.tflite = fold1)
 *   - ensemble mode: average softmax across multiple .tflite files
 *
 * Moment-matching TTA (optional): aligns each live window's per-channel
 * mean/std to the training-set stats stored in manifest.json before inference.
 * Mirrors analysis/quick_wins.py align_moment().
 *
 * Input shape : [1, 128, 10] float32  (batch, seq_len, n_features)
 * Output shape: [1, 3] float32        (softmax over CLASSES)
 *
 * Requires SELECT_TF_OPS native libs (bundled via tensorflow-lite-select-tf-ops).
 */
class GruClassifier private constructor(
    private val interpreters: List<Interpreter>,
    val classes: Array<String>,
    val mode: Mode,
    private val trainMean: FloatArray?,
    private val trainStd: FloatArray?,
    val momentMatchingEnabled: Boolean
) {
    enum class Mode { SINGLE, ENSEMBLE }

    companion object {
        private const val TAG = "GruClassifier"
        private const val SEQ_LEN = 128
        private const val N_FEATURES = 10

        fun load(context: Context, ensemble: Boolean = true): GruClassifier {
            val manifestJson = context.assets.open("manifest.json")
                .bufferedReader().use { it.readText() }
            val manifest = JSONObject(manifestJson)
            val classes = arrayOf(
                manifest.getJSONArray("classes").getString(0),
                manifest.getJSONArray("classes").getString(1),
                manifest.getJSONArray("classes").getString(2)
            )

            val modelNames: List<String> = if (ensemble) {
                val arr = manifest.getJSONArray("ensemble")
                (0 until arr.length()).map { arr.getString(it) }
            } else {
                listOf(manifest.getString("single"))
            }

            // Moment-matching stats (optional)
            val trainMean: FloatArray? = if (manifest.has("feature_mean")) {
                val arr = manifest.getJSONArray("feature_mean")
                FloatArray(arr.length()) { i -> arr.getDouble(i).toFloat() }
            } else null
            val trainStd: FloatArray? = if (manifest.has("feature_std")) {
                val arr = manifest.getJSONArray("feature_std")
                FloatArray(arr.length()) { i -> arr.getDouble(i).toFloat() }
            } else null
            val momentOn = trainMean != null && trainStd != null &&
                    manifest.optBoolean("moment_matching", false)

            try {
                TensorFlowLite.init()
            } catch (e: Throwable) {
                Log.w(TAG, "TensorFlowLite.init() warning: ${e.message}")
            }
            try {
                System.loadLibrary("tensorflowlite_jni_select_ops")
            } catch (e: Throwable) {
                Log.w(TAG, "select_ops load warning (may already be loaded): ${e.message}")
            }

            val interps = modelNames.map { name ->
                val model = loadModelFile(context, name)
                val opts = Interpreter.Options().apply { setNumThreads(2) }
                Interpreter(model, opts).also {
                    Log.i(TAG, "Loaded $name  in=${it.getInputTensor(0).shape().joinToString()}  " +
                            "out=${it.getOutputTensor(0).shape().joinToString()}")
                }
            }

            val mode = if (ensemble) Mode.ENSEMBLE else Mode.SINGLE
            Log.i(TAG, "GruClassifier ready mode=$mode n=${interps.size} " +
                    "momentMatching=$momentOn classes=${classes.joinToString()}")
            return GruClassifier(interps, classes, mode, trainMean, trainStd, momentOn)
        }

        private fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
            context.assets.openFd(assetName).use { fd ->
                FileInputStream(fd.fileDescriptor).use { input ->
                    val channel = input.channel
                    return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
        }
    }

    /**
     * Moment-match a (128, 10) window to training stats.
     * Only aligns the first [validFrames] rows (non-pad); trailing zero-pad left alone.
     * Formula (per channel j): x' = (x - win_mean_j) / (win_std_j + eps) * train_std_j + train_mean_j
     */
    private fun momentMatch(window: Array<FloatArray>, validFrames: Int): Array<FloatArray> {
        if (!momentMatchingEnabled || trainMean == null || trainStd == null || validFrames < 2) {
            return window
        }
        val out = Array(window.size) { FloatArray(window[0].size) }
        // Copy pad region as-is
        for (t in validFrames until window.size) {
            System.arraycopy(window[t], 0, out[t], 0, window[t].size)
        }
        // Per-channel stats over valid frames only
        for (j in 0 until N_FEATURES) {
            var sum = 0.0
            for (t in 0 until validFrames) sum += window[t][j]
            val winMean = (sum / validFrames).toFloat()
            var sq = 0.0
            for (t in 0 until validFrames) {
                val d = window[t][j] - winMean
                sq += d * d
            }
            val winStd = sqrt((sq / validFrames).toFloat()).coerceAtLeast(1e-6f)
            val tMean = trainMean[j]
            val tStd = trainStd[j]
            for (t in 0 until validFrames) {
                out[t][j] = ((window[t][j] - winMean) / winStd) * tStd + tMean
            }
        }
        return out
    }

    /**
     * Run inference on one window.
     * @param window Array[1][128][10] from AngleBuffer.toInputTensor()
     * @param validFrames number of real (non-pad) frames in the window
     * @return FloatArray(size=3) softmax probabilities (ensemble-averaged, moment-matched if enabled)
     */
    fun infer(window: Array<Array<FloatArray>>, validFrames: Int = SEQ_LEN): FloatArray {
        require(window.size == 1 && window[0].size == SEQ_LEN && window[0][0].size == N_FEATURES) {
            "bad window shape ${window.size}x${window[0].size}x${window[0][0].size}"
        }

        // Apply moment matching on the inner (128, 10) slice
        val matched = momentMatch(window[0], validFrames)

        val inputBuf = ByteBuffer.allocateDirect(1 * SEQ_LEN * N_FEATURES * 4)
            .order(ByteOrder.nativeOrder())
        for (t in 0 until SEQ_LEN) {
            for (f in 0 until N_FEATURES) {
                inputBuf.putFloat(matched[t][f])
            }
        }
        inputBuf.rewind()

        val outputBuf = ByteBuffer.allocateDirect(1 * classes.size * 4)
            .order(ByteOrder.nativeOrder())

        val acc = FloatArray(classes.size)
        for (interp in interpreters) {
            outputBuf.clear()
            interp.run(inputBuf, outputBuf)
            outputBuf.rewind()
            for (i in 0 until classes.size) {
                acc[i] += outputBuf.float
            }
        }
        val n = interpreters.size.toFloat()
        for (i in acc.indices) acc[i] /= n

        return acc
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

    fun close() {
        interpreters.forEach { it.close() }
    }
}
