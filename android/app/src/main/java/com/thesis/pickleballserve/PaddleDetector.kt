package com.thesis.pickleballserve

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Paddle YOLOv26s int8 TFLite runner — letterbox + decode mirrors
 * landing.YoloRunner; box selection mirrors pick_det() in
 * scripts/extract_paddle_angles.py (nearest box center to the right wrist
 * within 300px scaled by frame width over the 1920px reference, highest
 * conf when the wrist is unknown, none when nothing qualifies).
 *
 * Optional GPU delegate: create this object and call detectAll() from a
 * single thread, because a GpuDelegate must be built and invoked on the same
 * thread. Initialization and per-inference failures both fall back to CPU,
 * so enabling the GPU can never leave the detector unusable.
 */
class PaddleDetector private constructor(
    private var tflite: Interpreter,
    private val confThr: Float,
    private val bytes: MappedByteBuffer,
    private val cacheDirectory: String,
    private val cpuThreads: Int,
) : AutoCloseable {

    data class Det(val box: FloatArray, val conf: Float) {
        val cx: Float get() = (box[0] + box[2]) / 2f
        val cy: Float get() = (box[1] + box[3]) / 2f
    }

    private data class Layout(val scale: Double, val left: Double, val top: Double)

    private val input = ByteBuffer.allocateDirect(INPUT_BYTES).order(ByteOrder.nativeOrder())
    private val inputFloat = input.asFloatBuffer()
    private val fArray = FloatArray(3 * 640 * 640)
    private var pxArray = IntArray(640 * 640)
    private var scaledBitmap: Bitmap? = null
    private var scaledCanvas: Canvas? = null
    private val out = arrayOf(Array(5) { FloatArray(8400) })
    private var lastBitmap: Bitmap? = null
    private var lastLayout: Layout? = null
    private var delegate: GpuDelegate? = null
    val backend: String get() = if (delegate != null) "GPU" else "CPU/XNNPACK"

    private fun letterbox(bmp: Bitmap): Layout {
        if (bmp === lastBitmap && lastLayout != null) return lastLayout!!
        val w = bmp.width
        val h = bmp.height
        val scale = min(640.0 / w, 640.0 / h)
        val nw = (w * scale).roundToInt()
        val nh = (h * scale).roundToInt()
        val left = ((640 - nw) / 2.0 - 0.1).roundToInt()
        val top = ((640 - nh) / 2.0 - 0.1).roundToInt()

        if (scaledBitmap == null || scaledBitmap!!.width != nw || scaledBitmap!!.height != nh) {
            scaledBitmap?.recycle()
            scaledBitmap = Bitmap.createBitmap(nw, nh, Bitmap.Config.ARGB_8888)
            scaledCanvas = Canvas(scaledBitmap!!)
        }
        scaledCanvas!!.drawBitmap(bmp, Rect(0, 0, w, h), Rect(0, 0, nw, nh), null)

        val n = nw * nh
        if (pxArray.size < n) pxArray = IntArray(n)
        scaledBitmap!!.getPixels(pxArray, 0, nw, 0, 0, nw, nh)

        val area = 640 * 640
        val v = 114f / 255f
        fArray.fill(v)
        var sy = 0
        while (sy < nh) {
            var sx = 0
            var dstIdx = (top + sy) * 640 + left
            var srcIdx = sy * nw
            while (sx < nw) {
                val c = pxArray[srcIdx++]
                fArray[dstIdx] = ((c shr 16) and 0xFF) / 255f
                fArray[area + dstIdx] = ((c shr 8) and 0xFF) / 255f
                fArray[2 * area + dstIdx] = (c and 0xFF) / 255f
                dstIdx++
                sx++
            }
            sy++
        }

        inputFloat.position(0)
        inputFloat.put(fArray)
        val l = Layout(scale, left.toDouble(), top.toDouble())
        lastBitmap = bmp
        lastLayout = l
        return l
    }

    private fun unX(v: Float, l: Layout): Double = (v * 640.0 - l.left) / l.scale
    private fun unY(v: Float, l: Layout): Double = (v * 640.0 - l.top) / l.scale

    fun detectAll(bmp: Bitmap): List<Det> {
        val l = letterbox(bmp)
        runInference()
        val rows = out[0]
        val dets = ArrayList<Det>()
        for (k in 0 until 8400) {
            val conf = rows[4][k]
            if (conf < confThr) continue
            val cx = unX(rows[0][k], l)
            val cy = unY(rows[1][k], l)
            val w = rows[2][k] * 640.0 / l.scale
            val h = rows[3][k] * 640.0 / l.scale
            dets.add(Det(floatArrayOf(
                (cx - w / 2).toFloat(), (cy - h / 2).toFloat(),
                (cx + w / 2).toFloat(), (cy + h / 2).toFloat()), conf))
        }
        return dets
    }

    /** Runs on GPU when enabled; an inference-time GPU failure demotes to CPU for good. */
    private fun runInference() {
        input.rewind()
        try {
            tflite.run(input, out)
        } catch (e: RuntimeException) {
            if (delegate == null) throw e
            Log.w(TAG, "paddle GPU inference failed; falling back to CPU", e)
            val g = delegate
            delegate = null
            try { tflite.close() } catch (_: Throwable) {}
            try { g?.close() } catch (_: Throwable) {}
            tflite = cpuInterpreter(bytes, cpuThreads)
            input.rewind()
            tflite.run(input, out)
        }
    }

    /** Logs the live interpreter's shapes; must read tflite, which enableGpu() may have swapped. */
    private fun logLoaded() {
        Log.i(TAG, "loaded paddle backend=$backend in=" +
                tflite.getInputTensor(0).shape().joinToString() +
                " out=" + tflite.getOutputTensor(0).shape().joinToString())
    }

    /** Best-effort GPU enable. Any failure leaves the detector on CPU. */
    private fun enableGpu() {
        val supported = try {
            CompatibilityList().use { it.isDelegateSupportedOnThisDevice }
        } catch (e: RuntimeException) {
            Log.w(TAG, "paddle GPU compatibility check failed; using CPU", e); false
        } catch (e: LinkageError) {
            Log.w(TAG, "paddle GPU library unavailable; using CPU", e); false
        }
        if (!supported) return

        var g: GpuDelegate? = null
        try {
            // Cache the compiled program so the multi-minute first-run compilation
            // happens once per install instead of once per launch (see landing.YoloRunner).
            val digest = MessageDigest.getInstance("SHA-256")
                .apply { update(bytes.duplicate().apply { rewind() }) }
            val token = "paddle-" + digest.digest().joinToString("") { "%02x".format(it) }
            g = GpuDelegate(GpuDelegateFactory.Options()
                .setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                .setSerializationParams(cacheDirectory, token))
            val interp = Interpreter(bytes, Interpreter.Options().apply {
                setNumThreads(cpuThreads)
                setUseXNNPACK(true)
                addDelegate(g)
            })
            val cpu = tflite
            delegate = g
            tflite = interp
            cpu.close()
        } catch (e: RuntimeException) {
            Log.w(TAG, "paddle GPU initialization failed; using CPU", e)
            try { g?.close() } catch (_: Throwable) {}
            delegate = null
        } catch (e: LinkageError) {
            Log.w(TAG, "paddle GPU library unavailable; using CPU", e)
            try { g?.close() } catch (_: Throwable) {}
            delegate = null
        }
    }

    override fun close() {
        try { tflite.close() } finally { closeGpu() }
    }

    private fun closeGpu() {
        try { delegate?.close() } catch (_: Throwable) {}
        delegate = null
    }

    companion object {
        private const val TAG = "PaddleDetector"
        private const val INPUT_BYTES = 1 * 3 * 640 * 640 * 4
        const val CONF_THR = 0.25f
        const val WRIST_MAX_PX = 300f
        const val REF_WIDTH_PX = 1920f

        /** pick_det(): nearest box center to wrist within maxPx; argmax conf if wrist unknown. */
        fun pick(dets: List<Det>, wristX: Float, wristY: Float, maxPx: Float): Det? {
            if (dets.isEmpty()) return null
            if (!wristX.isNaN() && !wristY.isNaN()) {
                var best: Det? = null
                var bestD = Float.MAX_VALUE
                for (d in dets) {
                    val dx = d.cx - wristX
                    val dy = d.cy - wristY
                    val dist = kotlin.math.sqrt(dx * dx + dy * dy)
                    if (dist < bestD) {
                        bestD = dist
                        best = d
                    }
                }
                return if (bestD <= maxPx) best else null
            }
            return dets.maxByOrNull { it.conf }
        }

        /**
         * useGpu: enable the GPU delegate when the device supports it. Creation and
         * every detectAll() call must then happen on the same thread. Failures at
         * either stage fall back to CPU/XNNPACK.
         */
        fun open(context: Context, numThreads: Int = 4, useGpu: Boolean = false): PaddleDetector {
            val bytes = load(context, "paddle_w8a32.tflite")
            val interp = cpuInterpreter(bytes, numThreads)
            val det = PaddleDetector(
                interp, CONF_THR, bytes, context.codeCacheDir.absolutePath, numThreads,
            )
            if (useGpu) det.enableGpu()
            det.logLoaded()
            return det
        }

        private fun cpuInterpreter(bytes: MappedByteBuffer, numThreads: Int) =
            Interpreter(bytes, Interpreter.Options().apply {
                setNumThreads(numThreads)
                setUseXNNPACK(true)
            })

        private fun load(context: Context, assetName: String): MappedByteBuffer {
            context.assets.openFd(assetName).use { fd ->
                FileInputStream(fd.fileDescriptor).use { input ->
                    val channel = input.channel
                    return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
        }
    }
}
