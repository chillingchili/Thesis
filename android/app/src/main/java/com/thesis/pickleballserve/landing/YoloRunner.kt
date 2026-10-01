package com.thesis.pickleballserve.landing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest

/** All operations, including open/close, belong to the same inference worker. */
class YoloRunner private constructor(
    private val ballModel: Model,
    private val courtModel: Model,
    private val ballConfThr: Double,
    private val roiModel: Model? = null,
) : AutoCloseable {
    data class BallDet(
        val cx: Double, val cy: Double, val conf: Double,
        val x0: Double, val y0: Double, val x1: Double, val y1: Double,
    )

    data class CourtDet(val boxConf: Double, val kptsXy: DoubleArray, val kptConf: DoubleArray)
    data class FrameDet(val ball: BallDet?, val court: CourtDet?)

    private val owner = Thread.currentThread()
    private val inputSize = ballModel.interpreter.getInputTensor(0).shape()[2]
    private val input = YoloInput(inputSize)
    private val ballOut = arrayOf(Array(5) { FloatArray(ballModel.interpreter.getOutputTensor(0).shape()[2]) })
    private val courtOut = arrayOf(Array(47) { FloatArray(courtModel.interpreter.getOutputTensor(0).shape()[2]) })
    private val roiInput = if (roiModel != null) YoloInput(160) else null
    private val roiOut = arrayOf(Array(5) { FloatArray(525) })
    val backend: String
        get() = "ball=${ballModel.backend}, court=${courtModel.backend}"

    fun detectFrame(bmp: Bitmap, rotation: Int = 0, includeCourt: Boolean = true, includeBall: Boolean = true,
        ballRegion: Rect? = null): FrameDet {
        check(Thread.currentThread() === owner) { "YOLO inference must run on its initialization thread" }
        // The decoder mutates a reusable bitmap. Prepare once per frame, never cache by identity.
        val start = SystemClock.elapsedRealtime()
        val cropped = if (includeBall && ballRegion != null && roiModel != null) {
            require(rotation == 0) { "ROI input must already be oriented" }
            Bitmap.createBitmap(bmp, ballRegion.left, ballRegion.top, ballRegion.width(), ballRegion.height())
        } else null
        val useRoi = cropped != null
        val layout = if (useRoi) roiInput!!.prepare(cropped!!, 0) else input.prepare(bmp, rotation)
        val prepared = SystemClock.elapsedRealtime()
        try {
            if (includeBall) {
                if (useRoi) roiModel!!.run(roiInput!!.buffer, roiOut) else ballModel.run(input.buffer, ballOut)
            }
        } finally { if (cropped != null && cropped !== bmp) cropped.recycle() }
        val ballFinished = SystemClock.elapsedRealtime()
        val decoded = if (includeBall) decodeBall(layout, if (useRoi) roiOut[0] else ballOut[0], if (useRoi) 160 else inputSize) else null
        val ball = if (useRoi && decoded != null) {
            val dx = ballRegion!!.left; val dy = ballRegion.top
            decoded.copy(cx = decoded.cx + dx, cy = decoded.cy + dy,
                x0 = decoded.x0 + dx, y0 = decoded.y0 + dy, x1 = decoded.x1 + dx, y1 = decoded.y1 + dy)
        } else decoded
        val court = if (includeCourt) {
            val courtLayout = if (useRoi) input.prepare(bmp, rotation) else layout
            courtModel.run(input.buffer, courtOut)
            decodeCourt(courtLayout)
        } else null
        val finished = SystemClock.elapsedRealtime()
        Log.d(TAG, "frame preprocess=${prepared - start}ms ball=${ballFinished - prepared}ms " +
            "court=${finished - ballFinished}ms $backend")
        return FrameDet(ball, court)
    }

    private fun unX(v: Float, l: YoloInput.Layout): Double = (v * inputSize - l.left) / l.scale
    private fun unY(v: Float, l: YoloInput.Layout): Double = (v * inputSize - l.top) / l.scale

    private fun decodeBall(l: YoloInput.Layout, rows: Array<FloatArray>, size: Int): BallDet? {
        val k = rows[4].indices.maxByOrNull { rows[4][it] } ?: return null
        val conf = rows[4][k].toDouble()
        if (conf < ballConfThr || !conf.isFinite()) return null
        val cx = (rows[0][k] * size - l.left) / l.scale
        val cy = (rows[1][k] * size - l.top) / l.scale
        val w = rows[2][k] * size / l.scale
        val h = rows[3][k] * size / l.scale
        return BallDet(cx, cy, conf, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    }

    private fun decodeCourt(l: YoloInput.Layout): CourtDet? {
        val rows = courtOut[0]
        val k = rows[4].indices.maxByOrNull { rows[4][it] } ?: return null
        val conf = rows[4][k].toDouble()
        if (conf < 0.5 || !conf.isFinite()) return null
        val xy = DoubleArray(28)
        val confidence = DoubleArray(14)
        for (p in 0..13) {
            xy[2 * p] = unX(rows[5 + 3 * p][k], l)
            xy[2 * p + 1] = unY(rows[6 + 3 * p][k], l)
            confidence[p] = rows[7 + 3 * p][k].toDouble()
        }
        return CourtDet(conf, xy, confidence)
    }

    override fun close() {
        check(Thread.currentThread() === owner) { "YOLO cleanup must run on its initialization thread" }
        try { ballModel.close() } finally {
            try { courtModel.close() } finally {
                try { roiModel?.close() } finally { input.close(); roiInput?.close() }
            }
        }
    }

    private class Model(
        private val bytes: MappedByteBuffer,
        private val name: String,
        private val cacheDirectory: String,
        private val cpuThreads: Int,
    ) : AutoCloseable {
        lateinit var interpreter: Interpreter
        private var delegate: GpuDelegate? = null
        val backend: String get() = if (delegate != null) "GPU" else "CPU/XNNPACK"

        fun open(useGpu: Boolean): Model {
            if (useGpu) {
                try {
                    // Independent delegates prevent one model from invalidating the other's state.
                    val digest = MessageDigest.getInstance("SHA-256").apply { update(bytes.duplicate().apply { rewind() }) }
                    val token = name + "-" + digest.digest().joinToString("") { "%02x".format(it) }
                    delegate = GpuDelegate(GpuDelegateFactory.Options()
                        .setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                        .setSerializationParams(cacheDirectory, token))
                    interpreter = Interpreter(bytes, Interpreter.Options().apply {
                        setNumThreads(4)
                        setUseXNNPACK(true)
                        addDelegate(delegate!!)
                    })
                    return this
                } catch (e: RuntimeException) {
                    Log.w(TAG, "$name GPU initialization failed; using CPU", e)
                } catch (e: LinkageError) {
                    Log.w(TAG, "$name GPU library unavailable; using CPU", e)
                }
                closeGpu()
            }
            interpreter = cpuInterpreter()
            return this
        }

        fun run(input: ByteBuffer, output: Any) {
            input.rewind()
            try {
                interpreter.run(input, output)
            } catch (e: RuntimeException) {
                if (delegate == null) throw e
                Log.w(TAG, "$name GPU inference failed; retrying on CPU", e)
                interpreter.close()
                closeGpu()
                interpreter = cpuInterpreter()
                input.rewind()
                interpreter.run(input, output)
            }
        }

        private fun cpuInterpreter() = Interpreter(bytes, Interpreter.Options().apply {
            setNumThreads(cpuThreads)
            setUseXNNPACK(true)
        })

        private fun closeGpu() {
            delegate?.close()
            delegate = null
        }

        override fun close() {
            try { if (::interpreter.isInitialized) interpreter.close() } finally { closeGpu() }
        }
    }

    companion object {
        private const val TAG = "YoloRunner"

        fun open(
            context: Context, ballConfThr: Double = 0.3, useGpu: Boolean = false,
            cpuThreads: Int = 2,
            realtimeMode: Boolean = false,
            onProgress: (String) -> Unit = {},
        ): YoloRunner {
            val gpuSupported = useGpu && try {
                CompatibilityList().use { it.isDelegateSupportedOnThisDevice }
            } catch (e: RuntimeException) {
                Log.w(TAG, "GPU compatibility check failed; using CPU", e)
                false
            } catch (e: LinkageError) {
                Log.w(TAG, "GPU API unavailable; using CPU", e)
                false
            }
            val ballAsset = if (realtimeMode) "ball_320_w8a32.tflite" else "ball_int8.tflite"
            val courtAsset = if (realtimeMode) "court_320_w8a32.tflite" else "court_ft_int8.tflite"
            val ball = Model(load(context, ballAsset), "ball", context.codeCacheDir.absolutePath, cpuThreads)
            val court = Model(load(context, courtAsset), "court", context.codeCacheDir.absolutePath, cpuThreads)
            val roi = if (realtimeMode) Model(load(context, "ball_roi160_w8a32.tflite"), "ball-roi", context.codeCacheDir.absolutePath, cpuThreads) else null
            try {
                onProgress("Preparing ball detector…")
                ball.open(gpuSupported)
                onProgress("Preparing court detector…")
                court.open(gpuSupported)
                roi?.open(gpuSupported)
                val models = mutableListOf(Triple(ball, 5, if (realtimeMode) 320 else 640),
                    Triple(court, 47, if (realtimeMode) 320 else 640))
                roi?.let { models.add(Triple(it, 5, 160)) }
                for ((model, channels, size) in models) {
                    val anchors = (size / 8) * (size / 8) + (size / 16) * (size / 16) + (size / 32) * (size / 32)
                    require(model.interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(1, 3, size, size)) &&
                        model.interpreter.getInputTensor(0).dataType() == DataType.FLOAT32) {
                        "Expected float32 NCHW YOLO input [1,3,$size,$size]"
                    }
                    require(model.interpreter.getOutputTensor(0).shape().contentEquals(intArrayOf(1, channels, anchors)) &&
                        model.interpreter.getOutputTensor(0).dataType() == DataType.FLOAT32) {
                        "Unexpected YOLO output tensor; check the exported model"
                    }
                }
                return YoloRunner(ball, court, ballConfThr, roi).also { Log.i(TAG, "loaded ${it.backend}, ROI=${roi != null}") }
            } catch (e: Exception) {
                try { ball.close() } finally { try { court.close() } finally { roi?.close() } }
                throw e
            } catch (e: LinkageError) {
                try { ball.close() } finally { try { court.close() } finally { roi?.close() } }
                throw e
            }
        }

        private fun load(context: Context, assetName: String): MappedByteBuffer =
            context.assets.openFd(assetName).use { fd ->
                FileInputStream(fd.fileDescriptor).use { stream ->
                    stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
    }
}
