package com.thesis.pickleballserve

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * MediaPipe Pose Landmarker (BlazePose) pipeline.
 *
 * Converts camera frames -> 33 landmarks -> 5 joint angles (10 floats)
 * via JointAngles.fromLandmarks(), then feeds AngleBuffer.
 *
 * Runs in LIVE_STREAM mode; results arrive asynchronously on listener thread.
 */
class PosePipeline(
    context: Context,
    private val useGpu: Boolean = false,
    private val onPose: (landmarks: Array<PointF>, bitmap: Bitmap?, timestampMs: Long, realFrame: Boolean) -> Unit,
    private val onMissing: () -> Unit = {},
    private val onError: (String) -> Unit = {}
) {
    companion object {
        private const val TAG = "PosePipeline"
        // Model asset: download pose_landmarker_lite.task (or _full / _heavy) into assets/
        private const val MODEL_ASSET = "pose_landmarker_lite.task"
        private const val BITMAP_RING = 32
    }

    private var activeGpu = useGpu
    /** Backend actually in use after start() (false when GPU init fell back to CPU). */
    val usingGpu: Boolean get() = activeGpu
    private var landmarker: PoseLandmarker? = null
    @Volatile private var running = false
    private var lastTimestampMs = 0L

    // Cache landmark array to avoid per-frame allocation
    private val landmarkPts = Array(33) { PointF() }

    private val ringLock = Any()
    private data class Frame(val timestamp: Long, val bitmap: Bitmap, val real: Boolean)
    private val bitmapRing = ArrayDeque<Frame>()

    fun start(context: Context): Boolean {
        if (landmarker != null) return true
        running = true
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .setDelegate(if (activeGpu) Delegate.GPU else Delegate.CPU)
                .build()
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setNumPoses(1)
                .setResultListener { result: PoseLandmarkerResult, inputImage: MPImage ->
                    try { if (running) handleResult(result, result.timestampMs()) }
                    finally { inputImage.close() }
                }
                .setErrorListener { e: RuntimeException ->
                    Log.e(TAG, "PoseLandmarker error", e)
                    onError(e.message ?: "pose error")
                }
                .build()
            landmarker = PoseLandmarker.createFromOptions(context, options)
            Log.i(TAG, "PoseLandmarker started model=$MODEL_ASSET gpu=$activeGpu")
            return true
        } catch (e: Throwable) {
            if (activeGpu) {
                activeGpu = false
                Log.w(TAG, "GPU pose initialization failed; falling back to CPU", e)
                return start(context)
            }
            Log.e(TAG, "Failed to start PoseLandmarker", e)
            onError("Failed to load pose model: ${e.message}. " +
                    "Ensure $MODEL_ASSET is in app/src/main/assets/.")
            running = false
            return false
        }
    }

    /** Feed a camera frame (from CameraX ImageAnalysis / Preview). */
    fun analyze(bitmap: Bitmap, realFrame: Boolean = true) {
        val lm = landmarker ?: return
        // LIVE_STREAM requires strictly increasing timestamps
        val now = SystemClock.uptimeMillis()
        val ts = if (now <= lastTimestampMs) lastTimestampMs + 1 else now
        lastTimestampMs = ts

        synchronized(ringLock) {
            bitmapRing.addLast(Frame(ts, bitmap, realFrame))
            while (bitmapRing.size > BITMAP_RING) bitmapRing.removeFirst()
        }

        try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            lm.detectAsync(mpImage, ts)
        } catch (e: Throwable) {
            Log.e(TAG, "detectAsync failed", e)
        }
    }

    private fun handleResult(result: PoseLandmarkerResult, timestampMs: Long) {
        if (result.landmarks().isEmpty()) {
            onMissing()
            return
        }
        val pose = result.landmarks()[0]  // first (only) pose
        // pose: List<NormalizedLandmark> size 33
        if (pose.size < 33) return

        for (i in 0 until 33) {
            val lm: NormalizedLandmark = pose[i]
            // Convert normalized [0,1] to a consistent coordinate space.
            // Angles are scale/translation invariant, so we can use normalized
            // coords directly (matches Python which uses pixel coords before norm —
            // interior/signed angles only depend on directions).
            landmarkPts[i].set(lm.x(), lm.y())
        }
        val frame = synchronized(ringLock) {
            bitmapRing.firstOrNull { it.timestamp == timestampMs }
        } ?: return // Never associate a pose with a different image or synthetic-frame flag.
        onPose(landmarkPts, frame.bitmap, timestampMs, frame.real)
    }

    fun stop() {
        running = false
        try { landmarker?.close() } catch (_: Throwable) {}
        landmarker = null
        synchronized(ringLock) { bitmapRing.clear() }
    }
}
