package com.thesis.pickleballserve

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.thesis.pickleballserve.databinding.ActivityMainBinding
import com.thesis.pickleballserve.landing.LandingActivity
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2

private const val RIGHT_WRIST = 16
private const val LATENCY_TAG = "LiveLatency"

/**
 * Intent extras that replace the camera with a recorded clip played through the
 * live path in real time. `replay_clip` is a file name inside files/latency_input;
 * `replay_loop` repeats it with a gap so a serve can be watched more than once.
 */
const val EXTRA_REPLAY_CLIP = "replay_clip"
const val EXTRA_REPLAY_LOOP = "replay_loop"

/**
 * Live demo: camera -> BlazePose -> joint angles -> [GRU | GRU+kNN5 hybrid]
 *
 * Modes (cycle via Toggle):
 *   SINGLE    - one GRU fold (fastest)
 *   ENSEMBLE  - avg of 5 GRU folds + moment matching
 *   HYBRID    - avg( GRU ensemble , kNN5-on-timing-features )
 */
class MainActivity : AppCompatActivity() {

    private enum class RunMode { SINGLE, ENSEMBLE, HYBRID }

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var paddleExecutor: ThreadPoolExecutor
    private var posePipeline: PosePipeline? = null
    private var clipReplay: ServeClipReplay? = null
    private var gru: GruClassifier? = null
    private var knn: KnnClassifier? = null
    private var shiftDetector: ShiftDetector? = null
    @Volatile private var paddleStream: PaddleStream? = null
    private var feedbackEngine: FeedbackEngine? = null
    @Volatile private var paddleFailed = false
    @Volatile private var resultGeneration = 0L
    private var posePriority = true
    private var revisedEnd = false
    private var lastPaddleSubmitMs = 0L
    private var paddleThrottled = 0L
    private val paddleBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    private val serveWindow = ServeWindow(preRollMs = 1000L)

    private var mode = RunMode.ENSEMBLE  // Original GRU selected before Coach C Session 2 release.

    private val minFramesForInfer = 48

    // --- Motion gate: buffer only records during an active serve ---
    private var gateActive = false
    private var lowMotionSinceMs = 0L    // uptimeMillis of first low-motion frame, 0 = not in tail
    private val motionThreshold = 0.07f  // floor for start arm (rad/frame mean)
    private val lowHoldMs = 350L         // wall-clock below end threshold = serve ended
    private val finalMinFrames = 32      // min frames for end-of-serve final inference
    private val maxActiveMs = 5300L      // force-end safety timeout (buffer only holds 128 anyway)
    private val paddleQuiesceMs = 250L   // wait for queued paddle samples at serve end
    private val refractoryMs = 1200L     // ignore re-triggers right after a serve
    private var refractoryUntilMs = 0L
    // adaptive idle stats: threshold = idleMean + k*idleDev (re-measured while idle,
    // so standing pose noise can never keep the gate stuck open)
    private var idleMean = 0f
    private var idleDev = 0.02f
    private var lastAngles: FloatArray? = null
    private val motionRate = MotionRate()
    private val motionRing = FloatArray(5)
    private var motionRingIdx = 0
    private var motionRingCount = 0

    // --- Live latency instrumentation. Tag "LiveLatency": ---
    //     serve start -> motion peak -> end detected -> GRU done -> feedback on screen.
    private var serveStartMs = 0L
    private var lastMotionPeakMs = 0L
    private var lastMotionPeak = 0f
    private var lastServeEndMs = 0L
    private var paddleDropped = 0L

    private val emaAlpha = 0.35f
    private var emaProbs: FloatArray? = null

    private val confHigh = 0.70f
    private val confMed = 0.50f

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        posePriority = intent.getBooleanExtra("research_pose_priority", true)
        revisedEnd = intent.getBooleanExtra("research_end_v2", false)
        Log.i(LATENCY_TAG, "policy pose_priority=$posePriority end_v2=$revisedEnd")
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        // Paddle detection (~600ms/frame on this phone) runs off the pose callback so
        // pose, the motion gate and end detection keep up with the camera. The queue
        // is bounded: when paddle falls behind it drops the oldest sample and keeps
        // the newest, since the frames around contact are the ones that matter.
        paddleExecutor = ThreadPoolExecutor(
            1, 1, 0L, TimeUnit.MILLISECONDS, LinkedBlockingQueue(6),
        ).apply {
            rejectedExecutionHandler = java.util.concurrent.RejectedExecutionHandler { r, exec ->
                exec.queue.poll()
                paddleDropped++
                exec.execute(r)
            }
        }

        try {
            gru = GruClassifier.load(this, ensemble = true)
        } catch (e: Throwable) {
            Log.e("MainActivity", "GRU load failed", e)
            binding.classLabel.text = "GRU LOAD ERR"
            binding.confLabel.text = e.message ?: "load failed"
        }

        try {
            knn = KnnClassifier.load(this)
        } catch (e: Throwable) {
            Log.e("MainActivity", "kNN load failed", e)
            // continue without hybrid
            if (mode == RunMode.HYBRID) mode = RunMode.ENSEMBLE
        }

        shiftDetector = ShiftDetector.load(this)
        // Loaded on the paddle worker, not the main thread: a GpuDelegate must be
        // created on the same thread that later invokes it (see LandingActivity,
        // which does the same for its YOLO runner). Until this completes, paddle
        // samples are silently skipped; pose, the gate and the GRU are unaffected.
        paddleExecutor.execute {
            val loaded = PaddleStream.load(applicationContext)
            paddleStream = loaded
            if (loaded == null) Log.e("MainActivity", "Paddle stream unavailable")
        }
        feedbackEngine = try {
            FeedbackEngine.fromJson(assets.open("feedback_rules.json").bufferedReader().use { it.readText() })
        } catch (e: Exception) {
            Log.e("MainActivity", "Feedback rules unavailable", e)
            null
        }

        updateModeLabel()

        binding.modeToggle.setOnClickListener {
            synchronized(this) {
            mode = when (mode) {
                RunMode.SINGLE -> RunMode.ENSEMBLE
                RunMode.ENSEMBLE -> if (knn != null) RunMode.HYBRID else RunMode.SINGLE
                RunMode.HYBRID -> RunMode.SINGLE
            }
            // Reload GRU for single vs ensemble
            try {
                gru?.close()
                gru = GruClassifier.load(this, ensemble = mode != RunMode.SINGLE)
            } catch (e: Throwable) {
                Toast.makeText(this, "GRU reload failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
            updateModeLabel()
            binding.resetButton.performClick()
            Toast.makeText(this, "Mode: $mode", Toast.LENGTH_SHORT).show()
            }
        }

        binding.resetButton.setOnClickListener {
            synchronized(this) {
            resultGeneration++
            serveWindow.clear()
            emaProbs = null
            gateActive = false
            lowMotionSinceMs = 0L
            refractoryUntilMs = 0L
            lastServeEndMs = 0L
            lastAngles = null
            motionRate.reset()
            motionRing.fill(0f)
            motionRingIdx = 0
            motionRingCount = 0
            shiftDetector?.clear()
            paddleStream?.clear()
            binding.framesLabel.text = "READY (waiting for serve)"
            binding.classLabel.text = "--"
            binding.confLabel.text = "Conf: --"
            binding.classLabel.setTextColor(Color.WHITE)
            binding.confLabel.setTextColor(Color.parseColor("#B0BEC5"))
            binding.shiftLabel.text = "WEIGHT SHIFT: --"
            binding.shiftLabel.setTextColor(Color.parseColor("#B0BEC5"))
            binding.paddleLabel.text = "PADDLE: --"
            binding.paddleLabel.setTextColor(Color.parseColor("#B0BEC5"))
            binding.feedbackLabel.text = "Record a serve to receive feedback."
            binding.probDriveBar.progress = 0
            binding.probLobBar.progress = 0
            binding.probTopspinBar.progress = 0
            binding.probDrive.text = "Drive   --%"
            binding.probLob.text = "Lob     --%"
            binding.probTopspin.text = "Topspin --%"
            }
        }

        binding.landingButton.setOnClickListener {
            startActivity(Intent(this, LandingActivity::class.java))
        }

        binding.replayButton.setOnClickListener {
            startActivity(Intent(this, ServeReplayActivity::class.java))
        }
        binding.latencyButton.setOnClickListener {
            startActivity(Intent(this, LatencyBenchmarkActivity::class.java))
        }

        posePipeline = PosePipeline(
            context = this,
            useGpu = intent.getBooleanExtra("research_pose_gpu", true),
            onPose = { pts, bmp, timestamp, real -> onPoseDetected(pts, bmp, timestamp, real) },
            onError = { msg ->
                runOnUiThread { binding.poseLabel.text = "POSE ERR: $msg" }
            }
        )

        val replayClip = intent.getStringExtra(EXTRA_REPLAY_CLIP)
        if (replayClip != null) {
            startClipReplay(replayClip, intent.getBooleanExtra(EXTRA_REPLAY_LOOP, false))
        } else if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }

        binding.framesLabel.text = "READY (waiting for serve)"
    }

    private fun updateModeLabel() {
        val mm = if (gru?.momentMatchingEnabled == true) "+MM " else ""
        val label = when (mode) {
            RunMode.SINGLE -> "MODE: SINGLE ${mm}(fold1)"
            RunMode.ENSEMBLE -> "MODE: ENS ${mm}(5 folds)"
            RunMode.HYBRID -> "MODE: HYBRID ${mm}(GRU+kNN5)"
        }
        binding.modeLabel.text = label
    }

    /**
     * Real-time demo without a player: a clip from files/latency_input drives the
     * live pose pipeline in place of the camera. Pose, the motion gate, the async
     * paddle worker, the GRU, feedback and the LiveLatency instrumentation are all
     * unchanged, so what the screen shows is the live path running.
     */
    private fun startClipReplay(clipName: String, loop: Boolean) {
        val file = File(filesDir, "latency_input/$clipName")
        if (!file.isFile) {
            binding.poseLabel.text = "REPLAY MISSING"
            binding.framesLabel.text = "Put $clipName in files/latency_input"
            Log.e("MainActivity", "Replay clip not found: ${file.absolutePath}")
            return
        }
        posePipeline?.start(this)
        binding.poseLabel.text = "Pose: replay"
        binding.framesLabel.text = if (loop) "REPLAY $clipName (looping)" else "REPLAY $clipName"
        clipReplay = ServeClipReplay(
            file = file,
            onFrame = { bitmap, real -> posePipeline?.analyze(bitmap, real) },
            onDone = { frames -> Log.i(LATENCY_TAG, "replay_done clip=$clipName frames_sent=$frames") },
            loop = loop,
        ).also { it.start() }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }
            val imageAnalysis = ImageAnalysis.Builder()
                // Pose is fed at analysis resolution, not preview resolution. 640x480
                // cuts pose cost ~3x versus an unpinned default and still resolves a
                // full-body serve; matches the landing stream's setting.
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { a ->
                    a.setAnalyzer(cameraExecutor) { ip -> analyzeFrame(ip) }
                }
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis
                )
                posePipeline?.start(this)
                binding.poseLabel.text = "Pose: starting..."
            } catch (e: Exception) {
                Log.e("MainActivity", "Camera bind failed", e)
                binding.poseLabel.text = "CAMERA ERR"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        imageProxy.close()
        posePipeline?.analyze(bitmap)
    }

    /**
     * Motion gate state machine:
     *  IDLE   - retain one second of real poses; no inference. Idle motion stats
     *           are re-measured here (EMA mean/dev) so the trigger threshold adapts
     *           to this device/person's pose noise. A refractory window right after
     *           a serve absorbs recovery swings. UI: READY (or DONE while refractory).
     *  ACTIVE - motion >= arm threshold started a serve: retain pre-roll and
     *           append real frames. Classify once at the end. Ends after lowHoldMs of wall-clock low motion OR the
     *           maxActiveMs force-timeout. Both are wall-clock so the end-detection
     *           delay is fixed regardless of how fast frames are processed. Final
     *           inference, then clear buffer + EMA but HOLD displayed labels.
     *
     * Paddle detection is NOT done here — it runs on paddleExecutor so this
     * callback stays fast enough to keep up with the camera.
     */
    @Synchronized
    private fun onPoseDetected(landmarks: Array<PointF>, bitmap: Bitmap?, timestampMs: Long, realFrame: Boolean) {
        val feats = JointAngles.fromLandmarks(landmarks) ?: return
        val displacement = computeMotion(feats)
        val motion = if (intent.getBooleanExtra("research_motion_rate", true))
            motionRate.update(displacement, timestampMs) else smoothMotion(displacement)
        val now = SystemClock.uptimeMillis()
        // Keep only fresh, real poses; the trigger sample is appended exactly once.
        if (realFrame && (gateActive || now >= refractoryUntilMs)) serveWindow.add(timestampMs, feats)

        if (!gateActive) {
            val inRefractory = now < refractoryUntilMs
            if (realFrame && !inRefractory && motion >= startArm()) {
                gateActive = true
                lowMotionSinceMs = 0L
                serveStartMs = now
                lastMotionPeakMs = now
                lastMotionPeak = motion
                paddleExecutor.queue.clear()
                paddleDropped = 0L
                paddleThrottled = 0L
                paddleBusy.set(false)
                resultGeneration++
                paddleFailed = false
                serveWindow.begin()
                emaProbs = null
                shiftDetector?.clear()
                paddleStream?.clear()
                shiftDetector?.let { s ->
                    landmarkSample(landmarks)?.let { (hx, bh) -> s.add(hx, bh) }
                }
                addPaddleSample(landmarks, bitmap)
                runOnUiThread {
                    binding.poseLabel.text = "Pose: OK"
                    binding.framesLabel.text = "Capturing serve (${serveWindow.size} poses)"
                    binding.feedbackLabel.text = "Analyzing this serve..."
                    binding.shiftLabel.text = "WEIGHT SHIFT: ANALYZING"
                    binding.paddleLabel.text = "PADDLE: ANALYZING"
                    binding.classLabel.text = "--"
                    binding.confLabel.text = "Conf: --"
                }
            } else {
                // absorb standing noise (and any recovery swing during refractory)
                // BEFORE comparing — the triggering frame itself never updates stats
                idleMean += 0.1f * (motion - idleMean)
                idleDev += 0.1f * (abs(motion - idleMean) - idleDev)
                runOnUiThread {
                    binding.poseLabel.text = "Pose: OK"
                    binding.framesLabel.text =
                        if (inRefractory) "DONE (result held)"
                        else "READY (waiting for serve)"
                }
            }
            return
        }

        // ACTIVE: current sample was already appended above.
        if (realFrame) shiftDetector?.let { s ->
            landmarkSample(landmarks)?.let { (hx, bh) -> s.add(hx, bh) }
        }
        if (motion > lastMotionPeak) {
            lastMotionPeak = motion
            lastMotionPeakMs = now
        }
        val quiet = if (revisedEnd) ServeEndPolicy.quiet(motion, endArm(), lastMotionPeak, now-lastMotionPeakMs) else motion < endArm()
        if (quiet) {
            if (lowMotionSinceMs == 0L) lowMotionSinceMs = now
            // already at resting level: keep adapting idle stats so a moved/
            // repositioned standing pose can't hold the gate open
            idleMean += 0.05f * (motion - idleMean)
            idleDev += 0.05f * (abs(motion - idleMean) - idleDev)
        } else {
            lowMotionSinceMs = 0L
        }
        // Post-contact frames carry no paddle evidence. Keep the wrist-speed
        // bookkeeping (contactFrame is derived from it) but stop paying for
        // detection on the tail that end detection is waiting out.
        if (realFrame) addPaddleSample(landmarks, if (lowMotionSinceMs == 0L) bitmap else null)

        val frames = serveWindow.size

        val ended = (lowMotionSinceMs != 0L && now - lowMotionSinceMs >= intent.getLongExtra("research_low_hold_ms", lowHoldMs)) ||
            now - serveStartMs >= maxActiveMs
        if (ended) {
            lastServeEndMs = now
            // Paddle runs off-thread; give already-queued detections a bounded
            // chance to land so the verdict still sees the contact window.
            val drainMs = awaitPaddleQuiescence(paddleQuiesceMs)
            val inferStart = SystemClock.uptimeMillis()
            // Serve ended: one final inference, then reset gate (labels stay visible)
            val finalPrediction = if (frames >= finalMinFrames) runInference(frames, finalResult = true) else null
            val inferDone = SystemClock.uptimeMillis()
            Log.i(LATENCY_TAG, "serve_ended " +
                "start=$serveStartMs peak=$lastMotionPeakMs end=$now frames=$frames " +
                "window_ms=${serveWindow.durationMs} window_policy=preroll_complete_real_v1 " +
                "start_to_peak=${lastMotionPeakMs - serveStartMs}ms " +
                "peak_to_end=${now - lastMotionPeakMs}ms " +
                "paddle_drain=${drainMs}ms paddle_dropped=$paddleDropped paddle_throttled=$paddleThrottled " +
                "final_infer=${inferDone - inferStart}ms " +
                "pred=${finalPrediction?.first} conf=${finalPrediction?.second}")
            updateShiftLabel()
            updatePaddleLabel()
            updateFeedback(finalPrediction)
            if (finalPrediction == null) runOnUiThread {
                binding.classLabel.text = "UNAVAILABLE"
                binding.confLabel.text = "Insufficient serve classification data"
                binding.probDriveBar.progress = 0
                binding.probLobBar.progress = 0
                binding.probTopspinBar.progress = 0
                binding.probDrive.text = "Drive   --%"
                binding.probLob.text = "Lob     --%"
                binding.probTopspin.text = "Topspin --%"
            }
            gateActive = false
            lowMotionSinceMs = 0L
            refractoryUntilMs = now + refractoryMs
            serveWindow.clear()
            paddleExecutor.queue.clear()
            emaProbs = null
            shiftDetector?.clear()
            paddleStream?.clear()
            runOnUiThread {
                binding.poseLabel.text = "Pose: OK"
                binding.framesLabel.text = "DONE (result held)"
            }
            return
        }

        runOnUiThread {
            binding.poseLabel.text = "Pose: OK"
            binding.framesLabel.text = if (frames < minFramesForInfer) {
                "WARMUP $frames / $minFramesForInfer"
            } else {
                "Capturing serve ($frames poses)"
            }
        }
        // Run the unchanged ensemble only at serve end; intermediate inference
        // competes with pose acquisition and changes no final feedback decision.
    }

    /** Start trigger: comfortably above measured idle noise. */
    private fun startArm(): Float =
        (motionThreshold.coerceAtLeast(idleMean + 3f * idleDev)).coerceAtMost(0.5f)

    /** End threshold: lower than start (hysteresis) but still above idle mean. */
    private fun endArm(): Float =
        (0.05f.coerceAtLeast(idleMean + 1.5f * idleDev)).coerceAtMost(0.5f)

    /** (hipCenterX, bodyHeight) from landmarks, or null if degenerate. */
    private fun landmarkSample(landmarks: Array<PointF>): Pair<Float, Float>? {
        if (landmarks.size < 33) return null
        val hipX = (landmarks[23].x + landmarks[24].x) / 2f
        val ankleY = (landmarks[27].y + landmarks[28].y) / 2f
        val bodyH = kotlin.math.abs(landmarks[0].y - ankleY)
        if (bodyH < 1e-6f) return null
        return hipX to bodyH
    }

    /** Deterministic shift verdict computed once at serve end, held on screen. */
    private fun updateShiftLabel() {
        val ok = shiftDetector?.passed()
        runOnUiThread {
            if (ok == null) {
                binding.shiftLabel.text = "WEIGHT SHIFT: UNAVAILABLE"
                binding.shiftLabel.setTextColor(Color.parseColor("#B0BEC5"))
            } else if (ok) {
                binding.shiftLabel.text = "WEIGHT SHIFT: \u2713"
                binding.shiftLabel.setTextColor(Color.parseColor("#4CAF50"))
            } else {
                binding.shiftLabel.text = "WEIGHT SHIFT: \u2717"
                binding.shiftLabel.setTextColor(Color.parseColor("#F44336"))
            }
        }
    }

    /** Paddle-orientation verdict computed once at serve end, held on screen. */
    private fun updatePaddleLabel() {
        val stream = paddleStream.takeUnless { paddleFailed }
        val state = stream?.let {
            FeedbackEngine.paddleState(it.computeForFeedback(), it.low, it.high, it.directionValidated)
        }
        val color = when {
            state == null -> Color.parseColor("#B0BEC5")
            state != FeedbackEngine.PaddleState.OPTIMAL -> Color.parseColor("#F44336")
            else -> Color.parseColor("#4CAF50")
        }
        val text = when (state) {
            null -> "PADDLE: UNAVAILABLE"
            FeedbackEngine.PaddleState.OPTIMAL -> "PADDLE: WITHIN RANGE"
            FeedbackEngine.PaddleState.OUTSIDE_BASELINE -> "PADDLE: OUTSIDE RANGE"
            FeedbackEngine.PaddleState.TOO_OPEN -> "PADDLE: TOO OPEN"
            FeedbackEngine.PaddleState.TOO_CLOSED -> "PADDLE: TOO CLOSED"
        }
        runOnUiThread {
            binding.paddleLabel.text = text
            binding.paddleLabel.setTextColor(color)
        }
    }

    /** Evaluate the same completed serve before any stream buffers are cleared. */
    private fun updateFeedback(prediction: Pair<String, Float>?) {
        val generation = resultGeneration
        val paddle = paddleStream.takeUnless { paddleFailed }?.let {
            FeedbackEngine.paddleState(it.computeForFeedback(), it.low, it.high, it.directionValidated)
        }
        val result = feedbackEngine?.evaluate(FeedbackEngine.Input(
            serveType = prediction?.first, gruConfidence = prediction?.second,
            shiftSufficient = shiftDetector?.passed(), paddle = paddle,
            independentMechanics = intent.getBooleanExtra("research_motion_rate", true) &&
                serveWindow.size >= finalMinFrames && serveWindow.durationMs >= 1500L,
            // LandingActivity analyzes a separate clip, not a synchronized secondary stream.
            placementActive = false,
        ))
        Log.i("Feedback", "version=${result?.version} status=${result?.status} rules=${result?.ruleIds} paddle=${paddle ?: "UNAVAILABLE"}")
        runOnUiThread {
            if (generation != resultGeneration) return@runOnUiThread
            binding.feedbackTitle.text = if (result?.validationStatus == "pending_coach_c")
                "COACHING FEEDBACK · RESEARCH PREVIEW" else "COACHING FEEDBACK"
            binding.feedbackLabel.text = result?.text ?: "Coaching feedback unavailable: rules could not be loaded."
            if (intent.hasExtra(EXTRA_REPLAY_CLIP)) {
                Log.i("FeedbackOutput", org.json.JSONObject().put("status", result?.status?.name)
                    .put("text", binding.feedbackLabel.text.toString()).toString())
            }
            // The number the 2s thesis target is judged on: end-detection -> text on screen.
            val shown = SystemClock.uptimeMillis()
            if (lastServeEndMs != 0L) {
                Log.i(LATENCY_TAG, "feedback_shown shown=$shown " +
                    "end_to_shown=${shown - lastServeEndMs}ms " +
                    "peak_to_shown=${shown - lastMotionPeakMs}ms " +
                    "serve_total=${shown - serveStartMs}ms")
            }
        }
    }

    /**
     * Hand a frame to the paddle worker. Detection costs ~600ms and must not block
     * the pose callback, so it is queued to a single-thread executor; the queue is
     * bounded and drops the oldest sample when paddle falls behind.
     *
     * bitmap == null keeps wrist-speed bookkeeping only (contactFrame is derived
     * from it) and stays on this thread — it is sub-microsecond.
     */
    private fun addPaddleSample(landmarks: Array<PointF>, bitmap: Bitmap?) {
        if (paddleFailed) return
        val stream = paddleStream ?: return
        val now = SystemClock.uptimeMillis()
        val permit = bitmap != null && (!posePriority ||
            (now-lastPaddleSubmitMs >= 350 && paddleExecutor.activeCount == 0 && !paddleBusy.get()))
        if (bitmap != null && !permit) paddleThrottled++
        val sample = stream.prepare(landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y, permit)
        if (sample == null || bitmap == null) return
        val generation = resultGeneration
        if (posePriority && !paddleBusy.compareAndSet(false, true)) { paddleDropped++; return }
        lastPaddleSubmitMs = now
        try {
            paddleExecutor.execute {
                try {
                    if (generation == resultGeneration) stream.detect(sample, bitmap)
                } catch (e: Exception) {
                    if (generation == resultGeneration) runOnUiThread { markPaddleFailed(e) }
                } finally { paddleBusy.set(false) }
            }
        } catch (e: Throwable) { paddleBusy.set(false); markPaddleFailed(e) }
    }

    private fun markPaddleFailed(e: Throwable) {
        paddleFailed = true
        paddleStream?.clear()
        Log.e("MainActivity", "Paddle unavailable for this serve; other streams continue", e)
    }

    /**
     * Wait for paddle work already queued on the single-thread worker, bounded so
     * a slow detector can never hold the feedback path open. Returns the actual
     * wait in ms; 0 if the barrier could not be scheduled.
     */
    private fun awaitPaddleQuiescence(timeoutMs: Long): Long {
        val started = SystemClock.uptimeMillis()
        val barrier = CountDownLatch(1)
        try {
            paddleExecutor.execute { barrier.countDown() }
        } catch (e: Throwable) {
            return 0L
        }
        try {
            if (!barrier.await(timeoutMs, TimeUnit.MILLISECONDS)) {
                Log.w(LATENCY_TAG, "paddle did not quiesce within ${timeoutMs}ms")
            }
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        return SystemClock.uptimeMillis() - started
    }

    /** Per-joint mean |wrapped delta angle| from previous frame, radians. */
    private fun computeMotion(feats: FloatArray): Float {
        val angles = FloatArray(5)
        for (k in 0 until 5) {
            angles[k] = atan2(feats[k * 2], feats[k * 2 + 1])
        }
        val prev = lastAngles
        lastAngles = angles
        if (prev == null) return 0f
        var sum = 0f
        for (k in 0 until 5) {
            var d = angles[k] - prev[k]
            while (d > PI) d -= 2f * PI.toFloat()
            while (d < -PI) d += 2f * PI.toFloat()
            sum += abs(d)
        }
        return sum / 5f
    }

    /** Rolling mean of the last 5 motion values (pose-noise smoothing). */
    private fun smoothMotion(inst: Float): Float {
        motionRing[motionRingIdx] = inst
        motionRingIdx = (motionRingIdx + 1) % motionRing.size
        if (motionRingCount < motionRing.size) motionRingCount++
        var acc = 0f
        for (i in 0 until motionRingCount) acc += motionRing[i]
        return acc / motionRingCount
    }

    private fun runInference(validFrames: Int, finalResult: Boolean = false): Pair<String, Float>? {
        val g = gru ?: return null
        val generation = resultGeneration
        val sampled = serveWindow.input()
        val window = sampled.tensor
        val inputFrames = sampled.validFrames
        try {
            val gruProbs = g.infer(window, validFrames = inputFrames)

            val knnProbs = when {
                mode == RunMode.HYBRID && knn != null -> {
                    val knnFeats = TimingFeatures.fromFlatWindow(
                        flattenWindow(window), inputFrames
                    )
                    knn!!.predictProba(knnFeats)
                }
                else -> null
            }
            val combined = HybridPrediction.combine(gruProbs, knnProbs)

            // EMA smoothing
            val prev = emaProbs
            val smoothed = if (finalResult || prev == null || prev.size != combined.size) {
                combined.copyOf()
            } else {
                FloatArray(combined.size) { i ->
                    emaAlpha * combined[i] + (1 - emaAlpha) * prev[i]
                }
            }
            emaProbs = smoothed

            val (label, conf) = g.predict(smoothed)
            val confColor = when {
                conf >= confHigh -> Color.parseColor("#4CAF50")
                conf >= confMed  -> Color.parseColor("#FFC107")
                else             -> Color.parseColor("#F44336")
            }

            runOnUiThread {
                if (generation != resultGeneration) return@runOnUiThread
                binding.classLabel.text = if (conf >= FeedbackEngine.MIN_CONFIDENCE) label.uppercase() else "UNCERTAIN"
                binding.classLabel.setTextColor(confColor)
                val finalSource = if (knnProbs != null) "Final hybrid" else "Final GRU"
                binding.confLabel.text = "${if (finalResult) finalSource else "Conf"}: ${"%.1f".format(conf * 100f)}%"
                binding.confLabel.setTextColor(confColor)
                binding.probDrive.text = "Drive   ${"%.0f".format(smoothed[0] * 100)}%"
                binding.probLob.text = "Lob     ${"%.0f".format(smoothed[1] * 100)}%"
                binding.probTopspin.text = "Topspin ${"%.0f".format(smoothed[2] * 100)}%"
                binding.probDriveBar.progress = (smoothed[0] * 100).toInt()
                binding.probLobBar.progress = (smoothed[1] * 100).toInt()
                binding.probTopspinBar.progress = (smoothed[2] * 100).toInt()
            }
            return g.predict(combined)
        } catch (e: Throwable) {
            Log.e("MainActivity", "Inference failed", e)
            runOnUiThread { binding.confLabel.text = "INFER ERR: ${e.message}" }
            return null
        }
    }

    /** Extract (validFrames * 10) flat slice for timing features. */
    private fun flattenWindow(window: Array<Array<FloatArray>>): FloatArray {
        // window[0] is (128, 10) — we need only valid prefix for timing, but
        // TimingFeatures uses validFrames param; pass full flat with zeros trailing.
        val seqLen = window[0].size
        val feat = window[0][0].size
        val out = FloatArray(seqLen * feat)
        for (t in 0 until seqLen) {
            for (f in 0 until feat) {
                out[t * feat + f] = window[0][t][f]
            }
        }
        return out
    }

    override fun onDestroy() {
        super.onDestroy()
        clipReplay?.stop()
        cameraExecutor.shutdown()
        paddleExecutor.shutdownNow()
        posePipeline?.stop()
        gru?.close()
        paddleStream?.close()
    }
}
