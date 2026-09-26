package com.thesis.pickleballserve

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2

/**
 * Live demo: camera -> BlazePose -> joint angles -> [GRU | GRU+kNN5 hybrid]
 *
 * Modes (cycle via Toggle):
 *   SINGLE    - one GRU fold (fastest)
 *   ENSEMBLE  - avg of 5 GRU folds + moment matching
 *   HYBRID    - avg( GRU ensemble , kNN5-on-timing-features )  <- best holdout
 */
class MainActivity : AppCompatActivity() {

    private enum class RunMode { SINGLE, ENSEMBLE, HYBRID }

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var posePipeline: PosePipeline? = null
    private var gru: GruClassifier? = null
    private var knn: KnnClassifier? = null
    private var shiftDetector: ShiftDetector? = null
    private val angleBuffer = AngleBuffer(seqLen = 128, featDim = 10)

    private var mode = RunMode.HYBRID  // default to hybrid (best holdout)
    private var inferCooldownMs = 0L
    private val minInferIntervalMs = 100L

    private val minFramesForInfer = 48

    // --- Motion gate: buffer only records during an active serve ---
    private var gateActive = false
    private var lowMotionStreak = 0
    private var activeFrameCount = 0
    private val motionThreshold = 0.07f   // floor for start arm (rad/frame mean)
    private val lowHoldFrames = 15        // ~0.5s below end threshold = serve ended
    private val finalMinFrames = 32       // min frames for end-of-serve final inference
    private val maxActiveFrames = 160     // force-end ~5.3s (buffer only holds 128 anyway)
    private val refractoryMs = 1200L      // ignore re-triggers right after a serve
    private var refractoryUntilMs = 0L
    // adaptive idle stats: threshold = idleMean + k*idleDev (re-measured while idle,
    // so standing pose noise can never keep the gate stuck open)
    private var idleMean = 0f
    private var idleDev = 0.02f
    private var lastAngles: FloatArray? = null
    private val motionRing = FloatArray(5)
    private var motionRingIdx = 0
    private var motionRingCount = 0

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
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

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

        updateModeLabel()

        binding.modeToggle.setOnClickListener {
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
            Toast.makeText(this, "Mode: $mode", Toast.LENGTH_SHORT).show()
        }

        binding.resetButton.setOnClickListener {
            angleBuffer.clear()
            emaProbs = null
            gateActive = false
            lowMotionStreak = 0
            activeFrameCount = 0
            refractoryUntilMs = 0L
            lastAngles = null
            motionRing.fill(0f)
            motionRingIdx = 0
            motionRingCount = 0
            shiftDetector?.clear()
            binding.framesLabel.text = "READY (waiting for serve)"
            binding.classLabel.text = "--"
            binding.confLabel.text = "Conf: --"
            binding.classLabel.setTextColor(Color.WHITE)
            binding.confLabel.setTextColor(Color.parseColor("#B0BEC5"))
            binding.shiftLabel.text = "WEIGHT SHIFT: --"
            binding.shiftLabel.setTextColor(Color.parseColor("#B0BEC5"))
            binding.probDriveBar.progress = 0
            binding.probLobBar.progress = 0
            binding.probTopspinBar.progress = 0
        }

        posePipeline = PosePipeline(
            context = this,
            onPose = { pts -> onPoseDetected(pts) },
            onError = { msg ->
                runOnUiThread { binding.poseLabel.text = "POSE ERR: $msg" }
            }
        )

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
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

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }
            val imageAnalysis = ImageAnalysis.Builder()
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
     *  IDLE   - standing still: no buffer appends, no inference. Idle motion stats
     *           are re-measured here (EMA mean/dev) so the trigger threshold adapts
     *           to this device/person's pose noise. A refractory window right after
     *           a serve absorbs recovery swings. UI: READY (or DONE while refractory).
     *  ACTIVE - motion >= arm threshold started a serve: append frames, infer as
     *           before. Ends when motion < end threshold for ~0.5s OR after
     *           maxActiveFrames force-timeout. Final inference, then clear buffer +
     *           EMA but HOLD displayed labels (result stays on screen).
     */
    private fun onPoseDetected(landmarks: Array<PointF>) {
        val feats = JointAngles.fromLandmarks(landmarks) ?: return
        val motion = smoothMotion(computeMotion(feats))
        val now = SystemClock.uptimeMillis()

        if (!gateActive) {
            val inRefractory = now < refractoryUntilMs
            if (!inRefractory && motion >= startArm()) {
                gateActive = true
                lowMotionStreak = 0
                activeFrameCount = 1
                angleBuffer.clear()
                emaProbs = null
                shiftDetector?.clear()
                angleBuffer.add(feats)
                shiftDetector?.let { s ->
                    landmarkSample(landmarks)?.let { (hx, bh) -> s.add(hx, bh) }
                }
                runOnUiThread {
                    binding.poseLabel.text = "Pose: OK"
                    binding.framesLabel.text = "WARMUP 1 / $minFramesForInfer"
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

        // ACTIVE: append serve frame
        angleBuffer.add(feats)
        shiftDetector?.let { s ->
            landmarkSample(landmarks)?.let { (hx, bh) -> s.add(hx, bh) }
        }
        activeFrameCount++
        if (motion < endArm()) {
            lowMotionStreak++
            // already at resting level: keep adapting idle stats so a moved/
            // repositioned standing pose can't hold the gate open
            idleMean += 0.05f * (motion - idleMean)
            idleDev += 0.05f * (abs(motion - idleMean) - idleDev)
        } else {
            lowMotionStreak = 0
        }
        val frames = angleBuffer.validFrames()

        val ended = lowMotionStreak >= lowHoldFrames || activeFrameCount >= maxActiveFrames
        if (ended) {
            // Serve ended: one final inference, then reset gate (labels stay visible)
            if (frames >= finalMinFrames) runInference(frames)
            updateShiftLabel()
            gateActive = false
            lowMotionStreak = 0
            activeFrameCount = 0
            refractoryUntilMs = now + refractoryMs
            angleBuffer.clear()
            emaProbs = null
            shiftDetector?.clear()
            runOnUiThread {
                binding.poseLabel.text = "Pose: OK"
                binding.framesLabel.text = "DONE (result held)"
            }
            return
        }

        val ready = frames >= minFramesForInfer && (now - inferCooldownMs >= minInferIntervalMs)
        runOnUiThread {
            binding.poseLabel.text = "Pose: OK"
            binding.framesLabel.text = if (frames < minFramesForInfer) {
                "WARMUP $frames / $minFramesForInfer"
            } else {
                "Frames: $frames / 128"
            }
        }
        if (ready) {
            inferCooldownMs = now
            runInference(frames)
        }
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
        val detector = shiftDetector ?: return  // config missing → label stays --
        val ok = detector.passed()
        runOnUiThread {
            if (ok == null) {
                binding.shiftLabel.text = "WEIGHT SHIFT: --"
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

    private fun runInference(validFrames: Int) {
        val g = gru ?: return
        val window = angleBuffer.toInputTensor()
        try {
            val gruProbs = g.infer(window, validFrames = validFrames)

            val combined = when {
                mode == RunMode.HYBRID && knn != null -> {
                    val knnFeats = TimingFeatures.fromFlatWindow(
                        flattenWindow(window), validFrames
                    )
                    val knnProbs = knn!!.predictProba(knnFeats)
                    // simple average
                    FloatArray(3) { i -> 0.5f * gruProbs[i] + 0.5f * knnProbs[i] }
                }
                else -> gruProbs
            }

            // EMA smoothing
            val prev = emaProbs
            val smoothed = if (prev == null || prev.size != combined.size) {
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
                binding.classLabel.text = label.uppercase()
                binding.classLabel.setTextColor(confColor)
                binding.confLabel.text = "Conf: ${"%.1f".format(conf * 100f)}%"
                binding.confLabel.setTextColor(confColor)
                binding.probDrive.text = "Drive   ${"%.0f".format(smoothed[0] * 100)}%"
                binding.probLob.text = "Lob     ${"%.0f".format(smoothed[1] * 100)}%"
                binding.probTopspin.text = "Topspin ${"%.0f".format(smoothed[2] * 100)}%"
                binding.probDriveBar.progress = (smoothed[0] * 100).toInt()
                binding.probLobBar.progress = (smoothed[1] * 100).toInt()
                binding.probTopspinBar.progress = (smoothed[2] * 100).toInt()
            }
        } catch (e: Throwable) {
            Log.e("MainActivity", "Inference failed", e)
            runOnUiThread { binding.confLabel.text = "INFER ERR: ${e.message}" }
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
        cameraExecutor.shutdown()
        posePipeline?.stop()
        gru?.close()
    }
}
