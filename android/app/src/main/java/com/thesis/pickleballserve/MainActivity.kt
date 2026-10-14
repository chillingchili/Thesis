package com.thesis.pickleballserve

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.provider.Settings
import android.view.View
import android.widget.LinearLayout
import android.widget.PopupMenu
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.RejectedExecutionException
import android.graphics.PointF
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.addCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.thesis.pickleballserve.databinding.ActivityMainBinding
import com.thesis.pickleballserve.landing.LandingActivity
import com.thesis.pickleballserve.landing.LandingResult
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
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
    private val sessionEpoch = SessionEpoch()
    private val callbackGeneration = ThreadLocal<Long>()
    private val resultGeneration: Long get() = sessionEpoch.current
    private val inputEpoch = AtomicLong()
    private val poseQueued = AtomicBoolean(false)
    private lateinit var analysisExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraUseCases: Array<androidx.camera.core.UseCase> = emptyArray()
    @Volatile private var foreground = false
    @Volatile private var destroyed = false
    @Volatile private var modelsReady = false
    private var inputRunning = false
    private var permissionRequested = false
    private var lastPoseSeenMs = 0L
    private var lastStatusMs = 0L
    private var posePriority = true
    private var revisedEnd = false
    private var lastPaddleSubmitMs = 0L
    private var paddleThrottled = 0L
    private val paddleBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    private val serveWindow = ServeWindow(preRollMs = 1000L)

    private var mode = RunMode.ENSEMBLE  // Original GRU selected before Coach C Session 2 release.

    /** Back camera is the study setup; front camera is offered as an input selection. */
    private var cameraFacing = CameraSelector.LENS_FACING_BACK

    /** Fullscreen hides the result row so the shot can be framed on the tripod. */
    private var fullscreen = false
    // GPU pose attempt for the live path (MediaPipe delegate with CPU fallback
    // inside PosePipeline). Default true preserves current behavior. The recorded
    // benchmark stays CPU-only as the stable reference.
    private var poseGpu = true
    /** Spoken results keep the player on the court; same strings as on screen. */
    private var speakResults = true
    private var tts: TextToSpeech? = null

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

    // --- Dual-phone landing handoff (offline QR, no network). ---
    // Phone B shares its bounce zone as a QR code; a scan either amends the
    // just-completed serve (snapshot still fresh) or waits as pending for the
    // next completed serve. The freshness bound keeps stale zones from
    // attaching to unrelated serves.
    private data class LastServe(
        val serveType: String?,
        val confidence: Float?,
        val shift: Boolean?,
        val paddle: FeedbackEngine.PaddleState?,
        val independentMechanics: Boolean,
        val recordNumber: Int,
        val completedMs: Long,
    )
    @Volatile private var lastServe: LastServe? = null
    @Volatile private var pendingLanding: Pair<LandingResult, Long>? = null
    @Volatile private var landingRefreshRecord: Int? = null
    private val landingFreshMs = 120_000L

    private val emaAlpha = 0.35f
    private var emaProbs: FloatArray? = null

    private val confHigh = 0.70f
    private val confMed = 0.50f

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startInput() else showPermissionRequired()
        }

    private val scanLanding =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
            if (destroyed) return@registerForActivityResult
            if (res.resultCode != RESULT_OK) return@registerForActivityResult
            val payload = res.data?.getStringExtra(ScanLandingActivity.EXTRA_LANDING)
                ?: return@registerForActivityResult
            val parsed = LandingResult.parse(payload)
            if (parsed == null) {
                Toast.makeText(this, getString(R.string.scan_invalid), Toast.LENGTH_LONG).show()
                return@registerForActivityResult
            }
            onLandingScanned(parsed)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        permissionRequested = savedInstanceState?.getBoolean("permissionRequested") ?: false
        posePriority = intent.getBooleanExtra("research_pose_priority", true)
        revisedEnd = intent.getBooleanExtra("research_end_v2", false)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        WindowCompat.getInsetsController(window, binding.root).isAppearanceLightNavigationBars = true
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newSingleThreadExecutor()
        paddleExecutor = ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, LinkedBlockingQueue(6))
        binding.menuButton.setOnClickListener { showMenu() }
        binding.previewView.setOnClickListener {
            setFullscreen(binding.resultsScroll.visibility == View.VISIBLE)
        }
        onBackPressedDispatcher.addCallback(this) {
            if (fullscreen) setFullscreen(false)
            else { isEnabled = false; onBackPressedDispatcher.onBackPressed() }
        }
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val availability = tts?.setLanguage(Locale.ENGLISH)
                if (availability == TextToSpeech.LANG_MISSING_DATA ||
                    availability == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w("MainActivity", "Spoken results unavailable: no English voice data")
                }
            } else Log.w("MainActivity", "Spoken results unavailable: TTS init failed")
        }
        binding.serveInfoButton.setOnClickListener {
            showInfo(R.string.serve_definitions_title, R.string.serve_definitions_body)
        }
        binding.hipInfoButton.setOnClickListener {
            showInfo(R.string.hip_info_title, R.string.hip_info_body)
        }
        binding.paddleInfoButton.setOnClickListener {
            showInfo(R.string.paddle_info_title, R.string.paddle_info_body)
        }
        paddleExecutor.execute {
            paddleStream = PaddleStream.load(applicationContext)
            if (paddleStream == null) Log.e("MainActivity", "Paddle stream unavailable")
        }
        analysisExecutor.execute {
            try {
                gru = GruClassifier.load(applicationContext, ensemble = true)
                knn = runCatching { KnnClassifier.load(applicationContext) }.getOrNull()
                shiftDetector = ShiftDetector.load(applicationContext)
                feedbackEngine = runCatching {
                    FeedbackEngine.fromJson(assets.open("feedback_rules.json").bufferedReader().use { it.readText() })
                }.getOrNull()
                modelsReady = true
                runOnUiThread {
                    if (!destroyed) {
                        updateModeLabel()
                        if (foreground) startInput()
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Model initialization failed", e)
                runOnUiThread { if (!destroyed) showFailure("Analysis unavailable", "The analysis models could not load. Retry to prepare them again.") }
            } catch (e: LinkageError) {
                Log.e("MainActivity", "Model runtime unavailable", e)
                runOnUiThread { if (!destroyed) showFailure("Analysis unavailable", "The model runtime could not start on this device.") }
            }
        }
    }

    private fun postUi(generation: Long = callbackGeneration.get() ?: resultGeneration, block: () -> Unit) {
        runOnUiThread { if (!destroyed && foreground && sessionEpoch.accepts(generation)) block() }
    }

    private fun setStatus(title: String, detail: String, busy: Boolean = false) {
        binding.statusLine.text = if (detail.isBlank()) title else "$title · $detail"
        binding.statusProgress.visibility = if (busy) View.VISIBLE else View.GONE
    }

    private fun clearCards(analyzing: Boolean = false) {
        val text = if (analyzing) getString(R.string.analyzing) else getString(R.string.waiting)
        binding.classLabel.text = text
        binding.confLabel.setText(R.string.confidence_empty)
        binding.shiftLabel.text = text
        binding.paddleLabel.text = text
        binding.feedbackTitle.setText(R.string.coaching_title)
        binding.feedbackLabel.text = if (analyzing) "Checking the measurements from this serve…" else getString(R.string.coaching_empty)
        listOf(binding.classLabel, binding.confLabel, binding.shiftLabel, binding.paddleLabel).forEach {
            it.setTextColor(ContextCompat.getColor(this, R.color.ink))
        }
        binding.probDriveBar.progress = 0; binding.probLobBar.progress = 0; binding.probTopspinBar.progress = 0
        binding.probDrive.text = "Drive —"; binding.probLob.text = "Lob —"; binding.probTopspin.text = "Topspin —"
    }

    /** Called only on the analysis worker; UI invalidation happens immediately. */
    private fun resetAnalysis() {
        serveWindow.clear(); emaProbs = null; gateActive = false; lowMotionSinceMs = 0L
        refractoryUntilMs = 0L; lastServeEndMs = 0L; lastAngles = null; motionRate.reset()
        motionRing.fill(0f); motionRingIdx = 0; motionRingCount = 0
        idleMean = 0f; idleDev = .02f; lastPoseSeenMs = 0L
        shiftDetector?.clear(); paddleStream?.clear()
    }

    /** Fullscreen framing for tripod setup; Back leaves it before leaving. */
    private fun setFullscreen(on: Boolean) {
        fullscreen = on
        binding.resultsScroll.visibility = if (on) View.GONE else View.VISIBLE
    }

    private fun showMenu() {
        val popup = PopupMenu(this, binding.menuButton)
        popup.menuInflater.inflate(R.menu.main_menu, popup.menu)
        popup.menu.findItem(R.id.action_fullscreen).isChecked = fullscreen
        popup.menu.findItem(R.id.action_speak).isChecked = speakResults
        popup.menu.findItem(R.id.action_gpu_pose).isChecked = poseGpu
        popup.menu.findItem(R.id.action_session).title =
            getString(R.string.session_serves) + " (${SessionStore.size()})"
        popup.menu.findItem(
            if (cameraFacing == CameraSelector.LENS_FACING_FRONT) R.id.action_camera_front
            else R.id.action_camera_back
        ).isChecked = true
        popup.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_setup -> {
                    MaterialAlertDialogBuilder(this).setTitle(R.string.setup_title).setMessage(R.string.setup_body)
                        .setPositiveButton(R.string.understood, null).show()
                    true
                }
                R.id.action_reset -> {
                    sessionEpoch.invalidate()
                    pendingLanding = null
                    landingRefreshRecord = null
                    clearCards()
                    analysisExecutor.execute { resetAnalysis() }
                    setStatus(getString(R.string.ready_title), getString(R.string.camera_hint))
                    true
                }
                R.id.action_fullscreen -> {
                    setFullscreen(!fullscreen)
                    item.isChecked = fullscreen
                    true
                }
                R.id.action_session -> {
                    startActivity(Intent(this, SessionActivity::class.java)); true
                }
                R.id.action_speak -> {
                    speakResults = !speakResults
                    item.isChecked = speakResults
                    if (!speakResults) tts?.stop()
                    true
                }
                R.id.action_gpu_pose -> {
                    poseGpu = !poseGpu
                    item.isChecked = poseGpu
                    // Pipeline reads the flag at creation; restart input so it applies now.
                    if (foreground && !destroyed && modelsReady) {
                        stopInput()
                        startInput()
                    }
                    Toast.makeText(this, getString(R.string.gpu_pose_note, if (poseGpu) "on" else "off"), Toast.LENGTH_LONG).show()
                    true
                }
                R.id.action_camera_back -> {
                    item.isChecked = true
                    if (cameraFacing != CameraSelector.LENS_FACING_BACK) switchCamera(CameraSelector.LENS_FACING_BACK)
                    true
                }
                R.id.action_camera_front -> {
                    item.isChecked = true
                    if (cameraFacing != CameraSelector.LENS_FACING_FRONT) switchCamera(CameraSelector.LENS_FACING_FRONT)
                    true
                }
                R.id.action_probabilities -> {
                    binding.probabilityPanel.visibility =
                        if (binding.probabilityPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
                    true
                }
                R.id.action_compare -> {
                    startActivity(Intent(this, ServeReplayActivity::class.java)); true
                }
                R.id.action_landing -> {
                    startActivity(Intent(this, LandingActivity::class.java)); true
                }
                R.id.action_scan_landing -> {
                    scanLanding.launch(Intent(this, ScanLandingActivity::class.java)); true
                }
                R.id.action_latency -> {
                    startActivity(Intent(this, LatencyBenchmarkActivity::class.java)); true
                }
                R.id.action_showcase -> {
                    startShowcase(); true
                }
                R.id.action_change_model -> { changeMode(); true }
                R.id.action_about -> {
                    MaterialAlertDialogBuilder(this).setTitle(R.string.about_tool).setMessage(R.string.about_body)
                        .setPositiveButton(R.string.understood, null).show()
                    true
                }
                else -> false
            }
        }
        popup.show()
    }

    private fun showInfo(titleRes: Int, bodyRes: Int) {
        MaterialAlertDialogBuilder(this).setTitle(titleRes).setMessage(bodyRes)
            .setPositiveButton(R.string.understood, null).show()
    }

    /** Switch live input cameras without losing the session; research replay keeps its own source. */
    private fun switchCamera(facing: Int) {
        if (showcaseIndex >= 0) {
            Toast.makeText(this, "Showcase running — it finishes on its own.", Toast.LENGTH_SHORT).show()
            return
        }
        if (intent.getStringExtra(EXTRA_REPLAY_CLIP) != null) {
            cameraFacing = facing
            Toast.makeText(this, R.string.camera_switch_note, Toast.LENGTH_LONG).show()
            return
        }
        cameraFacing = facing
        val provider = cameraProvider
        if (!inputRunning || provider == null || destroyed || !foreground) {
            Toast.makeText(this, R.string.camera_switch_note, Toast.LENGTH_SHORT).show()
            return
        }
        try {
            provider.unbind(*cameraUseCases)
            val epoch = inputEpoch.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder().setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
            analysis.setAnalyzer(cameraExecutor) { image ->
                if (foreground && !destroyed && epoch == inputEpoch.get()) analyzeFrame(image) else image.close()
            }
            provider.bindToLifecycle(this, cameraSelector(), preview, analysis)
            cameraUseCases = arrayOf(preview, analysis)
        } catch (e: Exception) {
            Log.e("MainActivity", "Camera switch failed", e)
            showFailure("Camera unavailable", "Close other camera apps, then retry.")
            return
        }
        sessionEpoch.invalidate()
        pendingLanding = null
        landingRefreshRecord = null
        analysisExecutor.execute { resetAnalysis() }
        clearCards()
        setStatus(getString(R.string.ready_title), getString(R.string.camera_switch_note))
    }

    private fun cameraSelector() = CameraSelector.Builder().requireLensFacing(cameraFacing).build()

    /** Reads the same strings shown on screen, so the player can stay on court. */
    private fun speak(record: ServeRecord) {
        if (!speakResults) return
        val engine = tts ?: return
        val conf = record.confidence?.let { "(${(it * 100).toInt()} percent). " } ?: ""
        val landing = record.landing?.let { " Landing $it." } ?: ""
        engine.speak("Serve ${record.number}. ${record.serveType}. $conf${record.feedback}$landing",
            TextToSpeech.QUEUE_FLUSH, null, "serve-${record.number}")
    }

    /** Short landing-only utterance so the player hears a late-attached zone on court. */
    private fun speakLanding(record: ServeRecord) {
        if (!speakResults) return
        val engine = tts ?: return
        val zone = record.landing ?: return
        engine.speak("Serve ${record.number}. Landing $zone.",
            TextToSpeech.QUEUE_ADD, null, "landing-${record.number}")
    }

    private fun changeMode() {
        if (!modelsReady) return
        sessionEpoch.invalidate()
        clearCards()
        setStatus("Changing comparison model…", "Your next attempt will use the selected model.", true)
        analysisExecutor.execute {
            resetAnalysis()
            val next = when (mode) {
                RunMode.SINGLE -> RunMode.ENSEMBLE
                RunMode.ENSEMBLE -> if (knn != null) RunMode.HYBRID else RunMode.SINGLE
                RunMode.HYBRID -> RunMode.SINGLE
            }
            try {
                val replacement = GruClassifier.load(applicationContext, ensemble = next != RunMode.SINGLE)
                gru?.close(); gru = replacement; mode = next
                postUi { updateModeLabel(); binding.probabilityPanel.visibility = View.VISIBLE
                    setStatus("Comparison model ready", "${mode.name.lowercase()} selected. Reset does not change the model.") }
            } catch (e: Exception) {
                postUi { setStatus("Model unchanged", "Could not load the comparison model. Your previous model remains active.") }
            }
        }
    }

    private fun showPermissionRequired() {
        inputRunning = false
        binding.statusState.setText(R.string.camera_permission_title)
        setStatus(getString(R.string.camera_permission_title), getString(R.string.camera_permission_note))
        binding.recoveryButton.visibility = View.VISIBLE
        val settings = permissionRequested && !shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)
        binding.recoveryButton.setText(if (settings) R.string.open_settings else R.string.allow_camera)
        binding.recoveryButton.setOnClickListener {
            if (settings) startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName")))
            else { permissionRequested = true; requestPermission.launch(Manifest.permission.CAMERA) }
        }
    }

    private fun showFailure(title: String, detail: String) {
        if (destroyed) return
        stopInput()
        clearCards()
        binding.statusState.text = title
        setStatus(title, detail)
        binding.recoveryButton.visibility = View.VISIBLE
        binding.recoveryButton.setText(R.string.retry)
        binding.recoveryButton.setOnClickListener { recreate() }
    }

    private fun startInput() {
        if (!foreground || destroyed || !modelsReady || inputRunning) return
        val replay = intent.getStringExtra(EXTRA_REPLAY_CLIP)
        if (replay == null && ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            showPermissionRequired(); return
        }
        inputRunning = true
        binding.root.keepScreenOn = true
        binding.recoveryButton.visibility = View.GONE
        setStatus(getString(R.string.initializing), "Starting the camera and pose tracker…", true)
        if (replay != null) startClipReplay(replay, intent.getBooleanExtra(EXTRA_REPLAY_LOOP, false)) else startCamera()
    }

    private fun createPose(epoch: Long): PosePipeline = PosePipeline(
        context = applicationContext, useGpu = intent.getBooleanExtra("research_pose_gpu", poseGpu),
        onPose = { pts, bmp, timestamp, real ->
            if (foreground && !destroyed && epoch == inputEpoch.get() && poseQueued.compareAndSet(false, true)) {
                val points = Array(pts.size) { PointF(pts[it].x, pts[it].y) }
                val generation = resultGeneration
                try { analysisExecutor.execute {
                    try { if (foreground && epoch == inputEpoch.get() && sessionEpoch.accepts(generation)) {
                        callbackGeneration.set(generation)
                        lastPoseSeenMs = SystemClock.uptimeMillis()
                        onPoseDetected(points, bmp, timestamp, real)
                    } } finally { callbackGeneration.remove(); poseQueued.set(false) }
                } } catch (_: RejectedExecutionException) { poseQueued.set(false) }
            }
        },
        onMissing = {
            if (foreground && epoch == inputEpoch.get() && poseQueued.compareAndSet(false, true)) {
                val generation = resultGeneration
                try { analysisExecutor.execute {
                    try { if (foreground && epoch == inputEpoch.get() && sessionEpoch.accepts(generation)) {
                        callbackGeneration.set(generation); onPoseMissing()
                    } }
                    finally { callbackGeneration.remove(); poseQueued.set(false) }
                } } catch (_: RejectedExecutionException) { poseQueued.set(false) }
            }
        },
        onError = { msg -> Log.e("MainActivity", msg)
            postUi { if (epoch == inputEpoch.get())
                showFailure("Pose tracking unavailable", "Retry analysis. Ensure this device supports the bundled models.") }
        }
    )

    private fun onPoseMissing() {
        if (!inputRunning) return
        if (fullresPaddleArmed()) showcaseSrcQueue.poll()
        val now = SystemClock.uptimeMillis()
        if (gateActive && lastPoseSeenMs != 0L && now - lastPoseSeenMs >= 1500L) {
            resetAnalysis()
            postUi { clearCards(); binding.classLabel.text = "Incomplete capture"
                binding.feedbackLabel.text = "The player left view before analysis completed. Keep your full body visible and try another serve."
                setStatus("Capture interrupted", getString(R.string.tracking_lost_note)) }
        }
        if (now - lastStatusMs >= 500L) {
            lastStatusMs = now
            postUi { binding.statusState.setText(R.string.tracking_lost) }
        }
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
    private fun startClipReplay(
        clipName: String,
        loop: Boolean,
        showVideo: Boolean = false,
        statusLabel: String? = null,
        onFinished: (() -> Unit)? = null,
    ) {
        val base = File(filesDir, "latency_input").canonicalFile
        val file = File(base, clipName).canonicalFile
        if (file.parentFile != base || !file.isFile) {
            showFailure("Replay unavailable", "The selected research clip is not available on this phone."); return
        }
        val epoch = inputEpoch.get()
        cameraExecutor.execute {
            if (!foreground || epoch != inputEpoch.get()) return@execute
            runCatching { posePipeline?.stop() }
            posePipeline = null
            val pipeline = createPose(epoch).also { posePipeline = it }
            if (!pipeline.start(applicationContext)) return@execute
            postUi { binding.statusState.text = statusLabel ?: "Recorded live-path replay"
                setStatus(if (statusLabel != null) "Showcasing $clipName" else "Preparing recorded serve", "Research replay uses the live analysis path.") }
            clipReplay = ServeClipReplay(file = file,
                onFrame = { bitmap, real, srcIndex ->
                    if (showVideo) runOnUiThread { if (!destroyed) binding.replayView.setImageBitmap(bitmap) }
                    if (foreground && epoch == inputEpoch.get()) runCatching { cameraExecutor.execute {
                        if (foreground && epoch == inputEpoch.get()) {
                            if ((showVideo || fullresPaddleArmed()) && real) showcaseSrcQueue.add(srcIndex)
                            posePipeline?.analyze(bitmap, real)
                        }
                    } }
                },
                onDone = { frames ->
                    Log.i(LATENCY_TAG, "replay_done clip=$clipName frames_sent=$frames")
                    onFinished?.invoke()
                }, loop = loop
            ).also { if (foreground && epoch == inputEpoch.get()) it.start() }
        }
    }

    // --- Showcase demo: recorded serves through the live path, hands-free. ---
    // One clip per displayable output (corrective, partial, low-confidence), so a
    // screen recording shows the normal app analyzing and inferring without a player.
    // Uses the same motion gate, GRU, paddle worker and feedback as the camera.
    // C002 is the only live-path clip with a measured paddle angle plus substantive
    // feedback (OPTIMAL in the GPU audit); C005 partial is stable on both backends;
    // C023 has twice the frames of C014 and stays substantive under both backends
    // (CPU low-confidence, GPU corrective-uncertain). Live outputs can still vary
    // slightly run to run with motion-gate timing.
    private var showcaseQueue: List<String> = emptyList()
    private var showcaseIndex = -1
    private val showcaseGapMs = 2500L
    // Showcase / research paddle config: relaxed spacing (150ms vs 350ms) and an
    // optional full-res contact-window second pass (same detector/rules, bigger
    // pixels). Enabled by the showcase demo or explicitly per replay via the
    // research_relaxed_paddle / research_fullres_paddle intent extras, so the
    // exact demo configuration stays evaluable on all 30 clips.
    private fun fullresPaddleArmed() = showcaseIndex >= 0 ||
        intent.getBooleanExtra("research_fullres_paddle", false)
    private var showcaseClipFile: File? = null
    private val showcaseSrcQueue = java.util.concurrent.ConcurrentLinkedQueue<Int>()
    private val showcaseUsableSrc = ArrayList<Int>()
    private val showcaseUsableWrist = ArrayList<PointF>()

    private fun startShowcase() {
        if (showcaseIndex >= 0) {
            Toast.makeText(this, "Showcase already running.", Toast.LENGTH_SHORT).show()
            return
        }
        if (!modelsReady) {
            Toast.makeText(this, "Models still loading — try the showcase again in a moment.", Toast.LENGTH_SHORT).show()
            return
        }
        val dir = File(filesDir, "latency_input")
        val clips = listOf("C002.mp4", "C005.mp4", "C023.mp4").filter { File(dir, it).isFile }
        if (clips.size < 3) {
            Toast.makeText(this, "Load C002, C005 and C023 first: Menu → Run recorded-file latency test → Load clips from phone storage.", Toast.LENGTH_LONG).show()
            return
        }
        stopInput()
        showcaseQueue = clips
        showcaseIndex = 0
        inputRunning = true
        binding.root.keepScreenOn = true
        clearCards()
        playShowcaseClip()
    }

    private fun playShowcaseClip() {
        if (!foreground) { finishShowcase(); return }
        val idx = showcaseIndex
        if (idx < 0 || idx >= showcaseQueue.size) { finishShowcase(); return }
        val clipName = showcaseQueue[idx]
        binding.replayView.visibility = View.VISIBLE
        showcaseClipFile = File(File(filesDir, "latency_input"), clipName)
        showcaseSrcQueue.clear()
        synchronized(showcaseUsableSrc) {
            showcaseUsableSrc.clear()
            showcaseUsableWrist.clear()
        }
        startClipReplay(
            clipName, loop = false, showVideo = true,
            statusLabel = "Showcase ${idx + 1}/${showcaseQueue.size} · $clipName",
            onFinished = {
                // Keep the finished serve's result on screen briefly for the
                // recording, then continue hands-free.
                binding.root.postDelayed({
                    if (destroyed || !foreground || showcaseIndex < 0) return@postDelayed
                    analysisExecutor.execute { resetAnalysis() }
                    showcaseIndex++
                    if (showcaseIndex < showcaseQueue.size) playShowcaseClip()
                    else finishShowcase()
                }, showcaseGapMs)
            },
        )
    }

    private fun finishShowcase() {
        showcaseIndex = -1
        showcaseQueue = emptyList()
        showcaseClipFile = null
        showcaseSrcQueue.clear()
        synchronized(showcaseUsableSrc) {
            showcaseUsableSrc.clear()
            showcaseUsableWrist.clear()
        }
        runOnUiThread {
            if (destroyed) return@runOnUiThread
            binding.replayView.setImageDrawable(null)
            binding.replayView.visibility = View.GONE
        }
        Toast.makeText(this, "Showcase complete — live camera resumed.", Toast.LENGTH_SHORT).show()
        stopInput()
        if (foreground && modelsReady) startInput()
    }

    private fun startCamera() {
        val epoch = inputEpoch.get()
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (!foreground || destroyed || epoch != inputEpoch.get()) return@addListener
            try {
                val provider = future.get().also { cameraProvider = it }
                val preview = Preview.Builder().build().also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }
                val analysis = ImageAnalysis.Builder().setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
                cameraExecutor.execute {
                    if (!foreground || epoch != inputEpoch.get()) return@execute
                    val pipeline = createPose(epoch).also { posePipeline = it }
                    if (!pipeline.start(applicationContext)) return@execute
                    postUi { binding.statusState.text = "Live camera · side view"
                        setStatus(getString(R.string.ready_title), getString(R.string.camera_hint)) }
                }
                analysis.setAnalyzer(cameraExecutor) { image ->
                    if (foreground && epoch == inputEpoch.get()) analyzeFrame(image) else image.close()
                }
                provider.unbind(*cameraUseCases)
                provider.bindToLifecycle(this, cameraSelector(), preview, analysis)
                cameraUseCases = arrayOf(preview, analysis)
            } catch (e: Exception) {
                Log.e("MainActivity", "Camera start failed", e)
                showFailure("Camera unavailable", "Close other camera apps, then retry.")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        try {
            val raw = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees
            val bitmap = if (rotation == 0) raw else Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height,
                Matrix().apply { postRotate(rotation.toFloat()) }, true).also { if (it !== raw) raw.recycle() }
            posePipeline?.analyze(bitmap)
        } catch (e: Exception) { Log.e("MainActivity", "Camera frame failed", e) }
        finally { imageProxy.close() }
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
        // Drop stale frames queued before stopInput (e.g. camera frames still in
        // flight when a showcase starts): they never entered the submit queue, so
        // they must not pop tracking entries or feed the gate/paddle bookkeeping.
        if (!inputRunning) return
        // Consume this frame's queued source index (every real submit pops exactly
        // once here or in onPoseMissing, keeping the usable-frame records aligned).
        val showcaseSrcIdx = if (fullresPaddleArmed() && realFrame) showcaseSrcQueue.poll() else null
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
                paddleFailed = false
                serveWindow.begin()
                emaProbs = null
                shiftDetector?.clear()
                paddleStream?.clear()
                // New usable-frame sequence: reset the full-res pass records with it.
                if (fullresPaddleArmed()) {
                    showcaseSrcQueue.clear()
                    synchronized(showcaseUsableSrc) {
                        showcaseUsableSrc.clear()
                        showcaseUsableWrist.clear()
                    }
                }
                shiftDetector?.let { s ->
                    landmarkSample(landmarks)?.let { (hx, bh) -> s.add(hx, bh) }
                }
                addPaddleSample(landmarks, bitmap, showcaseSrcIdx)
                postUi {
                    binding.statusState.text = "Pose: OK"
                    clearCards(analyzing = true)
                    setStatus("Analyzing your serve…", "Keep the setup and follow-through in view.", true)
                }
            } else {
                // absorb standing noise (and any recovery swing during refractory)
                // BEFORE comparing — the triggering frame itself never updates stats
                idleMean += 0.1f * (motion - idleMean)
                idleDev += 0.1f * (abs(motion - idleMean) - idleDev)
                postUi {
                    binding.statusState.text = "Pose: OK"
                    if (!inRefractory) binding.statusLine.setText(R.string.ready_title)
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
        if (realFrame) addPaddleSample(landmarks, if (lowMotionSinceMs == 0L) bitmap else null, showcaseSrcIdx)

        val frames = serveWindow.size

        val ended = (lowMotionSinceMs != 0L && now - lowMotionSinceMs >= intent.getLongExtra("research_low_hold_ms", lowHoldMs)) ||
            now - serveStartMs >= maxActiveMs
        if (ended) {
            lastServeEndMs = now
            // Paddle runs off-thread; give already-queued detections a bounded
            // chance to land so the verdict still sees the contact window.
            val drainMs = awaitPaddleQuiescence(paddleQuiesceMs)
            // Showcase-only full-res paddle second pass (same model/rules, bigger
            // pixels): re-detects the contact window when the live worker missed it.
            if (showcaseIndex >= 0 || intent.getBooleanExtra("research_fullres_paddle", false)) showcaseFullResPaddle()
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
            updateFeedback(finalPrediction)
            if (finalPrediction == null) postUi {
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
            postUi {
                binding.statusState.text = "Pose: OK"
                binding.statusProgress.visibility = View.GONE
            }
            return
        }

        postUi {
            binding.statusState.text = "Pose: OK"
            binding.statusLine.text = "Analyzing your serve…"
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

    /** Snapshot all measurements once, before the session buffers are cleared. */
    private fun updateFeedback(prediction: Pair<String, Float>?) {
        val generation = callbackGeneration.get() ?: resultGeneration
        val hipValue = shiftDetector?.compute()
        val hipPassed = shiftDetector?.passed()
        val stream = paddleStream.takeUnless { paddleFailed }
        val angle = stream?.computeForFeedback()
        val paddle = stream?.let { FeedbackEngine.paddleState(angle, it.low, it.high, it.directionValidated) }
        val independentMechanics = intent.getBooleanExtra("research_motion_rate", true) &&
            serveWindow.size >= finalMinFrames && serveWindow.durationMs >= 1500L
        // A landing scanned while this serve was being analyzed attaches now;
        // anything older already expired inside consumePendingLanding.
        val landed = consumePendingLanding(SystemClock.uptimeMillis())
        val result = feedbackEngine?.evaluate(FeedbackEngine.Input(
            serveType = prediction?.first, gruConfidence = prediction?.second,
            shiftSufficient = hipPassed, paddle = paddle,
            independentMechanics = independentMechanics,
            placementActive = landed != null, placementZone = landed?.zone,
        ))
        val endMs = lastServeEndMs
        val peakMs = lastMotionPeakMs
        val startMs = serveStartMs
        Log.i("Feedback", "version=${result?.version} status=${result?.status} rules=${result?.ruleIds} paddle=${paddle ?: "UNAVAILABLE"} landing=${landed?.zone}")
        val hipText = ServePresentation.hip(hipValue, hipPassed)
        val paddleText = ServePresentation.paddle(angle, paddle)
        val titleText = feedbackTitleFor(result, prediction?.second)
        val bodyText = result?.text ?: "Coaching rules could not be loaded. Your available measurements are shown above."
        val record = ServeRecord(
            number = SessionStore.nextNumber(),
            time = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date()),
            serveType = prediction?.let { ServePresentation.classification(it.first, it.second) } ?: "Unavailable",
            confidence = prediction?.second,
            hip = hipText,
            paddle = paddleText,
            feedbackTitle = titleText,
            feedback = bodyText,
            landing = landed?.zone,
        )
        SessionStore.add(record)
        lastServe = LastServe(prediction?.first, prediction?.second, hipPassed, paddle,
            independentMechanics, record.number, SystemClock.uptimeMillis())
        speak(record)
        postUi(generation) {
            binding.shiftLabel.text = hipText
            binding.paddleLabel.text = paddleText
            binding.feedbackTitle.text = titleText
            binding.feedbackLabel.text = bodyText
            setStatus("Analysis complete", getString(R.string.results_note))
            if (intent.hasExtra(EXTRA_REPLAY_CLIP)) Log.i("FeedbackOutput", org.json.JSONObject()
                .put("status", result?.status?.name).put("text", binding.feedbackLabel.text.toString()).toString())
            val shown = SystemClock.uptimeMillis()
            if (endMs != 0L) Log.i(LATENCY_TAG, "feedback_shown shown=$shown end_to_shown=${shown-endMs}ms peak_to_shown=${shown-peakMs}ms serve_total=${shown-startMs}ms")
        }
    }

    /** Coaching header for a feedback result; shared by the live and late-landing paths. */
    private fun feedbackTitleFor(result: FeedbackEngine.Result?, confidence: Float?): String = when {
        result == null -> "COACHING UNAVAILABLE"
        result.status == FeedbackEngine.Status.CORRECTIVE && (confidence ?: 0f) < FeedbackEngine.MIN_CONFIDENCE -> "GENERAL MECHANICS · NEXT ATTEMPT"
        result.status == FeedbackEngine.Status.LOW_CONFIDENCE -> "MORE EVIDENCE NEEDED"
        else -> getString(R.string.coaching_title)
    }

    /** Take the saved landing when the scan is still fresh; drop it and warn when expired. */
    private fun consumePendingLanding(now: Long): LandingResult? {
        val saved = pendingLanding ?: return null
        pendingLanding = null
        if (now - saved.second > landingFreshMs) {
            postUi { Toast.makeText(this, getString(R.string.landing_expired), Toast.LENGTH_LONG).show() }
            return null
        }
        return saved.first
    }

    /**
     * Dual-phone handoff arrival (UI thread). A fresh snapshot means the scan
     * belongs to the just-completed serve: re-evaluate with the landing zone
     * and amend the logged record. Otherwise the zone waits for the next serve.
     */
    private fun onLandingScanned(result: LandingResult) {
        val now = SystemClock.uptimeMillis()
        val snap = lastServe
        if (snap != null && now - snap.completedMs <= landingFreshMs) {
            Toast.makeText(this, getString(R.string.landing_attached, result.zone, snap.recordNumber),
                Toast.LENGTH_LONG).show()
            analysisExecutor.execute { attachLanding(snap, result) }
        } else {
            pendingLanding = result to now
            Toast.makeText(this, getString(R.string.landing_saved_next, result.zone),
                Toast.LENGTH_LONG).show()
        }
    }

    /** Re-run the logged serve's fusion with the scanned zone; never touches live state. */
    private fun attachLanding(snap: LastServe, result: LandingResult) {
        val engine = feedbackEngine ?: return
        val re = engine.evaluate(FeedbackEngine.Input(
            serveType = snap.serveType, gruConfidence = snap.confidence,
            shiftSufficient = snap.shift, paddle = snap.paddle,
            independentMechanics = snap.independentMechanics,
            placementActive = true, placementZone = result.zone,
        ))
        val title = feedbackTitleFor(re, snap.confidence)
        if (!SessionStore.update(snap.recordNumber) {
            it.copy(landing = result.zone, feedbackTitle = title, feedback = re.text)
        }) return
        Log.i("Feedback", "landing_attached serve=${snap.recordNumber} zone=${result.zone} status=${re.status}")
        landingRefreshRecord = snap.recordNumber
        runOnUiThread { renderLandingRefresh() }
    }

    /** Refresh the feedback cards after a late landing; deferred to onResume when backgrounded. */
    private fun renderLandingRefresh() {
        val number = landingRefreshRecord ?: return
        if (destroyed || !foreground) return
        if (lastServe?.recordNumber != number) { landingRefreshRecord = null; return }
        val record = SessionStore.snapshot().find { it.number == number }
            ?: run { landingRefreshRecord = null; return }
        landingRefreshRecord = null
        binding.feedbackTitle.text = record.feedbackTitle
        binding.feedbackLabel.text = record.feedback
        setStatus("Analysis complete", "Landing ${record.landing} attached · ${getString(R.string.results_note)}")
        speakLanding(record)
    }

    /**
     * Hand a frame to the paddle worker. Detection costs ~600ms and must not block
     * the pose callback, so it is queued to a single-thread executor; the queue is
     * bounded and drops the oldest sample when paddle falls behind.
     *
     * bitmap == null keeps wrist-speed bookkeeping only (contactFrame is derived
     * from it) and stays on this thread — it is sub-microsecond.
     */
    /**
     * Showcase-only second paddle pass: when the live worker measured nothing,
     * re-decode the contact window at source resolution and run the same detector
     * through PaddleStream.addWindow, so the angles land in the verdict's window.
     * Mirrors CoreLatencyBenchmark's two-pass (contact +/-5 usable frames, even
     * stride, one forward batch so pixels match). Runs synchronously on the
     * analysis worker; trailing replay frames are idle and cannot retrigger.
     */
    private fun showcaseFullResPaddle() {
        val stream = paddleStream ?: return
        if (paddleFailed) return
        if (stream.computeForFeedback() != null) return
        val file = showcaseClipFile?.takeIf { it.isFile }
            ?: intent.getStringExtra(EXTRA_REPLAY_CLIP)?.let {
                File(File(filesDir, "latency_input"), File(it).name).canonicalFile
            }?.takeIf {
                it.parentFile == File(filesDir, "latency_input").canonicalFile && it.isFile
            } ?: return
        // Snapshot under lock: clears run on the UI thread between clips.
        data class WinFrame(val usable: Int, val src: Int, val wrist: PointF)
        val frames: List<WinFrame> = synchronized(showcaseUsableSrc) {
            val n = showcaseUsableSrc.size
            if (n == 0 || showcaseUsableWrist.size != n) return
            val contact = try { stream.contactFrame() } catch (_: Exception) { return }
            if (contact < 0 || contact >= n) return
            ((contact - 5)..(contact + 5)).filter { it in 0 until n && it % 2 == 0 }
                .map { WinFrame(it, showcaseUsableSrc[it], showcaseUsableWrist[it]) }
        }
        if (frames.isEmpty()) return
        val start = SystemClock.uptimeMillis()
        var measured = 0
        var contact = -1
        val retriever = MediaMetadataRetriever()
        try {
            if (android.os.Build.VERSION.SDK_INT < 28) return
            retriever.setDataSource(file.absolutePath)
            contact = try { stream.contactFrame() } catch (_: Exception) { -1 }
            val srcFirst = frames.first().src
            val srcLast = frames.last().src
            if (srcFirst < 0 || srcLast < srcFirst) return
            val params = MediaMetadataRetriever.BitmapParams().apply {
                preferredConfig = Bitmap.Config.ARGB_8888
            }
            val batch = retriever.getFramesAtIndex(srcFirst, srcLast - srcFirst + 1, params)
            try {
                for (f in frames) {
                    val bmp = batch.getOrNull(f.src - srcFirst) ?: continue
                    stream.addWindow(f.usable, f.wrist.x, f.wrist.y, bmp)
                }
            } finally {
                for (b in batch) runCatching { if (!b.isRecycled) b.recycle() }
            }
            measured = if (stream.computeForFeedback() != null) 1 else 0
        } catch (e: Exception) {
            Log.w("MainActivity", "Showcase full-res paddle pass failed; keeping live result", e)
        } finally {
            runCatching { retriever.release() }
        }
        Log.i(LATENCY_TAG, "showcase_paddle_pass2 contact=$contact window=${frames.size} measured=$measured ms=${SystemClock.uptimeMillis() - start}")
    }

    private fun addPaddleSample(landmarks: Array<PointF>, bitmap: Bitmap?, showcaseSrcIdx: Int? = null) {
        if (paddleFailed) return
        val stream = paddleStream ?: return
        // Record usable-frame source evidence for the showcase full-res pass.
        // This runs exactly once per PaddleStream usable frame, so the records
        // stay parallel to its internal usable-frame sequence.
        if (fullresPaddleArmed() && showcaseSrcIdx != null) {
            synchronized(showcaseUsableSrc) {
                showcaseUsableSrc.add(showcaseSrcIdx)
                showcaseUsableWrist.add(PointF(landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y))
            }
        }
        val now = SystemClock.uptimeMillis()
        // Paddle submit spacing. The 150ms showcase experiment recovered no extra
        // paddle over the full-res pass but cost availability (27 -> 18 outputs on
        // the full set; 9/10 recovered at 350ms), so both showcase and normal live
        // use 350ms. research_relaxed_paddle keeps the 150ms arm reproducible.
        val spacingMs = if (intent.getBooleanExtra("research_relaxed_paddle", false)) 150L else 350L
        val permit = bitmap != null && (!posePriority ||
            (now-lastPaddleSubmitMs >= spacingMs && paddleExecutor.activeCount == 0 && !paddleBusy.get()))
        if (bitmap != null && !permit) paddleThrottled++
        val sample = stream.prepare(landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y, permit)
        if (sample == null || bitmap == null) return
        val generation = callbackGeneration.get() ?: resultGeneration
        if (posePriority && !paddleBusy.compareAndSet(false, true)) { paddleDropped++; return }
        lastPaddleSubmitMs = now
        try {
            paddleExecutor.execute {
                try {
                    if (generation == resultGeneration) stream.detect(sample, bitmap)
                } catch (e: Exception) {
                    runCatching { analysisExecutor.execute {
                        if (generation == resultGeneration && !destroyed) markPaddleFailed(e)
                    } }
                } finally { paddleBusy.set(false) }
            }
        } catch (_: RejectedExecutionException) { paddleBusy.set(false); paddleDropped++ }
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
        val generation = callbackGeneration.get() ?: resultGeneration
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
            val confColor = ContextCompat.getColor(this,
                if (conf >= FeedbackEngine.MIN_CONFIDENCE) R.color.accent else R.color.warning)

            postUi {
                if (generation != resultGeneration) return@postUi
                binding.classLabel.text = ServePresentation.classification(label, conf)
                binding.classLabel.setTextColor(confColor)
                val finalSource = if (knnProbs != null) "Final hybrid" else "Final GRU"
                binding.confLabel.text = "${"%.1f".format(conf * 100f)}% confidence · ${if (conf < FeedbackEngine.MIN_CONFIDENCE) "serve type uncertain" else "model prediction"}"
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
            postUi { binding.confLabel.text = "INFER ERR: ${e.message}" }
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

    override fun onResume() {
        super.onResume()
        foreground = true
        if (modelsReady) startInput()
        // A landing scanned while the scanner covered this screen amends here.
        renderLandingRefresh()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean("permissionRequested", permissionRequested)
        super.onSaveInstanceState(outState)
    }

    private fun stopInput() {
        inputRunning = false
        inputEpoch.incrementAndGet()
        sessionEpoch.invalidate()
        binding.root.keepScreenOn = false
        cameraProvider?.unbind(*cameraUseCases)
        cameraUseCases = emptyArray()
        clipReplay?.stop(); clipReplay = null
        cameraExecutor.execute { posePipeline?.stop(); posePipeline = null }
        analysisExecutor.execute { resetAnalysis() }
    }

    override fun onStop() {
        foreground = false
        if (modelsReady && binding.statusProgress.visibility == View.VISIBLE) {
            clearCards()
            binding.feedbackLabel.text = "The attempt was interrupted. Start a new serve when you return."
        }
        stopInput()
        showcaseIndex = -1
        showcaseQueue = emptyList()
        showcaseClipFile = null
        showcaseSrcQueue.clear()
        synchronized(showcaseUsableSrc) {
            showcaseUsableSrc.clear()
            showcaseUsableWrist.clear()
        }
        binding.replayView.setImageDrawable(null)
        binding.replayView.visibility = View.GONE
        binding.statusState.setText(R.string.paused)
        binding.statusProgress.visibility = View.GONE
        super.onStop()
    }

    override fun onDestroy() {
        destroyed = true
        sessionEpoch.invalidate()
        clipReplay?.stop()
        runCatching { tts?.stop(); tts?.shutdown() }
        tts = null
        // Close each model after in-flight work, on its owning worker.
        cameraExecutor.execute { posePipeline?.stop(); posePipeline = null }
        cameraExecutor.shutdown()
        analysisExecutor.execute {
            gru?.close(); gru = null
            paddleExecutor.queue.clear()
            paddleExecutor.execute { paddleStream?.close(); paddleStream = null }
            paddleExecutor.shutdown()
        }
        analysisExecutor.shutdown()
        super.onDestroy()
    }
}
