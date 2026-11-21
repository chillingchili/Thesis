package com.thesis.pickleballserve

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.util.TypedValue
import android.view.View
import android.widget.ImageView
import androidx.appcompat.widget.PopupMenu
import android.widget.TableRow
import android.widget.TextView
import androidx.annotation.StringRes
import androidx.camera.core.AspectRatio
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.doOnNextLayout
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar
import com.thesis.pickleballserve.databinding.DialogFactsBinding
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

    /** SETUP = camera setup check (no serve capture); LIVE = capture + result cards. */
    private enum class Screen { SETUP, LIVE }
    /** Status pill dot: green ready/result, amber busy/partial, red needs attention. */
    private enum class Pill { GOOD, BUSY, PARTIAL, ALERT }
    private enum class Check { OK, WARN, PENDING }

    private var screen = Screen.SETUP
    /** Screen to reopen after a recreation (Theme change); null on a fresh launch. */
    private var restoredScreen: Screen? = null
    /** Read on the analysis worker: the motion gate only arms during a live session. */
    @Volatile private var sessionLive = false
    /** True while the pill shows "Player not visible", so the next pose restores it. */
    @Volatile private var trackingLostShown = false
    @Volatile private var lastFramingMs = 0L
    @Volatile private var lastFraming: FramingCheck.Result? = null
    @Volatile private var frameAspect = 0f
    /** Last final class probabilities (analysis worker), rendered with the result. */
    private var lastProbs: FloatArray? = null
    private var lastInferenceFailed = false
    @Volatile private var lastRecord: ServeRecord? = null
    /** Classifier label shown on the Research tools menu row. */
    private var modeLabelText = ""
    /** A finished result is on the panel (kept, dimmed, while the next serve is analyzed). */
    private var resultShown = false
    /** Last measured values, for the help dialogs' "This serve" row. */
    private var lastHipValue: Float? = null
    private var lastHipPassed: Boolean? = null
    private var lastPaddleAngle: Double? = null
    private var lastPaddleState: FeedbackEngine.PaddleState? = null

    // --- Hands-free start: hold still in the outline, then a spoken countdown. ---
    private var framingOkSinceMs = 0L
    private var countdownEndMs = 0L
    private val autoStartHoldMs = 1000L
    private val autoStartCountdownMs = 3000L
    private val autoStartTick = Runnable { tickAutoStart() }
    private val autoStartGraceMs = 400L
    private var framingBadSinceMs = 0L
    private var lastSpokenCount = -1
    /** Set by a manual return to setup; cleared once the player has left the outline. */
    private var autoStartBlocked = false

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
    /** Review clips (memory only). Off for research replays so latency runs stay unchanged. */
    private val clipRecorder = ClipRecorder()
    private val clipsEnabled get() = intent.getStringExtra(EXTRA_REPLAY_CLIP) == null
    @Volatile private var pendingClip: ServeClip? = null

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
                notify(R.string.scan_invalid)
                return@registerForActivityResult
            }
            onLandingScanned(parsed)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        permissionRequested = savedInstanceState?.getBoolean("permissionRequested") ?: false
        // A Theme change recreates this screen: keep the user's session settings across it.
        savedInstanceState?.let { s ->
            mode = runCatching { RunMode.valueOf(s.getString("mode") ?: mode.name) }.getOrDefault(mode)
            cameraFacing = s.getInt("cameraFacing", cameraFacing)
            poseGpu = s.getBoolean("poseGpu", poseGpu)
            speakResults = s.getBoolean("speakResults", speakResults)
            restoredScreen = runCatching { Screen.valueOf(s.getString("screen") ?: "") }.getOrNull()
        }
        posePriority = intent.getBooleanExtra("research_pose_priority", true)
        revisedEnd = intent.getBooleanExtra("research_end_v2", false)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        // Camera screen runs immersive: system bars hide and return on a swipe.
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowCompat.getInsetsController(window, binding.root).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }
        val gutter = dp(8)
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val cut = insets.getInsets(WindowInsetsCompat.Type.displayCutout())
            view.setPadding(gutter + cut.left, gutter + cut.top, gutter + cut.right, gutter + cut.bottom)
            insets
        }
        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newSingleThreadExecutor()
        paddleExecutor = ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, LinkedBlockingQueue(6))
        binding.menuButton.setOnClickListener { showMenu() }
        binding.previewView.setOnClickListener {
            if (screen == Screen.LIVE) setFullscreen(!fullscreen)
        }
        // "Start now" is the fallback for when someone is at the phone; the
        // session normally starts by itself (see tickAutoStart).
        binding.startSessionButton.setOnClickListener { if (modelsReady) applyMode(Screen.LIVE) }
        binding.tryAgainButton.setOnClickListener { newAttempt() }
        binding.setupTipsButton.setOnClickListener { applyMode(Screen.SETUP) }
        binding.backToSetupButton.setOnClickListener { applyMode(Screen.SETUP) }
        binding.allServesButton.setOnClickListener { startActivity(Intent(this, SessionActivity::class.java)) }
        binding.readAloudSwitch.isChecked = speakResults
        binding.readAloudSwitch.setOnCheckedChangeListener { _, on ->
            speakResults = on
            if (!on) tts?.stop()
        }
        onBackPressedDispatcher.addCallback(this) {
            when {
                fullscreen -> setFullscreen(false)
                // Live -> setup first, so a stray Back on stage never closes the app.
                screen == Screen.LIVE && showcaseIndex < 0 && intent.getStringExtra(EXTRA_REPLAY_CLIP) == null ->
                    applyMode(Screen.SETUP)
                else -> { isEnabled = false; onBackPressedDispatcher.onBackPressed() }
            }
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
        binding.hipInfoButton.setOnClickListener { showHipHelp() }
        binding.paddleInfoButton.setOnClickListener { showPaddleHelp() }
        // Every launch opens on the camera setup check (Figure 7.2a); research
        // replays skip it and run the live path directly.
        clearCards()
        applyMode(if (intent.getStringExtra(EXTRA_REPLAY_CLIP) != null) Screen.LIVE else restoredScreen ?: Screen.SETUP)
        paddleExecutor.execute {
            paddleStream = PaddleStream.load(applicationContext)
            if (paddleStream == null) Log.e("MainActivity", "Paddle stream unavailable")
        }
        analysisExecutor.execute {
            try {
                gru = GruClassifier.load(applicationContext, ensemble = mode != RunMode.SINGLE)
                knn = runCatching { KnnClassifier.load(applicationContext) }.getOrNull()
                shiftDetector = ShiftDetector.load(applicationContext)
                feedbackEngine = runCatching {
                    FeedbackEngine.fromJson(assets.open("feedback_rules.json").bufferedReader().use { it.readText() })
                }.getOrNull()
                modelsReady = true
                runOnUiThread {
                    if (!destroyed) {
                        updateModeLabel()
                        renderSetupChecks()
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

    /**
     * One status pill: dot + one line. The detail line shows only for problems
     * (ALERT); otherwise it is spoken via the content description.
     */
    private fun setStatus(title: String, detail: String, busy: Boolean = false, pill: Pill? = null) {
        val tone = pill ?: when {
            busy -> Pill.BUSY
            binding.recoveryButton.visibility == View.VISIBLE -> Pill.ALERT
            else -> Pill.GOOD
        }
        trackingLostShown = false
        binding.statusLine.text = title
        binding.statusDetail.text = detail
        binding.statusDetail.visibility = if (detail.isNotBlank() && tone == Pill.ALERT) View.VISIBLE else View.GONE
        binding.statusProgress.visibility = if (busy) View.VISIBLE else View.GONE
        binding.statusDot.visibility = if (busy) View.GONE else View.VISIBLE
        binding.statusDot.backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(this, when (tone) {
            Pill.GOOD -> R.color.dot_good
            Pill.BUSY, Pill.PARTIAL -> R.color.dot_busy
            Pill.ALERT -> R.color.dot_alert
        }))
        binding.statusPill.contentDescription = if (detail.isBlank()) title else "$title. $detail"
    }

    /** Idle status for the current screen. */
    private fun readyStatus() {
        if (screen == Screen.SETUP) setStatus(getString(R.string.setup_check), "")
        else setStatus(getString(R.string.status_ready), getString(R.string.camera_hint))
    }

    /**
     * One shell for both modes: the camera never moves; the panel swaps its
     * contents. SETUP shows the checks and starts by itself; LIVE arms the
     * motion gate and shows the cue and three metric rows.
     */
    private fun applyMode(next: Screen) {
        val wasLive = sessionLive
        screen = next
        sessionLive = next == Screen.LIVE
        if (fullscreen) setFullscreen(false)
        cancelAutoStart()
        val setup = next == Screen.SETUP
        // Back to setup from a session: do not jump straight back to LIVE while the player is in frame.
        if (setup && wasLive) autoStartBlocked = true
        if (!setup) SessionStore.markStart(SystemClock.elapsedRealtime())
        binding.setupPanel.visibility = if (setup) View.VISIBLE else View.GONE
        binding.resultsScroll.visibility = if (setup) View.GONE else View.VISIBLE
        binding.setupTitle.visibility = if (setup) View.VISIBLE else View.GONE
        binding.allServesButton.visibility = if (setup) View.GONE else View.VISIBLE
        binding.readAloudSwitch.visibility = if (setup) View.GONE else View.VISIBLE
        binding.framingHint.visibility = if (setup) View.VISIBLE else View.GONE
        updateServesButton()
        if (wasLive != sessionLive) {
            // Leaving or entering a session never carries a half-captured serve across.
            sessionEpoch.invalidate()
            if (::analysisExecutor.isInitialized) analysisExecutor.execute { resetAnalysis() }
            binding.statusProgress.visibility = View.GONE
            if (!resultShown) clearCards()
            else setResultDimmed(false)
        }
        renderSetupChecks()
        if (binding.recoveryButton.visibility != View.VISIBLE &&
            (modelsReady || binding.statusProgress.visibility != View.VISIBLE)) readyStatus()
    }

    private fun updateServesButton() {
        val n = SessionStore.size()
        binding.allServesButton.text = if (n == 0) getString(R.string.no_serves_yet) else getString(R.string.all_serves, n)
    }

    /** Setup checks: only what the app can actually measure. Drives hands-free start. */
    private fun renderSetupChecks(framing: FramingCheck.Result? = lastFraming) {
        val rear = cameraFacing == CameraSelector.LENS_FACING_BACK
        setCheck(binding.checkCameraIcon, binding.checkCameraText, if (rear) Check.OK else Check.WARN,
            getString(if (rear) R.string.check_camera else R.string.check_camera_front))
        val seen = framing != null && framing != FramingCheck.Result.NO_PLAYER
        setCheck(binding.checkPlayerIcon, binding.checkPlayerText,
            when { framing == null -> Check.PENDING; seen -> Check.OK; else -> Check.WARN },
            getString(if (seen) R.string.check_player else R.string.check_player_missing))
        setCheck(binding.checkBodyIcon, binding.checkBodyText,
            when { !seen -> Check.PENDING; framing!!.ok -> Check.OK; else -> Check.WARN },
            if (seen) framing!!.message else getString(R.string.stand_in_outline))
        binding.framingGuide.ok = framing?.ok == true
        binding.framingHint.text = if (framing?.ok == true) framing.message
            else framing?.takeIf { seen }?.message ?: getString(R.string.stand_in_outline)
        binding.startSessionButton.isEnabled = modelsReady
        binding.startSessionButton.setText(if (modelsReady) R.string.start_now else R.string.setup_waiting_models)
        // A manual return to setup stays put until the player has left the outline once.
        if (framing != null && !framing.ok) autoStartBlocked = false
        if (framing?.ok == true) maybeStartAutoStart() else if (framingOkSinceMs == 0L) cancelAutoStart()
    }

    private fun autoStartAllowed() = screen == Screen.SETUP && !autoStartBlocked && modelsReady && inputRunning && foreground &&
        !destroyed && showcaseIndex < 0 && intent.getStringExtra(EXTRA_REPLAY_CLIP) == null

    private fun maybeStartAutoStart() {
        if (!autoStartAllowed() || framingOkSinceMs != 0L) return
        framingOkSinceMs = SystemClock.uptimeMillis()
        binding.root.removeCallbacks(autoStartTick)
        binding.root.postDelayed(autoStartTick, 200)
    }

    /** Hold still ~1 s inside the outline, then a 3 s countdown; leaving the outline cancels. */
    private fun tickAutoStart() {
        binding.root.removeCallbacks(autoStartTick)
        if (!autoStartAllowed()) { cancelAutoStart(); return }
        val now = SystemClock.uptimeMillis()
        // Grace period: one jittery frame at the outline edge must not reset the countdown.
        if (lastFraming?.ok != true) {
            if (framingBadSinceMs == 0L) framingBadSinceMs = now
            if (now - framingBadSinceMs > autoStartGraceMs) { cancelAutoStart(); return }
        } else framingBadSinceMs = 0L
        if (now - framingOkSinceMs < autoStartHoldMs) { binding.root.postDelayed(autoStartTick, 200); return }
        if (countdownEndMs == 0L) countdownEndMs = now + autoStartCountdownMs
        val left = ((countdownEndMs - now + 999) / 1000).toInt()
        if (left <= 0) {
            cancelAutoStart()
            applyMode(Screen.LIVE)
            speakText(getString(R.string.spoken_ready), force = true)
            return
        }
        if (left != lastSpokenCount) { lastSpokenCount = left; speakText(left.toString(), force = true) }
        setStatus(getString(R.string.hold_still, left), "", busy = true)
        binding.startSessionNote.text = getString(R.string.starting_in, left)
        binding.root.postDelayed(autoStartTick, 200)
    }

    private fun cancelAutoStart() {
        binding.root.removeCallbacks(autoStartTick)
        val wasCounting = countdownEndMs != 0L
        framingOkSinceMs = 0L
        framingBadSinceMs = 0L
        countdownEndMs = 0L
        lastSpokenCount = -1
        binding.startSessionNote.setText(if (autoStartBlocked) R.string.starts_after_step_out else R.string.starts_by_itself)
        if (wasCounting && screen == Screen.SETUP && binding.recoveryButton.visibility != View.VISIBLE) readyStatus()
    }

    private fun setCheck(icon: ImageView, text: TextView, state: Check, label: String) {
        val (res, tint) = when (state) {
            Check.OK -> R.drawable.ic_check_circle to R.color.ok
            Check.WARN -> R.drawable.ic_info_outline to R.color.warn
            Check.PENDING -> R.drawable.ic_circle to R.color.icon_secondary
        }
        icon.setImageResource(res)
        icon.imageTintList = ColorStateList.valueOf(color(tint))
        text.text = label
        text.contentDescription = when (state) {
            Check.OK -> "Done: $label"; Check.WARN -> "Needs attention: $label"; Check.PENDING -> label
        }
    }

    /** Throttled framing check from the analysis worker (also drives the outline color). */
    private fun postFraming(result: FramingCheck.Result, force: Boolean = false) {
        val now = SystemClock.uptimeMillis()
        if (!force && now - lastFramingMs < 250L && result == lastFraming) return
        lastFramingMs = now
        if (result == lastFraming) return
        lastFraming = result
        runOnUiThread { if (!destroyed) renderSetupChecks(result) }
    }

    /** "Try again": clear the shown serve and wait for the next one. */
    private fun newAttempt() {
        sessionEpoch.invalidate()
        pendingLanding = null
        landingRefreshRecord = null
        clearCards()
        analysisExecutor.execute { resetAnalysis() }
        readyStatus()
    }

    private fun color(id: Int) = ContextCompat.getColor(this, id)
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    /** The last result stays readable, dimmed, while the next serve is analyzed. */
    private fun setResultDimmed(dim: Boolean) {
        val a = if (dim) 0.45f else 1f
        binding.feedbackLabel.alpha = a; binding.cueBadge.alpha = a
        binding.feedbackNotes.alpha = a; binding.metricRows.alpha = a
    }

    /**
     * Empty or analyzing panel. With a result already on screen, analyzing only
     * dims it; otherwise one message replaces the metric rows.
     */
    private fun clearCards(analyzing: Boolean = false) {
        serveAnalyzing = analyzing
        binding.feedbackActions.visibility = View.GONE
        if (analyzing && resultShown) { setResultDimmed(true); return }
        resultShown = false
        setResultDimmed(false)
        setCue(getString(if (analyzing) R.string.status_analyzing else R.string.waiting_title))
        binding.cueBadge.visibility = View.GONE
        binding.feedbackNotes.visibility = View.GONE
        binding.waitingMsg.visibility = if (analyzing) View.GONE else View.VISIBLE
        binding.metricRows.visibility = View.GONE
        binding.backToSetupButton.visibility = if (analyzing) View.GONE else View.VISIBLE
        listOf(binding.classLabel, binding.shiftLabel, binding.paddleLabel).forEach {
            it.setText(R.string.card_empty); it.setTextColor(color(R.color.text_primary))
        }
        binding.confLabel.text = ""
        listOf(binding.serveChip, binding.hipChip, binding.paddleChip).forEach { it.visibility = View.GONE }
    }

    /** Incomplete capture: the only state that offers Try again. */
    private fun showIncomplete() {
        clearCards()
        setCue(getString(R.string.incomplete_capture_note))
        binding.waitingMsg.visibility = View.GONE
        binding.backToSetupButton.visibility = View.GONE
        binding.feedbackActions.visibility = View.VISIBLE
    }

    /** A serve is between its start and its result; a Theme change would drop it. */
    private var serveAnalyzing = false

    /** Big cue at 24sp; long cues shrink after layout until the metric rows fit (fitCue). */
    private fun setCue(text: String) {
        // Live shows the first cue only, as a sentence; Read aloud and the review keep every cue.
        val first = text.lines().firstOrNull { it.isNotBlank() }?.removePrefix("• ")?.trim() ?: ""
        cueSentences = first.split(Regex("(?<=[.!?])\\s+")).filter { it.isNotBlank() }
        binding.feedbackLabel.contentDescription = text.replace("• ", "")
        binding.feedbackLabel.maxLines = Int.MAX_VALUE
        binding.feedbackLabel.ellipsize = null
        binding.feedbackLabel.setTextSize(TypedValue.COMPLEX_UNIT_SP, 24f)
        binding.feedbackLabel.text = first
        binding.resultsScroll.doOnNextLayout { fitCue(0) }
    }

    /** Sentences of the live cue; a long cue drops its opening diagnosis before anything is clipped. */
    private var cueSentences: List<String> = emptyList()

    /**
     * Keeps every metric row visible while the cue stays readable at 2–3 m: first
     * drops the engine note line, then steps the cue down to a 20sp floor, then
     * clamps it to three lines. Read aloud and the review screen keep the full text.
     */
    private fun fitCue(step: Int) {
        if (destroyed || step > 8) return
        val panel = binding.resultsScroll
        val rows = binding.metricRows
        if (panel.height == 0 || rows.visibility != View.VISIBLE) return
        // The parent squeezes rows into what is left, so compare their natural height.
        rows.measure(View.MeasureSpec.makeMeasureSpec(rows.width, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED))
        val overflow = rows.top + rows.measuredHeight > panel.height - panel.paddingBottom
        if (!overflow) return
        val sp = binding.feedbackLabel.textSize / resources.displayMetrics.scaledDensity
        // Changes go out of the layout pass (post), then re-check after the relayout.
        panel.post {
            val label = binding.feedbackLabel
            when {
                binding.feedbackNotes.visibility == View.VISIBLE -> binding.feedbackNotes.visibility = View.GONE
                sp > 20.5f -> label.setTextSize(TypedValue.COMPLEX_UNIT_SP, (sp - 2f).coerceAtLeast(20f))
                // Keep the instruction: drop the leading diagnosis sentence(s) first.
                cueSentences.size > 1 -> {
                    cueSentences = cueSentences.drop(1)
                    label.text = cueSentences.joinToString(" ")
                }
                label.maxLines > 3 -> { label.maxLines = 3; label.ellipsize = android.text.TextUtils.TruncateAt.END }
                label.maxLines > 2 -> label.maxLines = 2
                else -> return@post
            }
            panel.doOnNextLayout { fitCue(step + 1) }
            panel.requestLayout()
        }
    }

    private fun showChip(chip: TextView, label: String?, tone: ServePresentation.Tone) {
        if (label == null) { chip.visibility = View.GONE; return }
        chip.text = label
        chip.visibility = View.VISIBLE
        val (bg, fg) = when (tone) {
            ServePresentation.Tone.GOOD -> R.color.ok_muted to R.color.ok
            ServePresentation.Tone.WARN -> R.color.warn_muted to R.color.warn
            ServePresentation.Tone.NEUTRAL -> R.color.accent_muted to R.color.text_secondary
        }
        chip.backgroundTintList = ColorStateList.valueOf(color(bg))
        chip.setTextColor(color(fg))
    }

    /** Badge wording on the live rows (status in words, never color alone). */
    private fun badgeLabel(chip: String?): String? = when (chip) {
        null -> null
        "Within range" -> getString(R.string.badge_in_range)
        else -> chip
    }

    /** Render one finished serve: cue, then the three rows with their reference bands (UI thread). */
    private fun renderResult(
        serve: ServePresentation.ServeType, confidence: Float?,
        hip: ServePresentation.Metric, paddle: ServePresentation.Metric,
        cues: String, generalCue: Boolean, notes: List<String>,
        hipValue: Float?, hipThreshold: Float?, paddleLow: Double?, paddleHigh: Double?,
    ) {
        resultShown = true
        setResultDimmed(false)
        serveAnalyzing = false
        setCue(cues)
        binding.cueBadge.visibility = if (generalCue) View.VISIBLE else View.GONE
        val noteText = notes.filter { it.isNotBlank() }.joinToString("\n")
        binding.feedbackNotes.text = noteText
        binding.feedbackNotes.visibility = if (noteText.isBlank()) View.GONE else View.VISIBLE
        binding.waitingMsg.visibility = View.GONE
        binding.backToSetupButton.visibility = View.GONE
        binding.feedbackActions.visibility = View.GONE
        binding.metricRows.visibility = View.VISIBLE

        binding.classLabel.text = serve.value
        binding.classLabel.contentDescription = "Serve type ${serve.value}. ${serve.detail}"
        binding.classLabel.setTextColor(color(if (serve.reliable) R.color.text_primary else R.color.warn))
        binding.confLabel.text = if (serve.reliable && confidence != null)
            "${ServePresentation.pct(confidence)}%" else ""
        showChip(binding.serveChip, when {
            serve.reliable -> null
            serve.value == "Not measured" -> getString(R.string.badge_partial)
            else -> getString(R.string.badge_low_confidence)
        }, ServePresentation.Tone.WARN)

        binding.shiftLabel.setTextColor(color(R.color.text_primary))
        binding.shiftLabel.text = hip.value
        showChip(binding.hipChip, badgeLabel(hip.chip), hip.tone)
        binding.hipRef.text = when {
            hipValue != null && hipValue.isFinite() && hipThreshold != null ->
                getString(R.string.hip_ref, hipValue, hipThreshold)
            hipThreshold != null -> String.format(Locale.getDefault(), "≥ %.3f", hipThreshold)
            else -> ""
        }
        binding.paddleLabel.setTextColor(color(R.color.text_primary))
        binding.paddleLabel.text = paddle.value
        showChip(binding.paddleChip, badgeLabel(paddle.chip) ?: getString(R.string.badge_partial),
            if (paddle.chip == null) ServePresentation.Tone.WARN else paddle.tone)
        binding.paddleRef.text = if (paddleLow != null && paddleHigh != null)
            getString(R.string.paddle_ref, paddleLow, paddleHigh) else ""
        updateServesButton()
    }

    /** Called only on the analysis worker; UI invalidation happens immediately. */
    private fun resetAnalysis() {
        serveWindow.clear(); emaProbs = null; gateActive = false; lowMotionSinceMs = 0L
        refractoryUntilMs = 0L; lastServeEndMs = 0L; lastAngles = null; motionRate.reset()
        motionRing.fill(0f); motionRingIdx = 0; motionRingCount = 0
        idleMean = 0f; idleDev = .02f; lastPoseSeenMs = 0L
        shiftDetector?.clear(); paddleStream?.clear()
        clipRecorder.reset()
    }

    /** Camera only (tap the camera): the panel hides for framing; Back or a tap restores it. */
    private fun setFullscreen(on: Boolean) {
        if (screen != Screen.LIVE && on) return
        fullscreen = on
        binding.panel.visibility = if (on) View.GONE else View.VISIBLE
        binding.framingHint.visibility = if (on || screen == Screen.SETUP) View.VISIBLE else View.GONE
    }

    private fun showMenu() {
        val popup = PopupMenu(this, binding.menuButton, android.view.Gravity.END)
        popup.menuInflater.inflate(R.menu.main_menu, popup.menu)
        popup.setForceShowIcon(true)
        if (screen == Screen.SETUP) popup.menu.findItem(R.id.action_setup).setTitle(R.string.setup_tips)
        popup.menu.findItem(R.id.action_gpu_pose).isChecked = poseGpu
        popup.menu.findItem(R.id.action_change_model).title = getString(R.string.classifier_row, modeLabelText)
        popup.menu.findItem(R.id.action_camera).title = getString(R.string.camera_menu) + " · " +
            getString(if (cameraFacing == CameraSelector.LENS_FACING_FRONT) R.string.camera_front_short else R.string.camera_back_short)
        popup.menu.findItem(
            if (cameraFacing == CameraSelector.LENS_FACING_FRONT) R.id.action_camera_front
            else R.id.action_camera_back
        ).isChecked = true
        popup.menu.findItem(when (savedNightMode()) {
            androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_NO -> R.id.action_theme_light
            androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_YES -> R.id.action_theme_dark
            else -> R.id.action_theme_system
        }).isChecked = true
        popup.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_setup -> {
                    if (screen == Screen.SETUP || showcaseIndex >= 0) showInfo(R.string.setup_title, R.string.setup_body)
                    else applyMode(Screen.SETUP)
                    true
                }
                R.id.action_reset -> { newAttempt(); true }
                R.id.action_gpu_pose -> {
                    poseGpu = !poseGpu
                    item.isChecked = poseGpu
                    // Pipeline reads the flag at creation; restart input so it applies now.
                    if (foreground && !destroyed && modelsReady) {
                        stopInput()
                        startInput()
                    }
                    notify(getString(R.string.gpu_pose_note, if (poseGpu) "on" else "off"))
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
                R.id.action_about -> { showAbout(); true }
                R.id.action_theme_system -> { applyNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM); true }
                R.id.action_theme_light -> { applyNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_NO); true }
                R.id.action_theme_dark -> { applyNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_YES); true }
                else -> false
            }
        }
        popup.show()
    }

    private fun savedNightMode(): Int = getSharedPreferences(ServeApp.PREFS, MODE_PRIVATE)
        .getInt(ServeApp.KEY_NIGHT_MODE, androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM)

    /** Theme menu: remember the choice and apply it; open screens recreate in the new theme. */
    private fun applyNightMode(mode: Int) {
        if (mode == savedNightMode()) return
        // Recreating mid-analysis would drop the serve being analyzed.
        if (serveAnalyzing) { notify(R.string.theme_busy); return }
        getSharedPreferences(ServeApp.PREFS, MODE_PRIVATE).edit().putInt(ServeApp.KEY_NIGHT_MODE, mode).apply()
        androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(mode)
    }

    private fun showInfo(titleRes: Int, bodyRes: Int) {
        MaterialAlertDialogBuilder(this).setTitle(titleRes).setMessage(bodyRes)
            .setPositiveButton(R.string.got_it, null).show()
    }

    /** Help/About dialog: one paragraph, a fact table, one footnote. Method details live here. */
    private fun showFacts(@StringRes title: Int, body: String?, facts: List<Pair<String, String>>,
                          note: String?, @StringRes button: Int = R.string.got_it) {
        val v = DialogFactsBinding.inflate(layoutInflater)
        v.factsBody.text = body ?: ""
        v.factsBody.visibility = if (body.isNullOrBlank()) View.GONE else View.VISIBLE
        for ((k, value) in facts) {
            val row = TableRow(this)
            row.addView(TextView(this).apply {
                text = k
                setTextAppearance(R.style.TextAppearance_Serve_Body)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
                setTextColor(color(R.color.text_secondary))
                setPadding(0, dp(2), dp(16), dp(2))
            })
            row.addView(TextView(this).apply {
                text = value
                setTextAppearance(R.style.TextAppearance_Serve_Title)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
                setPadding(0, dp(2), 0, dp(2))
            })
            v.factsTable.addView(row)
        }
        v.factsNote.text = note ?: ""
        v.factsNote.visibility = if (note.isNullOrBlank()) View.GONE else View.VISIBLE
        MaterialAlertDialogBuilder(this).setTitle(title).setView(v.root)
            .setPositiveButton(button, null).show()
    }

    private fun showHipHelp() {
        val thr = shiftDetector?.thr
        val value = lastHipValue
        val now = if (value != null && value.isFinite()) String.format(Locale.getDefault(), "%.2f · %s", value,
            if (lastHipPassed == true) "Good" else "Limited") else getString(R.string.not_measured_yet)
        showFacts(R.string.hip_info_title, getString(R.string.hip_help_body), listOf(
            getString(R.string.fact_this_serve) to now,
            getString(R.string.fact_threshold) to (thr?.let { String.format(Locale.getDefault(), "≥ %.3f", it) } ?: "—"),
            getString(R.string.fact_source) to getString(R.string.hip_help_source),
        ), getString(R.string.hip_help_note))
    }

    private fun showPaddleHelp() {
        val stream = paddleStream
        val angle = lastPaddleAngle
        val now = if (angle != null && angle.isFinite()) String.format(Locale.getDefault(), "%.1f° · %s", angle,
            if (lastPaddleState == FeedbackEngine.PaddleState.OPTIMAL) "Within band" else "Outside band")
            else getString(R.string.not_measured_yet)
        val band = stream?.let { String.format(Locale.getDefault(), "%.1f–%.1f°", it.low, it.high) } ?: "—"
        showFacts(R.string.paddle_info_title, getString(R.string.paddle_help_body), listOf(
            getString(R.string.fact_this_serve) to now,
            getString(R.string.fact_band) to band,
            getString(R.string.fact_source) to getString(R.string.paddle_help_source),
        ), getString(R.string.paddle_help_note))
    }

    private fun showAbout() {
        showFacts(R.string.about_tool, null, listOf(
            getString(R.string.about_classifier) to getString(R.string.about_classifier_value),
            getString(R.string.about_input) to getString(R.string.about_input_value),
            getString(R.string.about_gate) to getString(R.string.about_gate_value),
            getString(R.string.about_general) to getString(R.string.about_general_value),
            getString(R.string.about_processing) to getString(R.string.about_processing_value),
        ), getString(R.string.about_note), R.string.close)
    }

    /** Transient feedback as a Material snackbar over the panel. */
    private fun notify(message: String, long: Boolean = true) {
        if (destroyed) return
        Snackbar.make(binding.root, message, if (long) Snackbar.LENGTH_LONG else Snackbar.LENGTH_SHORT).show()
    }
    private fun notify(@StringRes res: Int, long: Boolean = true) = notify(getString(res), long)

    /** Status speech; force = always spoken (session start cues the player hears from the baseline). */
    private fun speakText(text: String, force: Boolean = false) {
        if (!speakResults && !force) return
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "status")
    }

    /** Switch live input cameras without losing the session; research replay keeps its own source. */
    private fun switchCamera(facing: Int) {
        if (showcaseIndex >= 0) {
            notify("Showcase running — it finishes on its own.", long = false)
            return
        }
        if (intent.getStringExtra(EXTRA_REPLAY_CLIP) != null) {
            cameraFacing = facing
            notify(R.string.camera_switch_note)
            return
        }
        cameraFacing = facing
        val provider = cameraProvider
        if (!inputRunning || provider == null || destroyed || !foreground) {
            notify(R.string.camera_switch_note, long = false)
            return
        }
        try {
            provider.unbind(*cameraUseCases)
            val epoch = inputEpoch.get()
            val preview = previewUseCase()
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
        lastFraming = null
        renderSetupChecks()
        readyStatus()
    }

    private fun cameraSelector() = CameraSelector.Builder().requireLensFacing(cameraFacing).build()

    /**
     * 4:3 preview to match the 640x480 analysis stream: with fitCenter the screen
     * shows the same field of view the pose model analyzes (no hidden crop).
     */
    @Suppress("DEPRECATION")
    private fun previewUseCase(): Preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3).build()
        .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

    /** Reads the same strings shown on screen, so the player can stay on court. */
    private fun speak(record: ServeRecord, force: Boolean = false) {
        if (!speakResults && !force) return
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
                postUi { updateModeLabel()
                    setStatus("Comparison model ready", "${mode.name.lowercase()} selected. Reset does not change the model.") }
            } catch (e: Exception) {
                postUi { setStatus("Model unchanged", "Could not load the comparison model. Your previous model remains active.") }
            }
        }
    }

    private fun showPermissionRequired() {
        inputRunning = false
        binding.recoveryButton.visibility = View.VISIBLE
        setStatus(getString(R.string.camera_permission_title), getString(R.string.camera_permission_note), pill = Pill.ALERT)
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
        binding.recoveryButton.visibility = View.VISIBLE
        setStatus(title, detail, pill = Pill.ALERT)
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
        // A new input source starts unchecked; never keep a green outline from before.
        lastFraming = null
        renderSetupChecks()
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
        postFraming(FramingCheck.Result.NO_PLAYER)
        if (gateActive && lastPoseSeenMs != 0L && now - lastPoseSeenMs >= 1500L) {
            resetAnalysis()
            postUi { showIncomplete()
                setStatus("Capture interrupted", getString(R.string.tracking_lost_note), pill = Pill.ALERT) }
        }
        if (sessionLive && !trackingLostShown && now - lastStatusMs >= 500L) {
            lastStatusMs = now
            postUi {
                if (binding.recoveryButton.visibility != View.VISIBLE) {
                    setStatus(getString(R.string.tracking_lost), getString(R.string.tracking_lost_note), pill = Pill.ALERT)
                    trackingLostShown = true
                }
            }
        }
    }

    private fun updateModeLabel() {
        val matched = if (gru?.momentMatchingEnabled == true) " · training-matched" else ""
        val label = when (mode) {
            RunMode.SINGLE -> "Single model · fastest$matched"
            RunMode.ENSEMBLE -> "Ensemble · 5 models$matched"
            RunMode.HYBRID -> "Hybrid · GRU + technique check$matched"
        }
        modeLabelText = label
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
            postUi { setStatus(statusLabel ?: "Recorded serve replay", "") }
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
    private var showcaseVerdictShown = false
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
            notify("Showcase already running.", long = false)
            return
        }
        if (!modelsReady) {
            notify("Models still loading. Try the showcase again in a moment.", long = false)
            return
        }
        val dir = File(filesDir, "latency_input")
        val clips = listOf("C002.mp4", "C005.mp4", "C023.mp4").filter { File(dir, it).isFile }
        if (clips.size < 3) {
            notify("Load C002, C005 and C023 first: Menu › Research tools › Recorded-file latency test › Load clips from phone storage.")
            return
        }
        applyMode(Screen.LIVE)
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
        showcaseVerdictShown = false
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
                // Advance on verdict, not on replay end: the full-res pass and
                // inference land seconds after the replay finishes. Keep the shown
                // result on screen briefly for the recording, then continue
                // hands-free. A serve that never completes still advances on timeout.
                val doneAt = SystemClock.uptimeMillis()
                fun step() {
                    if (destroyed || !foreground || showcaseIndex < 0 || showcaseIndex != idx) return
                    val elapsed = SystemClock.uptimeMillis() - doneAt
                    if ((showcaseVerdictShown && elapsed >= showcaseGapMs) || elapsed >= 15000) {
                        analysisExecutor.execute { resetAnalysis() }
                        showcaseIndex++
                        if (showcaseIndex < showcaseQueue.size) playShowcaseClip()
                        else finishShowcase()
                    } else binding.root.postDelayed({ step() }, 500)
                }
                step()
            },
        )
    }

    private fun finishShowcase() {
        showcaseIndex = -1
        showcaseQueue = emptyList()
        showcaseVerdictShown = false
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
        notify("Showcase complete. Live camera resumed.", long = false)
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
                val preview = previewUseCase()
                val analysis = ImageAnalysis.Builder().setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
                cameraExecutor.execute {
                    if (!foreground || epoch != inputEpoch.get()) return@execute
                    val pipeline = createPose(epoch).also { posePipeline = it }
                    if (!pipeline.start(applicationContext)) return@execute
                    postUi { readyStatus() }
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
        if (realFrame) {
            bitmap?.let { b ->
                val aspect = b.width.toFloat() / b.height
                if (aspect != frameAspect) { frameAspect = aspect
                    runOnUiThread { if (!destroyed) binding.framingGuide.sourceAspect = aspect } }
            }
            postFraming(FramingCheck.evaluate(landmarks))
        }
        if (trackingLostShown) postUi {
            if (trackingLostShown) { if (gateActive) setStatus(getString(R.string.status_analyzing), "", true) else readyStatus() }
        }
        // Setup check: pose drives the framing guide only. Idle stats still adapt so
        // the first live serve triggers against this person's standing noise.
        if (!sessionLive) {
            idleMean += 0.1f * (motion - idleMean)
            idleDev += 0.1f * (abs(motion - idleMean) - idleDev)
            return
        }
        if (realFrame && clipsEnabled && bitmap != null) clipRecorder.offer(bitmap, landmarks, timestampMs)
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
                clipRecorder.begin()
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
                if (showcaseIndex >= 0) Log.i(LATENCY_TAG, "serve_started clip=${showcaseQueue.getOrNull(showcaseIndex)} time=$now")
                shiftDetector?.let { s ->
                    landmarkSample(landmarks)?.let { (hx, bh) -> s.add(hx, bh) }
                }
                addPaddleSample(landmarks, bitmap, showcaseSrcIdx)
                postUi {
                    clearCards(analyzing = true)
                    setStatus(getString(R.string.status_analyzing), "", true)
                }
            } else {
                // absorb standing noise (and any recovery swing during refractory)
                // BEFORE comparing — the triggering frame itself never updates stats
                idleMean += 0.1f * (motion - idleMean)
                idleDev += 0.1f * (abs(motion - idleMean) - idleDev)
                // The last result's status stays up until the next serve starts.
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
            pendingClip = if (clipsEnabled) clipRecorder.finish(lastMotionPeakMs) else null
            updateFeedback(finalPrediction)
            gateActive = false
            lowMotionSinceMs = 0L
            refractoryUntilMs = now + refractoryMs
            serveWindow.clear()
            paddleExecutor.queue.clear()
            emaProbs = null
            shiftDetector?.clear()
            paddleStream?.clear()
            return
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
        // Card view of the same values (display wording only).
        val probs = lastProbs.takeIf { prediction != null }
        val serveCard = ServePresentation.serveTypeCard(prediction?.first, prediction?.second, probs)
        val hipCard = ServePresentation.hipCard(hipValue, hipPassed)
        val paddleCard = ServePresentation.paddleCard(angle, paddle, stream?.low, stream?.high)
        val partial = ServePresentation.isPartial(serveCard, hipCard, paddleCard)
        val cues = result?.let { ServePresentation.coachingText(it.messages) } ?: bodyText
        // Same condition as the "GENERAL MECHANICS" title: subtype coaching withheld below the gate.
        val generalCue = result?.status == FeedbackEngine.Status.CORRECTIVE &&
            (prediction?.second ?: 0f) < FeedbackEngine.MIN_CONFIDENCE
        val hipThreshold = shiftDetector?.thr
        val paddleLow = stream?.low
        val paddleHigh = stream?.high
        val notes = (result?.notifications ?: emptyList()) +
            listOfNotNull(getString(R.string.analysis_error).takeIf { lastInferenceFailed })
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
            elapsed = SessionStore.elapsedLabel(SystemClock.elapsedRealtime()),
            label = prediction?.first,
            probs = probs?.takeIf { it.size == 3 }?.toList(),
            hipValue = hipValue, hipPassed = hipPassed, hipThreshold = hipThreshold,
            paddleAngle = angle, paddleState = paddle, paddleLow = paddleLow, paddleHigh = paddleHigh,
            generalCue = generalCue,
            analysisMs = if (endMs != 0L) SystemClock.uptimeMillis() - endMs else null,
        )
        SessionStore.add(record)
        pendingClip?.let { SessionStore.putClip(record.number, it) }
        pendingClip = null
        lastServe = LastServe(prediction?.first, prediction?.second, hipPassed, paddle,
            independentMechanics, record.number, SystemClock.uptimeMillis())
        lastRecord = record
        speak(record)
        postUi(generation) {
            if (showcaseIndex >= 0) showcaseVerdictShown = true
            lastHipValue = hipValue; lastHipPassed = hipPassed
            lastPaddleAngle = angle; lastPaddleState = paddle
            renderResult(serveCard, prediction?.second, hipCard, paddleCard, cues, generalCue, notes,
                hipValue, hipThreshold, paddleLow, paddleHigh)
            val analysisMs = if (endMs != 0L) SystemClock.uptimeMillis() - endMs else null
            setStatus(ServePresentation.resultStatus(record.number, partial, analysisMs), "",
                pill = if (partial) Pill.PARTIAL else Pill.GOOD)
            binding.root.announceForAccessibility("$titleText. $bodyText")
            binding.root.performHapticFeedback(android.view.HapticFeedbackConstants.CONFIRM)
            if (intent.hasExtra(EXTRA_REPLAY_CLIP)) Log.i("FeedbackOutput", org.json.JSONObject()
                .put("status", result?.status?.name).put("text", bodyText).toString())
            val shown = SystemClock.uptimeMillis()
            if (endMs != 0L) Log.i(LATENCY_TAG, "feedback_shown shown=$shown end_to_shown=${shown-endMs}ms peak_to_shown=${shown-peakMs}ms serve_total=${shown-startMs}ms")
        }
    }

    /** Coaching header for a feedback result; shared by the live and late-landing paths. */
    private fun feedbackTitleFor(result: FeedbackEngine.Result?, confidence: Float?): String = when {
        result == null -> "COACHING UNAVAILABLE"
        result.status == FeedbackEngine.Status.INSUFFICIENT_DATA -> "MORE EVIDENCE NEEDED"
        result.status == FeedbackEngine.Status.CORRECTIVE && (confidence ?: 0f) < FeedbackEngine.MIN_CONFIDENCE -> "GENERAL MECHANICS · NEXT ATTEMPT"
        result.status == FeedbackEngine.Status.LOW_CONFIDENCE -> "MORE EVIDENCE NEEDED"
        else -> getString(R.string.coaching_title)
    }

    /** Take the saved landing when the scan is still fresh; drop it and warn when expired. */
    private fun consumePendingLanding(now: Long): LandingResult? {
        val saved = pendingLanding ?: return null
        pendingLanding = null
        if (now - saved.second > landingFreshMs) {
            postUi { notify(R.string.landing_expired) }
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
            notify(getString(R.string.landing_attached, result.zone, snap.recordNumber))
            analysisExecutor.execute { attachLanding(snap, result) }
        } else {
            pendingLanding = result to now
            notify(getString(R.string.landing_saved_next, result.zone))
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
        val general = re.status == FeedbackEngine.Status.CORRECTIVE &&
            (snap.confidence ?: 0f) < FeedbackEngine.MIN_CONFIDENCE
        if (!SessionStore.update(snap.recordNumber) {
            it.copy(landing = result.zone, feedbackTitle = title, feedback = re.text, generalCue = general)
        }) return
        Log.i("Feedback", "landing_attached serve=${snap.recordNumber} zone=${result.zone} status=${re.status}")
        // Same split as a fresh result: messages are the cue, notifications the note line.
        landingRefresh = LandingRefresh(ServePresentation.coachingText(re.messages), re.notifications, general)
        landingRefreshRecord = snap.recordNumber
        runOnUiThread { renderLandingRefresh() }
    }

    private class LandingRefresh(val cue: String, val notes: List<String>, val generalCue: Boolean)
    @Volatile private var landingRefresh: LandingRefresh? = null

    /** Refresh the feedback cards after a late landing; deferred to onResume when backgrounded. */
    private fun renderLandingRefresh() {
        val number = landingRefreshRecord ?: return
        if (destroyed || !foreground) return
        if (lastServe?.recordNumber != number) { landingRefreshRecord = null; return }
        val record = SessionStore.snapshot().find { it.number == number }
            ?: run { landingRefreshRecord = null; return }
        landingRefreshRecord = null
        val refresh = landingRefresh.also { landingRefresh = null }
        lastRecord = record
        // The record is already amended; only repaint when that serve's result is still on
        // screen (not after Try again, and not while the next serve is being analyzed).
        if (!resultShown || binding.statusProgress.visibility == View.VISIBLE || refresh == null) return
        setCue(refresh.cue.ifBlank { record.feedback })
        val noteText = refresh.notes.filter { it.isNotBlank() }.joinToString("\n")
        binding.feedbackNotes.text = noteText
        binding.feedbackNotes.visibility = if (noteText.isBlank()) View.GONE else View.VISIBLE
        binding.cueBadge.visibility = if (refresh.generalCue) View.VISIBLE else View.GONE
        setStatus("Serve ${record.number} · landing ${record.landing} attached", "")
        binding.root.announceForAccessibility("Landing ${record.landing} attached. ${record.feedback}")
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
        lastProbs = null
        lastInferenceFailed = false
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
            // Shown with the result by updateFeedback (final inference only, so
            // smoothed == combined here and the bars match the reported class).
            lastProbs = smoothed.copyOf()
            return g.predict(combined)
        } catch (e: Throwable) {
            Log.e("MainActivity", "Inference failed", e)
            lastInferenceFailed = true
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
        updateServesButton()
        if (SessionStore.size() == 0 && resultShown) { resultShown = false; clearCards() }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean("permissionRequested", permissionRequested)
        outState.putString("mode", mode.name)
        outState.putInt("cameraFacing", cameraFacing)
        outState.putBoolean("poseGpu", poseGpu)
        outState.putBoolean("speakResults", speakResults)
        outState.putString("screen", screen.name)
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
        cancelAutoStart()
        if (modelsReady && binding.statusProgress.visibility == View.VISIBLE && sessionLive) {
            clearCards()
            setCue("The attempt was interrupted. Start a new serve when you return.")
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
        setStatus(getString(R.string.paused), "")
        super.onStop()
    }

    override fun onDestroy() {
        destroyed = true
        sessionEpoch.invalidate()
        clipReplay?.stop()
        clipRecorder.close()
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
