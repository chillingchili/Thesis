package com.thesis.pickleballserve.landing

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.thesis.pickleballserve.databinding.ActivityLandingBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.RejectedExecutionException
import org.opencv.android.OpenCVLoader

class LandingActivity : AppCompatActivity() {
    private lateinit var binding: ActivityLandingBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var inferenceExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    // Interpreters belong to inferenceExecutor; tracking/decoder belong to cameraExecutor.
    private var yolo: YoloRunner? = null
    private val inferenceBusy = AtomicBoolean(false)
    private data class Detection(val generation: Int, val index: Int, val result: YoloRunner.FrameDet)
    private val results = ConcurrentLinkedQueue<Detection>()
    @Volatile private var generation = 0
    private var lastCourtRequestMs = -10_000L
    private var lastDetectionRequestMs = -10_000L
    private var realtime = RealtimeTracker()
    private val normalizer = FrameNormalizer()
    internal data class ReplayMetrics(val frames: Int, val elapsedMs: Long, val sourceMs: Long,
        val displayed: Int, val ballFrames: Int, val courtFrames: Int, val processingP95Ms: Double, val lateFrames: Int)
    @Volatile internal var replayMetrics: ReplayMetrics? = null
        private set
    private val displayedFrames = AtomicInteger()
    private var ballFrames = 0
    private var courtFrames = 0
    private val frameWorkMs = ArrayList<Double>()
    private val recordTrace by lazy { intent.getBooleanExtra("recordTrace", false) }
    private val ballSamples = HashMap<Int, Pair<Double, Double>>()
    @Volatile internal var replayBallSamples: Map<Int, Pair<Double, Double>> = emptyMap()
        private set
    private var tracker = BallTracker(30.0)
    private var courtTracker = CourtTracker(estimator = NativeHomography::estimate)
    private var fps = 30.0
    private var minI = 0
    private var frameIdx = 0
    private var firstCameraTimestamp: Long? = null
    private var hit: BounceDetector.Hit? = null
    private var hitZone: String? = null
    private var resultLogged = false
    @Volatile private var replayStarted = false
    @Volatile private var stopped = false
    @Volatile private var paused = false
    @Volatile private var detectionFailure: String? = null
    private val trail = ArrayList<Double>(160)
    private val uiUpdatePending = AtomicBoolean(false)

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted && !replayStarted && !stopped) startCamera()
            else if (!granted) Toast.makeText(this, "Camera permission required for live analysis", Toast.LENGTH_LONG).show()
        }

    private val pickVideo =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null && !stopped) startReplay(null, uri, 0.0)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLandingBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.root.keepScreenOn = true
        cameraExecutor = Executors.newSingleThreadExecutor()
        inferenceExecutor = Executors.newSingleThreadExecutor()
        binding.loadVideoButton.isEnabled = false
        binding.landingStatus.text = "Loading analysis models…"
        binding.loadVideoButton.setOnClickListener { pickVideo.launch(arrayOf("video/*")) }
        val video = intent.getStringExtra("video")
        val fpsExtra = intent.getFloatExtra("fps", 0f).toDouble()
        inferenceExecutor.execute {
            if (stopped) return@execute
            try {
                check(OpenCVLoader.initLocal()) { "Could not initialize frame tracking" }
                org.opencv.core.Core.setNumThreads(1)
                // Leave GPU headroom for Android rendering; no four-minute GPU compilation on this phone.
                android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
                yolo = YoloRunner.open(applicationContext, useGpu = false, cpuThreads = 2,
                    realtimeMode = true, onProgress = { status ->
                    postUi { binding.landingStatus.text = status }
                })
                postUi {
                    binding.loadVideoButton.isEnabled = true
                    if (video != null) {
                        startReplay(video, null, fpsExtra)
                    } else {
                        binding.landingStatus.text = "Live: starting camera…"
                        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                            startCamera()
                        } else requestPermission.launch(Manifest.permission.CAMERA)
                    }
                }
            } catch (e: Exception) {
                modelLoadFailed(e)
            } catch (e: LinkageError) {
                modelLoadFailed(e)
            }
        }
    }

    private fun modelLoadFailed(e: Throwable) {
        Log.e(TAG, "model load failed", e)
        postUi { binding.landingStatus.text = "MODEL ERR: ${e.message}" }
    }

    private fun resetTracking(sourceFps: Double, replay: Boolean) {
        generation++
        results.clear()
        realtime.close()
        realtime = RealtimeTracker(sourceFps)
        replayMetrics = null
        displayedFrames.set(0)
        ballFrames = 0
        courtFrames = 0
        frameWorkMs.clear()
        ballSamples.clear()
        replayBallSamples = emptyMap()
        lastCourtRequestMs = -10_000L
        lastDetectionRequestMs = -10_000L
        fps = sourceFps
        minI = if (replay) (fps * 0.5).toInt() else 0
        tracker = BallTracker(fps)
        courtTracker = CourtTracker(estimator = NativeHomography::estimate)
        frameIdx = 0
        firstCameraTimestamp = null
        hit = null
        hitZone = null
        resultLogged = false
        trail.clear()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            if (stopped || replayStarted) return@addListener
            try {
                val provider = providerFuture.get()
                cameraProvider = provider
                val preview = Preview.Builder().setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_4_3).build().also {
                    binding.landingPreview.scaleType = androidx.camera.view.PreviewView.ScaleType.FIT_CENTER
                    it.setSurfaceProvider(binding.landingPreview.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(cameraExecutor) { proxy -> analyzeProxy(proxy) }
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "camera start failed", e)
                binding.landingStatus.text = "Camera unavailable. You can still load a video."
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeProxy(proxy: ImageProxy) {
        var bitmap: Bitmap? = null
        try {
            if (replayStarted || stopped || paused) return
            val timestamp = proxy.imageInfo.timestamp
            val start = firstCameraTimestamp ?: timestamp.also { firstCameraTimestamp = it }
            // Keep source time when CameraX drops frames under load.
            val idx = maxOf(frameIdx, ((timestamp - start) * fps / 1_000_000_000.0).toInt())
            frameIdx = idx + 1
            bitmap = proxy.toBitmap()
            processFrame(idx, bitmap, proxy.imageInfo.rotationDegrees, live = true)
        } catch (e: Exception) {
            Log.e(TAG, "analyze error", e)
            postUi { binding.landingStatus.text = "ANALYSIS ERR: ${e.message}" }
        } finally {
            bitmap?.recycle()
            proxy.close()
        }
    }

    private fun processFrame(idx: Int, frame: Bitmap, rot: Int, live: Boolean, status: String? = null) {
        val startedNs = SystemClock.elapsedRealtimeNanos()
        val image = normalizer.prepare(frame, rot)
        tracker.advance(idx)
        realtime.advance(idx, image)
        consumeDetections()
        val court = realtime.currentCourt
        if (court != null && idx % 5 == 0) {
            courtTracker.track(court.kptsXy, BooleanArray(14) { court.kptConf[it] >= 0.4 })
        } else if (court == null) {
            courtTracker = CourtTracker(estimator = NativeHomography::estimate)
        }
        requestDetection(idx, image)
        val ball = realtime.currentBall
        if (recordTrace && ball != null) ballSamples[idx] = ball.cx to ball.cy
        if (ball != null) ballFrames++
        if (courtTracker.homography != null) courtFrames++
        realtime.saveHomography(courtTracker.homography)
        val wp = ball?.let { courtTracker.transform(it.cx, it.cy) }
        if (ball != null && wp != null && tracker.add(idx, ball.cx, ball.cy, wp[0], wp[1])) {
            if (wp[0] in -4.0..24.0 && wp[1] in -4.0..48.0) {
                trail.add(wp[0])
                trail.add(wp[1])
                while (trail.size > 160) { trail.removeAt(0); trail.removeAt(0) }
            }
        }
        publishFrame(image, 0, ball, live, status ?: "Frame $idx · ${if (court == null) "acquiring court" else "tracking court"} · landing=${hitZone ?: "pending"}")
        if (live && hit == null) {
            val window = (fps * 4).toInt()
            val start = maxOf(0, idx + 1 - window)
            val n = idx + 1 - start
            val result = BounceDetector.detect(
                toDoubleRange(tracker.ballY, start, n),
                toDoubleRange(tracker.worldX, start, n),
                toDoubleRange(tracker.worldY, start, n),
                maxOf(0, minI - start), fps,
            )
            if (result != null) {
                hit = result.copy(frame = result.frame + start)
                hitZone = Zone.classify(hit!!.xFt, hit!!.yFt)
                publishResult("live")
            }
        }
        if (!live) frameWorkMs.add((SystemClock.elapsedRealtimeNanos() - startedNs) / 1_000_000.0)
    }

    private fun consumeDetections() {
        while (true) {
            val update = results.poll() ?: break
            if (update.generation != generation) continue
            val observations = realtime.correct(update.index, update.result)
            realtime.currentCourt?.let { courtTracker.track(it.kptsXy, BooleanArray(14) { p -> it.kptConf[p] >= 0.4 }) }
            for (observation in observations) {
                tracker.correct(observation.index, observation.ball.cx, observation.ball.cy, observation.homography)
            }
        }
    }

    private fun requestDetection(idx: Int, image: Bitmap) {
        val now = SystemClock.elapsedRealtime()
        val region = realtime.suggestedRegion()
        if (now - lastDetectionRequestMs < if (region == null) 300 else 100) return
        if (stopped || paused || detectionFailure != null || !inferenceBusy.compareAndSet(false, true)) return
        val snapshot = image.copy(Bitmap.Config.ARGB_8888, false)
        val session = generation
        lastDetectionRequestMs = now
        val includeCourt = now - lastCourtRequestMs >= 3000 && (region == null || now - lastCourtRequestMs >= 5000)
        if (includeCourt) lastCourtRequestMs = now
        try { inferenceExecutor.execute {
            try {
                if (stopped || session != generation) return@execute
                val runner = yolo ?: return@execute
                val ball = runner.detectFrame(snapshot, includeCourt = false, ballRegion = region)
                if (recordTrace) Log.d("LandingTrace", "MODEL frame=$idx ball=${ball.ball}")
                if (!stopped && session == generation) results.add(Detection(session, idx, ball))
                if (includeCourt && !stopped && session == generation) {
                    val court = runner.detectFrame(snapshot, includeCourt = true, includeBall = false)
                    if (!stopped && session == generation) results.add(Detection(session, idx, court))
                }
            } catch (e: Exception) {
                Log.e(TAG, "background detection failed", e)
                detectionFailure = "DETECTION ERR: ${e.message}"
            } catch (e: LinkageError) {
                Log.e(TAG, "detector library failed", e)
                detectionFailure = "DETECTION ERR: ${e.message}"
            } finally { snapshot.recycle(); inferenceBusy.set(false) }
        } } catch (e: RejectedExecutionException) {
            snapshot.recycle()
            inferenceBusy.set(false)
            if (!stopped) throw e
        }
    }

    private fun publishFrame(frame: Bitmap, rotation: Int, ball: YoloRunner.BallDet?, live: Boolean, status: String) {
        if (stopped || !uiUpdatePending.compareAndSet(false, true)) return
        val width = if (rotation % 180 != 0) frame.height else frame.width
        val height = if (rotation % 180 != 0) frame.width else frame.height
        val kpts = courtTracker.emaKpts?.copyOf()
        val valid = courtTracker.validMask.copyOf()
        val trailSnapshot = trail.toDoubleArray()
        val hasResult = resultLogged
        val session = generation
        // Never display the decoder's borrowed bitmap, or mutate/recycle a displayed bitmap.
        val shown = if (live) null else if (rotation == 0) {
            frame.copy(Bitmap.Config.ARGB_8888, false)
        } else error("Display requires an oriented frame")
        runOnUiThread {
            try {
                if (stopped || session != generation || (live && replayStarted)) {
                    shown?.recycle()
                    return@runOnUiThread
                }
                shown?.let { binding.replayFrame.setImageBitmap(it) }
                displayedFrames.incrementAndGet()
                binding.landingOverlay.apply {
                    frameW = width
                    frameH = height
                    this.ball = ball
                    this.kpts = kpts
                    kptValid = valid
                    trailFt = trailSnapshot
                    invalidate()
                }
                if (!live || !hasResult) binding.landingStatus.text = detectionFailure ?: status
            } finally { uiUpdatePending.set(false) }
        }
    }

    private fun startReplay(path: String?, uri: Uri?, fpsExtra: Double) {
        replayStarted = true
        cameraProvider?.unbindAll()
        binding.landingPreview.visibility = View.GONE
        binding.replayFrame.visibility = View.VISIBLE
        binding.loadVideoButton.isEnabled = false
        binding.landingOverlay.hit = null
        binding.landingOverlay.hitZone = null
        binding.landingOverlay.ball = null
        binding.landingOverlay.kpts = null
        binding.landingOverlay.trailFt = doubleArrayOf()
        binding.landingOverlay.invalidate()
        binding.landingStatus.text = "Loading video…"
        cameraExecutor.execute {
            if (stopped) return@execute
            try {
                val retriever = MediaMetadataRetriever()
                val metadataFps: Double
                val metadataDurationMs: Double
                try {
                    if (path != null) retriever.setDataSource(path) else retriever.setDataSource(this, uri!!)
                    val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toDoubleOrNull() ?: 0.0
                    metadataDurationMs = duration
                    val frameCount = if (android.os.Build.VERSION.SDK_INT >= 28) {
                        retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toDoubleOrNull() ?: 0.0
                    } else 0.0
                    metadataFps = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toDoubleOrNull()
                        ?: if (duration > 0 && frameCount > 0) frameCount * 1000.0 / duration else 0.0
                } finally { retriever.release() }
                FastVideoDecoder(path, uri, this@LandingActivity, isCancelled = { stopped }, maxFrameRate = 30.0).use { decoder ->
                    val sourceFps = listOf(fpsExtra, decoder.frameRate, metadataFps, 30.0).first { it.isFinite() && it > 0 }
                    val f = sourceFps / kotlin.math.ceil(sourceFps / 30.5)
                    resetTracking(f, replay = true)
                    val estimatedTotal = (metadataDurationMs * f / 1000.0).toInt()
                    var i = 0
                    val startMs = SystemClock.elapsedRealtime()
                    var firstPts: Long? = null
                    var displayStartMs = 0L
                    var lateFrames = 0
                    while (!stopped) {
                        if (paused) {
                            val pausedAt = SystemClock.elapsedRealtime()
                            while (paused && !stopped) SystemClock.sleep(10)
                            displayStartMs += SystemClock.elapsedRealtime() - pausedAt
                        }
                        val frame = decoder.nextFrame() ?: break
                        if (firstPts == null) {
                            firstPts = decoder.presentationTimeUs
                            displayStartMs = SystemClock.elapsedRealtime()
                        }
                        val due = displayStartMs + (decoder.presentationTimeUs - firstPts!!) / 1000
                        while (!stopped && SystemClock.elapsedRealtime() < due) {
                            SystemClock.sleep(minOf(10, due - SystemClock.elapsedRealtime()).coerceAtLeast(1))
                        }
                        if (stopped) break
                        if (SystemClock.elapsedRealtime() - due > 50) lateFrames++
                        val progress = if (estimatedTotal > 0) "${minOf(100, i * 100 / estimatedTotal)}% · " else ""
                        val elapsed = maxOf(1L, SystemClock.elapsedRealtime() - startMs)
                        val speed = i * 1000.0 / elapsed
                        processFrame(i, frame, decoder.rotation, live = false,
                            status = "Playing + tracking · ${progress}frame $i · ${"%.1f".format(speed)} frames/s")
                        // The playback clock begins when the first prepared frame is published,
                        // excluding one-time decoder/OpenCV setup from all subsequent deadlines.
                        if (i == 0) displayStartMs = SystemClock.elapsedRealtime()
                        i++
                    }
                    if (stopped) return@use
                    val elapsed = maxOf(1L, SystemClock.elapsedRealtime() - startMs)
                    Log.i(TAG, "Processed $i frames in ${elapsed}ms. FPS: ${i * 1000.0 / elapsed}")
                    val sortedWork = frameWorkMs.sorted()
                    replayBallSamples = ballSamples.toMap()
                    val sourceMs = if (firstPts == null) 0L else
                        (decoder.presentationTimeUs - firstPts!!) / 1000 + Math.round(1000.0 / f)
                    replayMetrics = ReplayMetrics(i, elapsed, sourceMs,
                        displayedFrames.get(), ballFrames, courtFrames,
                        sortedWork.getOrElse((sortedWork.size * 0.95).toInt()) { 0.0 }, lateFrames)
                    Log.i(TAG, "BENCHMARK $replayMetrics")
                    // Incorporate the last in-flight observation before deciding the final landing.
                    val drained = inferenceExecutor.submit { }
                    while (!drained.isDone && !stopped) { consumeDetections(); SystemClock.sleep(5) }
                    consumeDetections()
                    if (stopped) return@use
                    hit = BounceDetector.detect(
                        toDouble(tracker.ballY, i), toDouble(tracker.worldX, i), toDouble(tracker.worldY, i), minI, f)
                    hitZone = hit?.let { Zone.classify(it.xFt, it.yFt) }
                    publishResult("replay total=$i")
                }
            } catch (e: Exception) {
                Log.e(TAG, "replay failed", e)
                postUi { binding.landingStatus.text = "REPLAY ERR: ${e.message}" }
            } finally {
                realtime.close()
                postUi { binding.loadVideoButton.isEnabled = true }
            }
        }
    }

    private fun publishResult(prefix: String) {
        if (resultLogged || stopped) return
        resultLogged = true
        val result = hit
        val zone = hitZone
        val line = if (result == null) "LANDING_RESULT none ($prefix)" else {
            "LANDING_RESULT frame=${result.frame} x=${"%.2f".format(result.xFt)} " +
                "y=${"%.2f".format(result.yFt)} zone=$zone ($prefix)"
        }
        Log.i(TAG, line)
        postUi {
            binding.landingOverlay.hit = result
            binding.landingOverlay.hitZone = zone
            binding.landingOverlay.invalidate()
            binding.landingStatus.text = detectionFailure ?: line.removePrefix("LANDING_RESULT ")
        }
    }

    private fun postUi(action: () -> Unit) { runOnUiThread { if (!stopped) action() } }
    private fun toDoubleRange(src: FloatArray, start: Int, n: Int) = DoubleArray(n) { src[start + it].toDouble() }
    private fun toDouble(src: FloatArray, n: Int) = DoubleArray(n) { src[it].toDouble() }

    override fun onPause() { paused = true; super.onPause() }
    override fun onResume() { super.onResume(); paused = false }

    override fun onDestroy() {
        stopped = true
        replayStarted = true
        cameraProvider?.unbindAll()
        // Queue cleanup behind the current frame; never close an interpreter during inference.
        cameraExecutor.execute { realtime.close(); normalizer.close() }
        cameraExecutor.shutdown()
        inferenceExecutor.execute { yolo?.close(); yolo = null }
        inferenceExecutor.shutdown()
        binding.replayFrame.setImageDrawable(null)
        super.onDestroy()
    }

    companion object {
        private const val TAG = "LandingActivity"
    }
}
