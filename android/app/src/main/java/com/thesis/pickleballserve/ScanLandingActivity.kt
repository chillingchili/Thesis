package com.thesis.pickleballserve

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.thesis.pickleballserve.databinding.ActivityScanLandingBinding
import com.thesis.pickleballserve.landing.LandingResult
import com.thesis.pickleballserve.landing.QrCode
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Serve-phone QR scanner for the dual-phone landing handoff.
 *
 * Decodes the landing phone's offline QR via CameraX + ZXing (no network).
 * Returns [EXTRA_LANDING] (a [LandingResult]-encoded payload) to MainActivity.
 * The manual zone picker is the sunlight-washout fallback: the operator reads
 * the zone off the landing phone's dialog and picks it here.
 */
class ScanLandingActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_LANDING = "landing_payload"
        private const val TAG = "ScanLanding"
        private const val DECODE_THROTTLE_MS = 300L
    }

    private lateinit var binding: ActivityScanLandingBinding
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private val finished = AtomicBoolean(false)
    private var lastDecodeMs = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityScanLandingBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.root.keepScreenOn = true
        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.cancelButton.setOnClickListener { finish() }
        binding.manualButton.setOnClickListener { showZonePicker() }
        startCamera()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (finished.get()) return@addListener
            try {
                val provider = future.get()
                cameraProvider = provider
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(binding.scanPreview.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(cameraExecutor, ::analyze)
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "camera start failed", e)
                binding.scanStatus.text = getString(R.string.camera_permission_note)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(proxy: ImageProxy) {
        var bitmap: Bitmap? = null
        try {
            if (finished.get()) return
            val now = SystemClock.elapsedRealtime()
            if (now - lastDecodeMs < DECODE_THROTTLE_MS) return
            lastDecodeMs = now
            bitmap = proxy.toBitmap()
            val text = QrCode.decodeBitmap(bitmap) ?: return
            val parsed = LandingResult.parse(text)
            if (parsed == null) {
                runOnUiThread {
                    if (!finished.get()) binding.scanStatus.text = getString(R.string.scan_bad_code)
                }
                return
            }
            finishWith(parsed.encode())
        } catch (e: Exception) {
            Log.e(TAG, "scan error", e)
        } finally {
            bitmap?.recycle()
            proxy.close()
        }
    }

    /** Sunlight fallback: pick the zone the landing phone displays. */
    private fun showZonePicker() {
        if (finished.get()) return
        val zones = LandingResult.ZONES.sorted().toTypedArray()
        MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.scan_manual))
            .setSingleChoiceItems(zones, -1) { dialog, which ->
                dialog.dismiss()
                finishWith(LandingResult(zones[which]).encode())
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun finishWith(payload: String) {
        if (!finished.compareAndSet(false, true)) return
        runOnUiThread {
            setResult(RESULT_OK, Intent().putExtra(EXTRA_LANDING, payload))
            finish()
        }
    }

    override fun onDestroy() {
        finished.set(true)
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
        super.onDestroy()
    }
}
