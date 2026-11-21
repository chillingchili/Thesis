package com.thesis.pickleballserve

import android.os.Bundle
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import com.thesis.pickleballserve.databinding.ActivityToolBinding
import java.io.File
import java.util.concurrent.Executors

class LatencyBenchmarkActivity : AppCompatActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    @Volatile private var cancelled = false
    @Volatile var finished = false
        private set
    @Volatile var failure: String? = null
        private set
    var resultFile: File? = null
        private set
    private lateinit var status: TextView
    private lateinit var run: Button
    private lateinit var runGpu: Button
    private lateinit var save: Button
    private val export = registerForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        if (uri != null) resultFile?.let { file -> contentResolver.openOutputStream(uri)?.use { stream -> file.inputStream().use { it.copyTo(stream) } } }
    }
    private val pickClips = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isEmpty()) return@registerForActivityResult
        val dir = File(filesDir, "latency_input").apply { mkdirs() }
        var copied = 0
        for (uri in uris) {
            val name = contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (idx >= 0 && cursor.moveToFirst()) cursor.getString(idx) else null
            } ?: continue
            try {
                contentResolver.openInputStream(uri)?.use { input ->
                    File(dir, File(name).name).outputStream().use { input.copyTo(it) }
                }
                copied++
            } catch (e: Exception) { status.text = "Could not copy $name: ${e.message}" }
        }
        val have = listOf("C001.mp4", "C002.mp4", "C007.mp4").filter { File(dir, it).isFile }
        val missing = listOf("C001.mp4", "C002.mp4", "C007.mp4").filter { !File(dir, it).isFile }
        status.text = "Copied $copied file(s). Ready: ${have.joinToString(", ").ifEmpty { "none" }}." +
            (if (missing.isEmpty()) " All 3 benchmark clips present." else " Still missing: ${missing.joinToString(", ")}.")
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = getString(R.string.latency_tool)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val ui = ActivityToolBinding.inflate(layoutInflater)
        setContentView(ui.root)
        enterImmersive(ui.root)
        ui.toolBack.setOnClickListener { finish() }
        ui.toolTitle.setText(R.string.latency_tool)
        ui.toolLead.setText(R.string.latency_lead)
        ui.toolStatusTitle.setText(R.string.tool_status)
        ui.toolFoot.setText(R.string.latency_foot)
        status = ui.toolStatus
        val dir = File(filesDir, "latency_input")
        status.text = listOf("C001.mp4", "C002.mp4", "C007.mp4").joinToString("\n") {
            if (File(dir, it).isFile) "$it · ready" else "$it · missing"
        }
        fun add(button: Button) {
            ui.toolButtons.addView(button, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = (8 * resources.displayMetrics.density).toInt() })
        }
        val outlined = com.google.android.material.R.attr.materialButtonOutlinedStyle
        val load = MaterialButton(this, null, outlined).apply { setText(R.string.latency_load); setOnClickListener { pickClips.launch(arrayOf("video/*")) } }
        run = MaterialButton(this).apply { setText(R.string.latency_run); setOnClickListener { start(poseGpu=false) } }
        runGpu = MaterialButton(this, null, outlined).apply { setText(R.string.latency_run_gpu); setOnClickListener { start(poseGpu=true) } }
        save = MaterialButton(this, null, outlined).apply { setText(R.string.latency_export); isEnabled=false; setOnClickListener { export.launch(resultFile?.name ?: "latency.json") } }
        add(load); add(run); add(runGpu); add(save)
        if (intent.getBooleanExtra("autostart", false)) start()
    }
    private fun start(poseGpu: Boolean = false) {
        run.isEnabled=false; runGpu.isEnabled=false; save.isEnabled=false; finished=false; failure=null
        worker.execute {
            try {
                val files=listOf("C001.mp4","C002.mp4","C007.mp4").map { File(filesDir,"latency_input/$it") }
                val result=CoreLatencyBenchmark(this).run(files, { cancelled }, { message -> runOnUiThread { if(!cancelled) status.text=message } }, poseGpu)
                resultFile=result
                runOnUiThread { if(!cancelled) { status.text="Test complete (${if (poseGpu) "GPU pose" else "CPU pose"}). Export results for analysis.\n${result.name}\nLive two-second target remains unmeasured."; save.isEnabled=true } }
            } catch(e:Exception) {
                failure=e.message ?: e.javaClass.simpleName
                runOnUiThread { if(!cancelled) status.text="Test stopped: $failure. Partial records remain saved." }
            } finally { finished=true; runOnUiThread { if(!cancelled) { run.isEnabled=true; runGpu.isEnabled=true } } }
        }
    }
    override fun onDestroy() { cancelled=true; worker.shutdown(); super.onDestroy() }
}
