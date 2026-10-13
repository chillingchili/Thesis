package com.thesis.pickleballserve

import android.os.Bundle
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
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
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = getString(R.string.latency_tool)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val layout = LinearLayout(this).apply { orientation=LinearLayout.VERTICAL; setPadding(24,24,24,24) }
        val back = Button(this).apply { text=getString(R.string.back_to_live); setOnClickListener { finish() } }
        status=TextView(this).apply { text="Recorded-clip latency test: 3 prepared clips, 3 rounds, one warm-up. Includes decode, pose, hip, paddle, GRU and feedback. This does not measure live capture-to-display delay."; textSize=17f }
        run=Button(this).apply { text="Run latency test"; setOnClickListener { start(poseGpu=false) } }
        runGpu=Button(this).apply { text="Run GPU variant test"; setOnClickListener { start(poseGpu=true) } }
        save=Button(this).apply { text="Export results"; isEnabled=false; setOnClickListener { export.launch(resultFile?.name ?: "latency.json") } }
        layout.addView(back); layout.addView(run); layout.addView(runGpu); layout.addView(save); layout.addView(status)
        setContentView(ScrollView(this).apply { addView(layout) })
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
