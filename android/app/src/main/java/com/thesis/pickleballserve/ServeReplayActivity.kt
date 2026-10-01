package com.thesis.pickleballserve

import android.graphics.PointF
import android.graphics.Bitmap
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.Executors
import kotlin.math.roundToLong

/** Replay an already recorded, trimmed serve with a fresh VIDEO tracker and exact sample times. */
class ServeReplayActivity : AppCompatActivity() {
    private val worker=Executors.newSingleThreadExecutor()
    private lateinit var status: TextView
    private lateinit var choose: Button
    private lateinit var save: Button
    @Volatile private var stopped=false
    private var comparison: String?=null
    private val pick=registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            choose.isEnabled=false;save.isEnabled=false;comparison=null
            status.text="Comparing the complete serve…"
            worker.execute {
                try {
                    val result=replay(uri)
                    runOnUiThread { if (!stopped) {
                        comparison=result.toString(2);status.text=result.getString("summary")
                        choose.isEnabled=true;save.isEnabled=true
                    } }
                } catch (e: Exception) { runOnUiThread { if (!stopped) {
                    status.text="Comparison unavailable: ${e.message}";choose.isEnabled=true
                } } }
            }
        }
    }
    private val export=registerForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        val text=comparison
        if (uri != null && text != null) worker.execute {
            try { contentResolver.openOutputStream(uri)?.bufferedWriter()?.use { it.write(text) }
                runOnUiThread { if (!stopped) status.append("\nComparison saved.") }
            } catch (e: Exception) { runOnUiThread { if (!stopped) status.append("\nSave failed: ${e.message}") } }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val layout=LinearLayout(this).apply { orientation=LinearLayout.VERTICAL;setPadding(24,24,24,24) }
        status=TextView(this).apply { text="Choose a video trimmed to one complete serve (up to 12 seconds).\nThis comparison keeps the beginning and end of the serve.";textSize=17f }
        choose=Button(this).apply { text="Choose serve video";setOnClickListener { pick.launch(arrayOf("video/*")) } }
        save=Button(this).apply { text="Save comparison";isEnabled=false;setOnClickListener { export.launch("serve-comparison.json") } }
        layout.addView(choose);layout.addView(save);layout.addView(status)
        setContentView(ScrollView(this).apply { addView(layout) })
    }

    private fun replay(uri: Uri): JSONObject {
        if (Build.VERSION.SDK_INT < 28) throw IllegalArgumentException("Recorded replay requires Android 9 or later")
        val hash=MessageDigest.getInstance("SHA-256")
        contentResolver.openInputStream(uri)!!.use { input ->
            val block=ByteArray(65536)
            while (true) { val n=input.read(block);if (n < 0) break;hash.update(block,0,n) }
        }
        val extractor=MediaExtractor()
        val retriever=MediaMetadataRetriever()
        val frames=ArrayList<ServeSequence.Frame>()
        val records=JSONArray()
        var aspect=1.0
        try {
            extractor.setDataSource(this,uri,null)
            val track=(0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/")==true
            } ?: error("No video track")
            extractor.selectTrack(track)
            val pts=ArrayList<Long>()
            while (extractor.sampleTime >= 0) { pts.add(extractor.sampleTime);if (!extractor.advance()) break }
            val times=pts.distinct().sorted()
            require(times.size >= 2 && times.size == pts.size) { "Unsupported duplicate video timestamps" }
            require(times.last()-times.first() <= 12_000_000L) { "Trim the video to one serve of at most 12 seconds" }
            retriever.setDataSource(this,uri)
            val count=retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()
            require(count == times.size) { "Video frame count and sample timestamps differ; export a standard MP4 clip" }
            val options=PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build())
                .setRunningMode(RunningMode.VIDEO).setNumPoses(1)
                .setMinPoseDetectionConfidence(.5f).setMinPosePresenceConfidence(.5f).setMinTrackingConfidence(.5f).build()
            PoseLandmarker.createFromOptions(this,options).use { pose ->
                var lastMs=-1L
                for (i in times.indices) {
                    check(!stopped) { "Comparison cancelled" }
                    val bitmap=retriever.getFrameAtIndex(i,MediaMetadataRetriever.BitmapParams().apply {
                        preferredConfig=Bitmap.Config.ARGB_8888
                    }) ?: error("Cannot decode frame $i")
                    try {
                        aspect=bitmap.width.toDouble()/bitmap.height
                        val timeMs=(times[i]-times.first())/1000.0
                        val taskMs=maxOf(lastMs+1,timeMs.roundToLong());lastMs=taskMs
                        val image=BitmapImageBuilder(bitmap).build()
                        val result=try { pose.detectForVideo(image,taskMs) } finally { image.close() }
                        val landmarks=result.landmarks().firstOrNull()
                        val xy=landmarks?.let { lm -> FloatArray(66) { k -> if (k%2==0) lm[k/2].x() else lm[k/2].y() } }
                        val vis=landmarks?.let { lm -> FloatArray(33) { lm[it].visibility().orElse(0f) } }
                        val angles=landmarks?.let { lm -> JointAngles.fromLandmarks(Array(33) { PointF(lm[it].x(),lm[it].y()) }) }
                        frames.add(ServeSequence.Frame(timeMs,angles,xy,vis))
                        records.put(JSONObject().put("timestamp_ms",timeMs).put("task_timestamp_ms",taskMs)
                            .put("xy",xy?.let { JSONArray(it.map(Float::toDouble)) } ?: JSONObject.NULL)
                            .put("visibility",vis?.let { JSONArray(it.map(Float::toDouble)) } ?: JSONObject.NULL)
                            .put("angles",angles?.let { JSONArray(it.map(Float::toDouble)) } ?: JSONObject.NULL))
                    } finally { bitmap.recycle() }
                }
            }
        } finally { extractor.release();retriever.release() }
        val sequence=ServeSequence.build(frames,aspect)
        val predictions=JSONObject()
        val summary=StringBuilder("Source frames: ${sequence.sourceFrames}\nUsable frames: ${sequence.validSourceFrames}\nComplete-serve quality: ${if(sequence.accepted) "passed" else "insufficient"}\n")
        val knn=KnnClassifier.load(this)
        val gru=GruClassifier.load(this,ensemble=true)
        try {
            val valid=frames.mapNotNull { it.angles }
            for ((name,source) in listOf("old_first128" to valid.take(128),"old_last128" to valid.takeLast(128))) {
                if (source.size < 32) continue
                val tensor=Array(1) { Array(128) { t -> source.getOrNull(t)?.copyOf() ?: FloatArray(10) } }
                val g=gru.infer(tensor,source.size)
                val k=knn.predictProba(TimingFeatures.fromFlatWindow(tensor[0].flatMap { it.asIterable() }.toFloatArray(),source.size))
                val p=HybridPrediction.combine(g,k)
                predictions.put(name,JSONObject().put("gru",JSONArray(g.map(Float::toDouble))).put("knn",JSONArray(k.map(Float::toDouble))).put("hybrid",JSONArray(p.map(Float::toDouble))))
                val (label,conf)=gru.predict(p);summary.append("$name: $label (${"%.1f".format(conf*100)}%)\n")
            }
        } finally { gru.close() }
        if (sequence.accepted) {
            val names=assets.list("serve_research")?.toSet() ?: emptySet()
            if ("manifest.json" in names) {
                val manifest=JSONObject(assets.open("serve_research/manifest.json").bufferedReader().use { it.readText() })
                val modelBytes=assets.open("serve_research/model.tflite").use { it.readBytes() }
                val model=Interpreter(ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder()).apply { put(modelBytes);rewind() },Interpreter.Options().setNumThreads(2))
                val feature=sequence.features(manifest.getString("condition"))
                val gp=FloatArray(3)
                try { model.run(arrayOf(feature),arrayOf(gp)) } finally { model.close() }
                val bank=KnnClassifier.fromBytes(assets.open("serve_research/knn_meta.json").bufferedReader().use { it.readText() },assets.open("serve_research/knn_train.bin").use { it.readBytes() })
                val kp=bank.predictProba(TimingFeatures.fromFlatWindow(sequence.angles.flatMap { it.asIterable() }.toFloatArray(),128))
                val policy=ProbabilityPolicy.fromJson(manifest.getJSONObject("policy"))
                val p=policy.apply(gp,kp)
                val index=p.indices.maxByOrNull { p[it] }!!
                predictions.put("research_complete_serve",JSONObject().put("condition",manifest.getString("condition"))
                    .put("model_sha256",manifest.getString("model_sha256"))
                    .put("selection_sha256",manifest.getString("selection_sha256"))
                    .put("gru",JSONArray(gp.map(Float::toDouble))).put("knn",JSONArray(kp.map(Float::toDouble)))
                    .put("calibrated_hybrid",JSONArray(p.map(Float::toDouble))).put("threshold",policy.threshold ?: JSONObject.NULL))
                summary.append("Research complete serve: ${bank.classes[index]} (${"%.1f".format(p[index]*100)}%)\n")
                if (policy.threshold == null) summary.append("Confidence acceptance is disabled by validation.\n")
            } else summary.append("Research model bundle is unavailable.\n")
        }
        return JSONObject().put("version","serve_phone_replay_v1").put("pose_mode","VIDEO")
            .put("video_sha256",hash.digest().joinToString("") { "%02x".format(it) })
            .put("aspect",aspect).put("frames",records).put("quality_accepted",sequence.accepted)
            .put("angle_window",JSONArray(sequence.angles.map { JSONArray(it.map(Float::toDouble)) }))
            .put("predictions",predictions).put("summary",summary.toString())
    }

    override fun onDestroy() { stopped=true;worker.shutdown();super.onDestroy() }
}
