package com.thesis.pickleballserve

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.Assert.*
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MobileMethodsTest {
    @Test fun comparePoseCpuGpu() {
        val c=InstrumentationRegistry.getInstrumentation().targetContext
        val frames=listOf("C001","C002","C007").flatMap { id ->
            val r=MediaMetadataRetriever()
            try {
                r.setDataSource(File(c.filesDir,"latency_input/$id.mp4").absolutePath)
                val n=r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)!!.toInt()
                listOf(1,2,3).map { k ->
                    val b=requireNotNull(r.getFrameAtIndex(n*k/4))
                    val scale=minOf(640f/b.width,480f/b.height)
                    Bitmap.createScaledBitmap(b,(b.width*scale).toInt(),(b.height*scale).toInt(),true)
                }
            } finally { r.release() }
        }
        val report=JSONObject().put("protocol","Same nine frames, three clips; VIDEO mode, CPU/GPU, first pass warm-up then three passes; decoding/model init excluded. Coordinate disagreement is not ground-truth accuracy.")
        val rows=JSONArray();report.put("runs",rows)
        val out=File(c.filesDir,"latency_results/pose-cpu-gpu.json");out.parentFile!!.mkdirs()
        for (delegate in listOf(Delegate.CPU,Delegate.GPU)) {
            try {
                val options=PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").setDelegate(delegate).build())
                    .setRunningMode(RunningMode.VIDEO).setNumPoses(1).build()
                PoseLandmarker.createFromOptions(c,options).use { model ->
                    var ts=0L
                    for (round in 0..3) for ((i,b) in frames.withIndex()) {
                        val start=SystemClock.elapsedRealtimeNanos()
                        val result=model.detectForVideo(BitmapImageBuilder(b).build(),++ts*33)
                        val elapsed=(SystemClock.elapsedRealtimeNanos()-start)/1e6
                        val landmarks=JSONArray()
                        result.landmarks().firstOrNull()?.forEach { landmarks.put(JSONArray(listOf(it.x(),it.y(),it.visibility().orElse(0f)))) }
                        rows.put(JSONObject().put("delegate",delegate.name).put("round",round).put("warmup",round==0).put("frame",i).put("ms",elapsed).put("landmarks",landmarks))
                        out.writeText(report.toString(2))
                    }
                }
            } catch(e:Throwable) { report.put(delegate.name+"_error",e.toString());out.writeText(report.toString(2)) }
        }
        report.put("complete",true);out.writeText(report.toString(2))
        assertTrue(rows.length()>0)
    }

    @Test fun benchmarkResearchGruCandidates() {
        val c=InstrumentationRegistry.getInstrumentation().targetContext
        val dir=File(c.filesDir,"mobile_candidates")
        val flat=JSONArray(File(dir,"input.json").readText())
        val input=Array(1){Array(128){t->FloatArray(10){j->flat.getDouble(t*10+j).toFloat()}}}
        val rows=JSONArray();val report=JSONObject().put("runs",rows)
        for (name in listOf("mobile32","distilled16")) {
            val bytes=File(dir,"$name.tflite").readBytes()
            val buffer=ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).put(bytes).apply { rewind() }
            Interpreter(buffer,Interpreter.Options().setNumThreads(2)).use { model ->
                for (i in 0..30) {
                    val output=Array(1){FloatArray(3)}
                    val start=SystemClock.elapsedRealtimeNanos();model.run(input,output)
                    val elapsed=(SystemClock.elapsedRealtimeNanos()-start)/1e6
                    rows.put(JSONObject().put("model",name).put("warmup",i==0).put("ms",elapsed).put("probabilities",JSONArray(output[0].toList())))
                    assertTrue(output[0].all { it.isFinite() })
                }
            }
        }
        File(c.filesDir,"latency_results/mobile-models.json").writeText(report.put("complete",true).toString(2))
    }
}
