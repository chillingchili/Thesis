package com.thesis.pickleballserve

import android.media.MediaMetadataRetriever
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Rect
import java.io.File

class PaddleCropTest {
    @Test fun preservesPixelsAndAnglesWithLessConversionWork() {
        if (android.os.Build.VERSION.SDK_INT < 28) return
        assertTrue(OpenCVLoader.initLocal())
        val context=InstrumentationRegistry.getInstrumentation().targetContext
        val rows=JSONArray()
        val detector=PaddleDetector.open(context)
        try {
            for (id in listOf("C001","C002","C007")) {
                val reader=MediaMetadataRetriever()
                try {
                    reader.setDataSource(File(context.filesDir,"latency_input/$id.mp4").absolutePath)
                    val count=reader.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)!!.toInt()
                    for (quarter in 1..3) {
                        val bitmap=requireNotNull(reader.getFrameAtIndex(count*quarter/4))
                        try {
                            val detections=detector.detectAll(bitmap)
                            val rects=detections.map { PaddleGeometry.expandBox(it.box,bitmap.width,bitmap.height) }
                                .filter { it.width>=8 && it.height>=8 }

                            for(rect in rects) {
                                val full=PaddleStream.bitmapGray(bitmap)
                                val expected=Mat(full,rect)
                                val actual=PaddleStream.cropGray(bitmap,rect)
                                try {
                                    assertEquals(0.0,Core.norm(expected,actual,Core.NORM_INF),0.0)
                                    assertEquals(PaddleGeometry.cropAngle(expected),PaddleGeometry.cropAngle(actual))
                                } finally { actual.release();expected.release();full.release() }
                                // Warm both paths, then alternate order to limit order bias.
                                val elapsed=DoubleArray(2)
                                for(round in 0..20) for(path in if(round%2==0) listOf(0,1) else listOf(1,0)) {
                                    val start=SystemClock.elapsedRealtimeNanos()
                                    if(path==0) {
                                        val gray=PaddleStream.bitmapGray(bitmap); val crop=Mat(gray,rect)
                                        crop.release();gray.release()
                                    } else { PaddleStream.cropGray(bitmap,rect).release() }
                                    if(round>0) elapsed[path]+=(SystemClock.elapsedRealtimeNanos()-start)/1e6
                                }
                                rows.put(JSONObject().put("clip",id).put("quarter",quarter).put("source_pixels",bitmap.width*bitmap.height)
                                    .put("crop_pixels",rect.width*rect.height).put("old_mean_ms",elapsed[0]/20).put("new_mean_ms",elapsed[1]/20))
                            }
                            // Full-frame createBitmap may return the original; never recycle caller-owned input.
                            PaddleStream.cropGray(bitmap,Rect(0,0,bitmap.width,bitmap.height)).release()
                            assertFalse(bitmap.isRecycled)
                        } finally { bitmap.recycle() }
                    }
                } finally { reader.release() }
            }
            assertTrue("No paddle crops checked",rows.length()>0)
            File(context.filesDir,"latency_results/paddle-crops.json").writeText(JSONObject().put("completed",true)
                .put("pixels_and_angles_identical",true).put("repeats_per_path",20).put("runs",rows).toString(2))
        } finally { detector.close() }
    }
}
