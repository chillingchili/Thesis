package com.thesis.pickleballserve.landing

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer

/** Returns a borrowed, unrotated bitmap, valid until nextFrame/close. Copy it for display. */
class FastVideoDecoder(
    path: String?, uri: Uri?, context: Context,
    private val isCancelled: () -> Boolean = { false },
    private val maxFrameRate: Double? = null,
) : AutoCloseable {
    private val extractor = MediaExtractor()
    private var codec: MediaCodec? = null
    private var inputEnded = false
    private var outputEnded = false
    private val bufferInfo = MediaCodec.BufferInfo()
    private var lastSelectedUs: Long? = null
    
    var width = 0
        private set
    var height = 0
        private set
    var rotation = 0
        private set
    var durationUs = 0L
        private set
    var frameRate = 0.0
        private set
    var presentationTimeUs = 0L
        private set

    private var pixelBuffer: IntArray? = null
    private var cachedBitmap: Bitmap? = null
    private var yArray: ByteArray? = null
    private var uArray: ByteArray? = null
    private var vArray: ByteArray? = null

    private fun getArray(buffer: ByteBuffer, cache: ByteArray?): ByteArray {
        val plane = buffer.duplicate()
        plane.rewind()
        val size = plane.remaining()
        val arr = if (cache == null || cache.size < size) ByteArray(size) else cache
        plane.get(arr, 0, size)
        return arr
    }

    init {
        try {
        if (path != null) {
            extractor.setDataSource(path)
        } else {
            extractor.setDataSource(context, uri!!, null)
        }
        
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val mime = format.getString(MediaFormat.KEY_MIME)
            if (mime?.startsWith("video/") == true) {
                width = format.getInteger(MediaFormat.KEY_WIDTH)
                height = format.getInteger(MediaFormat.KEY_HEIGHT)
                durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0L
                if (format.containsKey(MediaFormat.KEY_ROTATION)) {
                    rotation = format.getInteger(MediaFormat.KEY_ROTATION)
                }
                if (format.containsKey(MediaFormat.KEY_FRAME_RATE)) {
                    frameRate = try { format.getInteger(MediaFormat.KEY_FRAME_RATE).toDouble() }
                        catch (_: ClassCastException) { format.getFloat(MediaFormat.KEY_FRAME_RATE).toDouble() }
                }
                // Rotation is applied exactly once, in preprocessing/display, not by the codec.
                format.setInteger(MediaFormat.KEY_ROTATION, 0)
                extractor.selectTrack(i)
                codec = MediaCodec.createDecoderByType(mime)
                // Configure to get YUV format Image output reliably across different SOCs
                format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                codec!!.configure(format, null, null, 0)
                codec!!.start()
                break
            }
        }
        requireNotNull(codec) { "The selected file has no video track" }
        } catch (e: Exception) {
            close()
            throw e
        }
    }

    fun nextFrame(): Bitmap? {
        val c = codec ?: return null
        val timeoutUs = 10000L
        
        while (!outputEnded && !isCancelled()) {
            // Keep the hardware decoder fed while tracking the previously returned frame.
            // Feeding just one input per output makes 60fps sources repeatedly wait for decode.
            var queued = 0
            while (!inputEnded && queued < 8 && !isCancelled()) {
                val inIndex = c.dequeueInputBuffer(0)
                if (inIndex < 0) break
                if (inIndex >= 0) {
                    val buffer = c.getInputBuffer(inIndex)
                    val size = extractor.readSampleData(buffer!!, 0)
                    if (size < 0) {
                        c.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputEnded = true
                    } else {
                        c.queueInputBuffer(inIndex, 0, size, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                    queued++
                }
            }
            
            val outIndex = c.dequeueOutputBuffer(bufferInfo, timeoutUs)
            if (outIndex >= 0) {
                // We got a frame!
                var bmp: Bitmap? = null
                try {
                    val select = maxFrameRate == null || lastSelectedUs == null ||
                        bufferInfo.presentationTimeUs - lastSelectedUs!! >= 1_000_000.0 / maxFrameRate - 500
                    val image = if (bufferInfo.size > 0 && select) {
                        requireNotNull(c.getOutputImage(outIndex)) { "Video decoder did not provide a YUV frame" }
                    } else null
                    if (image != null) {
                        try { bmp = yuvToBitmap(image) } finally { image.close() }
                    }
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputEnded = true
                    }
                } finally {
                    c.releaseOutputBuffer(outIndex, false)
                }
                if (bmp != null) {
                    presentationTimeUs = bufferInfo.presentationTimeUs
                    lastSelectedUs = presentationTimeUs
                    return bmp
                }
            } else if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                // Format changed
            }
        }
        return null
    }

    private fun yuvToBitmap(image: Image): Bitmap {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        
        yArray = getArray(yBuffer, yArray)
        uArray = getArray(uBuffer, uArray)
        vArray = getArray(vBuffer, vArray)
        val yBytes = yArray!!
        val uBytes = uArray!!
        val vBytes = vArray!!
        
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vPixelStride = vPlane.pixelStride
        
        val crop = image.cropRect
        val w = crop.width()
        val h = crop.height()
        
        // Fast downsampling if the video is huge (e.g. 1920x1080)
        // YOLO resizes everything to 640x640 anyway.
        val step = ((maxOf(w, h) + 639) / 640).coerceAtLeast(1)
        val outW = (w + step - 1) / step
        val outH = (h + step - 1) / step
        
        var pixels = pixelBuffer
        if (pixels == null || pixels.size != outW * outH) {
            pixels = IntArray(outW * outH)
            pixelBuffer = pixels
        }
        
        var pos = 0
        for (y in 0 until h step step) {
            val sourceY = crop.top + y
            val yOffset = sourceY * yRowStride
            val uvOffset = (sourceY / 2) * uRowStride
            val vRowOffset = (sourceY / 2) * vPlane.rowStride
            
            for (x in 0 until w step step) {
                val sourceX = crop.left + x
                var yVal = (yBytes[yOffset + sourceX * yPixelStride].toInt() and 0xFF) - 16
                if (yVal < 0) yVal = 0
                val uVal = (uBytes[uvOffset + (sourceX / 2) * uPixelStride].toInt() and 0xFF) - 128
                val vVal = (vBytes[vRowOffset + (sourceX / 2) * vPixelStride].toInt() and 0xFF) - 128
                
                val y1192 = 1192 * yVal
                var r = (y1192 + 1634 * vVal)
                var g = (y1192 - 833 * vVal - 400 * uVal)
                var b = (y1192 + 2066 * uVal)
                
                r = if (r < 0) 0 else if (r > 262143) 255 else r shr 10
                g = if (g < 0) 0 else if (g > 262143) 255 else g shr 10
                b = if (b < 0) 0 else if (b > 262143) 255 else b shr 10
                
                pixels[pos++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        
        var bmp = cachedBitmap
        if (bmp == null || bmp.width != outW || bmp.height != outH) {
            cachedBitmap?.recycle()
            bmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
            cachedBitmap = bmp
        }
        bmp.setPixels(pixels, 0, outW, 0, 0, outW, outH)
        
        return bmp
    }

    override fun close() {
        outputEnded = true
        try { codec?.stop() } catch (e: Exception) {}
        try { codec?.release() } catch (e: Exception) {}
        codec = null
        try { extractor.release() } catch (e: Exception) {}
        cachedBitmap?.recycle()
        cachedBitmap = null
    }
}
