package com.thesis.pickleballserve.landing

import android.graphics.Bitmap
import com.google.zxing.BarcodeFormat
import com.google.zxing.BinaryBitmap
import com.google.zxing.RGBLuminanceSource
import com.google.zxing.common.BitMatrix
import com.google.zxing.common.HybridBinarizer
import com.google.zxing.qrcode.QRCodeReader
import com.google.zxing.qrcode.QRCodeWriter

/**
 * Offline QR encode/decode for the dual-phone landing transfer.
 * ZXing core only: no network, no Play Services, thesis-offline compliant.
 *
 * [decodePixels] is pure JVM so the round trip is unit-tested;
 * [decodeBitmap]/[encodeBitmap] are thin Android wrappers over it.
 */
object QrCode {
    fun encodeMatrix(text: String, size: Int = 264): BitMatrix =
        QRCodeWriter().encode(text, BarcodeFormat.QR_CODE, size, size)

    fun encodeBitmap(text: String, size: Int = 512): Bitmap {
        val matrix = encodeMatrix(text, size.coerceIn(264, 1024))
        val bitmap = Bitmap.createBitmap(matrix.width, matrix.height, Bitmap.Config.ARGB_8888)
        for (y in 0 until matrix.height) {
            for (x in 0 until matrix.width) {
                bitmap.setPixel(x, y, if (matrix[x, y]) 0xFF000000.toInt() else 0xFFFFFFFF.toInt())
            }
        }
        return bitmap
    }

    /** Decoded text, or null when no QR code is visible. Never throws. */
    fun decodeBitmap(bitmap: Bitmap): String? {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return decodePixels(pixels, bitmap.width, bitmap.height)
    }

    fun decodePixels(pixels: IntArray, width: Int, height: Int): String? = try {
        val source = RGBLuminanceSource(width, height, pixels)
        QRCodeReader().decode(BinaryBitmap(HybridBinarizer(source))).text
    } catch (_: Exception) {
        null
    }

    fun matrixToPixels(matrix: BitMatrix): IntArray {
        val out = IntArray(matrix.width * matrix.height)
        for (y in 0 until matrix.height) {
            for (x in 0 until matrix.width) {
                out[y * matrix.width + x] = if (matrix[x, y]) 0xFF000000.toInt() else 0xFFFFFFFF.toInt()
            }
        }
        return out
    }
}
