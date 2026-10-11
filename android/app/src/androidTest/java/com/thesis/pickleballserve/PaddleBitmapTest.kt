package com.thesis.pickleballserve

import android.graphics.Bitmap
import android.graphics.Color
import org.junit.Assert.*
import org.junit.Test
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

class PaddleBitmapTest {
    @Test fun thinEdgesUseFloatPointsInMinimumRectangleFallback() {
        assertTrue(OpenCVLoader.initLocal())
        val gray=Mat.zeros(100,100,CvType.CV_8UC1)
        try {
            Imgproc.line(gray,Point(15.0,15.0),Point(75.0,85.0),Scalar(255.0),1)
            Imgproc.line(gray,Point(75.0,85.0),Point(85.0,25.0),Scalar(255.0),1)
            val result=PaddleGeometry.cropAngle(gray)
            assertTrue(result.status in setOf("ok","deg","no_contour"))
            result.angle?.let { assertTrue(it.isFinite() && it >= 0 && it < 180) }
        } finally { gray.release() }
    }
    @Test fun argbPixelsConvertToGrayWithoutTypeErrorOrChannelSwap() {
        assertTrue(OpenCVLoader.initLocal())
        val bitmap=Bitmap.createBitmap(intArrayOf(Color.RED,Color.GREEN,Color.BLUE,Color.WHITE),4,1,Bitmap.Config.ARGB_8888)
        val gray=PaddleStream.bitmapGray(bitmap)
        try {
            assertEquals(CvType.CV_8UC1,gray.type())
            val expected=doubleArrayOf(76.0,150.0,29.0,255.0)
            for(i in expected.indices) assertEquals(expected[i],gray.get(0,i)[0],1.0)
            assertFalse(bitmap.isRecycled)
        } finally { gray.release();bitmap.recycle() }
    }
}
