# Xiaomi paddle preprocessing optimization

Installed and tested on connected Xiaomi 23021RAAEG, Android 14. Crop preparation mean fell from 5.199 ms to 1.376 ms (73.5% reduction). This is only the grayscale/crop stage, not full inference or end-to-end latency.

The app now converts the detected paddle region plus a one-pixel border, instead of the full image. The border preserves OpenCV 3x3 Gaussian blur behavior. The returned Mat retains its backing allocation; caller-owned full-frame bitmaps are not recycled.

Checked 25 detected crops from nine frames across C001, C002 and C007. Twenty timed repetitions per path, per crop; alternating order after warm-up. Pixel equality and paddle-angle/status equality passed exactly. Frames with no detections remained in the test but produced no timing rows.

Android build, JVM tests and lint passed. Three device regression tests passed. No model weights, thresholds, frame sampling, serve end detection or frozen Coach C feedback changed.

## Alternatives tested

- threads_1_2_4.json: 1 threads: 1197.5 ms, 2 threads: 784.3 ms, 4 threads: 596.2 ms.
- threads_4_6_8.json: 4 threads: 580.7 ms, 6 threads: 609.8 ms, 8 threads: 607.1 ms.
- Retained four CPU threads. Other tested counts were slower. Detection counts, boxes (0.1 pixel tolerance) and confidence (0.0001 tolerance) passed across the thread comparisons.
- Rejected experimental expansion of quantized weights: output values changed in a desktop smoke check (maximum absolute tensor difference 0.405). Candidate was never installed or made the default.
- Initial crop implementation passed pixel equality but failed angle equality because OpenCV reads outside an ROI during blur. The final implementation preserves the needed border and passed both checks.

## Limits and next work

This targeted test does not establish a new full-video time or the live two-second thesis threshold. The earlier 68.6-second median remains the historical baseline, not a measurement of this APK. The saving is roughly 3.82 ms per converted detected crop; neural paddle inference still dominates. Larger gains require changing inference scheduling or validating a lighter detector, with separate checks for missing paddle measurements and feedback changes.

Reproduce with direct instrumentation using PaddleCropTest and PaddleBitmapTest. PaddleThreadsTest accepts the instrumentation argument `threads` such as `4,2,1` or `4,6,8`. Prepared private clips follow android/LATENCY_TEST.md. All comparisons ran on this phone; gains are not established for other devices.
