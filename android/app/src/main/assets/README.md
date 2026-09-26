# Assets

Place these files in this directory before building:

## Required: GRU TFLite models (already copied from models/tflite/)
- gru_fold1.tflite
- gru_fold2.tflite
- gru_fold3.tflite
- gru_fold4.tflite
- gru_fold5.tflite
- gru_single.tflite
- manifest.json

If missing, re-export from project root:
```
python scripts/export_tflite.py
copy models\tflite\*.tflite android\app\src\main\assets\
copy models\tflite\manifest.json android\app\src\main\assets\
```

## Required: MediaPipe Pose Landmarker task file
Download ONE of these and place in this folder:

- pose_landmarker_lite.task  (fastest, recommended for demo)
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

- pose_landmarker_full.task  (more accurate, slower)
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

PowerShell example (lite):
```
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "android\app\src\main\assets\pose_landmarker_lite.task"
```

The app expects `pose_landmarker_lite.task` by default (see PosePipeline.kt MODEL_ASSET).
