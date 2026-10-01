"""Compare exported court coordinate precision on the same serve frame."""
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

base = Path(__file__).resolve().parents[1]
models = [base / 'android/app/src/main/assets/court_ft_int8.tflite',
          base / 'outputs/mobile_candidates/rejected_apk_assets/court_320_int8.tflite',
          base / 'outputs/mobile_candidates/court320_w8a32/best_w8a32.tflite']
cap = cv2.VideoCapture(str(base / 'android/app/src/androidTest/assets/serve-replay.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES, 93)
_, frame = cap.read()
frame = cv2.resize(frame, (640, 360))
for path in models:
    if not path.exists():
        continue
    model = tf.lite.Interpreter(model_path=str(path), num_threads=4)
    model.allocate_tensors()
    inp = model.get_input_details()[0]
    size = int(inp['shape'][2])
    scale = size / 640
    top = round((size - 360 * scale) / 2 - 0.1)
    image = np.full((size, size, 3), 114, dtype=np.uint8)
    resized = cv2.resize(frame, (size, round(360 * scale)))
    image[top:top + resized.shape[0]] = resized
    value = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    model.set_tensor(inp['index'], value.transpose(2, 0, 1)[None])
    model.invoke()
    result = model.get_tensor(model.get_output_details()[0]['index'])[0]
    prediction = result[:, result[4].argmax()]
    kpts = prediction[5:].reshape(14, 3).copy()
    kpts[:, 0] *= 640
    kpts[:, 1] = (kpts[:, 1] * size - top) / scale
    print(path.name, 'size=', size, 'conf=', prediction[4], '\nkpts=', kpts.tolist(), flush=True)
    world = np.array([[0,0],[20,0],[20,44],[0,44],[0,15],[20,15],[0,29],[20,29],
                      [10,0],[10,22],[10,29],[10,44],[0,22],[20,22]], dtype=np.float32)
    valid = kpts[:,2] >= .4
    matrix, mask = cv2.findHomography(kpts[valid,:2], world[valid], cv2.RANSAC, 3)
    print('projected ball', cv2.perspectiveTransform(np.array([[[481.,187.5]]],np.float32), matrix),
          'inliers', mask.reshape(-1).tolist(), flush=True)
