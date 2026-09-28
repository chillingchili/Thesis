"""Export trained models to int8 TFLite for Android (LiteRT)."""
from ultralytics import YOLO

jobs = [
    ("ball", r"runs\detect\ball_yolo26s\weights\best.pt", r"datasets\ball\data.yaml"),
    ("court", r"runs\pose\court_yolo26s\weights\best.pt", r"datasets\court\data.yaml"),
]

for name, weights, data in jobs:
    m = YOLO(weights)
    path = m.export(format="tflite", int8=True, imgsz=640, data=data)
    print(f"{name} TFLite -> {path}", flush=True)
