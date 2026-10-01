"""Export trained models to int8 TFLite for Android (LiteRT).

Usage: python scripts/export_yolo_tflite.py [name ...]   (default: all)
"""
import sys

from ultralytics import YOLO

jobs = [
    ("ball", r"runs\detect\ball_yolo26s\weights\best.pt", r"datasets\ball\data.yaml"),
    ("court", r"runs\pose\court_ft\weights\best.pt", r"datasets\court_ft\data.yaml"),
    ("paddle", r"runs\detect\paddle_ft\weights\best.pt", r"datasets\paddle_ft\data.yaml"),
]

if len(sys.argv) > 1:
    jobs = [j for j in jobs if j[0] in sys.argv[1:]]

for name, weights, data in jobs:
    m = YOLO(weights)
    path = m.export(format="tflite", int8=True, imgsz=640, data=data)
    print(f"{name} TFLite -> {path}", flush=True)
