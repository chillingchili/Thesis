"""Held-out test-split evaluation for ball + court models (CPU)."""
from ultralytics import YOLO

ball = YOLO(r"runs\detect\ball_yolo26s\weights\best.pt")
rb = ball.val(
    data=r"datasets\ball\data.yaml",
    split="test",
    imgsz=640,
    device="cpu",
    batch=16,
    name="ball_test",
    exist_ok=True,
)
print(
    "BALL TEST: "
    f"P={rb.box.mp:.4f} R={rb.box.mr:.4f} mAP50={rb.box.map50:.4f} mAP50-95={rb.box.map:.4f}",
    flush=True,
)

court = YOLO(r"runs\pose\court_yolo26s\weights\best.pt")
rc = court.val(
    data=r"datasets\court\data.yaml",
    split="test",
    imgsz=640,
    device="cpu",
    batch=16,
    name="court_test",
    exist_ok=True,
)
print(
    "COURT TEST: "
    f"box mAP50={rc.box.map50:.4f} | "
    f"pose P={rc.pose.mp:.4f} R={rc.pose.mr:.4f} mAP50={rc.pose.map50:.4f} mAP50-95={rc.pose.map:.4f}",
    flush=True,
)
