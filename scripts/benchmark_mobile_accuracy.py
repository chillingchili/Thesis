"""Check small-input detection/ROI accuracy before deploying a mobile model.

Use existing labeled ball calibration images; retain the original checkpoints.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=30)
    args = parser.parse_args()
    model = YOLO(ROOT / 'runs/detect/ball_yolo26s/weights/best.pt')
    images = sorted((ROOT / 'datasets/ball_cal/images').glob('*.jpg'))
    images = images[::max(1, len(images) // args.n)][:args.n]
    stats = {str(size): {'matched': 0, 'labels': 0, 'false_positives': 0} for size in [640, 320]}
    roi = {'matched': 0, 'labels': 0}
    for path in images:
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        labels = []
        for line in (ROOT / 'datasets/ball_cal/labels' / (path.stem + '.txt')).read_text().splitlines():
            _, x, y, bw, bh = map(float, line.split())
            labels.append((x * w, y * h, bw * w, bh * h))
        for size in [640, 320]:
            result = model.predict(image, imgsz=size, conf=0.3, device=0, verbose=False)[0]
            centers = result.boxes.xywh.cpu().numpy()
            used = set()
            for x, y, bw, bh in labels:
                matches = [i for i, box in enumerate(centers) if i not in used and
                           abs(box[0] - x) <= max(5, bw) and abs(box[1] - y) <= max(5, bh)]
                if matches:
                    used.add(matches[0])
                    stats[str(size)]['matched'] += 1
                stats[str(size)]['labels'] += 1
            stats[str(size)]['false_positives'] += len(centers) - len(used)
        # A 160-pixel ROI at original 640-input scale retains the ball's pixel size.
        scale = 640.0 / max(w, h)
        small = cv2.resize(image, (round(w * scale), round(h * scale)))
        sh, sw = small.shape[:2]
        for x, y, bw, bh in labels:
            cx, cy = x * scale, y * scale
            left = min(max(0, round(cx) - 80), max(0, sw - 160))
            top = min(max(0, round(cy) - 80), max(0, sh - 160))
            crop = small[top:top + 160, left:left + 160]
            result = model.predict(crop, imgsz=160, conf=0.3, device=0, verbose=False)[0]
            centers = result.boxes.xywh.cpu().numpy()
            roi['labels'] += 1
            if any(abs(box[0] + left - cx) <= max(3, bw * scale) and
                   abs(box[1] + top - cy) <= max(3, bh * scale) for box in centers):
                roi['matched'] += 1
    report = {'images': len(images), 'full_frame': stats, 'roi160': roi}
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
