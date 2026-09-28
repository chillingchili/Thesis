"""Render a court test image with numbered keypoints to identify kpt indices."""
from pathlib import Path

import cv2

imgs = sorted(Path(r"datasets\court\images\test").glob("*.jpg"))
lbls = sorted(Path(r"datasets\court\labels\test").glob("*.txt"))

out = Path("data")
for i, (im_p, lb_p) in enumerate(zip(imgs, lbls)):
    if i >= 3:
        break
    im = cv2.imread(str(im_p))
    line = lb_p.read_text().splitlines()[0].split()
    vals = [float(v) for v in line]
    h, w = im.shape[:2]
    for k in range(14):
        x = int(vals[5 + k * 3] * w)
        y = int(vals[6 + k * 3] * h)
        cv2.circle(im, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(im, str(k), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    dst = out / f"kpt_ids_{i}.png"
    cv2.imwrite(str(dst), im)
    print(dst)
