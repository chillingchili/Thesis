"""Montage of contact frames with all audit detections + wrist marker.

Colors: green = conf>=0.25, orange = 0.1<=conf<0.25, red circle = right wrist.
Used to visually estimate how often the true paddle is detected at contact.

Usage: python scripts/make_audit_montage.py --audit outputs/paddle_angles/audit \
         --videos data/training/raw/coach --out outputs/paddle_angles/audit_contact_montage.jpg
"""
import argparse
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    files = sorted(Path(args.audit).glob("*.json"))
    random.Random(args.seed).shuffle(files)
    tiles = []
    for fp in files[: args.n]:
        r = json.loads(fp.read_text(encoding="utf-8"))
        contact = r["contact"]
        fr = [x for x in r["frames"] if x["f"] == contact]
        if not fr:
            continue
        fr = fr[0]
        vp = glob.glob(f"{args.videos}/**/{r['clip_id']}.mp4", recursive=True)
        if not vp:
            continue
        cap = cv2.VideoCapture(vp[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, contact)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        for b in fr["boxes"]:
            x1, y1, x2, y2, c = (int(v) for v in b)
            col = (0, 255, 0) if c >= 0.25 else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"{c:.2f}", (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if fr["wrist"]:
            w = (int(fr["wrist"][0]), int(fr["wrist"][1]))
            cv2.circle(img, w, 10, (0, 0, 255), 2)
            cv2.putText(img, "W", (w[0] + 12, w[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, r["clip_id"].replace("CoachA_", ""),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(cv2.resize(img, (640, 360)))

    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"wrote {len(tiles)} tiles -> {args.out}")


if __name__ == "__main__":
    main()
