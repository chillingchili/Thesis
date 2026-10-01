"""Summarize contact-window audit JSONs: detection coverage near the wrist.

Usage: python scripts/summarize_audit.py --audit outputs/paddle_angles/audit [--conf 0.25] [--r 150 250]
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--r", type=int, nargs="+", default=[150, 250])
    args = ap.parse_args()

    files = sorted(Path(args.audit).glob("*.json"))
    n_frames = n_anydet = 0
    near = {r: 0 for r in args.r}
    n_wrist = 0
    clips = clips_anydet = clips_gated = 0
    for fp in files:
        rec = json.loads(fp.read_text(encoding="utf-8"))
        contact = rec["contact"]
        clips += 1
        clip_det = clip_gated = False
        for fr in rec["frames"]:
            if abs(fr["f"] - contact) > 3:
                continue
            n_frames += 1
            dets = [b for b in fr["boxes"] if b[4] >= args.conf]
            if dets:
                n_anydet += 1
                clip_det = True
            wrist = fr.get("wrist")
            if wrist is None:
                continue
            n_wrist += 1
            for r in args.r:
                if any(
                    ((b[0] + b[2]) / 2 - wrist[0]) ** 2 + ((b[1] + b[3]) / 2 - wrist[1]) ** 2
                    <= r * r
                    for b in dets
                ):
                    near[r] += 1
                    if r == args.r[-1]:
                        clip_gated = True
        clips_anydet += clip_det
        clips_gated += clip_gated

    print(f"audit={args.audit} conf>={args.conf}")
    print(f"clips={clips} contact-window frames(+-3)={n_frames} (wrist known {n_wrist})")
    print(f"frame any-det: {n_anydet}/{n_frames} = {100*n_anydet/max(1,n_frames):.1f}%")
    for r in args.r:
        print(f"frame det within {r}px of wrist: {near[r]}/{n_wrist} = {100*near[r]/max(1,n_wrist):.1f}%")
    print(f"clips any-det: {clips_anydet}/{clips} = {100*clips_anydet/max(1,clips):.1f}%")
    print(f"clips gated(by last r): {clips_gated}/{clips} = {100*clips_gated/max(1,clips):.1f}%")


if __name__ == "__main__":
    main()
