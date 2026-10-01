"""Drag-and-drop ball + homography demo app.

Run:  python scripts/balltrack_app.py   -> opens http://127.0.0.1:7860
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from balltrack_pipeline import ROOT, get_models  # noqa: E402

get_models()

from balltrack_pipeline import process_video  # noqa: E402


def run(video_path: str, stride: int, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Drop a video first.")
    summary = process_video(video_path, stride=int(stride), progress=progress)
    landing = summary.get("landing")
    if landing:
        text = (
            f"**Landing zone:** {landing['zone']}  \n"
            f"**Court position:** {landing['x_ft']} ft (width) × {landing['y_ft']} ft (depth)  \n"
            f"**Bounce frame:** {landing['frame']}  |  **Processed:** {summary['frames']} frames "
            f"in {summary['seconds']}s (stride {summary['stride']})"
        )
    else:
        text = (
            f"**No landing detected**  \n**Processed:** {summary['frames']} frames "
            f"in {summary['seconds']}s (stride {summary['stride']})"
        )
    return summary["out"], text


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Video(sources=["upload", "webcam"], label="Drop video"),
        gr.Slider(1, 5, value=1, step=1, label="Frame stride (higher = faster, coarser)"),
    ],
    outputs=[
        gr.Video(label="Ball tracking + homography"),
        gr.Markdown(""),
    ],
    title="Pickleball Ball Tracking + Homography",
    description=(
        "Left panel: broadcast view with court keypoints, ball detection and trajectory. "
        "Right panel: bird's-eye court via homography with the projected ball trail, "
        "detected landing point and the 7-zone classification (Deep/Short-Left/Right + Faults). "
        "Processing runs on CPU — expect roughly 10-20 min per 5-minute video at stride 1; "
        "use a higher stride for quick previews."
    ),
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
