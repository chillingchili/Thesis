"""
playlist_keypoints.py

Interactive playlist player for 2D BlazePose keypoint sequences (.npy).

Controls:
  N           : Next file in playlist
  P           : Previous file in playlist
  Space       : Play / Pause
  Right Arrow : Next frame (when paused)
  Left Arrow  : Previous frame (when paused)
  R           : Reset / Replay current file
  Q or Esc    : Exit viewer

Usage:
  python npyvis.py path/to/folder_with_npy_files
  python npyvis.py path/to/folder_with_npy_files --fps 30
"""

import argparse
import os
import glob
import cv2
import numpy as np

# Standard BlazePose 33-landmark skeletal connectivity
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
]

def render_frame(
    kpts: np.ndarray,
    frame_idx: int,
    total_frames: int,
    is_paused: bool,
    width: int,
    height: int,
) -> np.ndarray:
    """Renders a single frame of keypoints and skeleton connections onto a canvas."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if np.isnan(kpts).any():
        cv2.putText(
            canvas,
            "NO POSE DETECTED",
            (width // 2 - 170, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        coords = {}
        for idx in range(kpts.shape[0]):
            x = int(np.clip(kpts[idx, 0] * width, 0, width - 1))
            y = int(np.clip(kpts[idx, 1] * height, 0, height - 1))
            coords[idx] = (x, y)

        for start_idx, end_idx in POSE_CONNECTIONS:
            pt1 = coords[start_idx]
            pt2 = coords[end_idx]
            
            if start_idx % 2 != 0 and end_idx % 2 != 0:
                bone_color = (255, 220, 0)   
            elif start_idx % 2 == 0 and end_idx % 2 == 0 and start_idx >= 12:
                bone_color = (0, 140, 255)   
            else:
                bone_color = (0, 255, 128)   

            cv2.line(canvas, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        for idx, (x, y) in coords.items():
            cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 2, (0, 0, 220), -1, cv2.LINE_AA)

    state_label = "PAUSED" if is_paused else "PLAYING"
    cv2.putText(
        canvas,
        f"Frame: {frame_idx + 1}/{total_frames}  [{state_label}]",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Space: Play/Pause | Left/Right: Step | R: Replay | Q: Quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )

    return canvas

def play_playlist(path: str, fps: int = 30, width: int = 800, height: int = 600):
    # Support both single files and directories transparently
    if os.path.isdir(path):
        playlist = sorted(glob.glob(os.path.join(path, "*.npy")))
    else:
        playlist = [path]

    if not playlist:
        print(f"No .npy files found in {path}")
        return

    file_idx = 0
    window_name = "BlazePose Playlist Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    frame_delay = max(1, int(1000 / fps))

    while 0 <= file_idx < len(playlist):
        npy_path = playlist[file_idx]
        try:
            data = np.load(npy_path)
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            file_idx += 1
            continue

        if data.ndim != 3 or data.shape[1] != 33 or data.shape[2] != 2:
            print(f"Skipping {npy_path}: Invalid shape {data.shape}")
            file_idx += 1
            continue

        num_frames = data.shape[0]
        frame_idx = 0
        paused = False
        skip_file = 0  # 1 for Next, -1 for Prev

        while True:
            frame_canvas = render_frame(
                kpts=data[frame_idx],
                frame_idx=frame_idx,
                total_frames=num_frames,
                is_paused=paused,
                width=width,
                height=height,
            )

            # Draw Playlist HUD Overlay
            filename = os.path.basename(npy_path)
            cv2.putText(
                frame_canvas,
                f"File {file_idx + 1}/{len(playlist)}: {filename}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_canvas,
                "N: Next File | P: Prev File",
                (20, height - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame_canvas)
            wait_time = 0 if paused else frame_delay
            key = cv2.waitKeyEx(wait_time)

            if key in (ord("q"), ord("Q"), 27):  # Q or Esc
                cv2.destroyAllWindows()
                return
            elif key in (ord("n"), ord("N")):
                skip_file = 1
                break
            elif key in (ord("p"), ord("P")):
                skip_file = -1
                break
            elif key == 32:  # Spacebar
                paused = not paused
            elif key in (ord("r"), ord("R")):
                frame_idx = 0
            elif key in (2555904, 65363, 83):  # Right arrow
                frame_idx = min(frame_idx + 1, num_frames - 1)
            elif key in (2424832, 65361, 81):  # Left arrow
                frame_idx = max(frame_idx - 1, 0)
            else:
                if not paused:
                    if frame_idx + 1 < num_frames:
                        frame_idx += 1
                    else:
                        paused = True  # Hold on last frame until replayed or skipped

        file_idx += skip_file

    cv2.destroyAllWindows()
    print("Playlist complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playlist Viewer for BlazePose .npy files.")
    parser.add_argument("path", help="Folder containing .npy files (or path to a single .npy file)")
    parser.add_argument("--fps", type=int, default=30, help="Playback frame rate")
    parser.add_argument("--width", type=int, default=800, help="Canvas width")
    parser.add_argument("--height", type=int, default=600, help="Canvas height")
    args = parser.parse_args()

    play_playlist(args.path, fps=args.fps, width=args.width, height=args.height)