YOLOv26s (ball + court + paddle)
================================

WHAT THIS IS
------------
datasets/ball/   11,120 images  ball detection  (YOLO detect labels)
datasets/court/  3,615 images   court keypoints (YOLO pose labels, 14 kpts)
datasets/paddle/ ~6,000 images  paddle detection (YOLO detect labels, 1 class)
weights/        pretrained YOLO weights (from repo root)
scripts/train_yolo_dgx.py   training entry point

WORKFLOW
--------
1) WinSCP (SFTP, same login as your ssh):
   Copy these INTO a folder on the DGX, e.g. ~/balltrack/
     datasets/                       (the whole folder, ~1.1 GB)
     weights/                        (contains yolo26s.pt, yolo26s-pose.pt,
                                       and yolo26n.pt for offline training)
     scripts/train_yolo_dgx.py

2) ssh into the DGX, then:

     cd ~/balltrack
     nvidia-smi                       # GPU must be listed, no error

     python3 -m venv venv
     source venv/bin/activate
     pip install ultralytics pyyaml
     python -c "import torch; print(torch.cuda.is_available())"   # must print True

3) Train:

   python train_yolo_dgx.py ball      # ball detector
   python train_yolo_dgx.py court     # court keypoint model
   python train_yolo_dgx.py paddle    # paddle detector

   Progress lives in:
     runs/detect/ball_yolo26s/results.csv
     runs/pose/court_yolo26s/results.csv
     runs/detect/paddle_yolo26s/results.csv
   Watch:  tail -f runs/detect/paddle_yolo26s/results.csv

   Rough time on DGX Spark: ball ~30-60 min, court ~20-40 min (100 epochs, 640px, batch 32).

4) When done, WinSCP back:
     runs/detect/ball_yolo26s/weights/best.pt
     runs/pose/court_yolo26s/weights/best.pt
     runs/detect/paddle_yolo26s/weights/best.pt
   Bring them home -> they get exported to TFLite on your PC for the app.

TROUBLESHOOTING
---------------
- torch.cuda.is_available() == False:
    pip install --upgrade torch        # DGX Spark needs a recent torch (>=2.8)
    still False -> ask school admin if there is an NGC PyTorch container
    they provide (many DGX setups ship one) and run train_yolo_dgx.py inside it.
- Out of memory: edit train_yolo_dgx.py, lower BATCH (32 -> 16).
- pip blocked by network policy: ask admin for the allowed PyPI mirror/proxy.
- Re-running is safe: same --name, exist_ok=True overwrites the run cleanly.
- Datasets folder already on DGX? skip step 1, just verify data.yaml paths
  (they are RELATIVE - works on any machine).

NOTES
-----
- data.yaml uses path: datasets/ball (and datasets/court) so it works
  as long as you run train_yolo_dgx.py from ~/balltrack (the folder
  that contains datasets/). NEVER use "path: ." - ultralytics resolves
  it to your current shell folder and training fails.
- Court model falls back to warm-starting from weights/yolo26s.pt if
  yolo26s-pose.pt cannot be downloaded on the DGX (air-gapped setups).
