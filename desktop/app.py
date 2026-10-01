"""Run with python -m desktop.app. Loopback-only video review application."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import threading
from urllib.parse import urlparse
import uuid
import webbrowser

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs/desktop"
MAX_UPLOAD = 250 * 1024 * 1024
EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ARTIFACTS = {"preview.mp4", "landing.mp4", "contact.jpg", "report.json", "keypoints.npy"}
jobs = {}
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="serve-analysis")


@asynccontextmanager
async def lifespan(app):
    yield
    with lock:
        for job in jobs.values():
            job["cancel"].set()
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Serve Lab", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method not in {"GET", "HEAD"} and origin and urlparse(origin).hostname not in {"127.0.0.1", "localhost", "testserver"}:
        return JSONResponse({"detail": "Only local requests are accepted."}, status_code=403)
    return await call_next(request)


def public(job):
    return {k: v for k, v in job.items() if k != "cancel"}


def work(job_id, primary, secondary, options):
    from desktop.pipeline import analyze, Cancelled
    def progress(value, message):
        with lock:
            jobs[job_id].update(progress=round(value, 3), message=message)
    with lock:
        job = jobs[job_id]
        job["status"] = "running"
        cancel = job["cancel"]
    try:
        result = analyze(primary, secondary, OUTPUTS / job_id, options, progress, cancel)
        result.update(primary_filename=job["filename"], secondary_filename=job.get("secondary_filename"), job_id=job_id)
        (OUTPUTS / job_id / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        with lock:
            job.update(status="complete", progress=1, message="Analysis complete", result=result)
    except Cancelled:
        with lock:
            job.update(status="cancelled", message="Analysis cancelled. Drop another clip to start again.")
    except Exception as exc:
        import traceback
        (OUTPUTS / job_id / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        with lock:
            job.update(status="failed", message=str(exc))


def save_upload(upload, directory, stem):
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in EXTENSIONS:
        raise HTTPException(400, "Choose an MP4, MOV, AVI, MKV, WebM or M4V video.")
    path = directory / (stem + extension)
    size = 0
    with path.open("wb") as handle:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "Each video must be smaller than 250 MB.")
            handle.write(chunk)
    if size == 0:
        raise HTTPException(400, "The video file is empty.")
    return path


@app.get("/api/health")
def health():
    files = {"Pose": "android/app/src/main/assets/pose_landmarker_lite.task",
             "GRU": "android/app/src/main/assets/gru_single.tflite",
             "kNN5": "android/app/src/main/assets/knn_train.bin",
             "Paddle": "runs/detect/paddle_ft/weights/best.pt",
             "Ball": "runs/detect/ball_yolo26s/weights/best.pt",
             "Court": "runs/pose/court_ft/weights/best.pt",
             "Rules": "android/app/src/main/assets/feedback_rules.json"}
    return {"app": "Serve Lab", "models": {name: (ROOT / path).is_file() for name, path in files.items()},
            "candidate_ready": all((ROOT / "models/serve_v2_fixed/tflite" / name).is_file()
                                   for name in ["manifest.json", "gru_single.tflite"] + [f"gru_fold{i}.tflite" for i in range(1,6)])}


@app.post("/api/jobs", status_code=202)
def create_job(video: UploadFile = File(...), secondary: UploadFile | None = File(None),
               start: float = Form(0), end: str = Form(""), model: str = Form("hybrid"),
               landing: str = Form("off"), moment_matching: bool = Form(True)):
    try:
        finish = float(end) if end.strip() else None
        if not math.isfinite(start) or start < 0 or (finish is not None and (not math.isfinite(finish) or finish <= start)):
            raise ValueError()
    except ValueError:
        raise HTTPException(400, "Choose a valid start and end time.")
    if model not in {"single", "ensemble", "hybrid", "v2_single", "v2_ensemble"} or landing not in {"off", "primary", "secondary"}:
        raise HTTPException(400, "Unsupported analysis mode.")
    if landing == "secondary" and secondary is None:
        raise HTTPException(400, "Add a second-camera video for landing analysis.")
    job_id = uuid.uuid4().hex
    with lock:
        if any(j["status"] in {"uploading", "queued", "running"} for j in jobs.values()):
            raise HTTPException(409, "An analysis is already running. Wait or cancel it first.")
        jobs[job_id] = dict(id=job_id, status="uploading", progress=0, message="Receiving clip",
                            filename=Path(video.filename or "video").name,
                            secondary_filename=Path(secondary.filename).name if secondary else None,
                            cancel=threading.Event())
    directory = OUTPUTS / job_id
    directory.mkdir(parents=True)
    try:
        primary = save_upload(video, directory, "input")
        second = save_upload(secondary, directory, "secondary") if secondary and landing == "secondary" else None
        options = dict(start=start, end=finish, model=model, landing=landing, moment_matching=moment_matching)
        with lock:
            jobs[job_id].update(status="queued", message="Preparing analysis")
        executor.submit(work, job_id, primary, second, options)
        return {"id": job_id}
    except Exception:
        with lock:
            jobs.pop(job_id, None)
        raise
    finally:
        video.file.close()
        if secondary:
            secondary.file.close()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, "Analysis not found. The application may have restarted.")
        return public(jobs[job_id])


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, "Analysis not found.")
        jobs[job_id]["cancel"].set()
    return {"message": "Cancellation requested; the current model operation will finish first."}


@app.get("/api/jobs/{job_id}/files/{filename}")
def artifact(job_id: str, filename: str):
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id) or filename not in ARTIFACTS:
        raise HTTPException(404)
    path = OUTPUTS / job_id / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=filename if filename.endswith((".json", ".npy")) else None)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--open", action="store_true", help="Open the local interface in your browser")
    args = parser.parse_args()
    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
