"""Finish dependent study stages after an already-running extraction completes.

Safe to resume: model training verifies saved configurations and hashes. This
does not deploy models, contact a coach, or write any original dataset labels.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    start=time.monotonic()
    marker=ROOT/"data/serve_study_v3/extraction_complete.json"
    while not marker.exists():
        if time.monotonic()-start>3*60*60:
            raise TimeoutError("Extraction did not finish within three hours; inspect extraction log before resuming")
        count=len(list((ROOT/"data/serve_study_v3/augmented").glob("*__a[12].npz")))
        print(f"Waiting for frozen extraction: {count}/418 variants",flush=True)
        time.sleep(30)
    for script,args in [("train_serve_study.py",["--conditions","angles_thesis_aug"]),
                        ("test_serve_study_integrity.py",[]),
                        ("evaluate_serve_study.py",[]),
                        ("inspect_serve_augmentation.py",[]),
                        ("report_serve_study.py",[])]:
        print(f"Running {script}",flush=True)
        subprocess.run([sys.executable,str(ROOT/"scripts"/script),*args],cwd=ROOT,check=True)
    print("All automated study stages complete; coach responses and physical phone validation remain pending.",flush=True)


if __name__=="__main__":main()
