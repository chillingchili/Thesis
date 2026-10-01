"""Finish evaluation, native bundle, integrity checks and complete-output package."""
import json
import shutil
import subprocess
import sys
import time
from serve_study import ROOT, write_json
from serve_refinement import DEST, OUTPUT


def run(script):
    print('RUN',script,flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    while not (DEST/'frozen_selection.json').exists():
        log=(ROOT/'serve_refinement_training.log').read_text(encoding='utf-8',errors='replace')
        if 'Traceback (most recent call last)' in log:raise RuntimeError('Training failed: see serve_refinement_training.log')
        time.sleep(5)
    run('evaluate_serve_refinement.py')
    assets=ROOT/'android/app/src/main/assets/serve_research';assets.mkdir(parents=True,exist_ok=True)
    for name in ['model.tflite','manifest.json','knn_meta.json','knn_train.bin']:
        shutil.copy2(DEST/'android_bundle'/name,assets/name)
    native_fixtures=ROOT/'android/app/src/androidTest/assets/serve_refinement.json'
    native_fixtures.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(DEST/'android_bundle/native_parity_cases.json',native_fixtures)
    for script in ['test_serve_refinement.py','test_serve_refinement_integrity.py','test_serve_study.py','test_serve_study_integrity.py']:
        run(script)
    write_json(OUTPUT/'python_verification.json',dict(refinement_tests=8,v3_tests=12,all_passed=True,
        commands=['python scripts/'+s for s in ['test_serve_refinement.py','test_serve_refinement_integrity.py','test_serve_study.py','test_serve_study_integrity.py']]))
    run('report_serve_refinement.py')
    print('REFINEMENT COMPLETE',flush=True)


if __name__=='__main__':main()
