# Evaluation protocol and interpretation

This evaluates the full-res-second-pass Android app on all 30 source clips used by
Coach C: GPU pose attempt, pose priority, motion-rate gating with guarded shared
mechanics, original end detector, 0.60 confidence, 32-real-pose minimum, paddle
submit spacing at the normal 350ms — plus a contact-window full-res second pass at
source resolution through the same detector and rules (`research_fullres_paddle`;
see `protocol.json` for APK and source SHA-256). 60s cooldown between clips.

One replay per clip, C001 through C030, restarting the app each time. Frames are
decoded before playback, then delivered at camera pace through the live pose,
gate, GRU, paddle and feedback path, with the second pass running synchronously
at serve end when the live worker measured nothing. Synthetic idle images run 1.5s
before and 3s after each clip, excluded from evidence. Recorded-video replay, not
physical camera capture.

Status summaries use the final displayed completed result per clip; all events are
preserved in results.json. CORRECTIVE/POSITIVE counts measure availability, not
validity. Recording categories are reference labels. Coach C ratings apply only to
the frozen messages. End-to-UI timing includes the second pass by design.

## Reproduce

With Xiaomi serial 1fb637d connected, from the repo root:

```powershell
$env:PATH = "$env:LOCALAPPDATA\Android\Sdk\platform-tools;" + $env:PATH
python scripts/run_coach_c_mobile_live.py --gpu --motion-rate --fullres-paddle --cooldown-secs 60 --output evaluation/coach_c/mobile_pass2_20261014
python scripts/summarize_coach_c_mobile_live.py evaluation/coach_c/mobile_pass2_20261014
```

Existing completed per-clip logs are reused; a different APK or flag set needs a
new output directory (the runner enforces this). Add an unlogged warmup replay
after install: the first clips after a fresh install currently miss (cold start).
