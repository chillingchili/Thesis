# Evaluation protocol and interpretation

This evaluates the showcase-configuration Android app on all 30 source clips used
by Coach C: GPU pose attempt, pose priority, motion-rate gating with guarded
shared mechanics, original end detector, 0.60 confidence, 32-real-pose minimum —
plus paddle submit spacing relaxed to 150ms (from 350ms) and a contact-window
full-res second pass at source resolution through the same detector and rules
(`research_relaxed_paddle` / `research_fullres_paddle` extras; see `protocol.json`
for APK and source SHA-256).

One replay per clip, C001 through C030, restarting the app each time. Frames are
decoded before playback, then delivered at camera pace through the live pose,
gate, GRU, paddle and feedback path, with the second pass running synchronously
at serve end when the live worker measured nothing. Synthetic idle images run 1.5s
before and 3s after each clip, excluded from evidence. Recorded-video replay, not
physical camera capture.

Status summaries use the final displayed completed result per clip; all events are
preserved in results.json. CORRECTIVE/POSITIVE counts measure substantive-output
availability, not demonstrated validity. Recording categories are reference
labels, not independent subtype annotations. Coach C ratings apply only to the
frozen messages he reviewed. End-to-UI timing includes the second pass by design.

## Reproduce

With Xiaomi serial 1fb637d connected, from the repo root:

```powershell
$env:PATH = "$env:LOCALAPPDATA\Android\Sdk\platform-tools;" + $env:PATH
python scripts/run_coach_c_mobile_live.py --gpu --motion-rate --relaxed-paddle --fullres-paddle --output evaluation/coach_c/mobile_showcase_20261014
python scripts/summarize_coach_c_mobile_live.py evaluation/coach_c/mobile_showcase_20261014
```

Existing completed per-clip logs are reused; a different APK or flag set needs a
new output directory (the runner enforces this).
