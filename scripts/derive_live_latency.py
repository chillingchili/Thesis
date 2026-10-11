"""Derive the live serve-end critical path from measured CoreLatencyBenchmark components.

The recorded benchmark times whole-clip processing, which is dominated by decode and
by per-frame pose over every source frame. That total is not live latency and never
will be. What *is* live latency is the work the app performs on the critical path
after the motion gate has decided the serve ended:

    lastServeEndMs = now                                   MainActivity.kt:389
    awaitPaddleQuiescence(paddleQuiesceMs)                 MainActivity.kt:392
    runInference(frames, finalResult = true)  -> gru_ms   MainActivity.kt:395
    updateFeedback(...)  -> final_measurements_and_feedback_ms
    log feedback_shown (inside runOnUiThread)              MainActivity.kt:523

Each term is either measured by the benchmark or bounded by app code. This script
sums them per measured run and reports the worst case, then states plainly what it
still cannot establish.

It does not claim the live target is verified. It claims the post-end critical path
has enough headroom that the remaining uncertainty is frame acquisition, not compute.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('out', type=Path)
    # App constants, kept as flags so a change in MainActivity.kt forces a re-derivation
    # rather than silently reusing a stale value.
    ap.add_argument('--low-hold-ms', type=int, default=350,
                    help='MainActivity.kt lowHoldMs: wall-clock below end threshold')
    ap.add_argument('--paddle-quiesce-ms', type=int, default=250,
                    help='MainActivity.kt paddleQuiesceMs: cap on paddle drain at serve end')
    ap.add_argument('--final-min-frames', type=int, default=32,
                    help='MainActivity.kt finalMinFrames: frames needed to classify at end')
    ap.add_argument('--ui-post-slack-ms', type=float, default=70.0,
                    help='One live pose-frame interval, allowed for runOnUiThread latency: '
                         'feedback_shown is logged inside the posted runnable, so this '
                         'is the only unmeasured term. 70ms is the pose cost measured at '
                         '1920x1080; live runs 640x480, so this over-estimates.')
    ap.add_argument('--live-analysis-width', type=int, default=640)
    ap.add_argument('--live-analysis-height', type=int, default=480)
    args = ap.parse_args()

    raw = args.input.read_bytes()
    report = json.loads(raw)
    runs = [r for r in report['runs'] if not r.get('warmup') and r.get('success')]
    if not runs:
        raise SystemExit('No successful measured runs')

    per_run = []
    for r in runs:
        pose_per_frame = r['pose_ms'] / r['decoded_frames']

        final_infer = r['gru_ms']
        feedback = r['final_measurements_and_feedback_ms']
        # Bounded, not measured: awaitPaddleQuiescence returns at the cap.
        drain = float(args.paddle_quiesce_ms)
        end_to_shown = drain + final_infer + feedback + args.ui_post_slack_ms
        # Conservative: charge the full source-resolution cost for the in-flight
        # frame. Live analysis is 640x480 so it is never slower than this. The
        # pixel-scaled figure is reported as context only, because BlazePose Lite
        # resizes to a fixed model input and its cost does not track input pixels.
        one_frame = pose_per_frame
        peak_to_shown = args.low_hold_ms + one_frame + end_to_shown
        per_run.append(dict(
            clip=r['clip'], round=r['round'],
            source_fps=r['decoded_frames'] * 1000 / r['source_duration_ms'],
            source_duration_ms=r['source_duration_ms'],
            pose_ms_per_frame_source=pose_per_frame,
            # Live frame rate is bounded above by the camera and below by pose cost
            # charged at source resolution (live is cheaper, never more expensive).
            live_fps_floor=1000 / pose_per_frame,
            live_fps_ceiling=30.0,
            # If the gate were open for the whole clip. It is not: it opens on motion
            # onset and closes 350ms after motion stops, so this is an upper window.
            live_frames_floor=r['source_duration_ms'] / 1000 * (1000 / pose_per_frame),
            live_frames_ceiling=r['source_duration_ms'] / 1000 * 30.0,
            final_infer_ms=final_infer, feedback_ms=feedback,
            paddle_drain_bound_ms=drain, ui_post_slack_ms=args.ui_post_slack_ms,
            end_to_shown_ms=end_to_shown, peak_to_shown_ms=peak_to_shown,
        ))

    def worst(key):
        return max(r[key] for r in per_run)

    def med(key):
        return statistics.median(r[key] for r in per_run)

    min_frames_floor = min(r['live_frames_floor'] for r in per_run)
    min_frames_ceiling = min(r['live_frames_ceiling'] for r in per_run)
    summary = dict(
        source_metric=report['metric'],
        source_sha256=hashlib.sha256(raw).hexdigest(),
        measured_runs=len(per_run),
        target_ms=2000.0,
        app_constants=dict(low_hold_ms=args.low_hold_ms,
                           paddle_quiesce_ms=args.paddle_quiesce_ms,
                           final_min_frames=args.final_min_frames),
        assumption=dict(ui_post_slack_ms=args.ui_post_slack_ms),
        end_to_shown_median_ms=med('end_to_shown_ms'),
        end_to_shown_worst_ms=worst('end_to_shown_ms'),
        peak_to_shown_worst_ms=worst('peak_to_shown_ms'),
        margin_over_target_x=2000.0 / worst('peak_to_shown_ms'),
        peak_to_shown_within_2s=worst('peak_to_shown_ms') <= 2000,
        final_min_frames_met_at_pose_floor=min_frames_floor >= args.final_min_frames,
        min_live_frames_floor=min_frames_floor,
        min_live_frames_ceiling=min_frames_ceiling,
        live_two_second_target_verified=False,
        post_end_within_2s=all(r['end_to_shown_ms'] <= 2000 for r in per_run),
        per_run=per_run,
    )

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'live_equivalent.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (args.out / 'source_raw.json').write_bytes(raw)

    def ms(v):
        return f'{v:.1f} ms'

    lines = [
        '# Live serve-end critical path derived from recorded benchmark',
        '',
        f"Source: {args.input} ({len(per_run)} successful measured runs).",
        '',
        '**This is a derivation from measured components, not an end-to-end live capture '
        'measurement. The live two-second target remains unverified.**', '',
        '## What is being derived', '',
        'The recorded benchmark times whole-clip processing, which is dominated by decode and',
        'by pose over every source frame. Live does neither on the critical path. What live',
        'does after the motion gate decides the serve ended is:', '',
        '```',
        'lastServeEndMs = now                        MainActivity.kt:389',
        'awaitPaddleQuiescence(paddleQuiesceMs)      MainActivity.kt:392  <= cap',
        'runInference(frames, finalResult=true)      MainActivity.kt:395  = gru_ms',
        'updateFeedback(...)                         MainActivity.kt:405  = feedback_ms',
        'log feedback_shown inside runOnUiThread     MainActivity.kt:523  = ui post',
        '```', '',
        'Note that the `lowHoldMs` end-detection delay is *excluded* from `end_to_shown`:',
        '`lastServeEndMs` is stamped when the gate fires, so `end_to_shown` starts there.',
        '`peak_to_shown` adds it back to describe player-visible delay from motion peak.', '',
        '## Worst case across all measured runs', '',
        '| Term | Worst | Basis |', '|---|---:|---|',
        f"| paddle drain | {ms(args.paddle_quiesce_ms)} | code bound (`paddleQuiesceMs`), not measured |",
        f"| final GRU inference | {ms(worst('final_infer_ms'))} | measured `gru_ms` |",
        f"| paddle verdict + feedback rules | {ms(worst('feedback_ms'))} | measured `final_measurements_and_feedback_ms` |",
        f"| UI post before `feedback_shown` | {ms(args.ui_post_slack_ms)} | assumption: one live pose-frame interval |",
        f"| **end_to_shown** | **{ms(worst('end_to_shown_ms'))}** | sum |",
        f"| + one in-flight pose frame | {ms(worst('pose_ms_per_frame_source'))} | upper bound at source resolution; live is {args.live_analysis_width}x{args.live_analysis_height} |",
        f"| + low-hold before end fires | {ms(args.low_hold_ms)} | code constant |",
        f"| **peak_to_shown** | **{ms(summary['peak_to_shown_worst_ms'])}** | player-visible worst case |",
        '',
        f"Target {ms(2000)}: peak_to_shown has **{summary['margin_over_target_x']:.1f}x** headroom.",
        '',
        '## Per run', '',
        '| clip | round | end_to_shown | peak_to_shown | final_infer | feedback | pose ms/frame | frames floor |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in sorted(per_run, key=lambda x: -x['peak_to_shown_ms']):
        lines.append(f"| {r['clip']} | {r['round']} | {ms(r['end_to_shown_ms'])} | {ms(r['peak_to_shown_ms'])} | "
                     f"{ms(r['final_infer_ms'])} | {ms(r['feedback_ms'])} | {r['pose_ms_per_frame_source']:.1f} | "
                     f"{r['live_frames_floor']:.0f} |")

    lines += [
        '',
        '## Frame accumulation', '',
        f"`finalMinFrames={args.final_min_frames}` gates whether the final GRU runs at all;",
        'below it, no classification is produced regardless of latency.', '',
        'Live pose rate is bounded above by the 30 fps camera and below by pose cost charged at',
        f'its measured 1920x1080 figure. Live analysis runs {args.live_analysis_width}x{args.live_analysis_height},',
        'so the real rate sits between those bounds. The table below assumes the gate stayed',
        'open for an entire clip; in practice it opens on motion onset and closes 350 ms after',
        'motion stops, so the real active window is shorter than the clip.', '',
        '| clip | duration | pose ms/frame | fps floor | fps ceiling | frames floor | frames ceiling |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in sorted(per_run, key=lambda x: x['clip']):
        lines.append(f"| {r['clip']} | {r['source_duration_ms']} ms | {r['pose_ms_per_frame_source']:.1f} | "
                     f"{r['live_fps_floor']:.1f} | {r['live_fps_ceiling']:.0f} | {r['live_frames_floor']:.0f} | "
                     f"{r['live_frames_ceiling']:.0f} |")
    lines += [
        '',
        f"The floor of {min_frames_floor:.0f} frames assumes the gate is open for a whole",
        'clip, which it is not: it opens on motion onset and closes 350 ms after motion',
        'stops. A ~2 s active window inside C002 would yield ~28 frames, below 32, and the',
        'final inference would be skipped entirely — no classification, whatever the',
        'latency. Live pose at 640x480 is cheaper than the 1920x1080 figure charged here,',
        'which is the only reason this may still clear. This is the least certain input in',
        'the derivation and the first thing a live measurement must settle.', '',
        '## What this does not establish', '',
        '- It does not measure capture-to-display latency end to end. No camera, no display,',
        '  no real motion gate were exercised.',
        '- The paddle drain is taken as its code cap, not observed. Under backlog the cap',
        '  guarantees the bound but also discards detections, which is a coaching-quality',
        '  question rather than a latency one.',
        '- The in-flight pose frame is charged at its measured 1920x1080 cost, and live runs',
        f'  {args.live_analysis_width}x{args.live_analysis_height}. That deliberately over-charges',
        '  the bound; it is not a measurement at the live resolution.',
        '- The UI post allowance is an assumption of one pose-frame interval.',
        '- The frame-accumulation floor assumes pose cost measured at 1920x1080 and a gate',
        '  window as long as the clip. Both are the least certain inputs here.',
        '',
        'A live measurement is still required to close the target. This only establishes that',
        'compute after the gate fires is not the obstacle.',
        '',
    ]
    (args.out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_run'}, indent=2))


if __name__ == '__main__':
    main()
