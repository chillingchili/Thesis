"""Summarize real CoreLatencyBenchmark records without claiming live latency."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    raw=args.input.read_bytes()
    report=json.loads(raw)
    measured=[r for r in report['runs'] if not r['warmup']]
    successful=[r for r in measured if r.get('success')]
    failed=[r for r in measured if not r.get('success')]
    values=sorted(r['total_processing_ms'] for r in successful)
    summary=dict(metric=report['metric'],source_sha256=hashlib.sha256(raw).hexdigest(),
        device=report['model'],soc=report['soc'],android=report['android'],
        completed=report.get('completed',False),measured_attempts=len(measured),successful=len(successful),failed=len(failed),
        warmups=sum(r['warmup'] for r in report['runs']),
        median_ms=statistics.median(values) if values else None,
        p95_nearest_rank_ms=values[math.ceil(.95*len(values))-1] if values else None,
        max_ms=max(values) if values else None,
        within_2s=sum(v<=2000 for v in values),
        within_2s_pct_of_attempts=100*sum(v<=2000 for v in values)/len(measured) if measured else None,
        live_two_second_target_verified=False,
        notes=['Median/p95/max describe successful runs only; failed attempts remain in the count and within-2s denominator.',
               'Nine measured runs across three fixed clips are an initial device benchmark, not a population-wide performance estimate.',
               'Fresh VIDEO pose tracker per clip; no live camera, motion gate, UI display timing or concurrent live inference.',
               'Initial GRU/paddle loading and per-run pose loading recorded separately; total_processing_ms excludes these.',
               'USB/charging, power and thermal conditions are retained in raw.json; no battery-state normalization was applied.'])
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'raw.json').write_bytes(raw)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fields=['clip','round','warmup','success','total_processing_ms','decode_ms','pose_ms','hip_accumulation_ms','paddle_ms','gru_ms','final_measurements_and_feedback_ms','pose_reset_load_ms','source_frames','source_duration_ms','decoded_frames','usable_frames','decoded_width','decoded_height','feedback_status','paddle_available','error']
    with (args.out/'runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(report['runs'])
    def sec(value): return 'Unavailable' if value is None else f'{value/1000:.3f} s'
    lines=['# Xiaomi recorded-core latency benchmark','',
        f"Device: {report['manufacturer']} {report['model']}; SoC {report['soc']}; Android {report['android']}; reported RAM {report['ram_bytes']/1024**3:.2f} GiB.",
        '', '**This measures recorded-file processing, not live capture-end-to-displayed-feedback latency. The thesis live two-second target remains unverified.**','',
        f"Measured attempts: {len(measured)}; successful: {len(successful)}; failures: {len(failed)}. Warm-up runs excluded: {summary['warmups']}.",'',
        '| Successful-run statistic | Full recorded processing |','|---|---:|',
        f"| Median | {sec(summary['median_ms'])} |",f"| P95 (nearest rank) | {sec(summary['p95_nearest_rank_ms'])} |",f"| Slowest | {sec(summary['max_ms'])} |",
        '',f"Completed within two seconds: {summary['within_2s']}/{len(measured)} measured attempts. This comparison does not establish the live target.",'',
        '## Mean component times on successful runs','', '| Component | Mean |','|---|---:|']
    for key in ['decode_ms','pose_ms','paddle_ms','hip_accumulation_ms','gru_ms','final_measurements_and_feedback_ms']:
        lines.append(f"| {key} | {sec(statistics.mean(r[key] for r in successful)) if successful else 'Unavailable'} |")
    lines+=['',f"Initial GRU/paddle/configuration loading: {sec(report.get('model_load_ms'))}. Per-clip pose construction is recorded separately in runs.csv.",'',
        '## Scope','',*['- '+note for note in summary['notes']],'',
        'Inputs are C001, C002 and C007, fixed before timing (first recorded Lob, Topspin and Drive selections). All frames are decoded at source dimensions. Clip identity and asset hashes are in raw.json. No Coach C answers affect timing.','',
        'Next: measure actual camera completion through final displayed feedback, including queued work and end-detection delay, on this same phone. Do not substitute GRU-only inference time for that measurement.','']
    (args.out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    assert args.input.read_bytes()==raw
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
