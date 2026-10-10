"""Copy one canonical Coach C evidence bundle into Git without rerunning inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/coach_c_evaluation_v2'
DEST = ROOT / 'evaluation/coach_c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    status = json.loads((SOURCE / 'status.json').read_text(encoding='utf-8'))
    baseline = ROOT / status['private_hybrid_run']
    active = ROOT / status['active_session2_run']
    received = ROOT / status['session1_response_audit']
    mapping = {}
    def add(source, target):
        mapping[target] = source
    for name in ['status.json', 'selection.json', 'researcher_key.csv', 'technical_checks.csv']:
        add(SOURCE / name, 'source/' + name)
    add(SOURCE / 'Pickleball Annotation - Coach C - Session 1 (30 Clips) (Responses).xlsx', 'session1/coach_c_responses.xlsx')
    for name in ['annotations.json', 'summary.json', 'REPORT.md']:
        add(received / name, 'session1/' + name)
    for name in ['freeze.json', 'summary.json', 'researcher_outputs.csv']:
        add(baseline / name, 'baseline/' + name)
    for name in ['freeze.json', 'wording.json', 'wording_comparison.csv', 'RESEARCHER_README.md']:
        add(active / name, 'session2/' + name)
    add(active / 'session2_packet_HOLD/feedback_strings.csv', 'session2/feedback_strings.csv')
    add(ROOT / status['evaluation_scope_amendment'], 'session2/EVALUATION_SCOPE_AMENDMENT.md')
    freeze = json.loads((baseline / 'freeze.json').read_text(encoding='utf-8'))
    for i in range(1, 31):
        rid = f'C{i:03d}'
        video = SOURCE / 'session1_staging/videos' / f'{rid}.mp4'
        assert sha(video) == freeze['clip_sha256'][rid]
        add(video, f'videos/{rid}.mp4')
        add(baseline / 'model_outputs' / rid / 'report.json', f'baseline/reports/{rid}.json')
        add(active / 'revised_reports' / f'{rid}.json', f'session2/reports/{rid}.json')
    records = []
    for name, source in mapping.items():
        target = DEST / name
        digest = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha(target) != digest:
            shutil.copy2(source, target)
        assert sha(source) == sha(target) == digest
        assert target.stat().st_size < 100 * 1024 * 1024
        records.append(dict(path=name, source=source.relative_to(ROOT).as_posix(), sha256=digest, bytes=target.stat().st_size))
    (DEST / 'provenance.json').write_text(json.dumps(dict(exported_utc=datetime.now(timezone.utc).isoformat(),
        files=records, note='Copies only; historical status in frozen artifacts is superseded by README and scope amendment.'), indent=2), encoding='utf-8')
    print(f'Copied and hash-verified {len(records)} evidence files ({sum(r["bytes"] for r in records)/1024**2:.1f} MB).')


if __name__ == '__main__':
    main()
