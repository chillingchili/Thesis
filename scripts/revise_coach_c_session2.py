"""Prepare approved Session 2 wording from frozen measurements; never read responses."""
from __future__ import annotations

import copy
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_STORED

from prepare_coach_c_session2 import EVAL, ROOT, sha, write_csv, write_json, assert_unchanged

VERSION = 'coach-c-actionable-wording-v2'
RULE_TEXT = {
    'C4_WEIGHT_TRANSFER': 'Your hips showed limited movement during the serve. On your next attempt, start with more weight on your back foot, then shift toward your front foot as you swing. Finish balanced.',
    'C6_PADDLE_REVIEW': 'Your paddle orientation differed from the reference serves. Try a slower practice swing, keep your grip relaxed, and maintain a consistent paddle position through contact.',
}
STATUS_TEXT = {
    'POSITIVE': 'Your measured hip movement and paddle orientation are within the reference ranges. Repeat the motion and focus on consistency.',
    'LOW_CONFIDENCE': 'The system could not confidently identify this serve. Record one complete serve, including your setup and follow-through.',
    'PARTIAL': 'No supported corrective cue for the available measurements. Overall serve form has not been assessed.',
    'INSUFFICIENT_DATA': 'Serve classification unavailable. Record another serve for coaching feedback.',
}
NOTE_TEXT = {
    'Paddle orientation unavailable.': 'Paddle orientation could not be measured. For the next recording, keep the paddle visible throughout the swing.',
}


def revised_feedback(old):
    feedback = copy.deepcopy(old)
    if old['status'] == 'CORRECTIVE':
        # Unknown/directional rules must fail rather than acquire invented wording.
        feedback['messages'] = [RULE_TEXT[rid] for rid in old['rule_ids']]
    else:
        feedback['messages'] = [STATUS_TEXT[old['status']]]
    feedback['notifications'] = [NOTE_TEXT.get(n, n) for n in old['notifications']]
    feedback.update(version=VERSION, validation_status='pending_coach_c')
    return feedback


def main():
    source = EVAL / 'private_session2_hybrid_20261010T045855Z'
    original_freeze = json.loads((source / 'freeze.json').read_text(encoding='utf-8'))
    assert_unchanged(original_freeze)
    # Hash the entire existing private run so its original outputs stay auditable.
    original_hashes = {p.relative_to(ROOT).as_posix(): sha(p)
                       for p in source.rglob('*') if p.is_file()}
    run = EVAL / ('private_session2_wording_v2_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir(exist_ok=False)
    packet = run / 'session2_packet_HOLD'
    (packet / 'videos').mkdir(parents=True)
    (run / 'revised_reports').mkdir()
    write_json(run / 'wording.json', dict(version=VERSION, rules=RULE_TEXT, statuses=STATUS_TEXT, notifications=NOTE_TEXT))
    rows, comparisons, cards, statuses = [], [], [], []
    for number in range(1, 31):
        rid = f'C{number:03d}'
        original = json.loads((source / 'model_outputs' / rid / 'report.json').read_text(encoding='utf-8'))
        revised = copy.deepcopy(original)
        revised['feedback'] = revised_feedback(original['feedback'])
        assert {k: v for k, v in revised.items() if k != 'feedback'} == {k: v for k, v in original.items() if k != 'feedback'}
        assert revised['feedback']['status'] == original['feedback']['status']
        assert revised['feedback']['rule_ids'] == original['feedback']['rule_ids']
        assert revised['input_sha256'] == original_freeze['clip_sha256'][rid]
        write_json(run / 'revised_reports' / f'{rid}.json', revised)
        message = '\n'.join(revised['feedback']['messages'] + revised['feedback']['notifications'])
        assert 'with your coach' not in message.lower()
        rows.append(dict(review_id=rid, feedback_string=message))
        comparisons.append(dict(review_id=rid, feedback_status=revised['feedback']['status'],
            old_feedback='\n'.join(original['feedback']['messages'] + original['feedback']['notifications']), new_feedback=message))
        statuses.append(revised['feedback']['status'])
        shutil.copy2(EVAL / 'session1_staging/videos' / f'{rid}.mp4', packet / 'videos' / f'{rid}.mp4')
        assert sha(packet / 'videos' / f'{rid}.mp4') == original_freeze['clip_sha256'][rid]
        cards.append(f'<section><h2>{rid}</h2><video controls preload="none" src="videos/{rid}.mp4"></video><p>{html.escape(message)}</p></section>')
    write_csv(packet / 'feedback_strings.csv', rows)
    write_csv(run / 'wording_comparison.csv', comparisons)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coach C — Session 2 feedback</title><style>
body{font:17px/1.6 system-ui,sans-serif;color:#202020;max-width:860px;margin:40px auto;padding:0 22px}
section{border-top:1px solid #bbb;padding:24px 0}video{width:100%;max-height:500px;background:#111}section p{white-space:pre-wrap}
</style><h1>Session 2: system feedback</h1>
<p>Watch each clip and assess the overall system feedback using the existing evaluation form. Match answers by clip ID. Do not consult Session 1 answers.</p>
''' + '\n'.join(cards) + '</html>'
    (packet / 'index.html').write_text(page, encoding='utf-8')
    (packet / 'README.md').write_text('Open index.html to view all 30 clips and feedback. Use your existing Session 2 form; no measurements or additional response form are required. Do not consult Session 1 answers.\n', encoding='utf-8')
    zip_path = run / 'PRIVATE_DO_NOT_SEND_session2_wording_v2.zip'
    with ZipFile(zip_path, 'w', ZIP_STORED) as z:
        for p in sorted(packet.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(packet).as_posix())
    with ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == {'index.html', 'README.md', 'feedback_strings.csv'} | {f'videos/C{i:03d}.mp4' for i in range(1, 31)}
        assert '<input' not in z.read('index.html').decode('utf-8')
    assert_unchanged(original_freeze)
    assert all(sha(ROOT / path) == digest for path, digest in original_hashes.items())
    manifest = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        change='User-approved wording revision applied uniformly to all original outputs before Coach C responses. No model, measurement, trigger, threshold, or clip selection changes.',
        source_run=source.relative_to(ROOT).as_posix(), original_files_sha256=original_hashes,
        source_freeze_sha256=sha(source / 'freeze.json'), wording_sha256=sha(run / 'wording.json'),
        generator_sha256=sha(Path(__file__)), clips=30, feedback_status_counts=dict(Counter(statuses)),
        session1_responses_read=False, session2_responses_read=False, release_status='HOLD_UNTIL_SESSION1_SUBMITTED',
        session2_packet_sha256=sha(zip_path), classifier_and_measurements_preserved=True,
        development_measurement_adapter_applied=False, android_wording_updated=False,
        accuracy=None, feedback_appropriateness=None,
        limitations=['Hip displacement is not physical weight distribution.', 'Paddle orientation is an image-plane estimate, not verified face tilt.', 'New practice suggestions await Coach C appropriateness review.'])
    manifest['revised_report_sha256'] = {p.name: sha(p) for p in sorted((run / 'revised_reports').glob('*.json'))}
    write_json(run / 'freeze.json', manifest)
    (run / 'RESEARCHER_README.md').write_text('''# Private revised Session 2 packet

This packet supersedes the original Session 2 wording for the planned review.
It reuses the frozen hybrid measurements and decision rules, with the approved
actionable wording. It does not apply the separate development measurement adapter.
The original run and Session 1 are preserved. wording_comparison.csv records every change.
The inherited report freeze/assets refer to the original inference; this directory's
freeze.json and wording.json identify the subsequent feedback revision.

Preparation can proceed during Session 1. Release the packet only after Session 1
answers are submitted and saved, to prevent feedback influencing independent labels.
Use the user's existing overall-feedback form. No manual frame/angle form is required.
Keep the wording fixed through Session 2 and report which version Coach C evaluated.
Document early private inference and this pre-review wording amendment in the thesis.
Neither revised wording nor successful packaging establishes improved accuracy or coaching.
''', encoding='utf-8')
    status_path = EVAL / 'status.json'
    shutil.copy2(status_path, run / 'status_before_wording_revision.json')
    status = json.loads(status_path.read_text(encoding='utf-8'))
    status.update(active_session2_run=run.relative_to(ROOT).as_posix(), session2_feedback_version=VERSION,
        session2_release_status='HOLD_UNTIL_SESSION1_SUBMITTED', session2_packet_sha256=sha(zip_path),
        session2_wording_amendment=manifest['change'], session2_uses_existing_user_form=True)
    write_json(status_path, status)
    print(json.dumps(dict(run=str(run), clips=len(rows), feedback_status_counts=dict(Counter(statuses)),
        original_artifacts_preserved=True, packet_verified=True), indent=2))


if __name__ == '__main__':
    main()
