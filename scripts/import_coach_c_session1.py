"""Read the submitted Session 1 workbook without altering source or frozen outputs."""
from collections import Counter
from datetime import datetime, timezone
import json
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from prepare_coach_c_session2 import EVAL, ROOT, sha, write_json

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_responses(path):
    with ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in si.findall('.//m:t', NS))
                       for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        sheets = [p for p in z.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', p)]
        assert len(sheets) == 1, 'Inspect additional worksheets before importing'
        rows = []
        for row in ET.fromstring(z.read(sheets[0])).findall('.//m:sheetData/m:row', NS):
            cells = {}
            for cell in row.findall('m:c', NS):
                assert cell.find('m:f', NS) is None, 'Unexpected formula in response data'
                value = cell.findtext('m:v', default='', namespaces=NS)
                if cell.get('t') == 's': value = strings[int(value)]
                if cell.get('t') == 'inlineStr': value = ''.join(t.text or '' for t in cell.findall('.//m:t', NS))
                if value != '': cells[re.sub(r'\d+', '', cell.get('r'))] = value
            if cells: rows.append((row.get('r'), cells))
    assert len(rows) == 2, 'Expected header and exactly one coach submission'
    headers, response = rows[0][1], rows[1][1]
    assert headers['A'] == 'Timestamp'
    expected = {f'C{i:03d} - {field}' for i in range(1, 31) for field in ['Form Judgment', 'Deviation Comment']}
    assert set(headers.values()) - {'Timestamp'} == expected
    assert len(headers) == 61
    lookup = {v: k for k, v in headers.items()}
    result = []
    for i in range(1, 31):
        rid = f'C{i:03d}'
        form_col, note_col = lookup[rid + ' - Form Judgment'], lookup[rid + ' - Deviation Comment']
        form, comment = response.get(form_col, ''), response.get(note_col, '')
        assert form in {'Good Form', 'Bad Form'}, f'Missing or unexpected judgment: {rid}'
        result.append(dict(review_id=rid, form_judgment=form, deviation_comment=comment,
            serve_type=None, serve_type_status='not_collected', source_sheet='Form Responses 1',
            judgment_cell=form_col + rows[1][0], comment_cell=note_col + rows[1][0]))
    return result, response.get('A')


def main():
    source = EVAL / 'Pickleball Annotation - Coach C - Session 1 (30 Clips) (Responses).xlsx'
    digest = sha(source)
    status_path = EVAL / 'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    active = ROOT / status['active_session2_run']
    protected_paths = [source, EVAL / 'coach_c_session1.zip', EVAL / 'session1_staging/annotations.csv']
    protected_paths += [p for p in active.rglob('*') if p.is_file()]
    protected = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected_paths}
    rows, timestamp = read_responses(source)
    judgments = Counter(r['form_judgment'] for r in rows)
    cross = Counter()
    for row in rows:
        report = json.loads((active / 'revised_reports' / (row['review_id'] + '.json')).read_text(encoding='utf-8'))
        row['frozen_feedback_status'] = report['feedback']['status']
        cross[(row['form_judgment'], row['frozen_feedback_status'])] += 1
    out = EVAL / ('session1_received_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    write_json(out / 'annotations.json', rows)
    summary = dict(source=source.relative_to(ROOT).as_posix(), source_sha256=digest,
        received_utc=datetime.now(timezone.utc).isoformat(), response_timestamp_excel_serial=timestamp,
        timestamp_timezone='not specified in workbook', response_rows=1, clips=30,
        form_judgment_counts=dict(judgments), good_form_clip_ids=[r['review_id'] for r in rows if r['form_judgment']=='Good Form'],
        nonempty_deviation_comments=sum(bool(r['deviation_comment'].strip()) for r in rows),
        independent_serve_type_labels=0, missing_serve_type_ids=[r['review_id'] for r in rows],
        classifier_accuracy_against_coach_c=None, feedback_appropriateness=None,
        form_judgment_by_feedback_status=[dict(form_judgment=k[0], feedback_status=k[1], count=v) for k,v in sorted(cross.items())],
        original_and_frozen_artifacts_sha256=protected,
        session2_release_status='HOLD_PENDING_SESSION1_SERVE_TYPE_LABELS',
        form_is_not_serve_type=True, labels_inferred_from_comments=False,
        user_confirmed_no_separate_serve_type_labels=True)
    write_json(out / 'summary.json', summary)
    table = '\n'.join(f'| {r["form_judgment"]} | {r["feedback_status"]} | {r["count"]} |' for r in summary['form_judgment_by_feedback_status'])
    (out / 'REPORT.md').write_text(f'''# Coach C Session 1 response audit

One submission contains all 30 form judgments: {judgments['Bad Form']} Bad Form and {judgments['Good Form']} Good Form.
Good Form: {', '.join(summary['good_form_clip_ids'])}. All 27 Bad Form judgments have comments.
The original workbook is unchanged. annotations.json preserves comments verbatim and records source cells.

## Missing classification labels

The workbook has no Drive/Lob/Topspin fields. The user confirmed there are no separate labels.
Classification accuracy, per-class precision/recall/F1 and a Coach C confusion matrix are unavailable.
Do not substitute recording categories, infer labels from free text, or interpret 27 Bad Form judgments as 27 classifier errors.
C001's comment disputes lob, C023 disputes drive, and C030 describes under spin rather than drive.
Preserve these comments; the last also raises a possible out-of-taxonomy label rather than forcing a three-class answer.

Before showing Session 2 feedback, request an independent label for each C001–C030:
Drive, Lob, Topspin, or an explicit Other/Unable to determine response where appropriate.
A short clip-ID list is sufficient; no new frame, paddle-angle, or mechanics measurement form is needed.
Keep original judgments intact and record the supplementary labels separately.

## Descriptive correspondence, not accuracy

| Coach form judgment | Frozen feedback status | Clips |
|---|---|---:|
{table}

The model does not supply a validated binary whole-form classifier. These counts cannot establish form-classification accuracy.
Session 2 must independently assess feedback appropriateness; these Session 1 comments are not Session 2 ratings.
Keep the approved wording and original predictions fixed after reading these responses.
Session 2 remains private pending the missing independent serve labels. No packet was sent.

Source SHA-256: `{digest}`.
''', encoding='utf-8')
    assert all(sha(ROOT / name) == expected for name, expected in protected.items())
    write_json(out / 'status_before_import.json', status)
    status.update(session1_response_received=True, session1_form_judgments_complete=True,
        session1_serve_labels_complete=False, coach_c_annotations_pending=True,
        session1_response_sha256=digest, session1_response_audit=out.relative_to(ROOT).as_posix(),
        session2_release_status=summary['session2_release_status'])
    write_json(status_path, status)
    print(json.dumps({k: v for k,v in summary.items() if k not in {'original_and_frozen_artifacts_sha256','missing_serve_type_ids'}}, indent=2))
    print('REPORT', out / 'REPORT.md')


if __name__ == '__main__':
    main()
