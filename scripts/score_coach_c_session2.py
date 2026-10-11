"""Score a Coach C Google Forms CSV/XLSX export without changing any responses.

Wide form exports use C001 - Feedback validity ... C030 - Feedback validity.
Long CSV/XLSX tables may use review_id, feedback_validity. Multiple wide
submissions require --response-row; rows are never silently merged or selected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
IDS = [f'C{i:03d}' for i in range(1, 31)]
CHOICES = ('Valid', 'Not valid', 'No coaching feedback to assess')
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xlsx_tables(path):
    """Read values without executing formulas or modifying the workbook."""
    tables = {}
    with ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in s.findall('.//m:t', NS))
                       for s in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        relationships = {r.get('Id'): r.get('Target') for r in
                         ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
                         if r.get('TargetMode') != 'External'}
        workbook = ET.fromstring(z.read('xl/workbook.xml'))
        for sheet in workbook.findall('m:sheets/m:sheet', NS):
            target = relationships.get(sheet.get('{' + REL + '}id'))
            if not target:
                continue
            member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            if not member.startswith('xl/'):
                raise ValueError('Unexpected worksheet location')
            rows = []
            for row in ET.fromstring(z.read(member)).findall('m:sheetData/m:row', NS):
                cells = {}
                for cell in row.findall('m:c', NS):
                    letters = re.fullmatch(r'([A-Z]+)\d+', cell.get('r', ''))
                    if not letters:
                        raise ValueError('Invalid Excel cell reference')
                    index = 0
                    for char in letters[1]:
                        index = index * 26 + ord(char) - 64
                    if cell.find('m:f', NS) is not None:
                        value = '__FORMULA_NOT_SUPPORTED__'
                    else:
                        value = cell.findtext('m:v', '', NS)
                        if cell.get('t') == 's':
                            value = strings[int(value)]
                        elif cell.get('t') == 'inlineStr':
                            value = ''.join(t.text or '' for t in cell.findall('.//m:t', NS))
                    cells[index - 1] = value
                if any(str(v).strip() for v in cells.values()):
                    rows.append((int(row.get('r')), [cells.get(i, '') for i in range(max(cells) + 1)]))
            tables[sheet.get('name')] = rows
    return tables


def read_ratings(path, response_row=None, sheet=None):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        if sheet:
            raise ValueError('--sheet applies only to XLSX')
        with path.open(encoding='utf-8-sig', newline='') as f:
            tables = {'CSV': [(n, row) for n, row in enumerate(csv.reader(f), 1) if any(c.strip() for c in row)]}
    elif path.suffix.lower() == '.xlsx':
        tables = xlsx_tables(path)
    else:
        raise ValueError('Use a .csv or .xlsx response export')
    if sheet:
        if sheet not in tables:
            raise ValueError(f'Worksheet not found: {sheet}')
        tables = {sheet: tables[sheet]}
    else:
        tables = {name: rows for name, rows in tables.items() if rows and
                  any('Feedback validity' in str(c) or str(c).strip() == 'feedback_validity' for c in rows[0][1])}
    if len(tables) != 1:
        raise ValueError('Expected one response table; supply --sheet for multiple matching worksheets')
    name, table = next(iter(tables.items()))
    if not table:
        raise ValueError('Response table is empty')
    headers = [str(v).strip() for v in table[0][1]]
    nonempty = [h for h in headers if h]
    if len(nonempty) != len(set(nonempty)):
        raise ValueError('Duplicate response column headings')
    records = table[1:]
    def values(row):
        if len(row) > len(headers) and any(str(c).strip() for c in row[len(headers):]):
            raise ValueError('Response values have no matching column headings')
        return dict(zip(headers, row + [''] * max(0, len(headers) - len(row))))
    raw, source_rows, timestamp = {}, {}, None
    if {'review_id', 'feedback_validity'}.issubset(headers):
        if response_row is not None:
            raise ValueError('--response-row cannot select a single clip from a long response table')
        for number, row in records:
            record = values(row)
            rid = str(record['review_id']).strip().upper()
            if rid not in IDS:
                raise ValueError(f'Unknown/missing clip ID at row {number}: {rid!r}')
            if rid in raw:
                raise ValueError(f'Duplicate clip ID: {rid}')
            raw[rid], source_rows[rid] = record['feedback_validity'], number
    else:
        expected = [rid + ' - Feedback validity' for rid in IDS]
        if not set(expected).issubset(headers):
            raise ValueError('Missing required C001–C030 Feedback validity columns; this may be the Session 1 export')
        extras = [h for h in headers if h.endswith(' - Feedback validity') and h not in expected]
        if extras:
            raise ValueError(f'Unexpected clip columns: {extras}')
        if response_row is not None:
            selected = [(n, r) for n, r in records if n == response_row]
        else:
            selected = records
        if len(selected) != 1:
            raise ValueError('Select exactly one submission with --response-row (header is row 1); no responses are combined')
        number, row = selected[0]
        record = values(row)
        timestamp = record.get('Timestamp')
        raw = {rid: record[rid + ' - Feedback validity'] for rid in IDS}
        source_rows = {rid: number for rid in IDS}
    canonical = {c.casefold(): c for c in CHOICES}
    ratings = {}
    for rid in IDS:
        value = str(raw.get(rid, '') or '').strip()
        if value and value.casefold() not in canonical:
            raise ValueError(f'Unknown rating for {rid}: {value!r}')
        ratings[rid] = canonical.get(value.casefold())
    return ratings, dict(sheet=name, selected_response_row=response_row,
                         source_rows=source_rows, timestamp_raw=timestamp,
                         timestamp_note='Original export value; timezone not inferred')


def load_outputs(folder):
    reports = {}
    for rid in IDS:
        p = folder / f'{rid}.json'
        report = json.loads(p.read_text(encoding='utf-8'))
        if report.get('review_id') != rid:
            raise ValueError(f'Frozen report ID mismatch: {rid}')
        reports[rid] = report
    return reports


def calculate(ratings, reports):
    if set(ratings) != set(IDS) or set(reports) != set(IDS):
        raise ValueError('Exactly C001–C030 must be accounted for')
    if any(value not in (*CHOICES, None) for value in ratings.values()):
        raise ValueError('Unexpected rating')
    counts = Counter(ratings.values())
    valid, invalid, no_feedback, missing = [counts[k] for k in (*CHOICES, None)]
    substantive = {'CORRECTIVE', 'POSITIVE'}
    cross = Counter((reports[rid]['feedback']['status'], ratings[rid] or 'Missing') for rid in IDS)
    notice_rated = [rid for rid in IDS if reports[rid]['feedback']['status'] not in substantive and ratings[rid] in CHOICES[:2]]
    advice_unassessed = [rid for rid in IDS if reports[rid]['feedback']['status'] in substantive and ratings[rid] == CHOICES[2]]
    return dict(total_clips=30, counts=dict(valid=valid, not_valid=invalid,
        no_coaching_feedback_to_assess=no_feedback, missing=missing), complete=missing == 0,
        observed_valid_of_30_pct=100 * valid / 30,
        final_valid_of_30_pct=100 * valid / 30 if not missing else None,
        assessed_denominator=valid + invalid,
        valid_among_assessed_pct=100 * valid / (valid + invalid) if valid + invalid else None,
        no_coaching_feedback_to_assess_pct=100 * no_feedback / 30,
        missing_pct=100 * missing / 30,
        corrective_or_positive_output_count=sum(reports[rid]['feedback']['status'] in substantive for rid in IDS),
        corrective_or_positive_output_coverage_pct=100 * sum(reports[rid]['feedback']['status'] in substantive for rid in IDS) / 30,
        model_status_counts=dict(Counter(r['feedback']['status'] for r in reports.values())),
        ratings_by_model_status=[dict(model_status=k[0], rating=k[1], count=v) for k,v in sorted(cross.items())],
        review_flags=dict(notice_only_output_rated_valid_or_not_valid=notice_rated,
                          substantive_output_marked_no_coaching=advice_unassessed),
        independent_classification_accuracy=None,
        notes=['No-coaching and missing are distinct; neither is scored as Valid.',
               'Review flags preserve coach answers unchanged; do not automatically recode them.',
               'Assessed-only validity is secondary and may be provisional when responses are incomplete.',
               'Source is one expert; clip counts do not establish inter-rater reliability or player-independent effectiveness.'])


def save_results(destination, source, source_hash, ratings, metadata, reports, report_dir, form_version):
    summary = calculate(ratings, reports)
    summary.update(source_file=source.name, source_sha256=source_hash, import_metadata=metadata,
        scored_utc=datetime.now(timezone.utc).isoformat(), declared_form_version=form_version,
        form_deployment_independently_verified=False,
        frozen_report_sha256={rid: sha(report_dir / f'{rid}.json') for rid in IDS},
        scorer_sha256=sha(__file__))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'scores.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (destination / 'clip_scores.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['review_id', 'feedback_validity', 'response_missing', 'model_feedback_status', 'predicted_serve', 'model_confidence', 'source_row'])
        writer.writeheader()
        for rid in IDS:
            r = reports[rid]
            writer.writerow(dict(review_id=rid, feedback_validity=ratings[rid] or '', response_missing=ratings[rid] is None,
                model_feedback_status=r['feedback']['status'], predicted_serve=r['serve'].get('label', ''),
                model_confidence=r['serve'].get('confidence', ''), source_row=metadata['source_rows'].get(rid, '')))
    c = summary['counts']
    secondary = summary['valid_among_assessed_pct']
    secondary_text = 'Unavailable (no Valid/Not valid judgments)' if secondary is None else f'{secondary:.1f}% ({c["valid"]}/{summary["assessed_denominator"]})'
    final_text = f'{summary["final_valid_of_30_pct"]:.1f}%' if summary['complete'] else 'Pending: incomplete submission'
    lines = ['# Coach C Session 2 scoring', '', f'Final Valid/30 result: **{final_text}**.', '',
        '| Response | Count | Percentage of all 30 |', '|---|---:|---:|']
    for label, value in [('Valid', c['valid']), ('Not valid', c['not_valid']), ('No coaching feedback to assess', c['no_coaching_feedback_to_assess']), ('Missing', c['missing'])]:
        lines.append(f'| {label} | {value} | {100*value/30:.1f}% |')
    lines += ['', f'Secondary validity among assessed responses: {secondary_text}.',
        f'Observed Valid/30 so far: {summary["observed_valid_of_30_pct"]:.1f}%; provisional if any responses are missing.',
        f'Frozen corrective-or-positive output coverage: {summary["corrective_or_positive_output_count"]}/30 ({summary["corrective_or_positive_output_coverage_pct"]:.1f}%). This is not coach-rated validity.', '',
        '## Consistency review', '',
        'Notice-only outputs rated Valid/Not valid: ' + (', '.join(summary['review_flags']['notice_only_output_rated_valid_or_not_valid']) or 'None') + '.',
        'Substantive outputs marked no coaching: ' + (', '.join(summary['review_flags']['substantive_output_marked_no_coaching']) or 'None') + '.',
        'These flags require interpretation, not automatic changes to the original answers.', '',
        'No independent subtype accuracy or binary whole-form accuracy is computed. Missing labels are not invented.',
        f'Source SHA-256: `{source_hash}`. Original export remains unchanged.',
        f'Declared presentation: `{form_version}`. Actual deployment must be confirmed by the researchers.', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('responses', type=Path)
    ap.add_argument('--response-row', type=int, help='Excel row or CSV logical record number including header; first submission is 2')
    ap.add_argument('--sheet')
    ap.add_argument('--reports-dir', type=Path, default=ROOT / 'evaluation/coach_c/session2_gru/reports')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--form-version', default='original-gru-v1 / classification-in-output-v1 / coach-c-actionable-wording-v2')
    args = ap.parse_args()
    try:
        original_hash = sha(args.responses)
        ratings, metadata = read_ratings(args.responses, args.response_row, args.sheet)
        reports = load_outputs(args.reports_dir)
        destination = args.out or ROOT / 'outputs/coach_c_session2_scores' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        summary = save_results(destination, args.responses, original_hash, ratings, metadata, reports, args.reports_dir, args.form_version)
        if sha(args.responses) != original_hash:
            raise ValueError('Source export changed during scoring; retain results for investigation')
    except (ValueError, FileExistsError) as exc:
        ap.error(str(exc))
    print(json.dumps(dict(output=str(destination), complete=summary['complete'], counts=summary['counts'], final_valid_of_30_pct=summary['final_valid_of_30_pct']), indent=2))


if __name__ == '__main__':
    main()
