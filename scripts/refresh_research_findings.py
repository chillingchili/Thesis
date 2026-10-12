"""Refresh public aggregate findings without rerunning models or copying coach data."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'research-site/assets'


def update_current_findings(data):
    scores_path = 'evaluation/coach_c/session2_gru/coach_c_scores_20261012/scores.json'
    phone_path = 'evaluation/phone_latency/note12_optimized_full_20261010/comparison.json'
    verification_path = 'evaluation/coach_c/session2_gru/coach_c_scores_20261012/form_verification.json'
    scores = json.loads((ROOT / scores_path).read_text())
    phone = json.loads((ROOT / phone_path).read_text())
    assert scores['complete'] and sum(scores['counts'].values()) == 30
    findings = {
        'snapshot': '2026-10-12',
        'selected_model': 'Original five-fold GRU ensemble; no kNN contribution',
        'expert_review': {k: scores[k] for k in [
            'total_clips', 'counts', 'final_valid_of_30_pct', 'assessed_denominator',
            'valid_among_assessed_pct', 'corrective_or_positive_output_count',
            'corrective_or_positive_output_coverage_pct', 'model_status_counts',
            'ratings_by_model_status']},
        'interpretation': 'Submitted ratings preserved. Eight notice-only outputs rated Valid; two corrective outputs marked no coaching. Rating percentages do not establish corrective-feedback accuracy or player improvement.',
        'form_verification': 'All 30 current editor outputs match frozen reports; submission-time presentation not independently reconstructed.',
        'phone_benchmark': {**phone, 'successful_measured_runs_per_configuration': 9,
            'distinct_clips': 3, 'timing_boundary': 'recorded-file opening to computed feedback; initial model setup excluded',
            'live_two_second_target': 'unverified'},
    }
    data.update(findings)
    data['coach_status'] = {'session1': 'complete', 'session2': 'complete_with_interpretation_limitation', 'prior_diagnostic_overlap': 24, 'selected_clips': 30}
    sources = data.setdefault('sources', {})
    for path in [scores_path, phone_path, verification_path]:
        sources[path] = {'sha256': hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
    for src, dest in [('docs/thesis/CHAPTERS_5_6.md', 'chapters-5-6.md'), ('docs/thesis/Pascua-Leones_Thesis_Current.md', 'thesis.md')]:
        shutil.copyfile(ROOT / src, ASSETS / 'sources' / dest)
        sources[src] = {'sha256': hashlib.sha256((ROOT / src).read_bytes()).hexdigest()}
    (ASSETS / 'sources/current-findings.json').write_text(json.dumps(findings, indent=2) + '\n', encoding='utf-8')
    return data


def main():
    data = update_current_findings(json.loads((ASSETS / 'research-data.json').read_text(encoding='utf-8')))
    serialized = json.dumps(data, ensure_ascii=True, indent=2)
    (ASSETS / 'research-data.json').write_text(serialized, encoding='utf-8')
    (ASSETS / 'research-data.js').write_text('window.RESEARCH_DATA = ' + serialized + ';\n', encoding='utf-8')
    print('Refreshed aggregate findings and current manuscript downloads.')


if __name__ == '__main__':
    main()
