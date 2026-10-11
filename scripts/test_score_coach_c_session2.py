"""Synthetic input tests only; these are not Coach C ratings."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from xml.sax.saxutils import escape
from zipfile import ZipFile

spec = importlib.util.spec_from_file_location('scorer', Path(__file__).with_name('score_coach_c_session2.py'))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.reports = s.load_outputs(s.ROOT / 'evaluation/coach_c/session2_gru/reports')
        self.headers = ['Timestamp'] + [rid + ' - Feedback validity' for rid in s.IDS]

    def csv(self, rows):
        p = self.base / 'responses.csv'
        with p.open('w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(rows)
        return p

    def test_full_set_and_assessed_denominators(self):
        ratings = dict(zip(s.IDS, ['Valid'] * 12 + ['Not valid'] * 5 + [s.CHOICES[2]] * 13))
        result = s.calculate(ratings, self.reports)
        self.assertEqual(40, result['final_valid_of_30_pct'])
        self.assertEqual(17, result['assessed_denominator'])
        self.assertAlmostEqual(100*12/17, result['valid_among_assessed_pct'])
        self.assertEqual(21, result['corrective_or_positive_output_count'])

    def test_missing_is_not_no_coaching_or_final_zero(self):
        ratings = {rid: None for rid in s.IDS}
        ratings['C001'] = s.CHOICES[2]
        result = s.calculate(ratings, self.reports)
        self.assertEqual(29, result['counts']['missing'])
        self.assertEqual(1, result['counts']['no_coaching_feedback_to_assess'])
        self.assertIsNone(result['final_valid_of_30_pct'])
        self.assertIsNone(result['valid_among_assessed_pct'])

    def test_all_no_coaching_retains_thirty(self):
        result = s.calculate({rid: s.CHOICES[2] for rid in s.IDS}, self.reports)
        self.assertTrue(result['complete'])
        self.assertEqual(0, result['final_valid_of_30_pct'])
        self.assertIsNone(result['valid_among_assessed_pct'])

    def test_wide_csv_and_explicit_multiple_submission_selection(self):
        p = self.csv([self.headers, ['first'] + ['Valid']*30, ['second'] + ['Not valid']*30])
        with self.assertRaisesRegex(ValueError, 'Select exactly one'):
            s.read_ratings(p)
        ratings, meta = s.read_ratings(p, response_row=3)
        self.assertTrue(all(v == 'Not valid' for v in ratings.values()))
        self.assertEqual('second', meta['timestamp_raw'])

    def test_column_order_does_not_change_clip_matching(self):
        p = self.csv([list(reversed(self.headers)), list(reversed(['date'] + ['Valid'] + ['Not valid']*29))])
        ratings, _ = s.read_ratings(p)
        self.assertEqual('Valid', ratings['C001'])
        self.assertEqual('Not valid', ratings['C030'])

    def test_duplicate_or_missing_questions_rejected(self):
        p = self.csv([self.headers[:-1] + [self.headers[1]], ['date'] + ['Valid']*30])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            s.read_ratings(p)
        p = self.csv([self.headers[:-1], ['date'] + ['Valid']*29])
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            s.read_ratings(p)

    def test_unknown_answer_rejected(self):
        p = self.csv([self.headers, ['date', 'Good Form'] + ['Valid']*29])
        with self.assertRaisesRegex(ValueError, 'Unknown rating'):
            s.read_ratings(p)

    def test_long_csv_duplicates_unknown_ids_and_missing_rows(self):
        p = self.csv([['review_id', 'feedback_validity'], ['C001', 'Valid']])
        ratings, _ = s.read_ratings(p)
        self.assertEqual(29, s.calculate(ratings, self.reports)['counts']['missing'])
        p = self.csv([['review_id', 'feedback_validity'], ['C001', 'Valid'], ['C001', 'Not valid']])
        with self.assertRaisesRegex(ValueError, 'Duplicate clip'):
            s.read_ratings(p)
        p = self.csv([['review_id', 'feedback_validity'], ['C031', 'Valid']])
        with self.assertRaisesRegex(ValueError, 'Unknown/missing clip'):
            s.read_ratings(p)

    def test_notice_ratings_preserved_and_flagged(self):
        ratings = {rid: 'Valid' for rid in s.IDS}
        result = s.calculate(ratings, self.reports)
        self.assertEqual(100, result['final_valid_of_30_pct'])
        self.assertEqual(9, len(result['review_flags']['notice_only_output_rated_valid_or_not_valid']))

    def xlsx(self, formula=False):
        p = self.base / 'responses.xlsx'
        ns = s.NS['m']
        rows = [[(h, True) for h in self.headers], [('46306.5', False)] + [('Valid', True)]*30]
        strings, xml_rows = [], []
        for number, row in enumerate(rows, 1):
            cells = []
            for col, (value, shared) in enumerate(row, 1):
                letters, n = '', col
                while n:
                    n, rest = divmod(n-1, 26)
                    letters = chr(65+rest) + letters
                if shared:
                    strings.append(value)
                    cell = f'<c r="{letters}{number}" t="s"><v>{len(strings)-1}</v></c>'
                else:
                    cell = f'<c r="{letters}{number}"><v>{value}</v></c>'
                if formula and number == 2 and col == 2:
                    cell = '<c r="B2"><f>1+1</f><v>2</v></c>'
                cells.append(cell)
            xml_rows.append(f'<row r="{number}">{"".join(cells)}</row>')
        with ZipFile(p, 'w') as z:
            z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{s.REL}"><sheets><sheet name="Form Responses 1" sheetId="1" r:id="rId1"/></sheets></workbook>')
            z.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
            z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}">' + ''.join(f'<si><t>{escape(v)}</t></si>' for v in strings) + '</sst>')
            z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>')
        return p

    def test_xlsx_shared_strings_and_numeric_timestamp(self):
        ratings, meta = s.read_ratings(self.xlsx())
        self.assertTrue(all(v == 'Valid' for v in ratings.values()))
        self.assertEqual('46306.5', meta['timestamp_raw'])

    def test_xlsx_formulas_not_interpreted_as_ratings(self):
        with self.assertRaisesRegex(ValueError, 'Unknown rating'):
            s.read_ratings(self.xlsx(formula=True))

    def test_reports_preserve_input_and_do_not_overwrite(self):
        p = self.csv([self.headers, ['date'] + ['Valid']*30])
        digest = s.sha(p)
        ratings, metadata = s.read_ratings(p)
        out = self.base / 'scores'
        s.save_results(out, p, digest, ratings, metadata, self.reports, s.ROOT/'evaluation/coach_c/session2_gru/reports', 'synthetic-test-only')
        self.assertEqual(digest, s.sha(p))
        summary = json.loads((out/'scores.json').read_text())
        self.assertEqual(30, summary['counts']['valid'])
        self.assertTrue((out/'REPORT.md').exists())
        with self.assertRaises(FileExistsError):
            s.save_results(out, p, digest, ratings, metadata, self.reports, s.ROOT/'evaluation/coach_c/session2_gru/reports', 'synthetic-test-only')


if __name__ == '__main__':
    unittest.main()
