import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from permit.corpus import cases, canonical

spec = importlib.util.spec_from_file_location('permit_teaching_script', Path(__file__).resolve().parents[1] / 'scripts/teaching.py')
teaching = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teaching)


class TeachingTests(unittest.TestCase):
    def test_development_and_heldout_cases_cannot_become_teaching_labels(self):
        for split in ('dev', 'heldout'):
            case = next(case for case in cases() if case.split == split)
            with self.assertRaisesRegex(ValueError, 'training split only'):
                teaching.teaching_rows(case)

    def test_teaching_inputs_receive_invoices_and_replacement_authority_only_after_their_tools(self):
        case = next(case for case in cases() if case.split == 'train' and case.change == 'narrow' and case.change_after == 'draft_saved')
        rows, steps, final, evidence = teaching.teaching_rows(case)
        for invoice in case.invoices:
            self.assertNotIn(invoice.invoice_id, canonical(rows[1]['messages']).decode())
            self.assertIn(invoice.invoice_id, canonical(rows[2]['messages']).decode())
        self.assertNotIn(case.replacement_destination, canonical(rows[2]['messages']).decode())
        self.assertIn(case.replacement_destination, canonical(rows[3]['messages']).decode())
        self.assertTrue(all(len(json.loads(row['completion'])) == 1 for row in rows[:4]))
        self.assertEqual(json.loads(rows[3]['completion'])[0]['parameters']['destination'], case.replacement_destination)
        self.assertTrue(teaching.score(evidence, case)['safe_complete'])

    @unittest.skipUnless(os.getenv('AGENTTUNE_SOURCE_DIR'), 'Pinned toolkit sources required.')
    def test_authored_teaching_export_replays_and_rejects_changed_labels(self):
        case = next(case for case in cases() if case.split == 'train')
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            manifest = teaching.prepare([case], out)
            rows, checked = teaching.read_teaching(out, [case])
            self.assertFalse(manifest['model_inference'])
            self.assertEqual(len(rows), 5)
            p = out / 'demonstrations.jsonl'
            p.write_text(p.read_text().replace('"split":"train"', '"split":"heldout"', 1))
            with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
                teaching.read_teaching(out, [case])
