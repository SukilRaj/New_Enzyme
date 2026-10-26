"""Independent checks of publication evidence and the new shared workspace."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lab'))
import server


class TrainingEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run = json.loads((ROOT / 'models/assay_v2/latest.json').read_text())['run_id']
        cls.folder = ROOT / 'models/assay_v2' / run
        cls.report = json.loads((cls.folder / 'report.json').read_text())
        cls.protocol = json.loads((cls.folder / 'protocol.json').read_text())
        cls.splits = json.loads((cls.folder / 'splits.json').read_text())

    def test_nested_group_and_study_boundaries(self):
        sequences = np.array(self.protocol['sequence_groups'])
        studies = np.array(self.protocol['studies'])
        for design, folds in self.splits.items():
            covered = []
            for fold in folds:
                train, test = fold['train'], fold['test']
                covered.extend(test)
                self.assertFalse(set(train) & set(test))
                self.assertFalse(set(sequences[train]) & set(sequences[test]))
                if design == 'sequence_and_study':
                    self.assertFalse(set(studies[train]) & set(studies[test]))
                for inner in fold['inner_splits']:
                    a, b = inner['train'], inner['validation']
                    self.assertTrue(set(a + b) <= set(train))
                    self.assertFalse(set(sequences[a]) & set(sequences[b]))
                    if design == 'sequence_and_study':
                        self.assertFalse(set(studies[a]) & set(studies[b]))
            self.assertEqual(sorted(covered), list(range(self.report['rows'])))

    def test_report_recomputed_independently_from_predictions(self):
        for design, expected in self.report['evaluation'].items():
            with (self.folder / f'{design}_oof.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            y = np.array([float(r['observed_um']) for r in rows])
            self.assertEqual(len(y), self.report['rows'])
            for mode, metrics in expected.items():
                p = np.array([float(r[mode]) for r in rows])
                self.assertTrue(np.isfinite(p).all())
                self.assertAlmostEqual(float(np.mean(np.abs(y-p))), metrics['mae_um'])
                self.assertAlmostEqual(float(np.sqrt(np.mean((y-p)**2))), metrics['rmse_um'])
                self.assertAlmostEqual(float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)), metrics['r2'])
                ratio = (1+p)/(1+y)
                self.assertAlmostEqual(float(np.mean((ratio>=.5)&(ratio<=2))), metrics['within_twofold_fraction'])

    def test_saved_fingerprints_and_complete_state(self):
        self.assertEqual(hashlib.sha256((self.folder/'model.pkl').read_bytes()).hexdigest(), self.report['model_sha256'])
        for name, expected in self.protocol['source_sha256'].items():
            path = ROOT / ('scripts' if name.startswith('train_') else 'lab') / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        self.assertEqual(json.loads((self.folder/'status.json').read_text())['status'], 'completed')

    def test_shell_and_embedded_module_routes(self):
        client = server.app.test_client()
        for path in ('/', '/pet', '/analysis', '/evolution', '/training'):
            response = client.get(path, base_url='http://127.0.0.1:8765')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'workspace-views', response.data)
            response.close()
            response = client.get(path+'?embedded=1', base_url='http://127.0.0.1:8765')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'data-workspace-module=', response.data)
            self.assertNotIn(b'id="workspace-views"', response.data)
            response.close()
        response = client.get('/api/training/status', base_url='http://127.0.0.1:8765')
        self.assertEqual(response.json['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
