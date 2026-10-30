"""Independent checks of publication evidence and the new shared workspace."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lab'))
import server
import condition_model
from assay_model import AssayRegressor, CONTEXT_NAMES, normalize_record, specs


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
                mse_um2 = float(np.mean((y-p)**2))
                self.assertAlmostEqual(mse_um2, metrics['mse_um2'])
                self.assertAlmostEqual(float(np.sqrt(mse_um2)), metrics['rmse_um'])
                self.assertAlmostEqual(float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)), metrics['r2'])
                ratio = (1+p)/(1+y)
                self.assertAlmostEqual(float(np.mean((ratio>=.5)&(ratio<=2))), metrics['within_twofold_fraction'])

    def test_saved_fingerprints_and_complete_state(self):
        self.assertEqual(hashlib.sha256((self.folder/'model.pkl').read_bytes()).hexdigest(), self.report['model_sha256'])
        for name, expected in self.protocol['source_sha256'].items():
            path = ROOT / ('scripts' if name.startswith('train_') else 'lab') / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        self.assertEqual(json.loads((self.folder/'status.json').read_text())['status'], 'completed')

    def test_raw_target_candidate_and_nonnegative_predictions(self):
        rng = np.random.default_rng(42)
        sequence = rng.normal(size=(10, 4))
        environment = np.column_stack((np.full(10, 40), np.full(10, 8), np.arange(1, 11)))
        context = np.zeros((10, 16))
        target = np.array([0, 3, 5, 9, 15, 30, 45, 80, 120, 400], dtype=float)
        groups = np.repeat(np.arange(5), 2)
        model = AssayRegressor(mode='sequence_environment', kind='ridge', strength=1,
                               target_transform='raw')
        model.fit(sequence, environment, context, target, groups)
        predictions = model.predict(sequence, environment, context)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())
        self.assertIn('target_transform', next(s for s in specs() if s.get('target_transform') == 'raw'))

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
        response = client.get('/api/training/predictions', base_url='http://127.0.0.1:8765')
        self.assertEqual(response.status_code, 200)
        if response.json['rows']:
            self.assertIn('sequence_environment_mse_um', response.json['rows'][0])
        run_id = self.folder.name
        if (self.folder / 'baseline_comparison.json').is_file():
            response = client.get(f'/api/training/artifacts/{run_id}/baseline_comparison.json',
                                  base_url='http://127.0.0.1:8765')
            self.assertEqual(response.status_code, 200)
            json.loads(response.data)
            response.close()
        response = client.get(f'/api/training/artifacts/{run_id}/secret.txt', base_url='http://127.0.0.1:8765')
        self.assertEqual(response.status_code, 404)

    def test_context_unit_conversion_is_explicit(self):
        def record(**conditions):
            return {'conditions': conditions, 'material': {}}
        self.assertAlmostEqual(normalize_record(
            record(enzyme_concentration_value=2, enzyme_concentration_unit='mM'))['enzyme_um'], 2000.0)
        self.assertAlmostEqual(normalize_record(
            record(enzyme_concentration_value=500, enzyme_concentration_unit='nM'))['enzyme_um'], 0.5)
        self.assertAlmostEqual(normalize_record(
            record(enzyme_concentration_value=250, enzyme_concentration_unit='µg/mL'))['enzyme_mg_ml'], 0.25)
        self.assertAlmostEqual(normalize_record(
            record(reaction_volume_value=500, reaction_volume_unit='µL'))['reaction_volume_ml'], 0.5)
        self.assertAlmostEqual(normalize_record(
            record(reaction_volume_value=1, reaction_volume_unit='L'))['reaction_volume_ml'], 1000.0)
        back_computed = normalize_record(record(
            enzyme_amount_value=1, enzyme_amount_unit='mg', reaction_volume_value=2, reaction_volume_unit='mL'))
        self.assertAlmostEqual(back_computed['enzyme_mg_ml'], 0.5)
        implausible = normalize_record({'conditions': {}, 'material': {'crystallinity_percent': 150}})
        self.assertTrue(np.isnan(implausible['crystallinity_percent']))
        self.assertEqual(set(normalize_record(record()).keys()) - {'material_form', 'preparation'}, set(CONTEXT_NAMES))

    def test_target_standardization_matches_declared_formula(self):
        factor = {'mM': 1000.0, 'µM': 1.0, 'µmol/L': 1.0, 'µg/mL': 1000.0 / 166.13}
        path = condition_model.PET_ROOT / 'data/processed/tpa_pet_ml_confirmed.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            source = list(csv.DictReader(stream))
        checked = 0
        for row in source:
            unit = row['target_unit'].strip()
            if unit not in factor:
                continue
            self.assertAlmostEqual(
                float(row['target_value_standardized']), float(row['target_value']) * factor[unit], places=6)
            checked += 1
        self.assertGreater(checked, 200)

    def test_saved_predictions_trace_to_current_dataset(self):
        data = condition_model.load_data()
        by_source = {int(row['source_row']): i for i, row in enumerate(data['rows'])}
        for design in self.splits:
            with (self.folder / f'{design}_oof.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(data['rows']))
            self.assertEqual(len({row['source_row'] for row in rows}), len(rows))
            for row in rows:
                index = by_source[int(row['source_row'])]
                self.assertEqual(data['rows'][index]['sequence_hash'], row['sequence_hash'])
                self.assertEqual(data['rows'][index]['literature_doi'], row['study'])
                self.assertAlmostEqual(data['target'][index], float(row['observed_um']))

    def test_context_audit_excludes_identifiers_and_units(self):
        audit = json.loads((self.folder / 'context_audit.json').read_text())
        self.assertEqual(audit['rows'], self.report['rows'])
        self.assertEqual(audit['coverage'], self.report['context_coverage'])
        self.assertEqual(audit['input_fields'], list(CONTEXT_NAMES))
        for forbidden in ('target_unit', 'protein_id', 'literature_doi', 'measurements'):
            self.assertIn(forbidden, audit['excluded'])

    def test_ui_reports_regression_metrics_honestly(self):
        html = (ROOT / 'lab/static/training.html').read_text(encoding='utf-8')
        script = (ROOT / 'lab/static/training.js').read_text(encoding='utf-8')
        self.assertIn('id="training-evidence-warning"', html)
        self.assertIn('no classification accuracy percentage', html)
        self.assertIn('µM²', html)
        self.assertIn('sequence_environment_mse_um', script)
        self.assertIn("r2<0", script)
        self.assertIn('not external validation', script)
        self.assertIn('Negative R', script)


if __name__ == '__main__':
    unittest.main()
