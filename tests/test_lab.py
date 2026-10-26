import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'lab'))
from engine import validate, run
(Path(__file__).resolve().parents[1]/'tmp').mkdir(exist_ok=True)

CONFIG = dict(engine='baseline', samples=100, seed=42, steps=3, population=4, epochs=1)


class ExperimentTests(unittest.TestCase):
    def test_invalid_config(self):
        for key, value in [('steps', 0), ('seed', -1), ('population', True), ('engine', 'fake')]:
            with self.assertRaises(ValueError):
                validate({**CONFIG, key: value})

    def test_reproducible_holdout_and_elitism(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as a, tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as b:
            first = run(CONFIG, a, lambda *a, **k: None, lambda: False)
            second = run(CONFIG, b, lambda *a, **k: None, lambda: False)
            self.assertEqual(first, second)
            split = json.loads((Path(a)/'split.json').read_text())
            self.assertFalse(set(split['train']) & set(split['test']))
            self.assertEqual(len(split['test']),20)
            best = [h['best'] for h in first['history']]
            self.assertEqual(best, sorted(best))
            self.assertTrue((Path(a)/'holdout.csv').exists())
            self.assertEqual(len({c['sequence'] for c in first['candidates']}),len(first['candidates']))
            self.assertFalse(first['manifest']['paper_reproduction'])

    def test_cancellation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as folder:
            with self.assertRaises(InterruptedError):
                run(CONFIG, folder, lambda *a, **k: None, lambda: True)


class ApiTests(unittest.TestCase):
    def setUp(self):
        import server
        self.server = server
        self.client = server.app.test_client()

    def test_local_access_and_invalid_input(self):
        self.assertEqual(self.client.get('/api/overview', base_url='http://evil.example').status_code,403)
        base='http://127.0.0.1:8765'
        with self.client.get('/', base_url=base) as response:
            self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.post('/api/runs',base_url=base,json=CONFIG).status_code,403)
        self.assertEqual(self.client.post('/api/runs',base_url=base,json={},headers={'X-Lab-Token':self.server.token}).status_code,400)
        self.assertEqual(self.client.get('/api/runs/example/download/baseline.pkl',base_url=base).status_code,404)


if __name__ == '__main__':
    unittest.main()

