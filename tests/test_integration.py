import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lab'))
import integration
import server


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self.base = 'http://127.0.0.1:8765'
        self.headers = {'X-Lab-Token': server.token}

    def test_routes_and_protection(self):
        for route in ('/', '/evolution', '/pet'):
            response = self.client.get(route, base_url=self.base)
            self.assertEqual(response.status_code, 200)
            response.close()
        self.assertEqual(self.client.get('/api/arbitrary', base_url=self.base).status_code, 404)
        self.assertEqual(self.client.post('/api/neural/recommend', base_url=self.base, json={}).status_code, 403)
        self.assertEqual(self.client.get('/api/research/pet-runs/not-a-run', base_url=self.base).status_code, 404)

    def test_pet_unavailable_is_explicit(self):
        with patch.object(integration, 'pet_call', side_effect=OSError('offline')):
            response = self.client.get('/api/neural/status', base_url=self.base)
        self.assertEqual(response.status_code, 503)
        self.assertIn('Start-Research.cmd', response.json['detail'])

    def test_success_persists_conditions_provenance_and_domain(self):
        with tempfile.TemporaryDirectory(dir=integration.ROOT / 'tmp') as folder, patch.object(integration, 'SAVED', Path(folder)):
            with patch.object(integration, 'pet_call', return_value=(b'{"test_metrics":{"r2":-2.3},"recommendations":[]}', 200, 'application/json')):
                response = self.client.post('/api/neural/recommend', base_url=self.base, headers=self.headers,
                                            json={'temperature':40,'ph':8,'duration':48,'top_n':3})
            self.assertEqual(response.status_code, 200)
            rid = response.json['research_run_id']
            saved = json.loads((Path(folder) / (rid + '.json')).read_text())
            self.assertEqual(saved['unit'], 'µM TPA')
            self.assertEqual(saved['domain'], 'PET')
            self.assertEqual(saved['config']['ph'], 8)
            self.assertEqual(len(saved['checkpoint_sha256']), 64)
            index = self.client.get('/api/research/runs', base_url=self.base).json
            self.assertTrue(any(r['domain']=='PET' and r['id']==rid for r in index))
            self.assertTrue(any(r['domain']=='avGFP' for r in index))
            artifact = self.client.get(f'/api/research/pet-runs/{rid}', base_url=self.base)
            self.assertEqual(artifact.status_code, 200)
            artifact.close()

    def test_failed_prediction_is_not_saved(self):
        with tempfile.TemporaryDirectory(dir=integration.ROOT / 'tmp') as folder, patch.object(integration, 'SAVED', Path(folder)):
            with patch.object(integration, 'pet_call', return_value=(b'{"detail":"invalid"}', 422, 'application/json')):
                response = self.client.post('/api/neural/recommend', base_url=self.base, headers=self.headers, json={})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_structure_body_limit_is_separate_from_small_requests(self):
        payload = {'protein_id':'1', 'pdb_text':'x'*20000}
        with patch.object(integration, 'pet_call', return_value=(b'{}',200,'application/json')):
            self.assertEqual(self.client.post('/api/structures/inspect', base_url=self.base, headers=self.headers, json=payload).status_code,200)
            self.assertEqual(self.client.post('/api/neural/recommend', base_url=self.base, headers=self.headers, json=payload).status_code,413)

    def test_fasta_is_rejected_before_coordinate_service(self):
        with patch.object(integration, 'pet_call') as downstream:
            response = self.client.post('/api/structures/inspect', base_url=self.base, headers=self.headers,
                                       json={'protein_id': '1', 'pdb_text': '\ufeff  >variant\nACDEFGHIKLMN\n'})
        self.assertEqual(response.status_code, 422)
        self.assertIn('Import a protein FASTA', response.json['detail'])
        downstream.assert_not_called()


if __name__ == '__main__':
    unittest.main()

