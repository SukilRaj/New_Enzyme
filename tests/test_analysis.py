import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lab'))
import analysis_api
import condition_model
import server
import structure_images


class ScientificContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=analysis_api.read_report()
        cls.data=condition_model.load_data()

    def test_fixed_and_nested_splits_have_no_sequence_overlap(self):
        protocol=json.loads((analysis_api.MODEL/'protocol.json').read_text())
        groups=self.data['groups'];partitions=protocol['split']
        self.assertEqual(sum(map(len,partitions.values())),len(groups))
        for name,rows in partitions.items():
            for other,others in partitions.items():
                if name!=other:self.assertFalse(set(groups[rows])&set(groups[others]))
        folds=json.loads((analysis_api.MODEL/'cv_splits.json').read_text())
        heldout=[]
        for fold in folds:
            train,test=fold['train'],fold['test'];heldout.extend(test)
            self.assertFalse(set(groups[train])&set(groups[test]))
            for inner in fold['inner_splits']:
                a,b=inner['train'],inner['validation']
                self.assertTrue(set(a+b)<=set(train))
                self.assertFalse(set(groups[a])&set(groups[b]))
        self.assertEqual(sorted(heldout),list(range(len(groups))))

    def test_saved_scores_recompute_from_predictions(self):
        fixed=pd.read_csv(analysis_api.MODEL/'heldout.csv')
        score=condition_model.metrics(fixed.observed_um.to_numpy(),fixed.selected_prediction_um.to_numpy())
        self.assertAlmostEqual(score['mae_um'],self.report['selected_test']['mae_um'])
        self.assertAlmostEqual(score['mse_um2'],score['rmse_um']**2)
        for entry in self.report['fixed_test']:
            column='legacy_prediction_um' if entry['mode']=='legacy' else entry['mode']
            if column in fixed:
                measured=condition_model.metrics(fixed.observed_um.to_numpy(),fixed[column].to_numpy())
                self.assertAlmostEqual(measured['rmse_um'],entry['metrics']['rmse_um'])
        grouped=pd.read_csv(analysis_api.MODEL/'grouped_oof.csv')
        for name,metrics in self.report['nested_group_cv'].items():
            observed=condition_model.metrics(grouped.observed_um.to_numpy(),grouped[name].to_numpy())
            self.assertAlmostEqual(observed['mae_um'],metrics['mae_um'])

    def test_validation_winner_and_model_fingerprint(self):
        choices=[r for r in self.report['fixed_test'] if r['mode'] in ('multimodal','sequence_environment')]
        expected=min(choices,key=lambda r:r['validation']['log_rmse'])
        self.assertEqual(self.report['selected'],expected['spec'])
        self.assertEqual(hashlib.sha256((analysis_api.MODEL/'model.pkl').read_bytes()).hexdigest(),self.report['model_sha256'])

    def test_invalid_conditions_and_unknown_engine(self):
        for extra in ({'ph':15},{'duration':0},{'temperature':float('nan')},{'ph':True},{'engine':'untrained'}):
            with self.assertRaises(ValueError):analysis_api.validate({'protein_id':'1',**extra})
        with self.assertRaises(ValueError):condition_model.environmental_features([[40,8,-1]])

    def test_zero_width_window_and_training_support(self):
        config=analysis_api.validate({'protein_id':'1','engine':'fusion','temperature_span':0,'ph_span':0,'duration_span':0})
        result=analysis_api.analyze(config)
        self.assertEqual(result['scenario_count'],1)
        self.assertEqual(len(result['surface']),1)
        self.assertEqual(len(result['ranking']),25)
        for row in result['ranking']:
            self.assertAlmostEqual(row['center_prediction_um'],row['robust_score_um'])
            self.assertEqual(row['scenario_std_um'],0)
        self.assertEqual(result['outside_training_range_fraction'],0)


class StructureImageTests(unittest.TestCase):
    def test_distance_units_and_png_dimensions(self):
        pixels,distances=structure_images.distance_image(np.array([[0,0,0],[3,4,0],[30,0,0]],dtype=float))
        self.assertEqual(distances[0,1],5)
        self.assertEqual(pixels[0,1],191)
        self.assertEqual(pixels[0,0],255)
        self.assertEqual(pixels[0,2],0)
        content=structure_images.png(pixels)
        self.assertEqual(content[:8],b'\x89PNG\r\n\x1a\n')
        self.assertEqual(struct.unpack('!II',content[16:24]),(3,3))

    def test_verified_render_is_deterministic_and_downloadable(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            output=Path(folder)
            with patch.object(structure_images,'OUTPUT',output),patch.object(analysis_api,'IMAGE_ROOT',output):
                first=structure_images.generate('1');second=structure_images.generate('1')
                self.assertEqual(first,second)
                self.assertEqual(first['residues'],len(structure_images.catalog()['1']['sequence']))
                for image in first['images']:
                    self.assertEqual(hashlib.sha256((output/'1'/image['name']).read_bytes()).hexdigest(),image['sha256'])
                response=server.app.test_client().get('/api/analysis/images/1/bundle.zip',base_url='http://127.0.0.1:8765')
                self.assertEqual(response.status_code,200)
                with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
                    self.assertEqual(set(archive.namelist()),{'front.svg','side.svg','top.svg','distance.png','contacts.png','manifest.json'})
                response.close()

    def test_missing_and_path_like_proteins_are_rejected(self):
        for pid in ('../1',None,'999999'):
            with self.assertRaises(ValueError):structure_images.generate(pid)
        missing=next(pid for pid in structure_images.catalog() if not (structure_images.PET_ROOT/f'data/structure_images/pdb/{pid}.pdb').is_file())
        with self.assertRaises(ValueError):structure_images.generate(missing)


class AnalysisApiTests(unittest.TestCase):
    def setUp(self):
        self.client=server.app.test_client();self.base='http://127.0.0.1:8765'

    def test_api_access_limits(self):
        response=self.client.get('/analysis',base_url=self.base)
        self.assertEqual(response.status_code,200);response.close()
        self.assertEqual(self.client.post('/api/analysis/run',json={'protein_id':'1'},base_url=self.base).status_code,403)
        self.assertEqual(self.client.get('/api/analysis/evaluation/model.pkl',base_url=self.base).status_code,404)
        self.assertEqual(self.client.get('/api/analysis/images/1/../../model.pkl',base_url=self.base).status_code,404)
        self.assertEqual(self.client.get('/api/analysis/runs/invalid',base_url=self.base).status_code,404)


if __name__=='__main__':unittest.main()
