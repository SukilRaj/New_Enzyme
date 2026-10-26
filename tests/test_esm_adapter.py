"""Offline integration test: tiny random weights, never evidence of model quality."""
from pathlib import Path
import sys
import tempfile
import unittest
import importlib.util
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lab'))
(Path(__file__).resolve().parents[1]/'tmp').mkdir(exist_ok=True)


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ('torch','transformers','polyleven')), 'Optional ESM dependencies not installed')
class ESMAdapterTests(unittest.TestCase):
    def test_real_tensor_pipeline_with_tiny_random_model(self):
        import torch
        from transformers import EsmConfig, EsmTokenizer, EsmModel, EsmForMaskedLM
        from engine import run, AA
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as folder:
            root = Path(folder)
            vocab = ['<cls>', '<pad>', '<eos>', '<unk>']+list(AA)+['X', '<mask>']
            (root/'vocab.txt').write_text('\n'.join(vocab))
            tokenizer = EsmTokenizer(str(root/'vocab.txt'))
            config = EsmConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,
                               num_attention_heads=2,intermediate_size=32,
                               pad_token_id=1,mask_token_id=len(vocab)-1,
                               position_embedding_type='rotary')
            output = root/'run'
            output.mkdir()
            with patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer), \
                 patch('transformers.EsmModel.from_pretrained',side_effect=lambda *a,**k:EsmModel(config)), \
                 patch('transformers.EsmForMaskedLM.from_pretrained',side_effect=lambda *a,**k:EsmForMaskedLM(config)):
                result = run(dict(engine='esm',samples=100,seed=1,steps=1,population=2,epochs=1),
                             output,lambda *a,**k:None,lambda:False)
            self.assertTrue((output/'decoder.pt').exists())
            self.assertEqual(result['metrics']['test_count'],20)
            self.assertTrue(result['candidates'])
            self.assertFalse(result['manifest']['paper_reproduction'])
            self.assertTrue(all(set(c['sequence'])<=set(AA) for c in result['candidates']))


if __name__=='__main__':
    unittest.main()

