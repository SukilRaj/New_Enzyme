"""Collect a self-contained scientific notebook bundle without frontend/services."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PET = Path(os.environ.get('ENZAIME_ROOT', r'D:\Final_Year_Project\EnzAIme_ML'))
BUNDLE = ROOT / 'panel_notebooks'


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    BUNDLE.mkdir(exist_ok=True)
    for name in ('condition_model.py', 'assay_model.py', 'structure_images.py', 'engine.py'):
        copy_file(ROOT/'lab'/name, BUNDLE/'lab'/name)
    for name in ('train_condition_models.py', 'train_assay_models.py'):
        copy_file(ROOT/'scripts'/name, BUNDLE/'scripts'/name)
    for directory in ('models/condition_analysis', 'models/assay_v2/train-20261024-160717',
                      'runs/20261023-162906-a42ad8', 'runs/20261024-092703-afed29',
                      'runs/20261024-093528-uncertainty-diversity'):
        for path in (ROOT/directory).glob('*'):
            if path.is_file(): copy_file(path, BUNDLE/directory/path.name)
    for directory in ('data/processed', 'data/neural_features', 'data/structure_images', 'models/neural'):
        for path in (PET/directory).rglob('*'):
            if path.is_file():
                copy_file(path, BUNDLE/'pet'/path.relative_to(PET))
    for relative in ('master_dataset.csv','data/tpa_pet_ml_dataset.csv','data/fireprotdb/model_protein_mutations.csv',
                     'reports/results/tpa_ambiguous_review.md'):
        copy_file(PET/relative,BUNDLE/'pet'/relative)
    for name in ('neural_oracle.py', 'neural_recommender.py','mutation.py','structure.py'):
        copy_file(PET/'backend/app/services'/name, BUNDLE/'pet/backend/app/services'/name)
    for name in ('train_neural_oracle.py', 'prepare_neural_features.py', 'prepare_structure_images.py',
                 'build_tpa_provenance_audit.py','build_confirmed_tpa_dataset.py'):
        copy_file(PET/'scripts'/name, BUNDLE/'pet/scripts'/name)
    upstream = ROOT/'Directed_Evolution-main'
    for path in upstream.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and (path.suffix == '.py' or path.name in ('README.md','LICENSE','avGFP.csv','avGFP_reference_sequence.txt')):
            copy_file(path, BUNDLE/'Directed_Evolution-main'/path.relative_to(upstream))
    for short, model in [('esm2_8m','facebook/esm2_t6_8M_UR50D'),('esm2_35m','facebook/esm2_t12_35M_UR50D')]:
        cache=ROOT/'models/huggingface/hub'/('models--'+model.replace('/','--'))
        revision=(cache/'refs/main').read_text().strip()
        for path in (cache/'snapshots'/revision).iterdir():
            if path.is_file(): copy_file(path,BUNDLE/'models'/short/path.name)
        (BUNDLE/'models'/short/'SOURCE.json').write_text(json.dumps({'model':model,'revision':revision},indent=2),encoding='utf-8')
    for name in ('BENCHMARK_COMPARISON_AND_GAPS.md','PROJECT_AND_JOURNAL_SUMMARY.md'):
        copy_file(ROOT/'docs/journal'/name,BUNDLE/'documentation'/name)
    copy_file(ROOT/'REPLICATION_STATUS.md',BUNDLE/'documentation/REPLICATION_STATUS.md')
    manifest={'schema_version':1,'name':'EnzAIme Helix panel notebooks',
              'description':'Scientific data, source and checkpoints; no frontend or running server needed.',
              'origin':{'helix':str(ROOT),'pet':str(PET)},'files':{}}
    for path in sorted(BUNDLE.rglob('*')):
        if path.is_file() and path.name != 'bundle_manifest.json' and 'outputs' not in path.parts and '__pycache__' not in path.parts:
            manifest['files'][path.relative_to(BUNDLE).as_posix()]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    (BUNDLE/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'root':str(BUNDLE),'files':len(manifest['files']),'megabytes':round(sum(v['bytes'] for v in manifest['files'].values())/1e6,2)},indent=2))


if __name__ == '__main__': main()
