"""Package the manuscript basis and selected evidence; not a standalone application."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PET = Path(os.environ.get('ENZAIME_ROOT', r'D:\Final_Year_Project\EnzAIme_ML'))
OUT = ROOT / 'docs/journal'


def main():
    output = OUT / 'ENZAIME_HELIX_JOURNAL_PACKAGE.zip'
    manifest = OUT / 'evidence_manifest.json'
    paths = set()
    for directory, extensions in [('lab', {'.py', '.js', '.css', '.html'}),
                                  ('scripts', {'.py'}), ('tests', {'.py'}),
                                  ('frontend/src', {'.js', '.jsx', '.css'}),
                                  ('frontend/tests', {'.js'}), ('frontend/dist', {'.html', '.js', '.css'})]:
        paths.update(p for p in (ROOT / directory).rglob('*') if p.is_file() and p.suffix in extensions)
    for name in ('README.md', 'VERIFICATION.md', 'REPLICATION_STATUS.md', 'launch_research.py',
                 'Start-Research.cmd', 'Check-Research.cmd', 'EnzAIme-Helix.code-workspace',
                 'requirements-core.txt', 'requirements-local.txt', 'requirements-report.txt',
                 'frontend/package.json', 'frontend/package-lock.json', '.vscode/tasks.json',
                 'docs/RUN_IN_VS_CODE.md', 'docs/SEQUENCE_AND_STRUCTURE_GUIDE.md',
                 'docs/ANALYSIS_METHODS_AND_RESULTS.md', 'docs/PROJECT_COMPARISON_AND_INTEGRATION.md'):
        paths.add(ROOT / name)
    evidence = set()
    for directory in ('models/condition_analysis', 'models/assay_v2/train-20261024-160717',
                      'runs/20261023-162906-a42ad8'):
        evidence.update(p for p in (ROOT / directory).glob('*') if p.is_file() and p.suffix in {'.json', '.csv'})
    for name in ('integration_verification.json', 'analysis_verification.json',
                 'workspace_verification.json', 'sequence_workflow_verification.json', 'project_preflight.json',
                 'fasta-import.png', 'reference-structure-inspection.png', 'unified-training.png'):
        evidence.add(ROOT / 'reports' / name)
    paths.update(evidence)
    paths.update(p for p in OUT.rglob('*') if p.is_file() and p != manifest and p.suffix != '.zip')
    paths.update(ROOT / name for name in ('models/condition_analysis/model.pkl', 'models/assay_v2/train-20261024-160717/model.pkl'))
    pet_paths = [PET / name for name in ('models/neural/manifest.json', 'models/neural/weights.pt',
                 'models/neural/held_out_predictions.json', 'data/processed/tpa_pet_ml_confirmed.csv',
                 'data/neural_features/index.json', 'data/structure_images/manifest.json',
                 'backend/app/workspace_app.py')]
    entries = []
    for origin, root, collection in [('helix', ROOT, sorted(paths)), ('pet', PET, pet_paths)]:
        for path in collection:
            if not path.is_file():
                raise FileNotFoundError(f'Required handover evidence missing: {path}')
            entries.append({'project': origin, 'path': path.relative_to(root).as_posix(),
                            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    content = {'created_at': datetime.now(timezone.utc).isoformat(), 'schema_version': 1,
               'purpose': 'Local manuscript evidence and implementation fingerprints. Checkpoints are hashed, not included in the journal ZIP.',
               'roots': {'helix': str(ROOT), 'pet': str(PET)}, 'files': entries}
    manifest.write_text(json.dumps(content, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, 'journal/' + path.relative_to(OUT).as_posix())
        for path in sorted(evidence):
            archive.write(path, 'evidence/' + path.relative_to(ROOT).as_posix())
        for name in ('RUN_IN_VS_CODE.md', 'SEQUENCE_AND_STRUCTURE_GUIDE.md'):
            archive.write(ROOT / 'docs' / name, 'evidence/docs/' + name)
        for path in pet_paths:
            if path.suffix == '.json':
                archive.write(path, 'evidence/pet/' + path.relative_to(PET).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        count = len(archive.namelist())
    print(json.dumps({'zip': str(output), 'files_in_zip': count, 'fingerprinted_files': len(entries),
                      'zip_bytes': output.stat().st_size, 'zip_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
