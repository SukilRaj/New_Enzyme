"""Generate executable teaching notebooks with the implementation visible in cells."""
import ast
import json
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'panel_notebooks'
TEMPLATES=ROOT/'scripts/panel_cells'


def source(path): return path.read_text(encoding='utf-8-sig')


def scientific_source(name):
    text=source(BUNDLE/'lab'/name)
    lines=text.splitlines()
    lines=[line for line in lines if not line.startswith(('ROOT =','PET_ROOT =','OUTPUT ='))]
    return '\n'.join(lines)


def definition(path,name):
    text=source(path)
    node=next(n for n in ast.parse(text).body if getattr(n,'name',None)==name)
    return '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])


COMMON=r'''
from pathlib import Path
import os, sys, json, hashlib, importlib.metadata, datetime, uuid

# Set this only if automatic discovery cannot find your attached Kaggle dataset.
BUNDLE_OVERRIDE = os.environ.get('ENZAIME_NOTEBOOK_ROOT', '')
def find_bundle():
    candidates = [Path(BUNDLE_OVERRIDE)] if BUNDLE_OVERRIDE else []
    candidates += [Path.cwd(), Path.cwd()/'panel_notebooks', Path.cwd().parent]
    if Path('/kaggle/input').exists():
        candidates += [p.parent for p in Path('/kaggle/input').rglob('bundle_manifest.json')]
    valid = list(dict.fromkeys(p.resolve() for p in candidates if (p/'bundle_manifest.json').is_file()))
    if not valid:
        raise FileNotFoundError('Extract EnzAIme_Helix_Notebook_Package.zip, or attach its extracted folder as a Kaggle dataset. Set BUNDLE_OVERRIDE to its panel_notebooks folder.')
    if BUNDLE_OVERRIDE and valid: return valid[0]
    if len(valid)>1: raise ValueError('Multiple bundles found. Set BUNDLE_OVERRIDE to the one to use.')
    return valid[0]

BUNDLE = ROOT = find_bundle()
PET_ROOT = BUNDLE/'pet'
OUTPUT_ROOT = Path(os.environ.get('ENZAIME_NOTEBOOK_OUTPUT',
    '/kaggle/working/enzaime_outputs' if Path('/kaggle/working').exists() else str(BUNDLE/'outputs')))
RUN_DIR = OUTPUT_ROOT/(NOTEBOOK_NAME+'-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
RUN_DIR.mkdir(parents=True, exist_ok=False)
os.environ['ENZAIME_ROOT'] = str(PET_ROOT)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, str(BUNDLE/'lab'))
sys.path.insert(0, str(BUNDLE/'Directed_Evolution-main'))
bundle_manifest = json.loads((BUNDLE/'bundle_manifest.json').read_text(encoding='utf-8'))
def verify_file(relative):
    path = BUNDLE/relative
    expected = bundle_manifest['files'][relative]['sha256']
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected: raise ValueError(f'Artifact hash mismatch: {relative}')
    return path

print('Data and saved models:', BUNDLE)
print('New outputs:', RUN_DIR)
print('Python:', sys.version.split()[0])
'''

DEPS=r'''
# On a fresh Kaggle session, set True once if the version check below requests it.
# After installation, restart the kernel/session and return this flag to False.
INSTALL_DEPENDENCIES = False
if INSTALL_DEPENDENCIES:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(BUNDLE/'requirements-kaggle.txt')])
    raise RuntimeError('Dependencies installed. Restart the kernel/session, set INSTALL_DEPENDENCIES=False, then Run All.')

expected_versions={'numpy':'1.26.4','scipy':'1.13.1','scikit-learn':'1.5.2','pandas':'2.2.3'}
mismatches={}
for name, version in expected_versions.items():
    try: actual=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: actual='missing'
    if actual != version: mismatches[name]={'installed':actual,'required':version}
if mismatches:
    raise RuntimeError(f'Checkpoint-compatible package versions required: {mismatches}. Run the installation cell with INSTALL_DEPENDENCIES=True, then restart the kernel.')

import copy, csv, pickle, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
_thread_limit = threadpool_limits(limits=2)
plt.rcParams.update({'figure.figsize':(9,4), 'axes.spines.top':False, 'axes.spines.right':False, 'font.size':10})
np.random.seed(42)

def evaluate_predictions(observed, predicted):
    y,p=np.asarray(observed,dtype=float),np.asarray(predicted,dtype=float)
    if y.shape!=p.shape or not np.isfinite(np.column_stack([y,p])).all():
        raise ValueError('Evaluation requires aligned, finite labels and predictions.')
    return {'MAE':float(mean_absolute_error(y,p)), 'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'R2':float(r2_score(y,p)) if len(y)>1 and np.ptp(y) else None,
            'Spearman':float(spearmanr(y,p).statistic) if np.ptp(y) and np.ptp(p) else None,
            'n':len(y)}

def save_figure(name):
    plt.tight_layout(); plt.savefig(RUN_DIR/(name+'.png'),dpi=160,bbox_inches='tight'); plt.show()

versions={name:importlib.metadata.version(name) for name in expected_versions}
(RUN_DIR/'environment.json').write_text(json.dumps({'python':sys.version,'packages':versions},indent=2))
display(pd.Series(versions,name='Installed version').to_frame())
'''


def parse_percent(text):
    cells=[];mode='code';buffer=[]
    def flush():
        content='\n'.join(buffer).strip()
        if content: cells.append(nbf.v4.new_markdown_cell(content) if mode=='markdown' else nbf.v4.new_code_cell(content))
    for line in text.splitlines():
        if line.startswith('# %%'):
            flush();buffer=[];mode='markdown' if '[markdown]' in line else 'code'
        else: buffer.append(line)
    flush();return cells


def main():
    def full_study(name,flag):
        text=source(BUNDLE/'scripts'/name)
        text=text.replace("ROOT = Path(__file__).resolve().parents[1]",'ROOT = WORK').replace("ROOT=Path(__file__).resolve().parents[1]",'ROOT=WORK')
        text=text.replace("if __name__=='__main__':",f'if {flag}:').replace("if __name__ == '__main__':",f'if {flag}:')
        return f"__file__ = str(WORK/'scripts/{name}')\n"+text
    engine=source(BUNDLE/'lab/engine.py').replace('ROOT = Path(__file__).resolve().parents[1]','ROOT = BUNDLE')
    replacements={'COMMON':COMMON,'DEPS':DEPS,
        'CONDITION_SOURCE':scientific_source('condition_model.py'),
        'ASSAY_SOURCE':scientific_source('assay_model.py'),
        'NEURAL_SOURCE':source(BUNDLE/'pet/backend/app/services/neural_oracle.py'),
        'DECODER_SOURCE':source(BUNDLE/'Directed_Evolution-main/de/predictors/attention/decoder.py'),
        'COORDINATES_SOURCE':'\n\n'.join(definition(BUNDLE/'lab/structure_images.py',n) for n in ('catalog','coordinates','distance_image','trace_svg')),
        'FULL_CONDITION_STUDY':full_study('train_condition_models.py','RUN_FULL_CONDITION_STUDY'),
        'FULL_ASSAY_STUDY':full_study('train_assay_models.py','RUN_FULL_ASSAY_STUDY'),
        'FULL_GFP_ENGINE':"__file__ = str(BUNDLE/'lab/engine.py')\n"+engine,
    }
    for template in sorted(TEMPLATES.glob('*.py')):
        text=source(template)
        for key,value in replacements.items(): text=text.replace('{{'+key+'}}',textwrap.dedent(value).strip())
        if '{{' in text: raise ValueError('Unresolved notebook template marker')
        notebook=nbf.v4.new_notebook(cells=parse_percent(text),metadata={
            'kernelspec':{'display_name':'Python 3 (EnzAIme)','language':'python','name':'python3'},
            'language_info':{'name':'python','version':'3.12'},
            'enzaime':{'source_template':template.name,'execution':'Top to bottom; no frontend/server required'}})
        nbf.validate(notebook)
        nbf.write(notebook,BUNDLE/(template.stem+'.ipynb'))
        # Export a readable, executable Python version with Markdown as comments.
        exported=[]
        for cell in notebook.cells:
            exported.append('# %% [markdown]\n'+'\n'.join('# '+l for l in cell.source.splitlines()) if cell.cell_type=='markdown' else '# %%\n'+cell.source)
        python='\n\n'.join(exported)+'\n';compile(python,template.stem,'exec')
        (BUNDLE/(template.stem+'.py')).write_text(python,encoding='utf-8')
        print(template.stem, len(notebook.cells),'cells')


if __name__=='__main__': main()
