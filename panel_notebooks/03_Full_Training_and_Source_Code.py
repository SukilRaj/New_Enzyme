# %% [markdown]
# # Full training studies and original scientific source
# 
# Use notebook 01 for the PET live demonstration and notebook 02 for GFP. This notebook exposes the original complete training scripts and the application's full GFP engine. Long studies are opt-in because they perform many nested fits. Every output is directed to a new writable work folder; the bundled saved results are preserved.
# 
# The original base-paper source is in `Directed_Evolution-main/`, the PET checkpoint architecture is in `pet/backend/app/services/neural_oracle.py`, and all condition/assay implementations are in `lab/`. No frontend is needed. The full condition study evaluates 25 configurations; the assay study evaluates 24 configurations under sequence and sequence/study grouping. Completion is a computational result, not proof of high accuracy.

# %%
NOTEBOOK_NAME='full_studies'
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

# %%
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

# %% [markdown]
# ## 1. Select the work to run
# Leave both flags False for a quick code walkthrough. To reproduce the complete studies, set the desired flag True and run from this cell downward. Runtime depends on CPU/GPU and may be many minutes. The assay study can use the bundled original condition split without rerunning the earlier condition study.

# %%
RUN_FULL_CONDITION_STUDY=False
RUN_FULL_ASSAY_STUDY=False
import shutil
WORK=RUN_DIR/'reproduction'
WORK.mkdir()
for folder in ('lab','scripts','models/condition_analysis'):
    shutil.copytree(BUNDLE/folder,WORK/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
print('Complete-study work folder:',WORK)
print('Condition study requested:',RUN_FULL_CONDITION_STUDY)
print('Assay study requested:',RUN_FULL_ASSAY_STUDY)

# %% [markdown]
# ## 2. Full condition/image training script
# All imports, candidate selection, nested folds, paired bootstrap, artifact persistence and metric computation are visible in the next cell. Only the working-directory assignment and execution guard differ from the supplied script. The implementations imported from `lab/` are shown in notebook 01 and shipped as source files.

# %%
__file__ = str(WORK/'scripts/train_condition_models.py')
"""One reproducible experiment: select on validation, report the untouched fixed test.

The old test has been inspected historically, so this is exploratory comparison,
not fresh external validation. No legacy checkpoint is overwritten.
"""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = WORK
sys.path.insert(0, str(ROOT / 'lab'))
from condition_model import PET_ROOT, fit_predict, load_data, metrics, specs


def main():
    data = load_data()
    legacy = json.loads((PET_ROOT / 'models/neural/manifest.json').read_text())
    split = {name: np.flatnonzero(np.isin(data['groups'], hashes)) for name, hashes in legacy['split_sequence_hashes'].items()}
    train, val, test = (split[k] for k in ('train','validation','test'))
    assert sum(map(len,split.values())) == len(data['rows']) == legacy['rows_used']
    assert not set(data['groups'][train]) & set(data['groups'][test])
    folder = ROOT / 'models/condition_analysis'
    folder.mkdir(parents=True, exist_ok=True)
    candidates = specs()
    protocol = {'created_at': time.time(), 'primary_selection_metric': 'validation log_rmse', 'candidates': candidates,
                'dataset_sha256': data['dataset_sha256'], 'split': {k:v.tolist() for k,v in split.items()},
                'split_sequence_hashes': legacy['split_sequence_hashes'],
                'note': 'Historical fixed holdout; exploratory, not independent external validation. Models train on original training split only.'}
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    fits = []
    for spec in candidates:
        model, predictions = fit_predict(data, spec, train, val)
        score = metrics(data['target'][val], predictions)
        fits.append((spec, model, score))
        print(f"Validation {spec['mode']} {spec['kind']} {spec['alpha']}: logRMSE={score['log_rmse']:.4f}", flush=True)
    # Preserve all ablations; winning hyperparameters and mode are chosen on validation only.
    chosen = {mode:min([f for f in fits if f[0]['mode']==mode], key=lambda f:f[2]['log_rmse']) for mode in sorted({s['mode'] for s in candidates})}
    eligible = [f for f in fits if f[0]['mode'] in ('sequence_environment', 'multimodal')]
    selected = min(eligible, key=lambda f:f[2]['log_rmse'])
    old_rows = json.loads((PET_ROOT/'models/neural/held_out_predictions.json').read_text())
    assert len(old_rows)==len(test)
    for i, row in zip(test, old_rows):
        assert data['rows'][i]['protein_id']==row['protein_id']
        assert np.isclose(data['target'][i], row['observed_um'], atol=.002, rtol=1e-5)
    results = [{'name':'Original EnzAIme neural checkpoint', 'mode':'legacy', 'metrics': metrics(data['target'][test], np.array([r['predicted_um'] for r in old_rows]))}]
    predictions_by_mode = {}
    for mode,(spec,model,validation) in chosen.items():
        pred = model.predict(*(data[k][test] for k in ('sequence','env','image')))
        results.append({'name': mode.replace('_',' + '), 'mode':mode,'spec':spec,'validation':validation,'metrics':metrics(data['target'][test],pred)})
        predictions_by_mode[mode]=pred
    baseline = np.repeat(np.expm1(np.mean(np.log1p(data['target'][train]))),len(test))
    results.append({'name':'Training log-mean baseline','mode':'training_mean','metrics':metrics(data['target'][test],baseline)})
    spec, model, val_metrics = selected
    with (folder/'model.pkl').open('wb') as stream:
        pickle.dump(model,stream)
    # Validation residual quantile is an empirical diagnostic, not a guaranteed calibrated interval.
    val_prediction = model.predict(*(data[k][val] for k in ('sequence','env','image')))
    residual = float(np.quantile(np.abs(np.log1p(val_prediction)-np.log1p(data['target'][val])), .9, method='higher'))
    selected_test = model.predict(*(data[k][test] for k in ('sequence','env','image')))
    # Protein-bootstrap paired comparison retains correlated assay rows within proteins.
    old = np.array([r['predicted_um'] for r in old_rows]); groups=data['groups'][test]; unique=np.unique(groups); rng=np.random.default_rng(42)
    differences=[]
    for _ in range(1000):
        sampled=rng.choice(unique,len(unique),replace=True); ids=np.concatenate([np.flatnonzero(groups==g) for g in sampled])
        differences.append(float(np.abs(selected_test[ids]-data['target'][test][ids]).mean()-np.abs(old[ids]-data['target'][test][ids]).mean()))
    # Nested group CV supplies a broader all-protein assessment without reusing the fixed test for tuning.
    # Restrict each ablation to the prespecified family; outer predictions are never selection input.
    cv_predictions={mode:np.full(len(data['rows']),np.nan) for mode in ('sequence','environment','sequence_environment','multimodal')}
    cv_manifest=[]
    for fold,(outer_train,outer_test) in enumerate(GroupKFold(5).split(data['env'],groups=data['groups']),1):
        inner_scores={i:[] for i,s in enumerate(candidates) if s['mode'] in cv_predictions}
        inner_manifests=[]
        for a,b in GroupKFold(3).split(outer_train,groups=data['groups'][outer_train]):
            a,b=outer_train[a],outer_train[b]
            inner_manifests.append({'train':a.tolist(),'validation':b.tolist()})
            for index in inner_scores:
                _,pred=fit_predict(data,candidates[index],a,b)
                inner_scores[index].extend((np.log1p(pred)-np.log1p(data['target'][b]))**2)
        selection={}
        for mode in cv_predictions:
            best=min((i for i in inner_scores if candidates[i]['mode']==mode),key=lambda i:float(np.mean(inner_scores[i])))
            _,pred=fit_predict(data,candidates[best],outer_train,outer_test)
            cv_predictions[mode][outer_test]=pred
            selection[mode]=candidates[best]
        cv_manifest.append({'fold':fold,'train':outer_train.tolist(),'test':outer_test.tolist(),'inner_splits':inner_manifests,'selection':selection})
        print(f'Nested grouped evaluation {fold}/5 complete',flush=True)
    cv_scores={mode:metrics(data['target'],pred) for mode,pred in cv_predictions.items()}
    report={'schema_version':1,'trained_at':time.time(),'encoder':data['encoder'],'dataset_sha256':data['dataset_sha256'],
            'rows':len(data['rows']),'proteins':len(np.unique(data['groups'])),'test_rows':len(test),'test_proteins':len(unique),
            'selected':spec,'selection_metric':'validation log_rmse','validation':val_metrics,'fixed_test':results,
            'selected_test':metrics(data['target'][test],selected_test),'nested_group_cv':cv_scores,
            'paired_mae_difference_95_interval_um':np.quantile(differences,[.025,.975]).tolist(),
            'paired_mae_difference_um':float(np.mean(np.abs(selected_test-data['target'][test])-np.abs(old-data['target'][test]))),
            'model_sha256':hashlib.sha256((folder/'model.pkl').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256((ROOT/'lab/condition_model.py').read_bytes()).hexdigest(),
            'environment_min':data['env'][train].min(0).tolist(),'environment_max':data['env'][train].max(0).tolist(),
            'environment_mean':data['env'][train].mean(0).tolist(),'environment_std':data['env'][train].std(0).clip(1e-6).tolist(),
            'validation_absolute_log_residual_q90':residual,
            'limitations':['Historical test already inspected; exploratory comparison.', '25 sequence groups; related families/studies may cross splits.', 'Validation chooses models; test is reported once per model.', 'Images exist for 13 proteins; image benefit is not established by model selection.', 'Empirical residual bands are not calibrated confidence intervals.', 'Assay material and enzyme dose are not model inputs.']}
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (folder/'cv_splits.json').write_text(json.dumps(cv_manifest,indent=2),encoding='utf-8')
    with (folder/'heldout.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['source_row','protein_id','observed_um','legacy_prediction_um','selected_prediction_um',*predictions_by_mode])
        for j,i in enumerate(test):
            writer.writerow([data['rows'][i]['source_row'],data['rows'][i]['protein_id'],data['target'][i],old[j],selected_test[j],*[p[j] for p in predictions_by_mode.values()]])
    with (folder/'grouped_oof.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['source_row','protein_id','observed_um',*cv_predictions])
        for i,row in enumerate(data['rows']):writer.writerow([row['source_row'],row['protein_id'],data['target'][i],*[p[i] for p in cv_predictions.values()]])
    print(json.dumps({'selected':spec,'selected_test':report['selected_test'],'legacy':results[0]['metrics'],'nested_group_cv':cv_scores},indent=2),flush=True)


if RUN_FULL_CONDITION_STUDY:
    with threadpool_limits(limits=2):main()

# %% [markdown]
# ## 3. Full assay-context training script
# This reproduces pre-assay context extraction, 24 prespecified configurations, training-only preprocessing, nested selection, strict connected sequence/study partitions, and the final all-data refit. The refit's predictions on its own training rows are not an independent evaluation.

# %%
__file__ = str(WORK/'scripts/train_assay_models.py')
"""Frozen v2 protocol: nested group validation, assay context, final full-data refit."""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import traceback
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT=WORK
sys.path.insert(0,str(ROOT/'lab'))
from assay_model import dataset, specs, fit_predict, score, group_error, CONTEXT_NAMES

BASE=ROOT/'models/assay_v2'


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def components(data):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for p,s in zip(data['groups'],data['studies']):
        parent[find('p:'+p)]=find('s:'+s)
    return np.array([find('p:'+p) for p in data['groups']])


def main():
    BASE.mkdir(exist_ok=True,parents=True)
    run_id='train-'+time.strftime('%Y%m%d-%H%M%S')
    folder=BASE/run_id;folder.mkdir()
    write(BASE/'latest.json',{'run_id':run_id})
    started=time.time();history=[]
    def update(stage,progress,detail,status='running'):
        history.append({'time':time.time(),'stage':stage,'detail':detail})
        write(folder/'status.json',dict(run_id=run_id,status=status,stage=stage,progress=progress,detail=detail,started=started,updated=time.time(),log=history[-80:]))
        print(f'{progress:5.1f}% {stage}: {detail}',flush=True)
    try:
        update('Audit',1,'Extracting only pre-assay fields; preserving every eligible observation.')
        data=dataset();candidates=specs();strict=components(data)
        old_protocol=json.loads((ROOT/'models/condition_analysis/protocol.json').read_text())
        train,val,test=(np.array(old_protocol['split'][k]) for k in ('train','validation','test'))
        coverage={k:int(np.isfinite(data['context'][:,i]).sum()) for i,k in enumerate(CONTEXT_NAMES)}
        protocol={'schema_version':2,'run_id':run_id,'created':time.time(),'candidates':candidates,
                  'selection_metric':'mean absolute log1p error, averaged equally over sequence groups',
                  'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'lab/assay_model.py',Path(__file__),ROOT/'lab/condition_model.py')},
                  'dataset_sha256':data['dataset_sha256'],'context_fields':CONTEXT_NAMES,'context_coverage':coverage,
                  'historical_split':old_protocol['split'],'sequence_groups':data['groups'].tolist(),'studies':data['studies'].tolist(),
                  'strict_groups':strict.tolist(),'strict_group_count':len(np.unique(strict)),
                  'target':'TPA concentration in micromolar; training target log1p(TPA)',
                  'notes':['All existing data is development data; no fresh external test exists.',
                           'No outcomes, target units, identifiers, free-text evidence or DOI are model features.',
                           'Missing or conflicting pre-assay fields are missing, not inferred from labels.',
                           'Feature set and candidate search fixed before v2 fitting.',
                           'Final serving model refits all 208 observations; its training fit is not a held-out score.']}
        write(folder/'protocol.json',protocol)
        write(folder/'context_audit.json',{'coverage':coverage,'rows':len(data['rows']),
            'input_fields':CONTEXT_NAMES,'material_forms':['film','powder','particle','fiber','unknown'],
            'excluded':['measurements','target_unit','protein_id','protein_name','literature_doi','evidence','reported_result','additional_conditions']})
        fits=[]
        for index,spec in enumerate(candidates):
            model,pred=fit_predict(data,spec,train,val)
            fits.append((spec,model,score(data['target'][val],pred,data['groups'][val])))
            update('Validation search',3+index/len(candidates)*12,f"{index+1}/{len(candidates)} · {spec['mode']} / {spec['kind']} / {spec['strength']}")
        chosen={mode:min((f for f in fits if f[0]['mode']==mode),key=lambda f:f[2]['group_log_mae']) for mode in {s['mode'] for s in candidates}}
        eligible=[f for f in fits if f[0]['mode'].startswith('sequence')]
        winner=min(eligible,key=lambda f:f[2]['group_log_mae'])
        with (folder/'historical_model.pkl').open('wb') as stream:pickle.dump(winner[1],stream)
        fixed={};fixed_predictions={}
        for mode,(spec,model,validation) in chosen.items():
            pred=model.predict(*(data[k][test] for k in ('sequence','env','context')))
            fixed[mode]={'spec':spec,'validation':validation,'metrics':score(data['target'][test],pred,data['groups'][test])}
            fixed_predictions[mode]=pred
        selection_fixed={'spec':winner[0],'validation':winner[2],'test':fixed[winner[0]['mode']]['metrics']}

        all_splits={};evaluations={};oof={};selection_scores={}
        designs=[('sequence',data['groups'],5)]
        if len(np.unique(strict))>=4:designs.append(('sequence_and_study',strict,min(5,len(np.unique(strict)))))
        for design_index,(design,groups,nfolds) in enumerate(designs):
            predictions={mode:np.full(len(groups),np.nan) for mode in chosen}
            predictions['training_median']=np.full(len(groups),np.nan)
            splits=[];candidate_errors={i:[] for i in range(len(candidates))}
            for fold,(a,b) in enumerate(GroupKFold(nfolds).split(data['env'],groups=groups),1):
                inner_scores={i:[] for i in range(len(candidates))};inners=[]
                inner_n=min(3,len(np.unique(groups[a])))
                for inner,(c,d) in enumerate(GroupKFold(inner_n).split(a,groups=groups[a]),1):
                    c,d=a[c],a[d];inners.append({'train':c.tolist(),'validation':d.tolist()})
                    for index,spec in enumerate(candidates):
                        _,p=fit_predict(data,spec,c,d)
                        # One loss per sequence, so repeated assays cannot dominate selection.
                        error=np.abs(np.log1p(p)-np.log1p(data['target'][d]))
                        for g in np.unique(data['groups'][d]):inner_scores[index].append(float(error[data['groups'][d]==g].mean()))
                    progress=16+(design_index+(fold-1+inner/inner_n)/nfolds)/len(designs)*72
                    update('Nested validation',progress,f'{design} · outer {fold}/{nfolds} · inner {inner}/{inner_n}')
                selections={}
                for mode in chosen:
                    best=min((i for i,s in enumerate(candidates) if s['mode']==mode),key=lambda i:np.mean(inner_scores[i]))
                    _,pred=fit_predict(data,candidates[best],a,b)
                    predictions[mode][b]=pred;selections[mode]=candidates[best]
                predictions['training_median'][b]=np.median(data['target'][a])
                for i,values in inner_scores.items():candidate_errors[i].extend(values)
                splits.append({'fold':fold,'train':a.tolist(),'test':b.tolist(),'inner_splits':inners,'selection':selections})
            all_splits[design]=splits
            evaluations[design]={mode:score(data['target'],p,data['groups']) for mode,p in predictions.items()}
            oof[design]=predictions;selection_scores[design]=candidate_errors
            write(folder/'splits.json',all_splits)
            with (folder/f'{design}_oof.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['source_row','protein_id','sequence_hash','study','observed_um',*predictions])
                for i,row in enumerate(data['rows']):writer.writerow([row['source_row'],row['protein_id'],row['sequence_hash'],row['literature_doi'],data['target'][i],*[p[i] for p in predictions.values()]])

        # Select the full-data recipe from pooled inner validation losses, not the outer errors.
        losses=selection_scores['sequence']
        best=min((i for i,s in enumerate(candidates) if s['mode'].startswith('sequence')),key=lambda i:np.mean(losses[i]))
        best_spec=candidates[best]
        update('Final refit',91,f"Training {best_spec} on all {len(data['rows'])} observations.")
        ids=np.arange(len(data['rows']));production,_=fit_predict(data,best_spec,ids,ids)
        with (folder/'model.pkl').open('wb') as stream:pickle.dump(production,stream)
        with (folder/'historical_predictions.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['source_row','protein_id','observed_um',*fixed_predictions])
            for j,i in enumerate(test):writer.writerow([data['rows'][i]['source_row'],data['rows'][i]['protein_id'],data['target'][i],*[p[j] for p in fixed_predictions.values()]])
        report={'schema_version':2,'run_id':run_id,'created':time.time(),'duration_seconds':time.time()-started,
                'rows':len(ids),'protein_groups':len(np.unique(data['groups'])),'studies':len(np.unique(data['studies'])),
                'strict_groups':len(np.unique(strict)),'dataset_sha256':data['dataset_sha256'],
                'model_sha256':hashlib.sha256((folder/'model.pkl').read_bytes()).hexdigest(),
                'production_spec':best_spec,'production_training_rows':len(ids),'historical_selection':selection_fixed,
                'fixed_test':fixed,'evaluation':evaluations,'context_coverage':coverage,
                'selection_metric':protocol['selection_metric'],'neural_ensemble_seeds':[42,43,44],
                'prespecified_configurations':len(candidates),'limitations':protocol['notes']+
                ['Assay context may be confounded with study; connected sequence/study grouping is reported separately.',
                 'No categorical accuracy score is claimed for this regression task.',
                 'More epochs cannot replace missing independent proteins and assay evidence.',
                 'Final model is a full-data research refit, not a newly validated biological oracle.']}
        write(folder/'report.json',report)
        update('Completed',100,'All planned folds, ablations and the full-data model refit completed.','completed')
        print(json.dumps({'run_id':run_id,'selected':best_spec,'historical':selection_fixed,'evaluation':evaluations},indent=2),flush=True)
    except BaseException as error:
        (folder/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        update('Failed',0,str(error),'failed');raise


if RUN_FULL_ASSAY_STUDY:
    with threadpool_limits(limits=2):main()

# %% [markdown]
# ## 4. Full GFP application engine
# The next cell contains the original application engine as executable definitions. It includes the complete data loader, configuration validation, frozen ESM/Attention1D training, greedy selection and three-head uncertainty/diversity option. It does not run a training job until `run(...)` is called. The 8M/35M local weights are used via a local Hugging Face cache assembled below, so the original source can run without a download.

# %%
__file__ = str(BUNDLE/'lab/engine.py')
"""Local avGFP experiments. Measured labels and model predictions stay separate."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from importlib.metadata import version, PackageNotFoundError

ROOT = BUNDLE
UPSTREAM = ROOT / 'Directed_Evolution-main'
sys.path.insert(0, str(UPSTREAM))
os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'huggingface'))
AA = 'ACDEFGHIKLMNPQRSTVWY'


def dataset():
    path = UPSTREAM / 'preprocessed_data/avGFP/avGFP.csv'
    with path.open() as f:
        rows = list(csv.DictReader(f))
    # Deduplicate before splitting, averaging repeated measurements.
    grouped = {}
    for row in rows:
        seq, score = row['sequence'], float(row['fitness'])
        if set(seq) <= set(AA):
            grouped.setdefault(seq, []).append(score)
    rows = [{'sequence': s, 'fitness': sum(y) / len(y)} for s, y in grouped.items()]
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    import importlib.util
    rows, sha = dataset()
    ys = [r['fitness'] for r in rows]
    return dict(dataset='avGFP', description='Green fluorescent protein', rows=len(rows),
                length=len(rows[0]['sequence']), minimum=min(ys), maximum=max(ys),
                sha256=sha, esm_available=all(importlib.util.find_spec(p) for p in ('torch','transformers','polyleven')),
                oracle_available=False, histogram=histogram(ys))


def histogram(values):
    lo, hi = min(values), max(values)
    bins = [0] * 24
    for value in values:
        bins[min(23, int((value-lo) / (hi-lo or 1) * 24))] += 1
    return bins


def validate(config):
    if config.get('engine') not in ('baseline', 'esm'):
        raise ValueError('Choose baseline or esm.')
    ranges = {'samples': (100, 50000), 'epochs': (1, 100), 'steps': (1, 60),
              'population': (2, 64), 'seed': (0, 2147483647)}
    out = {'engine': config['engine']}
    for name, (lo, hi) in ranges.items():
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer between {lo} and {hi}.')
        out[name] = value
    out['model'] = config.get('model', 'facebook/esm2_t12_35M_UR50D')
    if out['model'] not in ('facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D'):
        raise ValueError('Unsupported ESM model.')
    out['selection'] = config.get('selection', 'greedy')
    if out['selection'] not in ('greedy', 'uncertainty_diversity'):
        raise ValueError('Choose greedy or uncertainty_diversity selection.')
    if out['engine'] == 'baseline' and out['selection'] != 'greedy':
        raise ValueError('Uncertainty-and-diversity selection requires the ESM engine.')
    return out


def run(config, folder, update, cancelled):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    cfg = validate(config)
    rng = random.Random(cfg['seed'])
    np.random.seed(cfg['seed'])
    rows, sha = dataset()
    rng.shuffle(rows)
    rows = rows[:cfg['samples']]
    train, test = train_test_split(rows, test_size=.2, random_state=cfg['seed'])
    folder = Path(folder)
    manifest = dict(config=cfg, dataset_sha256=sha, train_count=len(train), test_count=len(test),
                    split='deduplicated sequences; seeded random 80/20; not mutation-distance split',
                    independent_oracle=False, paper_reproduction=False)
    manifest['selection_method'] = ('greedy predicted-fitness selection' if cfg['selection'] == 'greedy'
                                    else '3-head bootstrap ensemble; mean + 0.5 SD, then 0.5-scaled normalized minimum Hamming-distance bonus')
    manifest['python_version'] = sys.version
    manifest['engine_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['packages'] = {}
    for package in ('numpy', 'scipy', 'scikit-learn', 'torch', 'transformers'):
        try:
            manifest['packages'][package] = version(package)
        except PackageNotFoundError:
            pass
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (folder / 'split.json').write_text(json.dumps({'train': [r['sequence'] for r in train],
                                                'test': [r['sequence'] for r in test]}))
    def check():
        if cancelled():
            raise InterruptedError('Cancelled by user')
    update('Preparing data', 5, f'{len(train):,} training / {len(test):,} held-out sequences')
    check()
    if cfg['engine'] == 'baseline':
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import Ridge
        import pickle
        def features(seqs):
            return [{f'{i}:{a}': 1 for i, a in enumerate(s)} for s in seqs]
        vectorizer = DictVectorizer()
        x = vectorizer.fit_transform(features([r['sequence'] for r in train]))
        model = Ridge(alpha=10, solver='lsqr')
        model.fit(x, [r['fitness'] for r in train])
        def predict(seqs):
            return model.predict(vectorizer.transform(features(seqs)))
        def predict_distribution(seqs):
            values = predict(seqs)
            return values, np.zeros_like(values)
        with (folder / 'baseline.pkl').open('wb') as f:
            pickle.dump((vectorizer, model), f)
        update('Evaluating model', 40, 'Position-specific one-hot ridge trained; this is not an LLM')
        propose = None
    else:
        import torch
        from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
        from de.predictors.attention.decoder import Decoder
        torch.manual_seed(cfg['seed'])
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        update('Loading ESM', 8, f"Loading {cfg['model']} on {device}; first use downloads weights")
        hf_home = Path(os.environ.get('HF_HOME', ROOT / 'models' / 'huggingface'))
        repo_cache = hf_home / 'hub' / ('models--' + cfg['model'].replace('/', '--'))
        try:
            revision = (repo_cache / 'refs' / 'main').read_text().strip()
        except OSError:
            revision = ''
        snapshot = repo_cache / 'snapshots' / revision
        cached_assets = ('config.json', 'model.safetensors', 'tokenizer_config.json',
                         'special_tokens_map.json', 'vocab.txt')
        local_files_only = bool(revision) and all((snapshot / name).is_file() for name in cached_assets)
        tokenizer = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=local_files_only)
        encoder = EsmModel.from_pretrained(cfg['model'], local_files_only=local_files_only).to(device).eval()
        manifest['model_revision'] = getattr(encoder.config, '_commit_hash', None)
        manifest['device'] = str(device)
        manifest['model_loaded_from'] = 'local cache' if local_files_only else 'Hugging Face Hub/cache'
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        # All avGFP sequences must share a length: no padding contributes to pooling.
        if len({len(r['sequence']) for r in rows}) != 1:
            raise ValueError('This local avGFP adapter requires fixed-length sequences.')
        def encode(seqs):
            inputs = tokenizer(seqs, return_tensors='pt', padding=True).to(device)
            with torch.no_grad():
                return encoder(**inputs).last_hidden_state
        ensemble_size = 3 if cfg['selection'] == 'uncertainty_diversity' else 1
        decoders = []
        for member in range(ensemble_size):
            torch.manual_seed(cfg['seed'] + member)
            decoder = Decoder(encoder.config.hidden_size, 128).to(device)
            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            # Bootstrap each head to estimate epistemic uncertainty across predictors.
            member_train = ([rng.choice(train) for _ in train]
                            if ensemble_size > 1 else list(train))
            for epoch in range(cfg['epochs']):
                decoder.train()
                order = list(member_train)
                rng.shuffle(order)
                losses = []
                for start in range(0, len(order), 4):
                    check()
                    batch = order[start:start+4]
                    features = encode([r['sequence'] for r in batch])
                    target = torch.tensor([r['fitness'] for r in batch], device=device)
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(decoder(features).flatten(), target)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    if start % 64 == 0:
                        update('Training Attention1D', 10+30*(member+(epoch+start/len(order))/cfg['epochs'])/ensemble_size,
                               f'Ensemble {member+1}/{ensemble_size}; epoch {epoch+1}/{cfg["epochs"]}; {start}/{len(order)} sequences')
                update('Training Attention1D', 10+30*(member+(epoch+1)/cfg['epochs'])/ensemble_size,
                       f'Ensemble {member+1}/{ensemble_size}, epoch {epoch+1}: training MSE {np.mean(losses):.4f}')
            decoder.eval()
            decoders.append(decoder)
        torch.save({'decoders': [d.state_dict() for d in decoders], 'model': cfg['model'],
                    'hidden_dim': 128, 'ensemble_size': ensemble_size}, folder/'decoder.pt')
        def predict_distribution(seqs):
            member_values = []
            for decoder in decoders:
                values = []
                for start in range(0, len(seqs), 4):
                    check()
                    with torch.no_grad():
                        values.extend(decoder(encode(seqs[start:start+4])).flatten().cpu().tolist())
                member_values.append(values)
            values = np.asarray(member_values)
            return values.mean(axis=0), values.std(axis=0)
        def predict(seqs):
            return predict_distribution(seqs)[0]
        mlm = None
        def propose(seq, position):
            nonlocal mlm
            if mlm is None:
                mlm = EsmForMaskedLM.from_pretrained(cfg['model'], local_files_only=local_files_only).to(device).eval()
            tokens = tokenizer(seq, return_tensors='pt').to(device)
            tokens['input_ids'][0, position+1] = tokenizer.mask_token_id
            with torch.no_grad():
                logits = mlm(**tokens).logits[0, position+1]
            allowed = [a for a in AA if a != seq[position]]
            ids = tokenizer.convert_tokens_to_ids(allowed)
            aa = allowed[int(logits[ids].argmax())]
            return seq[:position]+aa+seq[position+1:]
    check()
    predictions = predict([r['sequence'] for r in test])
    targets = np.array([r['fitness'] for r in test])
    correlation = float(spearmanr(targets, predictions).statistic)
    metrics = dict(rmse=float(np.sqrt(mean_squared_error(targets, predictions))),
                   mae=float(mean_absolute_error(targets, predictions)),
                   spearman=correlation if np.isfinite(correlation) else None,
                   train_count=len(train), test_count=len(test))
    with (folder/'holdout.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence','measured_fitness','predicted_fitness'])
        writer.writerows((r['sequence'], r['fitness'], float(y)) for r,y in zip(test,predictions))
    update('Searching variants', 50, 'Held-out evaluation saved. Search uses predicted fitness only.', metrics=metrics)
    # Start from a training sequence only; held-out labels never guide selection.
    parent = min(train, key=lambda r: abs(r['fitness']-np.median([r['fitness'] for r in train])))['sequence']
    pool = [parent]
    scores = predict(pool)
    uncertainties = predict_distribution(pool)[1]
    history = []
    if cfg['engine'] == 'esm':
        from de.samplers.maskers import ImportanceMasker2
        importance = ImportanceMasker2(k=1, max_subs=1)
    for step in range(cfg['steps']):
        check()
        candidates = list(pool)
        for i in range(cfg['population']*2):
            check()
            seq = rng.choice(pool)
            position = rng.randrange(len(seq))
            if propose and i % 2:
                _, poses = importance.run([seq])
                position = poses[0][0]
            if propose:
                candidate = propose(seq, position)
            else:
                aa = rng.choice([a for a in AA if a != seq[position]])
                candidate = seq[:position]+aa+seq[position+1:]
            candidates.append(candidate)
        candidates = list(dict.fromkeys(candidates))
        values, candidate_uncertainties = predict_distribution(candidates)
        if cfg['selection'] == 'uncertainty_diversity':
            acquisition = values + 0.5 * candidate_uncertainties
            distances = [[0] * len(candidates) for _ in candidates]
            for i in range(len(candidates)):
                for j in range(i):
                    distance = sum(a != b for a,b in zip(candidates[i], candidates[j]))
                    distances[i][j] = distances[j][i] = distance
            chosen = []
            available = set(range(len(candidates)))
            while available and len(chosen) < cfg['population']:
                def selection_score(index):
                    if not chosen:
                        diversity = 0.0
                    else:
                        min_distance = min(distances[index][j] for j in chosen)
                        diversity = 0.5 * min_distance / len(candidates[index])
                    return acquisition[index] + diversity
                index = max(available, key=lambda i: (selection_score(i), -i))
                chosen.append(index)
                available.remove(index)
            indices = np.asarray(chosen, dtype=int)
            selected_scores = np.asarray([values[i] + 0.5*candidate_uncertainties[i] +
                                          (0.5 * min(distances[i][j] for j in chosen[:k]) /
                                           len(candidates[i]) if k else 0.0)
                                          for k,i in enumerate(indices)])
        else:
            indices = np.argsort(-values, kind='stable')[:cfg['population']]
            selected_scores = values[indices]
        pool = [candidates[int(i)] for i in indices]
        scores = values[indices]
        uncertainties = candidate_uncertainties[indices]
        pairwise = [sum(a != b for a,b in zip(pool[i], pool[j])) for i in range(len(pool)) for j in range(i+1,len(pool))]
        history.append(dict(step=step+1, best=float(max(scores)), mean=float(np.mean(scores)), unique=len(pool),
                            mean_uncertainty=float(np.mean(uncertainties)),
                            mean_pairwise_distance=float(np.mean(pairwise)) if pairwise else 0.0))
        update('Searching variants', 50+49*(step+1)/cfg['steps'],
               f'Round {step+1}/{cfg["steps"]}: best predicted fitness {max(scores):.4f}', history=history)
    candidates = [dict(sequence=s, predicted_fitness=float(y), uncertainty=float(u), selection_score=float(sel),
                       mutations='; '.join(f'{a}{i+1}{b}' for i,(a,b) in enumerate(zip(parent,s)) if a != b) or 'Parent',
                       distance=sum(a != b for a,b in zip(parent,s)))
                  for s,y,u,sel in zip(pool,scores,uncertainties,selected_scores)]
    with (folder/'candidates.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    result = dict(metrics=metrics, history=history, candidates=candidates, parent=parent, manifest=manifest,
                  scatter=[dict(measured=float(y),predicted=float(p)) for y,p in zip(targets[:300],predictions[:300])])
    (folder/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result

# %% [markdown]
# ## 5. Optional exact application-engine run
# This call uses the same source as the web application but supplies a console progress callback. Set `RUN_APPLICATION_GFP=True` to execute it manually. Its 100-sequence/one-epoch default is only an execution check; increase the configuration before a planned study. Notebook 02 supplies the clearer matched predictive comparison.

# %%
RUN_APPLICATION_GFP=False
APPLICATION_ENGINE='baseline'  # 'esm' loads the packaged frozen ESM model.
if RUN_APPLICATION_GFP:
    import os
    # Copy only the chosen model into a writable cache with its recorded revision.
    local_model=BUNDLE/'models/esm2_8m';origin=json.loads((local_model/'SOURCE.json').read_text())
    hf_root=RUN_DIR/'huggingface';repo=hf_root/'hub'/('models--'+origin['model'].replace('/','--'))
    shutil.copytree(local_model,repo/'snapshots'/origin['revision'],dirs_exist_ok=True)
    (repo/'refs').mkdir(exist_ok=True);(repo/'refs/main').write_text(origin['revision'])
    os.environ['HF_HOME']=str(hf_root)
    application_output=RUN_DIR/'application_gfp';application_output.mkdir()
    config={'engine':APPLICATION_ENGINE,'samples':100,'epochs':1,'steps':1,'population':2,
            'seed':42,'model':'facebook/esm2_t6_8M_UR50D','selection':'greedy'}
    def progress(stage,percentage,detail,**kwargs): print(f'{percentage:.0f}% {stage}: {detail}')
    application_result=run(config,application_output,progress,lambda:False)
    display(pd.DataFrame([application_result['metrics']]))
else:
    print('Application-engine training skipped by configuration; the complete function definitions above are available.')

# %% [markdown]
# ## 6. Complete source inventory and saved evidence
# Model definitions and training scripts are shipped alongside the notebooks so the panel can inspect and run them directly. The base-paper code includes tasks beyond this demonstration; their external data, independent-oracle checkpoints and full reproduction are not supplied or claimed. The package includes all data/checkpoints required for the default notebook demonstrations.

# %%
inventory=[]
for folder in ('lab','scripts','pet/scripts','pet/backend','Directed_Evolution-main'):
    for path in (BUNDLE/folder).rglob('*.py'):
        inventory.append({'File':str(path.relative_to(BUNDLE)),'Lines':len(path.read_text(encoding='utf-8-sig').splitlines())})
display(pd.DataFrame(inventory))
status={'condition_study_executed':RUN_FULL_CONDITION_STUDY,'assay_study_executed':RUN_FULL_ASSAY_STUDY,
        'application_gfp_executed':RUN_APPLICATION_GFP,'work_folder':str(WORK)}
(RUN_DIR/'execution_status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
