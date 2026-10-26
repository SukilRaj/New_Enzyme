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

ROOT = Path(__file__).resolve().parents[1]
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


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
