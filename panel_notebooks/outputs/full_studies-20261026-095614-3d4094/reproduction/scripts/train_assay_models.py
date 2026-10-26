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

ROOT=Path(__file__).resolve().parents[1]
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


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
