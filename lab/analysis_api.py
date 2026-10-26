"""Environmental response analysis, evaluated models, and structure rendering."""
import csv
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import secrets
import threading
import time
import zipfile
import numpy as np
from flask import Blueprint, Response, jsonify, request, send_from_directory
from condition_model import ROOT, PET_ROOT, load_data
from structure_images import OUTPUT as IMAGE_ROOT, catalog, generate

bp = Blueprint('analysis', __name__)
MODEL = ROOT / 'models/condition_analysis'
SAVED = ROOT / 'analysis_runs'
image_lock = threading.Lock()
PAPER = {'title':'Protein Design by Directed Evolution Guided by Large Language Models',
         'doi':'10.1109/TEVC.2024.3439690', 'task':'avGFP',
         'mfs':9.50,'mfs_std':0.37,'afs':9.33,'afs_std':0.40,
         'source':'Tables II and V, printed pages 423–424 (PDF pages 6–7).',
         'comparison_status':'not_directly_comparable',
         'reason':'Paper uses a separate evaluation oracle. Our local GFP surrogate and PET TPA prediction errors are different measurements.'}


def read_report():
    path=MODEL/'report.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


@lru_cache(maxsize=1)
def resources(model_hash):
    path=MODEL/'model.pkl'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=model_hash:
        raise ValueError('Model fingerprint mismatch. Rebuild the analysis experiment.')
    data=load_data()
    report=read_report()
    if data['dataset_sha256']!=report['dataset_sha256']:
        raise ValueError('Dataset changed since training. Rebuild the analysis model.')
    # Only the locally trained, fingerprint-checked model path is deserialized.
    with path.open('rb') as stream:model=pickle.load(stream)
    protocol=json.loads((MODEL/'protocol.json').read_text(encoding='utf-8'))
    return model,data,protocol


@bp.get('/analysis')
def studio():
    from workspace import render
    return render('analysis','analysis.html')


@bp.get('/api/analysis/status')
def status():
    report=read_report()
    structures=json.loads((PET_ROOT/'data/structure_images/manifest.json').read_text())
    image_ids={r['protein_id'] for r in structures['matched']}
    proteins=[{'id':pid,'name':row['protein_name'],'length':len(row['sequence']),'structure_available':pid in image_ids} for pid,row in catalog().items()]
    return jsonify(ready=report is not None,report=report,paper=PAPER,proteins=proteins)


def validate(payload):
    if not isinstance(payload,dict):raise ValueError('Expected analysis configuration.')
    pid=payload.get('protein_id')
    if pid not in catalog():raise ValueError('Select an available protein.')
    engine=payload.get('engine','legacy')
    if engine not in ('legacy','fusion'):raise ValueError('Choose original or experimental fusion model.')
    config={'protein_id':pid,'engine':engine}
    ranges={'temperature':(0,100),'ph':(0,14),'duration':(.01,720), 'temperature_span':(0,20),'ph_span':(0,2),'duration_span':(0,240)}
    defaults={'temperature':40,'ph':8,'duration':48,'temperature_span':5,'ph_span':.5,'duration_span':12}
    for name,(lo,hi) in ranges.items():
        value=payload.get(name,defaults[name])
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError(f'{name} must be a finite number between {lo} and {hi}.')
        config[name]=float(value)
    return config


def analyze(config):
    report=read_report()
    if report is None:raise FileNotFoundError('Train the analysis model: .venv\\Scripts\\python.exe scripts\\train_condition_models.py')
    model,data,protocol=resources(report['model_sha256'])
    center=np.array([config[k] for k in ('temperature','ph','duration')])
    spans=np.array([config[k] for k in ('temperature_span','ph_span','duration_span')])
    lower=np.maximum(center-spans,[0,0,.01]);upper=np.minimum(center+spans,[100,14,720])
    axes=[np.unique(np.linspace(lo,hi,3)) for lo,hi in zip(lower,upper)]
    grid=np.array(np.meshgrid(*axes,indexing='ij')).reshape(3,-1).T
    grid_with_center=np.vstack([grid,center])
    legacy_predictions={}
    def query_legacy(pids,environment):
        from integration import pet_call
        raw,code,_=pet_call('environment/predict',json.dumps({'protein_ids':pids,'conditions':environment.tolist()}).encode())
        if code!=200:raise ValueError('Original PET model query failed; restart Start-Research.cmd and check the PET service.')
        return json.loads(raw)['predictions']
    if config['engine']=='legacy':
        legacy_predictions=query_legacy(list(catalog()),grid_with_center)
    train=np.array(protocol['split']['train'])
    mean=np.array(report['environment_mean']);std=np.array(report['environment_std'])
    train_min=np.array(report['environment_min']);train_max=np.array(report['environment_max'])
    train_z=(data['env'][train]-mean)/std
    def support(env):
        z=(env-mean)/std
        distances=np.linalg.norm(z[:,None,:]-train_z[None,:,:],axis=-1).min(1)
        return ((env<train_min)|(env>train_max)).any(1),distances
    grid_outside,grid_distance=support(grid)
    def predict(pid,env):
        if config['engine']=='legacy':
            return np.array(query_legacy([pid],env)[pid])
        feat=data['cache'][pid]
        return model.predict(np.repeat(feat['sequence'][None],len(env),axis=0),env,np.repeat(feat['image'][None],len(env),axis=0))
    rankings=[]
    for pid,row in catalog().items():
        predictions=np.array(legacy_predictions[pid][:-1]) if legacy_predictions else predict(pid,grid)
        count=sum(data['rows'][i]['protein_id']==pid for i in train)
        center_score=legacy_predictions[pid][-1] if legacy_predictions else predict(pid,center[None])[0]
        rankings.append({'protein_id':pid,'protein_name':row['protein_name'],'center_prediction_um':float(center_score),
                         'robust_score_um':float(np.quantile(predictions,.1)), 'mean_prediction_um':float(predictions.mean()),
                         'min_prediction_um':float(predictions.min()),'max_prediction_um':float(predictions.max()),
                         'scenario_std_um':float(predictions.std()),'training_observations':count,
                         'structure_used':config['engine']=='fusion' and report['selected']['mode']=='multimodal' and bool(data['cache'][pid]['image'][-1])})
    rankings.sort(key=lambda r:r['robust_score_um'],reverse=True)
    pid=config['protein_id']
    surface_t=np.unique(np.linspace(lower[0],upper[0],7));surface_ph=np.unique(np.linspace(lower[1],upper[1],7))
    surface_grid=np.array([[t,ph,center[2]] for t in surface_t for ph in surface_ph])
    surface_scores=predict(pid,surface_grid)
    surface_outside,surface_distance=support(surface_grid)
    surface=[dict(temperature=float(e[0]),ph=float(e[1]),predicted_tpa_um=float(y),outside_training_range=bool(outside),nearest_training_distance=float(distance)) for e,y,outside,distance in zip(surface_grid,surface_scores,surface_outside,surface_distance)]
    sweeps={}
    for axis,name in enumerate(('temperature','ph','duration')):
        env=np.repeat(center[None],13,axis=0);env[:,axis]=np.linspace(lower[axis],upper[axis],13)
        pred=predict(pid,env);outside,_=support(env)
        sweeps[name]=[{'value':float(e[axis]),'predicted_tpa_um':float(y),'outside_training_range':bool(o)} for e,y,o in zip(env,pred,outside)]
    observed=[]
    for i,row in enumerate(data['rows']):
        if row['protein_id']==pid:
            observed.append({'temperature':float(data['env'][i,0]),'ph':float(data['env'][i,1]),'duration':float(data['env'][i,2]),'measured_tpa_um':float(data['target'][i]),'used_in_training':i in set(train.tolist())})
    selected=next(r for r in rankings if r['protein_id']==pid)
    best=surface[int(np.argmax(surface_scores))]
    return {'config':config,'selected_protein':selected,'ranking':rankings,'surface':surface,'sweeps':sweeps,
            'observed_evidence':observed,'best_grid_point':best,'scenario_count':len(grid),
            'outside_training_range_fraction':float(grid_outside.mean()),'nearest_training_distance_max':float(grid_distance.max()),
            'training_condition_ranges':dict(zip(('temperature','ph','duration'),[[float(a),float(b)] for a,b in zip(train_min,train_max)])),
            'model':report['selected'] if config['engine']=='fusion' else {'mode':'legacy_attention1d'},
            'model_sha256':report['model_sha256'] if config['engine']=='fusion' else hashlib.sha256((PET_ROOT/'models/neural/weights.pt').read_bytes()).hexdigest(),'dataset_sha256':report['dataset_sha256'],
            'score_definition':'10th percentile of predicted TPA over the discrete condition grid; not a biological measurement or calibrated lower confidence bound.',
            'limitations':['Environmental sweeps are model sensitivity, not causal effects.', 'Highest grid score is a hypothesis, not a verified optimum.', 'GFP dataset has no condition labels; these PET effects are not applied to GFP.', 'No reliable mutation-effect estimates are inferred from catalog-protein ranking.']}


@bp.post('/api/analysis/run')
def run_analysis():
    try:config=validate(request.get_json());result=analyze(config)
    except (ValueError,TypeError) as error:return jsonify(error=str(error)),400
    except (OSError,KeyError) as error:return jsonify(error=str(error)),503
    rid=f'analysis-{int(time.time())}-{secrets.token_hex(6)}'
    result.update(id=rid,created=time.time(),status='completed',domain='PET environment',unit='µM TPA',
                  artifact_url=f'/api/analysis/runs/{rid}')
    SAVED.mkdir(exist_ok=True)
    (SAVED/(rid+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return jsonify(result)


@bp.get('/api/analysis/runs/<rid>')
def download(rid):
    import re
    if not re.fullmatch(r'analysis-[0-9]+-[a-f0-9]{12}',rid):return jsonify(error='Unknown analysis'),404
    return send_from_directory(SAVED,rid+'.json',as_attachment=True)


@bp.post('/api/analysis/images')
def images():
    body=request.get_json()
    try:
        with image_lock:result=generate(body.get('protein_id') if isinstance(body,dict) else None)
    except ValueError as error:return jsonify(error=str(error)),400
    return jsonify(result)


@bp.get('/api/analysis/images/<pid>/<name>')
def image_file(pid,name):
    if not pid.isdecimal() or name not in ('front.svg','side.svg','top.svg','distance.png','contacts.png','manifest.json','bundle.zip'):
        return jsonify(error='Unknown structure artifact'),404
    if name=='bundle.zip':
        folder=IMAGE_ROOT/pid
        if not (folder/'manifest.json').is_file():return jsonify(error='Generate images first'),404
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
            for item in ('front.svg','side.svg','top.svg','distance.png','contacts.png','manifest.json'):
                archive.write(folder/item,arcname=item)
        return Response(stream.getvalue(),mimetype='application/zip',headers={'Content-Disposition':f'attachment; filename=protein-{pid}-views.zip'})
    return send_from_directory(IMAGE_ROOT/pid,name)


@bp.get('/api/analysis/evaluation/<name>')
def evaluation_file(name):
    if name not in ('report.json','protocol.json','heldout.csv','grouped_oof.csv','cv_splits.json'):
        return jsonify(error='Unknown evaluation artifact'),404
    return send_from_directory(MODEL,name,as_attachment=True)
