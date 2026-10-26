"""Training status and fingerprinted artifact browsing for the research studio."""
import csv
import json
from pathlib import Path
import re
from flask import Blueprint, jsonify, send_from_directory
from workspace import render

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'models/assay_v2'
bp=Blueprint('training',__name__)
LABELS={'enzyme_um':'Enzyme concentration (µM)','enzyme_mg_ml':'Enzyme concentration (mg/mL)',
        'enzyme_mg_g_pet':'Enzyme loading (mg/g PET)','enzyme_vol_percent':'Enzyme volume fraction (%)',
        'substrate_mg_ml':'Substrate concentration (mg/mL)','reaction_volume_ml':'Reaction volume (mL)',
        'crystallinity_percent':'PET crystallinity (%)','agitation_rpm':'Agitation speed (RPM)'}


def current_run():
    pointer=BASE/'latest.json'
    if not pointer.is_file():return None,None
    run_id=json.loads(pointer.read_text(encoding='utf-8')).get('run_id','')
    if not re.fullmatch(r'train-\d{8}-\d{6}',run_id):return None,None
    folder=BASE/run_id
    return (run_id,folder) if folder.is_dir() else (None,None)


@bp.get('/training')
def page():return render('training','training.html')


@bp.get('/api/training/status')
def status():
    run_id,folder=current_run()
    if not folder or not (folder/'status.json').is_file():return jsonify(status='not_started',progress=0,report=None)
    state=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    result={'run_id':run_id,'status':state['status'],'stage':state['stage'],'progress':state['progress'],
            'detail':state['detail'],'log':state.get('log',[])[-12:], 'report':None,
            'artifacts_url':f'/api/training/artifacts/{run_id}/'}
    if (folder/'report.json').is_file():result['report']=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    return jsonify(result)


@bp.get('/api/training/predictions')
def predictions():
    run_id,folder=current_run()
    if not folder or not (folder/'sequence_oof.csv').is_file():return jsonify(rows=[])
    with (folder/'sequence_oof.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    return jsonify(rows=[{'protein_id':r['protein_id'],'observed_um':float(r['observed_um']),
                          'sequence_environment_um':float(r['sequence_environment']),
                          'sequence_context_um':float(r['sequence_context'])} for r in rows])


@bp.get('/api/training/features')
def features():
    run_id,folder=current_run()
    if not folder or not (folder/'report.json').is_file():return jsonify(fields=LABELS,coverage={},rows=0)
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    return jsonify(fields=LABELS,coverage=report['context_coverage'],rows=report['rows'])


@bp.get('/api/training/artifacts/<run_id>/<name>')
def artifact(run_id,name):
    if not re.fullmatch(r'train-\d{8}-\d{6}',run_id) or name not in (
        'report.json','protocol.json','context_audit.json','splits.json','sequence_oof.csv',
        'sequence_and_study_oof.csv','historical_predictions.csv','status.json'):
        return jsonify(error='Unknown training artifact'),404
    return send_from_directory(BASE/run_id,name,as_attachment=True)
