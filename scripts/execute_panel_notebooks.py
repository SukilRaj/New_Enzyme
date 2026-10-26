"""Execute panel notebooks in fresh real Jupyter kernels and preserve outputs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'panel_notebooks'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--only',default='',help='Notebook filename prefix, for a targeted repair check.')
    args=parser.parse_args()
    kernels=ROOT/'tmp/panel_jupyter/kernels/enzaime-panel'
    kernels.mkdir(parents=True,exist_ok=True)
    (kernels/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                                  'display_name':'EnzAIme panel validation','language':'python'}),encoding='utf-8')
    os.environ['JUPYTER_PATH']=str(kernels.parents[1])
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'tmp/panel_jupyter/runtime')
    Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    # Test an input/output separation just as on a read-only Kaggle input mount.
    os.environ['ENZAIME_NOTEBOOK_OUTPUT']=str(BUNDLE/'outputs')
    (BUNDLE/'executed').mkdir(exist_ok=True)
    report_path=ROOT/'reports/notebook_execution.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {'notebooks':{}}
    for path in sorted(BUNDLE.glob(args.only+'*.ipynb')):
        notebook=nbformat.read(path,as_version=4)
        start=time.time();print('START '+path.name,flush=True)
        def progress(cell,cell_index,**kwargs):
            if cell.cell_type=='code': print(f'  cell {cell_index+1}/{len(notebook.cells)}: '+cell.source.splitlines()[0][:100],flush=True)
        client=NotebookClient(notebook,kernel_name='enzaime-panel',timeout=1200,iopub_timeout=180,
                              allow_errors=False,resources={'metadata':{'path':str(BUNDLE)}},on_cell_start=progress)
        success=False
        try:
            client.execute();success=True
        finally:
            nbformat.write(notebook,BUNDLE/'executed'/path.name)
            errors=[output for cell in notebook.cells if cell.cell_type=='code' for output in cell.get('outputs',[]) if output.output_type=='error']
            images=sum('image/png' in output.get('data',{}) for cell in notebook.cells if cell.cell_type=='code' for output in cell.get('outputs',[]))
            report['notebooks'][path.name]={'success':success,'seconds':round(time.time()-start,2),
                'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
                'executed_code_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells),
                'embedded_plots':images,'errors':[{'name':e.ename,'message':e.evalue} for e in errors]}
            report['checked_at']=datetime.now(timezone.utc).isoformat()
            report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print('PASS '+path.name+f' ({time.time()-start:.1f}s)',flush=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
