"""Build publication figures and metric tables from saved prediction artifacts."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/journal'
FIG = OUT / 'figures'


def save(fig, name):
    fig.savefig(FIG/f'{name}.png', dpi=250, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG/f'{name}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    r1 = json.loads((ROOT/'models/condition_analysis/report.json').read_text())
    run = json.loads((ROOT/'models/assay_v2/latest.json').read_text())['run_id']
    folder = ROOT/'models/assay_v2'/run
    r2 = json.loads((folder/'report.json').read_text())
    rows = []
    for entry in r1['fixed_test']:
        rows.append(dict(study='condition_v1',cohort='historical_18',method=entry['mode'],**entry['metrics']))
    for mode, metrics in r1['nested_group_cv'].items():
        rows.append(dict(study='condition_v1',cohort='nested_sequence_208',method=mode,**metrics))
    for design, entries in r2['evaluation'].items():
        for mode, metrics in entries.items():
            rows.append(dict(study='assay_v2',cohort=f'nested_{design}_208',method=mode,**metrics))
    fields=['study','cohort','method','rows','mae_um','rmse_um','r2','log_rmse','spearman','group_log_mae',
            'median_absolute_error_um','p90_absolute_error_um','within_twofold_fraction']
    with (OUT/'metric_tables.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#26362e','text.color':'#26362e',
                         'svg.fonttype':'none'})
    modes=['sequence_environment','sequence_context','context','training_median']
    labels=['Sequence + environment','Sequence + assay context','Environment + assay context','Training median']
    colors=['#285d45','#729785','#a9bdb0','#cacfc9']
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.4),sharey=True,layout='constrained')
    for ax,design,title in zip(axes,('sequence','sequence_and_study'),('A  Sequence groups','B  Sequence and study groups')):
        values=[r2['evaluation'][design][mode]['mae_um'] for mode in modes]
        ax.barh(np.arange(4),values,color=colors,height=.58)
        ax.set_yticks(np.arange(4),labels);ax.set_xlim(0,2800)
        ax.set_xlabel('Out-of-fold MAE (µM TPA)');ax.set_title(title,loc='left',fontweight='bold',pad=14)
        ax.xaxis.grid(True,alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(v+35,i,f'{v:,.1f}',va='center',fontsize=9)
    axes[0].invert_yaxis()
    save(fig,'grouped_results')
    with (folder/'sequence_oof.csv').open(newline='',encoding='utf-8') as stream:predictions=list(csv.DictReader(stream))
    y=np.array([float(p['observed_um']) for p in predictions]);p=np.array([float(p['sequence_environment']) for p in predictions])
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.5),layout='constrained')
    ax=axes[0];ax.scatter(1+y,1+p,s=19,alpha=.7,color='#285d45',edgecolors='none')
    bound=max(1+y.max(),1+p.max())*1.1
    ax.plot([1,bound],[1,bound],linestyle='--',color='#777777',linewidth=1)
    ax.set(xscale='log',yscale='log',xlabel='1 + observed TPA (numeric µM)',ylabel='1 + predicted TPA (numeric µM)',
           xlim=(1,bound),ylim=(1,bound))
    ax.set_title('A  Observed versus predicted',loc='left',fontweight='bold',pad=14);ax.grid(alpha=.15)
    ax=axes[1];ax.scatter(1+y,p-y,s=19,alpha=.7,color='#36678c',edgecolors='none')
    ax.axhline(0,color='#777777',linestyle='--',linewidth=1)
    ax.set(xscale='log',yscale='symlog',xlabel='1 + observed TPA (numeric µM)',ylabel='Predicted minus observed TPA (µM)')
    ax.set_yscale('symlog',linthresh=50)
    ax.set_title('B  Residuals on raw concentrations',loc='left',fontweight='bold',pad=14);ax.grid(alpha=.15)
    save(fig,'prediction_diagnostics')
    fig,ax=plt.subplots(figsize=(11.6,5.5));ax.set_xlim(0,12);ax.set_ylim(0,6);ax.axis('off')
    def box(x,y,w,h,title,detail,color='#edf3ed'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#83978a',linewidth=1))
        ax.text(x+w/2,y+h*.7,title,ha='center',va='center',fontsize=12,fontweight='bold')
        ax.text(x+w/2,y+h*.31,detail,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#54665b',lw=1.5))
    box(2.4,4.8,7.2,.95,'Shared local interface','Navigation · analyses · structures · training · experiment registry')
    box(3.4,3.2,5.2,.95,'Flask integration service  :8765','Condition analysis · orchestration · gateway')
    box(.3,1.2,5.2,1.25,'PET worker  :8766','Catalog TPA prediction · structure inspection\nCached features and reference coordinates')
    box(6.5,1.2,5.2,1.25,'GFP engine','One-hot or ESM predictor · mutation proposals\nFitness-guided search and candidate export')
    box(2.1,.08,7.8,.72,'Saved research evidence','Configurations · units · splits · predictions · hashes',color='#f3f3ef')
    arrow((6,4.8),(6,4.15));arrow((4.6,3.2),(2.9,2.45));arrow((7.4,3.2),(9.1,2.45))
    arrow((2.9,1.2),(4.2,.80));arrow((9.1,1.2),(7.8,.80))
    fig.tight_layout();save(fig,'architecture')
    (OUT/'FIGURES_AND_METRICS.md').write_text(
        '# Figure and metric provenance\n\n'
        'Figures are reproduced by `.venv\\Scripts\\python.exe scripts\\build_journal_evidence.py`. '
        'They use the saved condition report and the assay run named by `models/assay_v2/latest.json`.\n\n'
        f'Assay run used: `{run}`.\n\n'
        '`grouped_results` shows pooled outer-fold MAE for both grouping designs. '
        '`prediction_diagnostics` uses all 208 sequence-grouped observations and their saved predictions. '
        '`architecture` is an implementation diagram. Each is supplied as PNG and editable SVG.\n\n'
        '`metric_tables.csv` preserves cohort labels. Never rank methods across different cohorts as if their errors were directly comparable. '
        '`within_twofold_fraction` uses the ratio of 1 + TPA; it is not raw-concentration accuracy.\n',encoding='utf-8')
    print(json.dumps({'rows_exported':len(rows),'assay_run':run,'figures':3,'directory':str(OUT)},indent=2))


if __name__=='__main__':main()
