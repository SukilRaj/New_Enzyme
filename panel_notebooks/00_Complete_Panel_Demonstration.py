# %% [markdown]
# # EnzAIme × Helix — complete panel demonstration
# 
# **Start here.** Run all cells from top to bottom. This notebook combines the complete PET/environment and GFP demonstrations in one file. It loads real packaged data and checkpoints, exposes preprocessing and model classes, trains fresh demonstration models, evaluates them, and plots the results. No web application is required.
# 
# The first half covers PET and temperature/pH/duration; the second covers GFP benchmark prediction and sequence search. Each half creates its own timestamped output folder. They are separate scientific tasks: the GFP records have no environmental labels. Small default training budgets keep the live demonstration practical. Saved historical results and newly trained demonstration results are labeled separately.
# 
# For a shorter session use notebook 01 or 02 separately. Notebook 03 contains the complete opt-in training studies and original application engine. Read README.md for local/VS Code/Kaggle setup and MODEL_AND_DATA_LOCATIONS.md for the exact artifact paths.

# %% [markdown]
# # EnzAIme × Helix: PET, environmental data, training and evaluation
# 
# **Panel demonstration notebook.** Run cells from top to bottom. The actual preprocessing, architecture, training, prediction and plotting code is visible below. The companion folder contains data, frozen ESM features and saved checkpoints; no frontend or server is used.
# 
# You will: audit the 221 source records; reproduce 208 eligible rows; inspect temperature/pH/time; load the original neural checkpoint; evaluate its historical holdout; train fresh condition models and an Attention1D head; call models with your own conditions; inspect grouped results; and render real reference structures. Quick demonstration fits are saved separately from the original results.
# 
# TPA concentration is a regression target in **µM**. A sequence edit does not establish better enzyme function. The cached structures describe reference proteins; this notebook does not predict mutant folds.

# %%
NOTEBOOK_NAME='pet'
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
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
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

# %% [markdown]
# ## 1. Environment and package versions
# Use Python 3.10–3.12 with the supplied requirements. Kaggle: attach the **extracted bundle** as a dataset. Input data stays read-only; new outputs go to `/kaggle/working/enzaime_outputs`. Install dependencies once if requested, restart the kernel, then Run All.

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
# ## 2. Locate the actual models and data
# The original neural checkpoint has a historical holdout. The condition surrogate also preserves its historical split. The assay-v2 `model.pkl` was refitted on all 208 rows: use its saved out-of-fold predictions for generalization assessment, not predictions on its training data.

# %%
model_locations={
    'Original PET Attention1D': 'pet/models/neural/weights.pt',
    'Condition/image surrogate': 'models/condition_analysis/model.pkl',
    'Assay-v2 full-data refit': 'models/assay_v2/train-20261024-160717/model.pkl',
    'Assay-v2 historical model': 'models/assay_v2/train-20261024-160717/historical_model.pkl',
}
display(pd.DataFrame([{'Model':name,'Actual file':str(verify_file(path)),
                       'KB':round((BUNDLE/path).stat().st_size/1024,1)} for name,path in model_locations.items()]))
data_path=verify_file('pet/data/processed/tpa_pet_ml_confirmed.csv')
raw=pd.read_csv(data_path,dtype={'protein_id':str},keep_default_na=False)
print('Raw assay rows:',len(raw),'| Catalog proteins:',raw.protein_id.nunique())
display(raw[['protein_id','protein_name','temperature_c','ph','duration_hours',
             'target_value_standardized','standardized_unit','literature_doi']].head(8))

# %% [markdown]
# ## 3. Complete preprocessing and condition-regression implementation
# The code validates finite measurements, removes nonpositive durations/negative targets and exact duplicate assay records, preserves source row IDs, verifies sequence-feature hashes, and pools cached per-residue ESM embeddings. Cached embeddings are frozen representations, not model predictions or labels.
# 
# Scaling and PCA are fitted within training subsets. The six environmental descriptors use only temperature, pH and duration. Image descriptors use existing sequence-matched geometry. DOI and protein IDs are split/provenance fields, not predictors.

# %%
"""Condition-aware ESM surrogates with explicit train/validation/test contracts."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from structure_images import descriptor

FEATURE_NAMES = ['temperature_c', 'ph', 'duration_hours']


def load_data():
    csv.field_size_limit(10_000_000)
    path = PET_ROOT / 'data/processed/tpa_pet_ml_confirmed.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        source = list(csv.DictReader(stream))
    rows, seen = [], set()
    for index, row in enumerate(source):
        vals = [float(row[k]) for k in FEATURE_NAMES + ['target_value_standardized']]
        sequence = row['sequence'].strip().upper()
        if not np.isfinite(vals).all() or vals[2] <= 0 or vals[3] < 0:
            continue
        key = (sequence, *vals, row.get('literature_doi',''), row.get('assay_index',''), row.get('measurement_index',''))
        if key in seen:
            continue
        seen.add(key)
        rows.append({**row, 'sequence': sequence, 'source_row': index, 'sequence_hash': hashlib.sha256(sequence.encode()).hexdigest()})
    feature_index = json.loads((PET_ROOT / 'data/neural_features/index.json').read_text())
    cache = {}
    for row in rows:
        pid = row['protein_id']
        if pid in cache:
            continue
        meta = feature_index['proteins'][pid]
        if meta['sequence_sha256'] != row['sequence_hash']:
            raise ValueError('Sequence and cached ESM features disagree.')
        path_feature = (PET_ROOT / 'data/neural_features' / meta['file']).resolve()
        if not path_feature.is_relative_to((PET_ROOT / 'data/neural_features').resolve()):
            raise ValueError('Invalid cached feature path.')
        with np.load(path_feature, allow_pickle=False) as f:
            seq = f['sequence'].astype(np.float64)
        if len(seq) != len(row['sequence']) or not np.isfinite(seq).all():
            raise ValueError('Invalid residue features.')
        cache[pid] = {'sequence': seq.mean(0), 'image': descriptor(pid)}
    return {'rows': rows, 'env': np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows]),
            'sequence': np.vstack([cache[r['protein_id']]['sequence'] for r in rows]),
            'image': np.vstack([cache[r['protein_id']]['image'] for r in rows]),
            'target': np.array([float(r['target_value_standardized']) for r in rows]),
            'groups': np.array([r['sequence_hash'] for r in rows]), 'cache': cache,
            'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'encoder': feature_index['encoder']}


def environmental_features(environment):
    environment = np.asarray(environment, dtype=float)
    if environment.ndim != 2 or environment.shape[1] != 3 or not np.isfinite(environment).all():
        raise ValueError('Expected finite temperature, pH and duration values.')
    t, ph, hours = environment.T
    if np.any(t <= -273.15) or np.any((ph < 0)|(ph > 14)) or np.any(hours <= 0):
        raise ValueError('Invalid assay conditions.')
    # Deterministic condition descriptors. No experimental response is used here.
    return np.column_stack([t, ph, np.log1p(hours), (ph-7)**2, 1000/(t+273.15), t*np.log1p(hours)/100])


class ConditionRegressor:
    def __init__(self, mode='sequence_environment', kind='ridge', alpha=100, seed=42):
        self.mode, self.kind, self.alpha, self.seed = mode, kind, alpha, seed

    def features(self, seq, env, image):
        parts = []
        if self.mode != 'environment':
            parts.append(self.pca.transform(self.seq_scaler.transform(seq)))
        if self.mode != 'sequence':
            parts.append(environmental_features(env))
        if self.mode in ('multimodal', 'availability_control'):
            parts.append(image if self.mode == 'multimodal' else image[:, -1:])
        return np.column_stack(parts)

    def fit(self, seq, env, image, y, groups):
        if self.mode != 'environment':
            _, first = np.unique(groups, return_index=True)
            self.seq_scaler = StandardScaler().fit(seq[first])
            n = min(5, len(first)-1, seq.shape[1])
            if n < 1:
                raise ValueError('At least two training sequence groups are required.')
            self.pca = PCA(n_components=n, svd_solver='full').fit(self.seq_scaler.transform(seq[first]))
        x = self.features(seq, env, image)
        self.scaler = StandardScaler().fit(x)
        if self.kind == 'ridge':
            self.model = Ridge(alpha=self.alpha)
        elif self.kind == 'extra_trees':
            self.model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=4, max_depth=6, random_state=self.seed, n_jobs=1)
        elif self.kind == 'boosting':
            self.model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, min_samples_leaf=10, l2_regularization=10, random_state=self.seed)
        else:
            raise ValueError('Unsupported regressor')
        self.model.fit(self.scaler.transform(x), np.log1p(y))
        return self

    def predict(self, seq, env, image):
        log_pred = self.model.predict(self.scaler.transform(self.features(seq, env, image)))
        # A finite numerical bound, fixed before evaluation; the lower bound enforces nonnegative TPA.
        return np.expm1(np.clip(log_pred, 0, 20))


def metrics(y, pred):
    return {'mae_um': float(mean_absolute_error(y, pred)), 'rmse_um': float(np.sqrt(mean_squared_error(y, pred))),
            'r2': float(r2_score(y, pred)) if len(y)>1 and np.ptp(y) else None,
            'log_rmse': float(np.sqrt(np.mean((np.log1p(y)-np.log1p(pred))**2))),
            'spearman': float(spearmanr(y, pred).statistic) if np.ptp(y) and np.ptp(pred) else None, 'rows': len(y)}


def fit_predict(data, spec, train, test):
    model = ConditionRegressor(**spec).fit(*(data[k][train] for k in ('sequence','env','image','target','groups')))
    return model, model.predict(*(data[k][test] for k in ('sequence','env','image')))


def specs():
    # Prespecified, bounded search: never extend this list in response to the test scores.
    return [{'mode': mode, 'kind': kind, 'alpha': alpha, 'seed': 42}
            for mode in ('sequence','environment','sequence_environment','availability_control','multimodal')
            for kind, alpha in [('ridge', 10), ('ridge',100), ('ridge',1000), ('extra_trees',100), ('boosting',100)]]

# %%
# Register the visible class under its original module name for checkpoint compatibility.
import condition_model as condition_module
condition_module.ConditionRegressor=ConditionRegressor
ConditionRegressor.__module__='condition_model'
data=load_data()
rows=pd.DataFrame(data['rows'])
for column in FEATURE_NAMES+['target_value_standardized']:
    rows[column]=pd.to_numeric(rows[column])
assert len(rows)==208 and len(np.unique(data['groups']))==25
assert data['dataset_sha256']==json.loads((PET_ROOT/'models/neural/manifest.json').read_text())['dataset_sha256']
print(f'Eligible: {len(rows)} / {len(raw)} rows; {len(raw)-len(rows)} removed or deduplicated.')
display(rows[FEATURE_NAMES+['target_value_standardized']].describe())
rows.to_csv(RUN_DIR/'eligible_pet_measurements.csv',index=False)

# %% [markdown]
# ## 4. Environmental data and target visualization
# Measured values are shown here. Later response surfaces are model predictions. Neither is a direct estimate of carbon emissions or real-world plastic removal.

# %%
fig,axes=plt.subplots(1,4,figsize=(15,3.4))
for ax,column,label in zip(axes,FEATURE_NAMES+['target_value_standardized'],
                          ['Temperature (°C)','pH','Duration (hours)','TPA (µM; log1p display)']):
    values=rows[column].to_numpy(float)
    ax.hist(np.log1p(values) if column=='target_value_standardized' else values,bins=18,color='#286b58',alpha=.85)
    ax.set(xlabel=label,ylabel='Measured rows')
save_figure('01_measured_environment_and_target')
fig,axes=plt.subplots(1,3,figsize=(13,3.5))
for ax,column in zip(axes,FEATURE_NAMES):
    ax.scatter(rows[column],rows.target_value_standardized,s=22,alpha=.55,color='#286b58')
    ax.set(xlabel=column,ylabel='Measured TPA (µM)',yscale='log')
save_figure('02_observed_condition_associations')

# %% [markdown]
# ## 5. Preserve the recorded training, validation and test groups
# This is the original historical split, which has already been inspected during development. It is useful for reproduction and teaching; it is not a new external confirmation set. No test labels are used for preprocessing, fitting, early stopping or hyperparameter selection below.

# %%
legacy_manifest=json.loads(verify_file('pet/models/neural/manifest.json').read_text())
split={key:np.flatnonzero(np.isin(data['groups'],hashes))
       for key,hashes in legacy_manifest['split_sequence_hashes'].items()}
train_idx,val_idx,test_idx=(split[k] for k in ('train','validation','test'))
for left,right in [('train','validation'),('train','test'),('validation','test')]:
    assert not set(data['groups'][split[left]]) & set(data['groups'][split[right]])
assert sorted(np.concatenate(list(split.values())).tolist())==list(range(len(rows)))
display(pd.DataFrame([{'Partition':k,'Rows':len(v),'Sequence groups':len(np.unique(data['groups'][v]))} for k,v in split.items()]))
(RUN_DIR/'split.json').write_text(json.dumps({k:v.tolist() for k,v in split.items()},indent=2))

# %% [markdown]
# ## 6. Original Attention1D architecture and batch function
# This is the architecture used by the installed PET checkpoint. Frozen residue embeddings are pooled by learned attention, then combined with normalized assay conditions. The original checkpoint declares `image_trained=false`, so its image branch is disabled during inference.

# %%
"""Paper-inspired Attention1D oracle with optional image-feature fusion.

Inputs are per-residue ESM-2 states, frozen ResNet18 image features, and
temperature/pH/duration. No Ridge fallback and no randomly initialized serving.
"""
import torch
from torch import nn


class AttentionOracle(nn.Module):
    def __init__(self, sequence_dim=480, image_dim=512, hidden=64):
        super().__init__()
        self.sequence = nn.Sequential(nn.Linear(sequence_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.attention = nn.Linear(hidden, 1)
        self.image = nn.Sequential(nn.Linear(image_dim, hidden), nn.ReLU())
        self.environment = nn.Sequential(nn.Linear(3, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 3 + 1, hidden), nn.ReLU(),
                                  nn.Dropout(0.15), nn.Linear(hidden, 1))

    def forward(self, sequence, mask, image, image_present, environment):
        if not mask.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one residue.")
        residues = self.sequence(sequence)
        weights = self.attention(residues).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
        pooled = (residues * weights.unsqueeze(-1)).sum(1)
        visual = self.image(image) * image_present.reshape(-1, 1)
        fused = torch.cat([pooled, visual, self.environment(environment),
                           image_present.reshape(-1, 1)], dim=-1)
        return self.head(fused).squeeze(-1), weights


def batch_features(items, device="cpu"):
    length = max(len(x["sequence"]) for x in items)
    dim = items[0]["sequence"].shape[-1]
    sequence = torch.zeros(len(items), length, dim, device=device)
    mask = torch.zeros(len(items), length, dtype=torch.bool, device=device)
    for i, item in enumerate(items):
        n = len(item["sequence"])
        sequence[i, :n] = torch.as_tensor(item["sequence"], device=device)
        mask[i, :n] = True
    return sequence, mask, torch.stack([torch.as_tensor(x["image"], device=device) for x in items])

# %%
torch.set_num_threads(2)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_index=json.loads(verify_file('pet/data/neural_features/index.json').read_text())
residue_cache={}
for pid,meta in feature_index['proteins'].items():
    path=verify_file('pet/data/neural_features/'+meta['file'])
    with np.load(path,allow_pickle=False) as z:
        residue_cache[pid]={name:z[name].astype(np.float32) for name in ('sequence','image')}

def load_pet_neural_model(checkpoint_path,metadata):
    model=AttentionOracle(**metadata['architecture']).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path,map_location=DEVICE,weights_only=True))
    return model.eval()

@torch.inference_mode()
def call_pet_neural(model,protein_ids,conditions,metadata):
    env=np.asarray(conditions,dtype=np.float32)
    if env.shape!=(len(protein_ids),3): raise ValueError('Use one [temperature, pH, hours] row per protein.')
    environmental_features(env)  # Validate bounds and finiteness.
    values=[]
    for start in range(0,len(protein_ids),16):
        pids=protein_ids[start:start+16]
        seq,mask,img=batch_features([residue_cache[str(pid)] for pid in pids],device=DEVICE)
        present=torch.tensor([float(feature_index['proteins'][str(pid)]['image_present'] and metadata['image_trained']) for pid in pids],device=DEVICE)
        norm=(env[start:start+16]-np.asarray(metadata['environment_mean'],np.float32))/np.asarray(metadata['environment_std'],np.float32)
        scores,_=model(seq,mask,img,present,torch.tensor(norm,device=DEVICE))
        values.extend(np.maximum(0,np.expm1(scores.cpu().numpy().astype(np.float64))).tolist())
    return np.asarray(values)

original_model=load_pet_neural_model(verify_file('pet/models/neural/weights.pt'),legacy_manifest)
print('Checkpoint loaded on',DEVICE,'| image branch trained:',legacy_manifest['image_trained'])
original_prediction=call_pet_neural(original_model,rows.iloc[test_idx].protein_id.tolist(),data['env'][test_idx],legacy_manifest)
recorded=json.loads(verify_file('pet/models/neural/held_out_predictions.json').read_text())
assert [r['protein_id'] for r in recorded]==rows.iloc[test_idx].protein_id.tolist()
np.testing.assert_allclose(original_prediction,[r['predicted_um'] for r in recorded],rtol=2e-4,atol=.005)
display(pd.DataFrame([evaluate_predictions(data['target'][test_idx],original_prediction)],index=['Original neural / historical test']))

# %% [markdown]
# ## 7. Train fresh condition models: training function, validation selection, test evaluation
# For a fast live demonstration, compare four feature sets using Ridge strengths fixed in advance. The complete 25-configuration condition study and 24-configuration assay study are in notebook 03. This small demonstration does not replace their saved journal results.

# %%
def train_condition_model(data,mode,train_ids,validation_ids,alphas=(10,100,1000)):
    choices=[]
    for alpha in alphas:
        model=ConditionRegressor(mode=mode,kind='ridge',alpha=alpha,seed=42)
        model.fit(*(data[k][train_ids] for k in ('sequence','env','image','target','groups')))
        pred=model.predict(*(data[k][validation_ids] for k in ('sequence','env','image')))
        validation_log_rmse=np.sqrt(np.mean((np.log1p(pred)-np.log1p(data['target'][validation_ids]))**2))
        choices.append((float(validation_log_rmse),alpha,model))
    loss,alpha,selected=min(choices,key=lambda item:item[0])
    return selected,{'mode':mode,'alpha':alpha,'validation_log_rmse':loss}

demo_models={}; selection=[]; comparisons=[]
comparisons.append({'Model':'Original neural',**evaluate_predictions(data['target'][test_idx],original_prediction)})
for mode in ('sequence','environment','sequence_environment','multimodal'):
    model,chosen=train_condition_model(data,mode,train_idx,val_idx)
    demo_models[mode]=model;selection.append(chosen)
    pred=model.predict(*(data[k][test_idx] for k in ('sequence','env','image')))
    comparisons.append({'Model':'Demo '+mode,**evaluate_predictions(data['target'][test_idx],pred)})
    with (RUN_DIR/(mode+'.pkl')).open('wb') as stream: pickle.dump(model,stream)
display(pd.DataFrame(selection));display(pd.DataFrame(comparisons))
pd.DataFrame(comparisons).to_csv(RUN_DIR/'demo_historical_test_comparison.csv',index=False)

# %% [markdown]
# ## 8. Train an Attention1D neural head and show its learning curve
# The encoder remains frozen: training uses the packaged ESM residue features. This cell trains a new head for 12 demonstration epochs, chooses weights by validation loss, and evaluates the test only afterward. Increase the budget for a planned study before inspecting its test result. The original checkpoint is preserved.

# %%
DEMO_EPOCHS=12
def train_pet_attention(data,train_ids,validation_ids,epochs=12,seed=42):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    meta=copy.deepcopy(legacy_manifest)
    meta.pop('test_metrics',None)
    meta['status']='fresh_panel_demonstration_checkpoint'
    meta['environment_mean']=data['env'][train_ids].mean(0).tolist()
    meta['environment_std']=data['env'][train_ids].std(0).clip(1e-6).tolist()
    meta['image_trained']=False
    model=AttentionOracle(**meta['architecture']).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    history=[];best=float('inf');state=None
    for epoch in range(epochs):
        model.train();losses=[]
        order=rng.permutation(train_ids)
        for start in range(0,len(order),16):
            ids=order[start:start+16]
            items=[residue_cache[pid] for pid in rows.iloc[ids].protein_id]
            seq,mask,img=batch_features(items,device=DEVICE)
            env=(data['env'][ids]-meta['environment_mean'])/meta['environment_std']
            target=torch.tensor(np.log1p(data['target'][ids]),dtype=torch.float32,device=DEVICE)
            optimizer.zero_grad();p,_=model(seq,mask,img,torch.zeros(len(ids),device=DEVICE),torch.tensor(env,dtype=torch.float32,device=DEVICE))
            loss=torch.nn.functional.mse_loss(p,target);loss.backward();optimizer.step()
            losses.append((loss.item(),len(ids)))
        model.eval();pred=call_pet_neural(model,rows.iloc[validation_ids].protein_id.tolist(),data['env'][validation_ids],meta)
        validation=float(np.mean((np.log1p(pred)-np.log1p(data['target'][validation_ids]))**2))
        history.append({'epoch':epoch+1,'train_log_mse':sum(v*n for v,n in losses)/len(train_ids),'validation_log_mse':validation})
        if validation<best: best=validation;state=copy.deepcopy(model.state_dict());meta['best_epoch']=epoch+1
    model.load_state_dict(state);model.eval()
    return model,meta,pd.DataFrame(history)

trained_neural,trained_meta,learning=train_pet_attention(data,train_idx,val_idx,DEMO_EPOCHS)
torch.save(trained_neural.state_dict(),RUN_DIR/'pet_attention_demo.pt')
(RUN_DIR/'pet_attention_demo_manifest.json').write_text(json.dumps(trained_meta,indent=2))
learning.plot(x='epoch',y=['train_log_mse','validation_log_mse'],ylabel='Mean squared log1p error',title='Fresh neural-head training: validation chooses the epoch')
save_figure('03_neural_learning_curve')
demo_neural_prediction=call_pet_neural(trained_neural,rows.iloc[test_idx].protein_id.tolist(),data['env'][test_idx],trained_meta)
trained_meta['test_metrics']=metrics(data['target'][test_idx],demo_neural_prediction)
(RUN_DIR/'pet_attention_demo_manifest.json').write_text(json.dumps(trained_meta,indent=2))
display(pd.DataFrame([evaluate_predictions(data['target'][test_idx],demo_neural_prediction)],index=['Fresh demonstration head']))

# %% [markdown]
# ## 9. Load the saved condition and assay-context checkpoints
# The full assay preprocessing implementation is below: numeric units, missingness, categorical preparation/material fields, imputation, scaling, model fitting and prediction. The saved full-data assay model can make research predictions, but its generalization score must come from stored outer-fold predictions.

# %%
"""Version 2: audited pre-assay context and robust, grouped protein regression."""
import json
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from condition_model import load_data, environmental_features, metrics

CONTEXT_NAMES = ['enzyme_um','enzyme_mg_ml','enzyme_mg_g_pet','enzyme_vol_percent',
                 'substrate_mg_ml','reaction_volume_ml','crystallinity_percent','agitation_rpm']
FORMS = ['film','powder','particle','fiber','unknown']
PREPARATIONS = ['purified','crude','unknown']


def obj(value):
    if isinstance(value, (dict,list)):return value
    try:return json.loads(value)
    except (ValueError,TypeError):return {}


def numeric(value):
    if isinstance(value,bool):return np.nan
    try:
        value=float(value)
        return value if np.isfinite(value) and value>=0 else np.nan
    except (TypeError,ValueError):return np.nan


def unit(value):
    return str(value).strip().replace('μ','u').replace('µ','u').lower().replace(' ','')


def normalize_record(record):
    # Explicit whitelist: never encode measurements, evidence prose, IDs or target units.
    c=obj(record.get('conditions')); m=obj(record.get('material'))
    if not isinstance(c,dict):c={}
    if not isinstance(m,dict):m={}
    out={name:np.nan for name in CONTEXT_NAMES}
    v=numeric(c.get('enzyme_concentration_value')); u=unit(c.get('enzyme_concentration_unit'))
    if u in ('nm','um','mm'):out['enzyme_um']=v*{'nm':.001,'um':1,'mm':1000}[u]
    if u in ('mg/ml','ug/ml','g/l'):out['enzyme_mg_ml']=v*{'mg/ml':1,'ug/ml':.001,'g/l':1}[u]
    if u=='mgenzyme/gpet':out['enzyme_mg_g_pet']=v
    if u in ('vol%','%(v/v)'):out['enzyme_vol_percent']=v
    volume=numeric(c.get('reaction_volume_value')); vu=unit(c.get('reaction_volume_unit'))
    if vu in ('ul','ml','l'):out['reaction_volume_ml']=volume*{'ul':.001,'ml':1,'l':1000}[vu]
    amount=numeric(c.get('enzyme_amount_value')); au=unit(c.get('enzyme_amount_unit'))
    if not np.isfinite(out['enzyme_mg_ml']) and au in ('ug','mg') and out['reaction_volume_ml']>0:
        out['enzyme_mg_ml']=amount*{'ug':.001,'mg':1}[au]/out['reaction_volume_ml']
    s=numeric(c.get('substrate_concentration_value')); su=unit(c.get('substrate_concentration_unit'))
    if su in ('g/l','mg/ml','ug/ml','mg/ul','g/ml'):
        out['substrate_mg_ml']=s*{'g/l':1,'mg/ml':1,'ug/ml':.001,'mg/ul':1000,'g/ml':1000}[su]
    if not np.isfinite(out['substrate_mg_ml']) and unit(c.get('substrate_amount_unit'))=='mg' and out['reaction_volume_ml']>0:
        out['substrate_mg_ml']=numeric(c.get('substrate_amount_value'))/out['reaction_volume_ml']
    out['crystallinity_percent']=numeric(m.get('crystallinity_percent'))
    if out['crystallinity_percent']>100:out['crystallinity_percent']=np.nan
    out['agitation_rpm']=numeric(c.get('agitation_speed_rpm'))
    form=str(m.get('form','')).lower()
    out['material_form']='film' if any(x in form for x in ('film','disk','foil')) else 'powder' if 'powder' in form else 'fiber' if 'fiber' in form else 'particle' if any(x in form for x in ('particle','granule','nanomaterial')) else 'unknown'
    preparation=str(c.get('enzyme_preparation','')).lower()
    out['preparation']='purified' if preparation=='purified' else 'crude' if any(x in preparation for x in ('crude','supernatant','lysate')) else 'unknown'
    return out


def row_context(row):
    records=obj(row.get('original_record_json')); records=records if isinstance(records,list) else [records]
    items=[normalize_record(r) for r in records if isinstance(r,dict)]
    result={}
    for key in CONTEXT_NAMES:
        values=[r[key] for r in items if np.isfinite(r[key])]
        result[key]=values[0] if values and np.allclose(values,values[0],rtol=1e-7,atol=1e-10) else np.nan
    for key in ('material_form','preparation'):
        values={r[key] for r in items if r[key]!='unknown'}
        result[key]=next(iter(values)) if len(values)==1 else 'unknown'
    return result


def context_matrix(contexts):
    rows=[]
    for c in contexts:
        nums=[numeric(c.get(k)) for k in CONTEXT_NAMES]
        rows.append([*np.log1p(nums),*[float(c.get('material_form','unknown')==v) for v in FORMS],
                     *[float(c.get('preparation','unknown')==v) for v in PREPARATIONS]])
    return np.array(rows)


def dataset():
    data=load_data()
    data['contexts']=[row_context(r) for r in data['rows']]
    data['context']=context_matrix(data['contexts'])
    # Study IDs are split variables only, never features.
    data['studies']=np.array([r['literature_doi'] for r in data['rows']])
    return data


class AssayRegressor:
    def __init__(self,mode='sequence_context',kind='extra',strength=4,seed=42):
        self.mode,self.kind,self.strength,self.seed=mode,kind,strength,seed

    def features(self,seq,env,context):
        parts=[environmental_features(env)]
        if self.mode.startswith('sequence'):
            parts.append(self.pca.transform(self.sequence_scaler.transform(seq)))
        if self.mode.endswith('context'):parts.append(context)
        return np.column_stack(parts)

    def fit(self,seq,env,context,y,groups):
        if self.mode.startswith('sequence'):
            _,ids=np.unique(groups,return_index=True)
            self.sequence_scaler=StandardScaler().fit(seq[ids])
            self.pca=PCA(n_components=min(8,len(ids)-1),svd_solver='full').fit(self.sequence_scaler.transform(seq[ids]))
        x=self.features(seq,env,context)
        self.imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True).fit(x)
        self.scaler=StandardScaler().fit(self.imputer.transform(x))
        x=self.scaler.transform(self.imputer.transform(x)); target=np.log1p(y)
        if self.kind=='extra':model=ExtraTreesRegressor(n_estimators=240,min_samples_leaf=self.strength,max_features=.85,random_state=self.seed,n_jobs=1)
        elif self.kind=='boost':model=HistGradientBoostingRegressor(max_iter=300,learning_rate=.035,max_leaf_nodes=7,min_samples_leaf=8,l2_regularization=self.strength,early_stopping=False,random_state=self.seed)
        elif self.kind=='ridge':model=Ridge(alpha=self.strength)
        elif self.kind=='kernel':model=KernelRidge(alpha=self.strength,kernel='rbf',gamma=1/x.shape[1])
        elif self.kind=='neural':
            self.models=[];self.iterations=[]
            for seed in (self.seed,self.seed+1,self.seed+2):
                member=MLPRegressor(hidden_layer_sizes=(32,16),activation='tanh',alpha=self.strength,
                    learning_rate_init=.002,max_iter=700,n_iter_no_change=70,tol=1e-5,random_state=seed,early_stopping=False)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',category=UserWarning);member.fit(x,target)
                self.models.append(member);self.iterations.append(member.n_iter_)
            return self
        else:raise ValueError('Unknown model')
        model.fit(x,target);self.models=[model]
        return self

    def predict(self,seq,env,context):
        x=self.scaler.transform(self.imputer.transform(self.features(seq,env,context)))
        predictions=np.array([m.predict(x) for m in self.models]).mean(0)
        return np.expm1(np.clip(predictions,0,20))


def specs():
    # A frozen experiment, not an adaptive search against held-out scores.
    choices=[('ridge',100),('ridge',1000),('extra',2),('extra',6),('boost',10),('kernel',1),('kernel',10),('neural',1)]
    return [dict(mode=mode,kind=kind,strength=strength,seed=42) for mode in ('sequence_environment','context','sequence_context') for kind,strength in choices]


def fit_predict(data,spec,train,test):
    m=AssayRegressor(**spec).fit(*(data[k][train] for k in ('sequence','env','context','target','groups')))
    return m,m.predict(*(data[k][test] for k in ('sequence','env','context')))


def group_error(y,pred,groups):
    error=np.abs(np.log1p(y)-np.log1p(pred))
    return float(np.mean([error[groups==g].mean() for g in np.unique(groups)]))


def score(y,pred,groups):
    m=metrics(y,pred);m['group_log_mae']=group_error(y,pred,groups)
    error=np.abs(y-pred)
    m.update(median_absolute_error_um=float(np.median(error)),p90_absolute_error_um=float(np.quantile(error,.9)),
             within_twofold_fraction=float(np.mean(np.abs(np.log1p(y)-np.log1p(pred))<=np.log(2))))
    return m

# %%
import assay_model as assay_module
assay_module.AssayRegressor=AssayRegressor
AssayRegressor.__module__='assay_model'
assay_data=dataset()
with verify_file('models/condition_analysis/model.pkl').open('rb') as stream: saved_condition=pickle.load(stream)
with verify_file('models/assay_v2/train-20261024-160717/model.pkl').open('rb') as stream: saved_assay=pickle.load(stream)
saved_condition_pred=saved_condition.predict(*(data[k][test_idx] for k in ('sequence','env','image')))
saved_condition_report=json.loads(verify_file('models/condition_analysis/report.json').read_text())
assert np.isclose(mean_absolute_error(data['target'][test_idx],saved_condition_pred),saved_condition_report['selected_test']['mae_um'])
display(pd.DataFrame([evaluate_predictions(data['target'][test_idx],saved_condition_pred)],index=['Saved condition/image model / historical test']))
display(pd.DataFrame({'Field':CONTEXT_NAMES,'Measured entries':[np.isfinite(assay_data['context'][:,i]).sum() for i in range(len(CONTEXT_NAMES))]}))

# %% [markdown]
# ## 10. Model-call function: enter your own assay conditions
# Catalog IDs are supported because their embeddings are provided and sequence-verified. A new protein or mutant needs newly computed embeddings and independent evaluation; editing its name cannot reuse another sequence's features. Here `engine='original'`, `'condition'` or `'assay'` selects the actual saved model.

# %%
def predict_pet(protein_id,temperature_c=40,ph=8,duration_hours=48,engine='original'):
    protein_id=str(protein_id)
    if protein_id not in data['cache']: raise ValueError('Protein ID is not in the verified feature catalog.')
    env=np.array([[temperature_c,ph,duration_hours]],dtype=float);environmental_features(env)
    if engine=='original': return float(call_pet_neural(original_model,[protein_id],env,legacy_manifest)[0])
    entry=data['cache'][protein_id]
    seq=entry['sequence'][None,:];img=entry['image'][None,:]
    if engine=='condition': return float(saved_condition.predict(seq,env,img)[0])
    if engine=='assay':
        if saved_assay.mode!='sequence_environment': raise ValueError('This checkpoint also needs explicit assay-context inputs.')
        return float(saved_assay.predict(seq,env,np.full((1,assay_data['context'].shape[1]),np.nan))[0])
    raise ValueError('Choose original, condition or assay.')

PROTEIN_ID='1'
TEMPERATURE_C=40
PH=8
DURATION_HOURS=48
display(pd.DataFrame([{'Engine':engine,'Protein':PROTEIN_ID,'Predicted TPA (µM)':predict_pet(PROTEIN_ID,TEMPERATURE_C,PH,DURATION_HOURS,engine)} for engine in ('original','condition','assay')]))
prediction_table=pd.DataFrame([{'protein_id':pid,'protein_name':rows.loc[rows.protein_id==pid,'protein_name'].iloc[0],
    'predicted_tpa_um':predict_pet(pid,TEMPERATURE_C,PH,DURATION_HOURS)} for pid in data['cache']]).sort_values('predicted_tpa_um',ascending=False)
display(prediction_table.head(10));prediction_table.to_csv(RUN_DIR/'catalog_predictions.csv',index=False)

# %% [markdown]
# ## 11. Environmental response surface, sensitivity and scenario ranking
# Predictions across the condition grid describe the fitted model, not a causal experiment. The 10th percentile across 27 scenarios is a scenario summary, not a confidence bound. Plots are labelled with the model and units.

# %%
temperatures=np.linspace(30,80,7);ph_values=np.linspace(7,9,7)
grid=np.array([[t,p,DURATION_HOURS] for p in ph_values for t in temperatures])
surface=call_pet_neural(original_model,[PROTEIN_ID]*len(grid),grid,legacy_manifest).reshape(7,7)
fig,ax=plt.subplots();im=ax.imshow(surface,origin='lower',aspect='auto',extent=[30,80,7,9],cmap='viridis')
ax.set(xlabel='Temperature (°C)',ylabel='pH',title=f'Original PET model · protein {PROTEIN_ID} · {DURATION_HOURS} h')
fig.colorbar(im,ax=ax,label='Predicted TPA (µM)');save_figure('04_condition_response_surface')
fig,axes=plt.subplots(1,3,figsize=(13,3.5))
for axis,(label,values) in enumerate([('Temperature (°C)',np.linspace(30,80,13)),('pH',np.linspace(7,9,13)),('Duration (h)',np.linspace(1,672,13))]):
    env=np.repeat([[TEMPERATURE_C,PH,DURATION_HOURS]],13,axis=0).astype(float);env[:,axis]=values
    pred=call_pet_neural(original_model,[PROTEIN_ID]*13,env,legacy_manifest)
    axes[axis].plot(values,pred,color='#286b58');axes[axis].set(xlabel=label,ylabel='Predicted TPA (µM)')
save_figure('05_condition_sensitivity')
scenario_grid=np.array([[t,p,h] for t in [35,40,45] for p in [7.5,8,8.5] for h in [36,48,60]])
scenario_rows=[]
for pid in data['cache']:
    p=call_pet_neural(original_model,[pid]*27,scenario_grid,legacy_manifest)
    scenario_rows.append({'protein_id':pid,'scenario_p10_um':np.quantile(p,.1),'scenario_mean_um':p.mean(),'scenario_sd_um':p.std()})
scenario_table=pd.DataFrame(scenario_rows).sort_values('scenario_p10_um',ascending=False)
display(scenario_table.head(10));scenario_table.to_csv(RUN_DIR/'environment_scenarios.csv',index=False)

# %% [markdown]
# ## 12. Recompute the journal comparisons from saved out-of-fold predictions
# Each observation was predicted while its sequence group was held out. The stricter assay design also holds connected sequence/study groups together. These are recorded evaluations, not the full-data model predicting its own training labels. Methods from different evaluation designs are not placed on one shared leaderboard.

# %%
grouped=pd.read_csv(verify_file('models/condition_analysis/grouped_oof.csv'))
grouped_scores=pd.DataFrame([{'Inputs':mode,**evaluate_predictions(grouped.observed_um,grouped[mode])} for mode in ('sequence','environment','sequence_environment','multimodal')])
display(grouped_scores)
grouped_scores.plot.bar(x='Inputs',y='MAE',legend=False,color='#286b58',ylabel='Out-of-fold MAE (µM)',rot=20)
save_figure('06_condition_ablation')
assay_report=json.loads(verify_file('models/assay_v2/train-20261024-160717/report.json').read_text())
assay_comparison=[]
for design in ('sequence','sequence_and_study'):
    frame=pd.read_csv(verify_file(f'models/assay_v2/train-20261024-160717/{design}_oof.csv'))
    for mode in ('sequence_environment','sequence_context','context','training_median'):
        result=evaluate_predictions(frame.observed_um,frame[mode])
        assert np.isclose(result['MAE'],assay_report['evaluation'][design][mode]['mae_um'])
        assay_comparison.append({'Grouping':design,'Inputs':mode,**result})
display(pd.DataFrame(assay_comparison));pd.DataFrame(assay_comparison).to_csv(RUN_DIR/'recomputed_grouped_results.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4))
y=grouped.observed_um.to_numpy();p=grouped.sequence_environment.to_numpy()
axes[0].scatter(1+y,1+p,alpha=.6,s=20);axes[0].plot([1,1+y.max()],[1,1+y.max()],'k--')
axes[0].set(xscale='log',yscale='log',xlabel='1 + measured TPA',ylabel='1 + out-of-fold predicted TPA')
axes[1].scatter(y,p-y,alpha=.6,s=20);axes[1].axhline(0,color='black',linestyle='--')
axes[1].set(xlabel='Measured TPA (µM)',ylabel='Residual: predicted − measured (µM)')
save_figure('07_heldout_predictions_and_residuals')

# %% [markdown]
# ## 13. Protein structures, distance/contact images and residue inspection
# The full coordinate parsing functions are included below. They read the first PDB model, select C-alpha atoms and require an exact reference-sequence match. The five images reuse those coordinates. Select protein 1 and reference position 121 to reproduce the earlier interface example. The reference residue stays S121 even if a separately exported sequence changes it to E.

# %%
AA=dict(zip('ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(),'ACDEFGHIKLMNPQRSTVWY'))
def catalog():
    csv.field_size_limit(10_000_000)
    with (PET_ROOT / 'data/processed/tpa_pet_ml_confirmed.csv').open(encoding='utf-8-sig', newline='') as stream:
        return {r['protein_id']: r for r in csv.DictReader(stream)}

def coordinates(protein_id):
    if not isinstance(protein_id, str) or not protein_id.isdecimal() or protein_id not in catalog():
        raise ValueError('Select a protein from the catalog.')
    path = PET_ROOT / 'data/structure_images/pdb' / (protein_id + '.pdb')
    if not path.is_file():
        raise ValueError('No verified structure is available for this protein. A sequence alone does not define a validated 3D image.')
    entries = {}
    model_seen = False
    for line in path.read_text().splitlines():
        record = line[:6].strip()
        if record == 'MODEL':
            if model_seen:
                break
            model_seen = True
        if record == 'ENDMDL':
            break
        if record != 'ATOM' or line[12:16].strip() != 'CA':
            continue
        chain, number, insertion = line[21:22], int(line[22:26]), line[26:27]
        amino = AA.get(line[17:20].strip(), 'X')
        xyz = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
        occupancy = float(line[54:60].strip() or 1)
        if not np.isfinite(xyz + [occupancy]).all() or occupancy < 0:
            raise ValueError('Nonfinite or invalid structure coordinates.')
        residues = entries.setdefault(chain, {})
        key = (number, insertion)
        order = (-occupancy, line[16:17].strip())
        old = residues.get(key)
        if old and old[1] != amino:
            raise ValueError('Ambiguous alternate residue identity.')
        if old is None or order < old[0]:
            residues[key] = (order, amino, xyz)
    reference = catalog()[protein_id]['sequence'].strip().upper()
    for values in entries.values():
        if ''.join(v[1] for v in values.values()) == reference:
            xyz = np.asarray([v[2] for v in values.values()], dtype=np.float64)
            return xyz, hashlib.sha256(path.read_bytes()).hexdigest()
    raise ValueError('Structure does not exactly match the complete reference sequence.')

def distance_image(xyz):
    distances = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    return np.uint8(np.clip(1 - distances / 20.0, 0, 1) * 255), distances

def trace_svg(xyz, axes):
    centered = xyz - xyz.mean(0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    for axis in range(3):
        pivot = np.argmax(np.abs(vectors[axis]))
        if vectors[axis, pivot] < 0:
            vectors[axis] *= -1
    points = (centered @ vectors.T)[:, axes]
    scale = 480 / max(float(np.ptp(points, axis=0).max()), 1)
    points = (points - (points.max(0) + points.min(0))/2) * scale + 300
    lines = []
    for i in range(len(points)-1):
        # Break chain segments rather than drawing over missing-coordinate gaps.
        if np.linalg.norm(xyz[i+1]-xyz[i]) > 5.0:
            continue
        hue = 150 + 90 * i / max(1, len(points)-2)
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        lines.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="hsl({hue:.2f} 60% 43%)" stroke-width="3" stroke-linecap="round"/>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600"><rect width="600" height="600" fill="#f4f7f2"/>' + ''.join(lines) + '</svg>'

# %%
xyz,structure_sha=coordinates(PROTEIN_ID)
pixels,distances=distance_image(xyz)
centered=xyz-xyz.mean(0);_,_,vectors=np.linalg.svd(centered,full_matrices=False);projected=centered@vectors.T
fig,axes=plt.subplots(1,5,figsize=(18,3.5))
for ax,(a,b),title in zip(axes[:3],[(0,1),(0,2),(1,2)],['Front','Side','Top']):
    for i in range(len(xyz)-1):
        if np.linalg.norm(xyz[i+1]-xyz[i])<=5:
            ax.plot(projected[i:i+2,a],projected[i:i+2,b],color=plt.cm.viridis(i/(len(xyz)-1)),linewidth=1)
    ax.set(title=title,aspect='equal');ax.axis('off')
axes[3].imshow(pixels,cmap='gray');axes[3].set(title='Distance: cap 20 Å',xlabel='Residue index',ylabel='Residue index')
axes[4].imshow(distances<=8,cmap='gray');axes[4].set(title='C-alpha proximity ≤8 Å',xlabel='Residue index')
save_figure('08_five_reference_structure_views')
RESIDUE_POSITION=121
if not 1<=RESIDUE_POSITION<=len(xyz): raise ValueError('Residue is outside this matched structure.')
fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='3d')
for i in range(len(xyz)-1):
    if np.linalg.norm(xyz[i+1]-xyz[i])<=5: ax.plot(*xyz[i:i+2].T,color=plt.cm.viridis(i/(len(xyz)-1)))
ax.scatter(*xyz[RESIDUE_POSITION-1],color='orange',s=90)
ax.set(title=f'Reference protein {PROTEIN_ID} · position {RESIDUE_POSITION}',xlabel='x (Å)',ylabel='y (Å)',zlabel='z (Å)')
save_figure('09_reference_structure_3d')
reference=rows.loc[rows.protein_id==PROTEIN_ID,'sequence'].iloc[0]
neighbors=np.flatnonzero((distances[RESIDUE_POSITION-1]<=8)&(np.arange(len(xyz))!=RESIDUE_POSITION-1))
display(pd.DataFrame({'Reference position':neighbors+1,'Residue':[reference[i] for i in neighbors],'C-alpha distance (Å)':distances[RESIDUE_POSITION-1,neighbors]}))

# %% [markdown]
# ## 14. Outputs and panel discussion
# Explain which target is measured, why sequence groups are kept together, why validation selects a model, and why training-set fit is not an unseen-protein score. Report negative R² and unsuccessful image ablations alongside any MAE improvement. The model files loaded above are real artifacts; new demonstration weights are separate files under the printed output directory.

# %%
summary={'notebook':'PET/environment','dataset_sha256':data['dataset_sha256'],'source_rows':len(raw),'eligible_rows':len(rows),
         'original_checkpoint_test':evaluate_predictions(data['target'][test_idx],original_prediction),
         'demo_neural_test':evaluate_predictions(data['target'][test_idx],demo_neural_prediction),
         'demo_neural_epochs':DEMO_EPOCHS,'demo_best_epoch':trained_meta['best_epoch'],
         'structure_sha256':structure_sha,'output_directory':str(RUN_DIR),'mutant_fold_predicted':False}
(RUN_DIR/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2));display(pd.DataFrame({'Saved output':[p.name for p in sorted(RUN_DIR.iterdir())]}))

# %% [markdown]
# # EnzAIme × Helix: GFP benchmark, ESM training and directed evolution
# 
# Run top to bottom. This notebook loads measured avGFP sequences, reproduces existing checkpoint inference, creates one shared train/validation/test split for a new small demonstration, trains a Ridge baseline and three ESM Attention1D heads, evaluates both on identical test sequences, and generates mutation candidates.
# 
# **The default 200-sequence run is for a panel demonstration.** Change sample size and budgets before starting a planned experiment. Its results are saved independently from the earlier experiments. Frozen ESM-2 8M and 35M weights are included; the default uses 8M so no model download is needed. The independent evaluator used by the base paper is not included. Search scores are predictions from the model being optimized. GFP data has sequence/fitness labels only; temperature/pH/time cannot be inferred from these records.

# %%
NOTEBOOK_NAME='gfp'
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
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
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
# ## 1. Dataset call, cleaning and fixed split
# Deduplicate sequences before splitting and average repeated measured values. The shared three-way split is made once before either model is fitted. Identical sequence strings never cross partitions. This is not a sequence-family or mutation-distance split.

# %%
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
AMINO_ACIDS='ACDEFGHIKLMNPQRSTVWY'
def load_gfp_dataset(path):
    raw=pd.read_csv(path)
    raw['sequence']=raw.sequence.astype(str).str.strip().str.upper()
    raw['fitness']=pd.to_numeric(raw.fitness,errors='coerce')
    valid=raw.sequence.str.fullmatch('['+AMINO_ACIDS+']+') & np.isfinite(raw.fitness)
    clean=raw.loc[valid].groupby('sequence',sort=False,as_index=False).fitness.mean()
    if clean.sequence.str.len().nunique()!=1: raise ValueError('The current encoder-head example expects equal-length sequences.')
    return raw,clean

gfp_path=verify_file('Directed_Evolution-main/preprocessed_data/avGFP/avGFP.csv')
gfp_raw,gfp=load_gfp_dataset(gfp_path)
SAMPLES=200
SEED=42
HEAD_EPOCHS=18
ENSEMBLE_SIZE=3
ESM_FOLDER='esm2_8m'  # Change to esm2_35m before training for the larger encoder.
if not 100<=SAMPLES<=len(gfp): raise ValueError('Choose a sample size from 100 through the dataset size.')
sample=gfp.sample(n=SAMPLES,random_state=SEED).reset_index(drop=True)
indices=np.arange(SAMPLES)
trainval,test_ids=train_test_split(indices,test_size=.2,random_state=SEED)
train_ids,validation_ids=train_test_split(trainval,test_size=.25,random_state=SEED)
for a,b in [(train_ids,validation_ids),(train_ids,test_ids),(validation_ids,test_ids)]:
    assert not set(sample.sequence.iloc[a]) & set(sample.sequence.iloc[b])
print('Available unique sequences:',len(gfp),'| source columns:',list(gfp_raw.columns))
display(sample.head())
display(pd.DataFrame({'Partition':['Train','Validation','Test'],'Rows':[len(train_ids),len(validation_ids),len(test_ids)]}))
split_record={name:sample.iloc[ids].sequence.tolist() for name,ids in [('train',train_ids),('validation',validation_ids),('test',test_ids)]}
(RUN_DIR/'split.json').write_text(json.dumps(split_record,indent=2))
plt.hist(gfp.fitness,bins=35,color='#286b58');plt.xlabel('Measured GFP fitness');plt.ylabel('Sequences')
save_figure('01_gfp_dataset_distribution')

# %% [markdown]
# ## 2. Load the original saved Ridge checkpoint and reproduce its evaluation
# This saved model used 1,600 training sequences and a separate 400-sequence test. It is a different experiment from the small demonstration below. A pickle is loaded only from this verified local bundle. Its feature vectorizer must accompany its estimator.

# %%
def position_features(sequences):
    return [{f'{i}:{letter}':1 for i,letter in enumerate(sequence)} for sequence in sequences]

baseline_path=verify_file('runs/20261023-162906-a42ad8/baseline.pkl')
with baseline_path.open('rb') as stream: saved_vectorizer,saved_ridge=pickle.load(stream)
saved_holdout=pd.read_csv(verify_file('runs/20261023-162906-a42ad8/holdout.csv'))
saved_predictions=saved_ridge.predict(saved_vectorizer.transform(position_features(saved_holdout.sequence)))
np.testing.assert_allclose(saved_predictions,saved_holdout.predicted_fitness,rtol=1e-6,atol=1e-6)
display(pd.DataFrame([evaluate_predictions(saved_holdout.measured_fitness,saved_predictions)],index=['Saved Ridge / original 400-row test']))
print('Actual saved checkpoint:',baseline_path)

# %% [markdown]
# ## 3. Training function for the live Ridge baseline
# The vectorizer and model are fitted on training sequences. Validation chooses alpha. The test is evaluated only after selection. All choices are specified here before fitting.

# %%
def train_gfp_ridge(frame,training,validation,alphas=(1,10,100)):
    vectorizer=DictVectorizer()
    train_x=vectorizer.fit_transform(position_features(frame.sequence.iloc[training]))
    validation_x=vectorizer.transform(position_features(frame.sequence.iloc[validation]))
    fitted=[]
    for alpha in alphas:
        model=Ridge(alpha=alpha,solver='lsqr').fit(train_x,frame.fitness.iloc[training])
        loss=mean_absolute_error(frame.fitness.iloc[validation],model.predict(validation_x))
        fitted.append((float(loss),alpha,model))
    loss,alpha,model=min(fitted,key=lambda row:row[0])
    return vectorizer,model,{'alpha':alpha,'validation_mae':loss}

vectorizer,ridge,ridge_selection=train_gfp_ridge(sample,train_ids,validation_ids)
ridge_test=ridge.predict(vectorizer.transform(position_features(sample.sequence.iloc[test_ids])))
with (RUN_DIR/'gfp_ridge_demo.pkl').open('wb') as stream: pickle.dump((vectorizer,ridge),stream)
print('Validation choice:',ridge_selection)

# %% [markdown]
# ## 4. Original Attention1D decoder code
# This decoder is from the downloaded paper implementation, which attributes its attention/convolution components to Microsoft's protein-sequence-models project. The pretrained encoder is frozen. Its learned representation is passed to a trainable task-specific decoder. For compatibility with the saved local decoder, the representation includes the encoder's special tokens and all sequences have the same length.

# %%
""" Code adopted from: `https://github.com/microsoft/protein-sequence-models` """

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv1d(nn.Conv1d):
    """ A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

    Shape:
        Input: (N, L, in_channels)
        input_mask: (N, L, 1), optional
        Output: (N, L, out_channels)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):
        """
        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            kernel_size (int): the kernel width
            stride (int): filter shift
            dilation (int): dilation factor
            groups (int): perform depth-wise convolutions
            bias (bool): adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Attention1d(nn.Module):

    def __init__(self, in_dim: int):
        super().__init__()
        self.layer = MaskedConv1d(in_dim, 1, 1)

    def forward(self, x, input_mask=None):
        n, ell, _ = x.shape
        attn = self.layer(x)
        attn = attn.view(n, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(~input_mask.view(n, -1).bool(),
                                     float('-inf'))
        attn = F.softmax(attn, dim=-1).view(n, -1, 1)
        out = (attn * x).sum(dim=1)
        return out


class Decoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention1d = Attention1d(in_dim=hidden_dim)
        self.dense_3 = nn.Linear(hidden_dim, hidden_dim)
        self.dense_4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.dense_1(x))
        x = torch.relu(self.dense_2(x))
        x = self.attention1d(x)
        x = torch.relu(self.dense_3(x))
        x = self.dense_4(x)
        return x

# %% [markdown]
# ## 5. Load the actual offline ESM model and encode sequences
# `model.safetensors` stores pretrained ESM weights. `decoder.pt` stores a separately trained task head. They are different files and both are needed for ESM inference. Set `ESM_FOLDER` above before encoding. The same pretrained weights are used for train, validation and test; no labels enter the frozen encoder.

# %%
import transformers
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
if transformers.__version__!='4.44.2':
    raise RuntimeError('Use transformers==4.44.2 from requirements-kaggle.txt for this checked workflow, then restart the kernel.')
torch.set_num_threads(2)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path=BUNDLE/'models'/ESM_FOLDER
verify_file(f'models/{ESM_FOLDER}/model.safetensors')
tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
encoder=EsmModel.from_pretrained(str(model_path),local_files_only=True).to(DEVICE).eval()
for parameter in encoder.parameters(): parameter.requires_grad_(False)
embedding_cache={}

@torch.inference_mode()
def encode_sequences(sequences,batch_size=8):
    sequences=list(sequences)
    if not sequences or any(set(s)-set(AMINO_ACIDS) for s in sequences): raise ValueError('Expected nonempty canonical protein sequences.')
    if len({len(s) for s in sequences})!=1: raise ValueError('Use equal-length sequences in this example.')
    missing=list(dict.fromkeys(s for s in sequences if s not in embedding_cache))
    for start in range(0,len(missing),batch_size):
        part=missing[start:start+batch_size]
        tokens=tokenizer(part,return_tensors='pt',padding=True).to(DEVICE)
        features=encoder(**tokens).last_hidden_state.detach().cpu()
        for sequence,feature in zip(part,features): embedding_cache[sequence]=feature.clone()
    return torch.stack([embedding_cache[s] for s in sequences])

started=time.time();features=encode_sequences(sample.sequence.tolist())
print('Encoder:',model_path,'| device:',DEVICE,'| feature shape:',tuple(features.shape))
print('Encoding seconds:',round(time.time()-started,2))

# %% [markdown]
# ## 6. Neural training function with validation-based checkpoint selection
# Each member uses a bootstrap of the training indices. Validation chooses the epoch; test labels are never used in training. Three-head disagreement is an uncalibrated uncertainty heuristic. Precomputing frozen features makes head training much faster than re-encoding sequences in every epoch.

# %%
@torch.inference_mode()
def predict_head_features(heads,x,batch_size=32):
    member_predictions=[]
    for head in heads:
        head.eval();values=[]
        for start in range(0,len(x),batch_size): values.extend(head(x[start:start+batch_size].to(DEVICE)).flatten().cpu().tolist())
        member_predictions.append(values)
    member_predictions=np.asarray(member_predictions)
    return member_predictions.mean(0),member_predictions.std(0)

def train_gfp_attention(x,y,training,validation,epochs=18,members=3,seed=42):
    heads=[];history=[]
    for member in range(members):
        torch.manual_seed(seed+member);rng=np.random.default_rng(seed+member)
        head=Decoder(x.shape[-1],128).to(DEVICE)
        optimizer=torch.optim.Adam(head.parameters(),lr=1e-3)
        member_train=rng.choice(training,size=len(training),replace=True) if members>1 else np.asarray(training)
        best=float('inf');best_state=None;stale=0
        for epoch in range(epochs):
            head.train();order=rng.permutation(member_train);losses=[]
            for start in range(0,len(order),16):
                ids=order[start:start+16];batch=x[ids].to(DEVICE)
                target=torch.tensor(y[ids],dtype=torch.float32,device=DEVICE)
                optimizer.zero_grad();pred=head(batch).flatten()
                loss=torch.nn.functional.mse_loss(pred,target);loss.backward();optimizer.step();losses.append((loss.item(),len(ids)))
            validation_pred,_=predict_head_features([head],x[validation])
            loss=float(mean_absolute_error(y[validation],validation_pred))
            history.append({'member':member+1,'epoch':epoch+1,'train_mse':sum(v*n for v,n in losses)/len(order),'validation_mae':loss})
            if loss<best: best=loss;best_state=copy.deepcopy(head.state_dict());stale=0
            else: stale+=1
            if stale>=6: break
        head.load_state_dict(best_state);head.eval();heads.append(head)
    return heads,pd.DataFrame(history)

heads,learning=train_gfp_attention(features,sample.fitness.to_numpy(float),train_ids,validation_ids,HEAD_EPOCHS,ENSEMBLE_SIZE,SEED)
checkpoint={'decoders':[{k:v.detach().cpu() for k,v in head.state_dict().items()} for head in heads],
            'input_dim':int(features.shape[-1]),'hidden_dim':128,'model_folder':ESM_FOLDER,'ensemble_size':len(heads),
            'encoder_source':json.loads((model_path/'SOURCE.json').read_text()),'seed':SEED,'samples':SAMPLES}
torch.save(checkpoint,RUN_DIR/'gfp_attention_demo.pt')
fig,axes=plt.subplots(1,2,figsize=(11,4))
for member,part in learning.groupby('member'):
    axes[0].plot(part.epoch,part.train_mse,label=f'Head {member}')
    axes[1].plot(part.epoch,part.validation_mae,label=f'Head {member}')
axes[0].set(xlabel='Epoch',ylabel='Training MSE');axes[1].set(xlabel='Epoch',ylabel='Validation MAE')
axes[0].legend();axes[1].legend();save_figure('02_gfp_learning_curves')

# %% [markdown]
# ## 7. Matched predictive evaluation and reload verification
# Both freshly trained approaches use exactly the same measured test labels. This small, single-seed comparison demonstrates the evaluation workflow; repeat a prespecified study on larger data before using it as primary journal evidence. The earlier 400-row baseline result is not part of this table.

# %%
def load_gfp_heads(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    loaded=[]
    for weights in state['decoders']:
        head=Decoder(state['input_dim'],state['hidden_dim']).to(DEVICE)
        head.load_state_dict(weights);loaded.append(head.eval())
    return loaded

gfp_test,spread=predict_head_features(heads,features[test_ids])
reloaded_heads=load_gfp_heads(RUN_DIR/'gfp_attention_demo.pt')
np.testing.assert_allclose(gfp_test,predict_head_features(reloaded_heads,features[test_ids])[0],rtol=1e-6,atol=1e-6)
comparison=pd.DataFrame([{'Model':'One-hot Ridge',**evaluate_predictions(sample.fitness.iloc[test_ids],ridge_test)},
                         {'Model':f'Frozen ESM + {len(heads)} Attention1D heads',**evaluate_predictions(sample.fitness.iloc[test_ids],gfp_test)}])
display(comparison);comparison.to_csv(RUN_DIR/'matched_predictor_comparison.csv',index=False)
holdout=sample.iloc[test_ids].copy();holdout['ridge_prediction']=ridge_test;holdout['esm_prediction']=gfp_test;holdout['ensemble_sd']=spread
holdout.to_csv(RUN_DIR/'matched_holdout_predictions.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,pred,label in zip(axes,[ridge_test,gfp_test],['One-hot Ridge','Frozen ESM + Attention1D']):
    y=sample.fitness.iloc[test_ids];ax.scatter(y,pred,s=28,alpha=.7);limits=[min(y.min(),pred.min()),max(y.max(),pred.max())]
    ax.plot(limits,limits,'k--');ax.set(xlabel='Measured fitness',ylabel='Predicted fitness',title=label)
save_figure('03_matched_gfp_evaluation')

# %% [markdown]
# ## 8. Call the GFP model manually and reproduce the old ESM head
# `predict_gfp` accepts sequence strings and returns mean predicted fitness plus head disagreement. These are not environmental predictions. The existing 8M decoder has its own 20-sequence test; it is reproduced separately when using the matching 8M encoder.

# %%
def predict_gfp(sequences):
    return predict_head_features(heads,encode_sequences(sequences))

example=sample.sequence.iloc[int(train_ids[0])]
mean,sd=predict_gfp([example])
print('Example sequence length:',len(example),'| predicted fitness:',float(mean[0]),'| ensemble SD:',float(sd[0]))
if ESM_FOLDER=='esm2_8m':
    old_path=verify_file('runs/20261024-092703-afed29/decoder.pt')
    old_state=torch.load(old_path,map_location='cpu',weights_only=True)
    old_heads=[]
    # The earlier single-head checkpoint used 'decoder'; later ensembles use 'decoders'.
    old_weights=old_state['decoders'] if 'decoders' in old_state else [old_state['decoder']]
    for state in old_weights:
        head=Decoder(encoder.config.hidden_size,old_state['hidden_dim']).to(DEVICE);head.load_state_dict(state);old_heads.append(head.eval())
    old_test=pd.read_csv(verify_file('runs/20261024-092703-afed29/holdout.csv'))
    old_pred,_=predict_head_features(old_heads,encode_sequences(old_test.sequence))
    np.testing.assert_allclose(old_pred,old_test.predicted_fitness,rtol=2e-4,atol=2e-4)
    display(pd.DataFrame([evaluate_predictions(old_test.measured_fitness,old_pred)],index=['Saved 8M demonstration / its original test']))

# %% [markdown]
# ## 9. Mutation proposal and directed-evolution functions
# This compact notebook adaptation proposes single substitutions using the frozen masked-language model, retains parents in the candidate pool, and selects a new pool by predicted fitness or an uncertainty/diversity heuristic. The original application's complete engine and the base paper's source are supplied in the companion source folders and notebook 03.
# 
# Candidate fitness is evaluated by the search predictor. Increasing that score is not proof of improved measured fluorescence. The uncertainty/diversity rule adds `0.5 × ensemble SD` and a `0.5 × normalized Hamming distance` term; this heuristic is not a validated novel algorithm or calibrated acquisition function.

# %%
mlm=EsmForMaskedLM.from_pretrained(str(model_path),local_files_only=True).to(DEVICE).eval()
@torch.inference_mode()
def propose_mutation(sequence,rng):
    position=int(rng.integers(len(sequence)))
    tokens=tokenizer(sequence,return_tensors='pt').to(DEVICE)
    tokens['input_ids'][0,position+1]=tokenizer.mask_token_id
    logits=mlm(**tokens).logits[0,position+1]
    allowed=[a for a in AMINO_ACIDS if a!=sequence[position]]
    ids=tokenizer.convert_tokens_to_ids(allowed)
    replacement=allowed[int(logits[ids].argmax().item())]
    return sequence[:position]+replacement+sequence[position+1:]

def hamming(a,b):
    if len(a)!=len(b): raise ValueError('Hamming distance requires equal lengths.')
    return sum(x!=y for x,y in zip(a,b))

def evolve_sequences(parent,predict_fn,steps=3,population=4,selection='greedy',seed=42):
    if selection not in ('greedy','uncertainty_diversity'): raise ValueError('Unknown selection rule.')
    rng=np.random.default_rng(seed);pool=[parent];history=[]
    for step in range(steps):
        candidates=list(pool)
        for sequence in pool:
            candidates.extend(propose_mutation(sequence,rng) for _ in range(2))
        candidates=list(dict.fromkeys(candidates))
        means,sd=predict_fn(candidates)
        if selection=='greedy': chosen=np.argsort(-means,kind='stable')[:population].tolist()
        else:
            chosen=[];remaining=set(range(len(candidates)))
            while remaining and len(chosen)<population:
                def acquisition(i):
                    diversity=min((hamming(candidates[i],candidates[j])/len(parent) for j in chosen),default=0)
                    return means[i]+.5*sd[i]+.5*diversity
                best=max(remaining,key=lambda i:(acquisition(i),-i));chosen.append(best);remaining.remove(best)
        pool=[candidates[i] for i in chosen]
        history.append({'round':step+1,'best_predicted_fitness':float(max(means[chosen])),
                        'mean_predicted_fitness':float(np.mean(means[chosen])),'unique_sequences':len(pool)})
    means,sd=predict_fn(pool)
    result=pd.DataFrame({'sequence':pool,'predicted_fitness':means,'ensemble_sd':sd,
                         'distance_from_parent':[hamming(parent,s) for s in pool],
                         'substitutions':[';'.join(f'{a}{i+1}{b}' for i,(a,b) in enumerate(zip(parent,s)) if a!=b) or 'Parent' for s in pool]})
    return result,pd.DataFrame(history)

training=sample.iloc[train_ids]
parent=training.iloc[np.argmin(np.abs(training.fitness.to_numpy()-training.fitness.median()))].sequence
results={};histories={}
for rule in ('greedy','uncertainty_diversity'):
    results[rule],histories[rule]=evolve_sequences(parent,predict_gfp,selection=rule,seed=SEED)
    results[rule].to_csv(RUN_DIR/f'{rule}_candidates.csv',index=False)
    display(Markdown(f'**{rule}: predicted candidates**'));display(results[rule].drop(columns='sequence'))
fig,ax=plt.subplots()
for rule,history in histories.items(): ax.plot(history['round'],history.best_predicted_fitness,marker='o',label=rule)
ax.set(xlabel='Search round',ylabel='Best predicted fitness',title='Search-model score; independent evaluator unavailable');ax.legend()
save_figure('04_predicted_search_progress')
with (RUN_DIR/'candidate_sequences.fasta').open('w') as stream:
    for rule,frame in results.items():
        for i,row in frame.iterrows(): stream.write(f'>{rule}_{i+1} predicted_fitness={row.predicted_fitness:.6f}\n{row.sequence}\n')

# %% [markdown]
# ## 10. Export provenance and explain the benchmark scope
# The new Ridge and ESM results share one dataset sample and split. Repeated seeds, larger budgets and a prespecified evaluation are needed for a stronger predictor study. The paper's independent-oracle design scores are a separate benchmark; do not compare them numerically with the MAE or optimized surrogate scores here. Environmental-response novelty requires additional measured condition labels.

# %%
summary={'dataset_sha256':hashlib.sha256(gfp_path.read_bytes()).hexdigest(),'available_sequences':len(gfp),
         'demonstration_samples':SAMPLES,'split_sizes':{k:len(v) for k,v in split_record.items()},
         'seed':SEED,'head_epochs_max':HEAD_EPOCHS,'ensemble_size':len(heads),'encoder':checkpoint['encoder_source'],
         'matched_comparison':comparison.to_dict(orient='records'),'independent_oracle':False,'paper_reproduction':False,
         'environmental_labels_present':False,'output_directory':str(RUN_DIR)}
(RUN_DIR/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2));display(pd.DataFrame({'Saved output':[p.name for p in sorted(RUN_DIR.iterdir())]}))
