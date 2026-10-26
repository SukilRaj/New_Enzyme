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
