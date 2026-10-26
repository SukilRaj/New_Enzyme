# %% [markdown]
# EnzAIme × Helix: PET, environmental data, training and evaluation

**Panel demonstration notebook.** Run cells from top to bottom. The actual preprocessing, architecture, training, prediction and plotting code is visible below. The companion folder contains data, frozen ESM features and saved checkpoints; no frontend or server is used.

You will: audit the 221 source records; reproduce 208 eligible rows; inspect temperature/pH/time; load the original neural checkpoint; evaluate its historical holdout; train fresh condition models and an Attention1D head; call models with your own conditions; inspect grouped results; and render real reference structures. Quick demonstration fits are saved separately from the original results.

TPA concentration is a regression target in **µM**. A sequence edit does not establish better enzyme function. The cached structures describe reference proteins; this notebook does not predict mutant folds.

# %%
NOTEBOOK_NAME='pet'
{{COMMON}}

# %% [markdown]
## 1. Environment and package versions
Use Python 3.10–3.12 with the supplied requirements. Kaggle: attach the **extracted bundle** as a dataset. Input data stays read-only; new outputs go to `/kaggle/working/enzaime_outputs`. Install dependencies once if requested, restart the kernel, then Run All.

# %%
{{DEPS}}

# %% [markdown]
## 2. Locate the actual models and data
The original neural checkpoint has a historical holdout. The condition surrogate also preserves its historical split. The assay-v2 `model.pkl` was refitted on all 208 rows: use its saved out-of-fold predictions for generalization assessment, not predictions on its training data.

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
## 3. Complete preprocessing and condition-regression implementation
The code validates finite measurements, removes nonpositive durations/negative targets and exact duplicate assay records, preserves source row IDs, verifies sequence-feature hashes, and pools cached per-residue ESM embeddings. Cached embeddings are frozen representations, not model predictions or labels.

Scaling and PCA are fitted within training subsets. The six environmental descriptors use only temperature, pH and duration. Image descriptors use existing sequence-matched geometry. DOI and protein IDs are split/provenance fields, not predictors.

# %%
{{CONDITION_SOURCE}}

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
## 4. Environmental data and target visualization
Measured values are shown here. Later response surfaces are model predictions. Neither is a direct estimate of carbon emissions or real-world plastic removal.

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
## 5. Preserve the recorded training, validation and test groups
This is the original historical split, which has already been inspected during development. It is useful for reproduction and teaching; it is not a new external confirmation set. No test labels are used for preprocessing, fitting, early stopping or hyperparameter selection below.

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
## 6. Original Attention1D architecture and batch function
This is the architecture used by the installed PET checkpoint. Frozen residue embeddings are pooled by learned attention, then combined with normalized assay conditions. The original checkpoint declares `image_trained=false`, so its image branch is disabled during inference.

# %%
{{NEURAL_SOURCE}}

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
## 7. Train fresh condition models: training function, validation selection, test evaluation
For a fast live demonstration, compare four feature sets using Ridge strengths fixed in advance. The complete 25-configuration condition study and 24-configuration assay study are in notebook 03. This small demonstration does not replace their saved journal results.

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
## 8. Train an Attention1D neural head and show its learning curve
The encoder remains frozen: training uses the packaged ESM residue features. This cell trains a new head for 12 demonstration epochs, chooses weights by validation loss, and evaluates the test only afterward. Increase the budget for a planned study before inspecting its test result. The original checkpoint is preserved.

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
## 9. Load the saved condition and assay-context checkpoints
The full assay preprocessing implementation is below: numeric units, missingness, categorical preparation/material fields, imputation, scaling, model fitting and prediction. The saved full-data assay model can make research predictions, but its generalization score must come from stored outer-fold predictions.

# %%
{{ASSAY_SOURCE}}

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
## 10. Model-call function: enter your own assay conditions
Catalog IDs are supported because their embeddings are provided and sequence-verified. A new protein or mutant needs newly computed embeddings and independent evaluation; editing its name cannot reuse another sequence's features. Here `engine='original'`, `'condition'` or `'assay'` selects the actual saved model.

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
## 11. Environmental response surface, sensitivity and scenario ranking
Predictions across the condition grid describe the fitted model, not a causal experiment. The 10th percentile across 27 scenarios is a scenario summary, not a confidence bound. Plots are labelled with the model and units.

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
## 12. Recompute the journal comparisons from saved out-of-fold predictions
Each observation was predicted while its sequence group was held out. The stricter assay design also holds connected sequence/study groups together. These are recorded evaluations, not the full-data model predicting its own training labels. Methods from different evaluation designs are not placed on one shared leaderboard.

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
## 13. Protein structures, distance/contact images and residue inspection
The full coordinate parsing functions are included below. They read the first PDB model, select C-alpha atoms and require an exact reference-sequence match. The five images reuse those coordinates. Select protein 1 and reference position 121 to reproduce the earlier interface example. The reference residue stays S121 even if a separately exported sequence changes it to E.

# %%
AA=dict(zip('ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(),'ACDEFGHIKLMNPQRSTVWY'))
{{COORDINATES_SOURCE}}

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
## 14. Outputs and panel discussion
Explain which target is measured, why sequence groups are kept together, why validation selects a model, and why training-set fit is not an unseen-protein score. Report negative R² and unsuccessful image ablations alongside any MAE improvement. The model files loaded above are real artifacts; new demonstration weights are separate files under the printed output directory.

# %%
summary={'notebook':'PET/environment','dataset_sha256':data['dataset_sha256'],'source_rows':len(raw),'eligible_rows':len(rows),
         'original_checkpoint_test':evaluate_predictions(data['target'][test_idx],original_prediction),
         'demo_neural_test':evaluate_predictions(data['target'][test_idx],demo_neural_prediction),
         'demo_neural_epochs':DEMO_EPOCHS,'demo_best_epoch':trained_meta['best_epoch'],
         'structure_sha256':structure_sha,'output_directory':str(RUN_DIR),'mutant_fold_predicted':False}
(RUN_DIR/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2));display(pd.DataFrame({'Saved output':[p.name for p in sorted(RUN_DIR.iterdir())]}))
