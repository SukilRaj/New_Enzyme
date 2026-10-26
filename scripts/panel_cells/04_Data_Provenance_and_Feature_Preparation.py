# %% [markdown]
# Dataset provenance and offline ESM feature preparation

This notebook shows how the supplied PET assay records become the confirmed training CSV, and how an amino-acid sequence becomes per-residue ESM features. It runs the original provenance and inclusion scripts against copies of the source records, then checks the regenerated dataset against the packaged one. These records are a collected literature dataset; this notebook does not repeat the original literature collection.

All writes go to a fresh output folder. The original ambiguous-record review is retained as a source artifact. Its six exclusions are not invented by a model. Notebook 01 then removes duplicate or ineligible observations from the 221 confirmed records to obtain 208 modeling rows.

# %%
NOTEBOOK_NAME='data_preparation'
{{COMMON}}

# %%
{{DEPS}}

# %% [markdown]
## 1. Copy the source records into a writable preparation folder

# %%
import shutil
PREP_ROOT=RUN_DIR/'pet_preparation'
for relative in ('master_dataset.csv','data/tpa_pet_ml_dataset.csv','reports/results/tpa_ambiguous_review.md'):
    src=verify_file('pet/'+relative);dest=PREP_ROOT/relative
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
print('Source records copied to',PREP_ROOT)

# %% [markdown]
## 2. Original analyte, unit-conversion and provenance audit
The next cell is the complete original audit script; only its root directory is changed. It reconstructs 227 candidate observations from the collected records, preserves source JSON/DOIs, checks analyte identity and conversion rules, and marks six records as ambiguous.

# %%
{{PROVENANCE_SOURCE}}

# %% [markdown]
## 3. Original confirmed-dataset selection
This cell applies the recorded review and writes the 221 confirmed TPA measurements. Ambiguous records are exported separately with their recorded reasons.

# %%
{{CONFIRMED_DATA_SOURCE}}

# %%
regenerated=pd.read_csv(PREP_ROOT/'data/processed/tpa_pet_ml_confirmed.csv',dtype=str,keep_default_na=False)
packaged=pd.read_csv(verify_file('pet/data/processed/tpa_pet_ml_confirmed.csv'),dtype=str,keep_default_na=False)
pd.testing.assert_frame_equal(regenerated,packaged)
display(regenerated[['protein_id','protein_name','temperature_c','ph','duration_hours','target_value_standardized','standardized_unit','literature_doi']].head())
print('Reproduced all',len(regenerated),'confirmed rows and all',len(regenerated.columns),'columns exactly as parsed.')
fig,ax=plt.subplots()
ax.bar(['Candidate records','Confirmed analyte','Eligible modeling rows'],[227,221,208],color=['#b2c9bc','#5d957d','#286b58'])
ax.set(ylabel='Rows',title='Recorded dataset preparation stages')
save_figure('01_dataset_preparation_counts')

# %% [markdown]
## 4. Complete offline sequence-feature function
The PET checkpoint expects the frozen **ESM-2 35M** encoder and 480 channels. The function below explicitly excludes the special start/end tokens, verifies standard amino acids and records the sequence hash. It regenerates one catalog protein for an on-the-spot check. Set `REBUILD_ALL_PROTEINS=True` before running to encode all 25 proteins. New files are saved in the output folder and do not replace the original cache.

Images in the old PET checkpoint are disabled (`image_trained=false`). We therefore use a zero image vector and mark it absent, matching the original sequence-only feature preparation. Structural visualizations are shown separately in notebook 01; they do not manufacture measured activity labels.

# %%
import torch
from transformers import AutoModel,AutoTokenizer
torch.set_num_threads(2)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esm_path=BUNDLE/'models/esm2_35m'
verify_file('models/esm2_35m/model.safetensors')
feature_tokenizer=AutoTokenizer.from_pretrained(str(esm_path),local_files_only=True)
feature_encoder=AutoModel.from_pretrained(str(esm_path),local_files_only=True).to(DEVICE).eval()
AA_STANDARD=set('ACDEFGHIKLMNPQRSTVWY')

@torch.inference_mode()
def encode_pet_sequence(sequence):
    sequence=str(sequence).strip().upper()
    if not sequence or len(sequence)>1022 or set(sequence)-AA_STANDARD:
        raise ValueError('Use 1–1022 standard amino acids, without a FASTA header.')
    tokens=feature_tokenizer(sequence,return_tensors='pt').to(DEVICE)
    hidden=feature_encoder(**tokens).last_hidden_state[0,1:len(sequence)+1]
    features=hidden.cpu().numpy().astype(np.float32)
    assert features.shape==(len(sequence),480)
    return features,hashlib.sha256(sequence.encode()).hexdigest()

REBUILD_ALL_PROTEINS=False
catalog_rows=packaged.drop_duplicates('protein_id').set_index('protein_id')
selected=list(catalog_rows.index) if REBUILD_ALL_PROTEINS else ['1']
feature_output=RUN_DIR/'neural_features';feature_output.mkdir()
feature_index={}
old_index=json.loads(verify_file('pet/data/neural_features/index.json').read_text())
for pid in selected:
    sequence=catalog_rows.loc[pid,'sequence']
    representation,digest=encode_pet_sequence(sequence)
    old=old_index['proteins'][pid]
    assert digest==old['sequence_sha256']
    with np.load(verify_file('pet/data/neural_features/'+old['file']),allow_pickle=False) as cached:
        difference=float(np.max(np.abs(representation-cached['sequence'])))
        np.testing.assert_allclose(representation,cached['sequence'],rtol=2e-3,atol=2e-3)
    name=pid+'.npz'
    np.savez_compressed(feature_output/name,sequence=representation,image=np.zeros(512,dtype=np.float32))
    feature_index[pid]={'file':name,'sequence_sha256':digest,'sequence_length':len(sequence),'image_present':False}
    print(f'Protein {pid}: {representation.shape}; maximum difference from original cache {difference:.6g}')
(feature_output/'index.json').write_text(json.dumps({'encoder':old_index['encoder'],'image_encoder':old_index['image_encoder'],
    'proteins':feature_index,'scope':'all catalog proteins' if REBUILD_ALL_PROTEINS else 'single-protein demonstration'},indent=2))
print('Feature files:',feature_output)

# %% [markdown]
## 5. Outputs and what happens next
Use notebook 01 to load the verified complete feature cache, clean the 221-row CSV, fit training-only transforms, train models and evaluate saved checkpoints. Use `encode_pet_sequence` to prepare new sequence features; an unseen sequence has no measured label merely because features can be computed. New mutant activity and a new 3D fold remain separate prediction/validation problems.

# %%
preparation_summary={'confirmed_rows':len(regenerated),'all_columns_match':True,'regenerated_proteins':selected,
    'encoder_source':json.loads((esm_path/'SOURCE.json').read_text()),'new_output_directory':str(RUN_DIR)}
(RUN_DIR/'summary.json').write_text(json.dumps(preparation_summary,indent=2))
display(pd.DataFrame({'Output':[str(p.relative_to(RUN_DIR)) for p in sorted(RUN_DIR.rglob('*')) if p.is_file()]}))
