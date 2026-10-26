# %% [markdown]
# Full training studies and original scientific source

Use notebook 01 for the PET live demonstration and notebook 02 for GFP. This notebook exposes the original complete training scripts and the application's full GFP engine. Long studies are opt-in because they perform many nested fits. Every output is directed to a new writable work folder; the bundled saved results are preserved.

The original base-paper source is in `Directed_Evolution-main/`, the PET checkpoint architecture is in `pet/backend/app/services/neural_oracle.py`, and all condition/assay implementations are in `lab/`. No frontend is needed. The full condition study evaluates 25 configurations; the assay study evaluates 24 configurations under sequence and sequence/study grouping. Completion is a computational result, not proof of high accuracy.

# %%
NOTEBOOK_NAME='full_studies'
{{COMMON}}

# %%
{{DEPS}}

# %% [markdown]
## 1. Select the work to run
Leave both flags False for a quick code walkthrough. To reproduce the complete studies, set the desired flag True and run from this cell downward. Runtime depends on CPU/GPU and may be many minutes. The assay study can use the bundled original condition split without rerunning the earlier condition study.

# %%
RUN_FULL_CONDITION_STUDY=False
RUN_FULL_ASSAY_STUDY=False
import shutil
WORK=RUN_DIR/'reproduction'
WORK.mkdir()
for folder in ('lab','scripts','models/condition_analysis'):
    shutil.copytree(BUNDLE/folder,WORK/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
print('Complete-study work folder:',WORK)
print('Condition study requested:',RUN_FULL_CONDITION_STUDY)
print('Assay study requested:',RUN_FULL_ASSAY_STUDY)

# %% [markdown]
## 2. Full condition/image training script
All imports, candidate selection, nested folds, paired bootstrap, artifact persistence and metric computation are visible in the next cell. Only the working-directory assignment and execution guard differ from the supplied script. The implementations imported from `lab/` are shown in notebook 01 and shipped as source files.

# %%
{{FULL_CONDITION_STUDY}}

# %% [markdown]
## 3. Full assay-context training script
This reproduces pre-assay context extraction, 24 prespecified configurations, training-only preprocessing, nested selection, strict connected sequence/study partitions, and the final all-data refit. The refit's predictions on its own training rows are not an independent evaluation.

# %%
{{FULL_ASSAY_STUDY}}

# %% [markdown]
## 4. Full GFP application engine
The next cell contains the original application engine as executable definitions. It includes the complete data loader, configuration validation, frozen ESM/Attention1D training, greedy selection and three-head uncertainty/diversity option. It does not run a training job until `run(...)` is called. The 8M/35M local weights are used via a local Hugging Face cache assembled below, so the original source can run without a download.

# %%
{{FULL_GFP_ENGINE}}

# %% [markdown]
## 5. Optional exact application-engine run
This call uses the same source as the web application but supplies a console progress callback. Set `RUN_APPLICATION_GFP=True` to execute it manually. Its 100-sequence/one-epoch default is only an execution check; increase the configuration before a planned study. Notebook 02 supplies the clearer matched predictive comparison.

# %%
RUN_APPLICATION_GFP=False
APPLICATION_ENGINE='baseline'  # 'esm' loads the packaged frozen ESM model.
if RUN_APPLICATION_GFP:
    import os
    # Copy only the chosen model into a writable cache with its recorded revision.
    local_model=BUNDLE/'models/esm2_8m';origin=json.loads((local_model/'SOURCE.json').read_text())
    hf_root=RUN_DIR/'huggingface';repo=hf_root/'hub'/('models--'+origin['model'].replace('/','--'))
    shutil.copytree(local_model,repo/'snapshots'/origin['revision'],dirs_exist_ok=True)
    (repo/'refs').mkdir(exist_ok=True);(repo/'refs/main').write_text(origin['revision'])
    os.environ['HF_HOME']=str(hf_root)
    application_output=RUN_DIR/'application_gfp';application_output.mkdir()
    config={'engine':APPLICATION_ENGINE,'samples':100,'epochs':1,'steps':1,'population':2,
            'seed':42,'model':'facebook/esm2_t6_8M_UR50D','selection':'greedy'}
    def progress(stage,percentage,detail,**kwargs): print(f'{percentage:.0f}% {stage}: {detail}')
    application_result=run(config,application_output,progress,lambda:False)
    display(pd.DataFrame([application_result['metrics']]))
else:
    print('Application-engine training skipped by configuration; the complete function definitions above are available.')

# %% [markdown]
## 6. Complete source inventory and saved evidence
Model definitions and training scripts are shipped alongside the notebooks so the panel can inspect and run them directly. The base-paper code includes tasks beyond this demonstration; their external data, independent-oracle checkpoints and full reproduction are not supplied or claimed. The package includes all data/checkpoints required for the default notebook demonstrations.

# %%
inventory=[]
for folder in ('lab','scripts','pet/scripts','pet/backend','Directed_Evolution-main'):
    for path in (BUNDLE/folder).rglob('*.py'):
        inventory.append({'File':str(path.relative_to(BUNDLE)),'Lines':len(path.read_text(encoding='utf-8-sig').splitlines())})
display(pd.DataFrame(inventory))
status={'condition_study_executed':RUN_FULL_CONDITION_STUDY,'assay_study_executed':RUN_FULL_ASSAY_STUDY,
        'application_gfp_executed':RUN_APPLICATION_GFP,'work_folder':str(WORK)}
(RUN_DIR/'execution_status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
