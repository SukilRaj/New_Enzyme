"""Batched condition queries using the unchanged original PET checkpoint."""
import hashlib
import numpy as np
import torch
from fastapi import HTTPException
from pydantic import BaseModel, Field, ConfigDict
from backend.app.api.neural import service
from backend.app.services.neural_oracle import batch_features


class EnvironmentBatch(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    protein_ids: list[str] = Field(min_length=1, max_length=25)
    conditions: list[list[float]] = Field(min_length=1, max_length=125)


def attach(app):
    @app.post('/api/environment/predict')
    def predict(request: EnvironmentBatch):
        try:
            env=np.array(request.conditions,dtype=np.float32)
            if env.ndim!=2 or env.shape[1]!=3 or not np.isfinite(env).all():
                raise ValueError('Conditions must be temperature, pH, duration triples.')
            if np.any((env[:,0]<0)|(env[:,0]>100)|(env[:,1]<0)|(env[:,1]>14)|(env[:,2]<=0)|(env[:,2]>720)):
                raise ValueError('Conditions are outside the analysis bounds.')
            model=service();predictions={}
            for pid in dict.fromkeys(request.protein_ids):
                if pid not in model.catalog:raise ValueError('Unknown protein.')
                meta=model.index['proteins'][pid]
                sequence=model.catalog[pid]['sequence'].strip().upper()
                if hashlib.sha256(sequence.encode()).hexdigest()!=meta['sequence_sha256']:
                    raise ValueError('Protein sequence and features disagree.')
                path=(model.features_dir/meta['file']).resolve()
                if not path.is_relative_to(model.features_dir.resolve()):raise ValueError('Invalid feature path.')
                with np.load(path,allow_pickle=False) as f:
                    item={'sequence':f['sequence'].astype(np.float32),'image':f['image'].astype(np.float32)}
                if not all(np.isfinite(value).all() for value in item.values()):
                    raise ValueError('Nonfinite feature values.')
                values=[]
                for start in range(0,len(env),32):
                    batch=env[start:start+32]
                    seq,mask,image=batch_features([item]*len(batch))
                    present=torch.full((len(batch),),float(meta['image_present'] and model.meta['image_trained']))
                    with torch.inference_mode():
                        scores,_=model.model(seq,mask,image,present,torch.tensor((batch-model.mean)/model.std))
                    raw=np.maximum(0,np.expm1(scores.numpy().astype(np.float64)))
                    if not np.isfinite(raw).all():raise ValueError('Nonfinite model response.')
                    values.extend(raw.tolist())
                predictions[pid]=values
            return {'predictions':predictions,'model':'Original EnzAIme ESM-2 + Attention1D checkpoint',
                    'image_trained':model.meta['image_trained']}
        except (ValueError,KeyError) as error:
            raise HTTPException(422,str(error)) from error
