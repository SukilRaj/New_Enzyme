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
    def __init__(self,mode='sequence_context',kind='extra',strength=4,seed=42,target_transform='log1p'):
        if target_transform not in ('log1p','raw'):
            raise ValueError('target_transform must be log1p or raw.')
        self.mode,self.kind,self.strength,self.seed=mode,kind,strength,seed
        self.target_transform=target_transform

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
        x=self.scaler.transform(self.imputer.transform(x))
        target=np.log1p(y) if self.target_transform=='log1p' else y
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
        if getattr(self,'target_transform','log1p')=='log1p':
            return np.expm1(np.clip(predictions,0,20))
        return np.maximum(predictions,0)


def specs():
    # A frozen experiment, not an adaptive search against held-out scores.
    choices=[('ridge',100),('ridge',1000),('extra',2),('extra',6),('boost',10),('kernel',1),('kernel',10),('neural',1)]
    candidates=[dict(mode=mode,kind=kind,strength=strength,seed=42)
                for mode in ('sequence_environment','context','sequence_context') for kind,strength in choices]
    candidates.extend([
        dict(mode='sequence_environment',kind='extra',strength=2,seed=42,target_transform='raw'),
        dict(mode='sequence_environment',kind='boost',strength=10,seed=42,target_transform='raw'),
    ])
    return candidates


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
