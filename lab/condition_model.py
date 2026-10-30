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

ROOT = Path(__file__).resolve().parents[1]
PET_ROOT = Path(os.environ.get('ENZAIME_ROOT', r'D:\Final_Year_Project\EnzAIme_ML'))
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
    mse = float(mean_squared_error(y, pred))
    return {'mae_um': float(mean_absolute_error(y, pred)), 'mse_um2': mse,
            'rmse_um': float(np.sqrt(mse)),
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
