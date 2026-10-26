"""Local avGFP experiments. Measured labels and model predictions stay separate."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from importlib.metadata import version, PackageNotFoundError

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'Directed_Evolution-main'
sys.path.insert(0, str(UPSTREAM))
os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'huggingface'))
AA = 'ACDEFGHIKLMNPQRSTVWY'


def dataset():
    path = UPSTREAM / 'preprocessed_data/avGFP/avGFP.csv'
    with path.open() as f:
        rows = list(csv.DictReader(f))
    # Deduplicate before splitting, averaging repeated measurements.
    grouped = {}
    for row in rows:
        seq, score = row['sequence'], float(row['fitness'])
        if set(seq) <= set(AA):
            grouped.setdefault(seq, []).append(score)
    rows = [{'sequence': s, 'fitness': sum(y) / len(y)} for s, y in grouped.items()]
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    import importlib.util
    rows, sha = dataset()
    ys = [r['fitness'] for r in rows]
    return dict(dataset='avGFP', description='Green fluorescent protein', rows=len(rows),
                length=len(rows[0]['sequence']), minimum=min(ys), maximum=max(ys),
                sha256=sha, esm_available=all(importlib.util.find_spec(p) for p in ('torch','transformers','polyleven')),
                oracle_available=False, histogram=histogram(ys))


def histogram(values):
    lo, hi = min(values), max(values)
    bins = [0] * 24
    for value in values:
        bins[min(23, int((value-lo) / (hi-lo or 1) * 24))] += 1
    return bins


def validate(config):
    if config.get('engine') not in ('baseline', 'esm'):
        raise ValueError('Choose baseline or esm.')
    ranges = {'samples': (100, 50000), 'epochs': (1, 100), 'steps': (1, 60),
              'population': (2, 64), 'seed': (0, 2147483647)}
    out = {'engine': config['engine']}
    for name, (lo, hi) in ranges.items():
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer between {lo} and {hi}.')
        out[name] = value
    out['model'] = config.get('model', 'facebook/esm2_t12_35M_UR50D')
    if out['model'] not in ('facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D'):
        raise ValueError('Unsupported ESM model.')
    out['selection'] = config.get('selection', 'greedy')
    if out['selection'] not in ('greedy', 'uncertainty_diversity'):
        raise ValueError('Choose greedy or uncertainty_diversity selection.')
    if out['engine'] == 'baseline' and out['selection'] != 'greedy':
        raise ValueError('Uncertainty-and-diversity selection requires the ESM engine.')
    return out


def run(config, folder, update, cancelled):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    cfg = validate(config)
    rng = random.Random(cfg['seed'])
    np.random.seed(cfg['seed'])
    rows, sha = dataset()
    rng.shuffle(rows)
    rows = rows[:cfg['samples']]
    train, test = train_test_split(rows, test_size=.2, random_state=cfg['seed'])
    folder = Path(folder)
    manifest = dict(config=cfg, dataset_sha256=sha, train_count=len(train), test_count=len(test),
                    split='deduplicated sequences; seeded random 80/20; not mutation-distance split',
                    independent_oracle=False, paper_reproduction=False)
    manifest['selection_method'] = ('greedy predicted-fitness selection' if cfg['selection'] == 'greedy'
                                    else '3-head bootstrap ensemble; mean + 0.5 SD, then 0.5-scaled normalized minimum Hamming-distance bonus')
    manifest['python_version'] = sys.version
    manifest['engine_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['packages'] = {}
    for package in ('numpy', 'scipy', 'scikit-learn', 'torch', 'transformers'):
        try:
            manifest['packages'][package] = version(package)
        except PackageNotFoundError:
            pass
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (folder / 'split.json').write_text(json.dumps({'train': [r['sequence'] for r in train],
                                                'test': [r['sequence'] for r in test]}))
    def check():
        if cancelled():
            raise InterruptedError('Cancelled by user')
    update('Preparing data', 5, f'{len(train):,} training / {len(test):,} held-out sequences')
    check()
    if cfg['engine'] == 'baseline':
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import Ridge
        import pickle
        def features(seqs):
            return [{f'{i}:{a}': 1 for i, a in enumerate(s)} for s in seqs]
        vectorizer = DictVectorizer()
        x = vectorizer.fit_transform(features([r['sequence'] for r in train]))
        model = Ridge(alpha=10, solver='lsqr')
        model.fit(x, [r['fitness'] for r in train])
        def predict(seqs):
            return model.predict(vectorizer.transform(features(seqs)))
        def predict_distribution(seqs):
            values = predict(seqs)
            return values, np.zeros_like(values)
        with (folder / 'baseline.pkl').open('wb') as f:
            pickle.dump((vectorizer, model), f)
        update('Evaluating model', 40, 'Position-specific one-hot ridge trained; this is not an LLM')
        propose = None
    else:
        import torch
        from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
        from de.predictors.attention.decoder import Decoder
        torch.manual_seed(cfg['seed'])
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        update('Loading ESM', 8, f"Loading {cfg['model']} on {device}; first use downloads weights")
        hf_home = Path(os.environ.get('HF_HOME', ROOT / 'models' / 'huggingface'))
        repo_cache = hf_home / 'hub' / ('models--' + cfg['model'].replace('/', '--'))
        try:
            revision = (repo_cache / 'refs' / 'main').read_text().strip()
        except OSError:
            revision = ''
        snapshot = repo_cache / 'snapshots' / revision
        cached_assets = ('config.json', 'model.safetensors', 'tokenizer_config.json',
                         'special_tokens_map.json', 'vocab.txt')
        local_files_only = bool(revision) and all((snapshot / name).is_file() for name in cached_assets)
        tokenizer = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=local_files_only)
        encoder = EsmModel.from_pretrained(cfg['model'], local_files_only=local_files_only).to(device).eval()
        manifest['model_revision'] = getattr(encoder.config, '_commit_hash', None)
        manifest['device'] = str(device)
        manifest['model_loaded_from'] = 'local cache' if local_files_only else 'Hugging Face Hub/cache'
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        # All avGFP sequences must share a length: no padding contributes to pooling.
        if len({len(r['sequence']) for r in rows}) != 1:
            raise ValueError('This local avGFP adapter requires fixed-length sequences.')
        def encode(seqs):
            inputs = tokenizer(seqs, return_tensors='pt', padding=True).to(device)
            with torch.no_grad():
                return encoder(**inputs).last_hidden_state
        ensemble_size = 3 if cfg['selection'] == 'uncertainty_diversity' else 1
        decoders = []
        for member in range(ensemble_size):
            torch.manual_seed(cfg['seed'] + member)
            decoder = Decoder(encoder.config.hidden_size, 128).to(device)
            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            # Bootstrap each head to estimate epistemic uncertainty across predictors.
            member_train = ([rng.choice(train) for _ in train]
                            if ensemble_size > 1 else list(train))
            for epoch in range(cfg['epochs']):
                decoder.train()
                order = list(member_train)
                rng.shuffle(order)
                losses = []
                for start in range(0, len(order), 4):
                    check()
                    batch = order[start:start+4]
                    features = encode([r['sequence'] for r in batch])
                    target = torch.tensor([r['fitness'] for r in batch], device=device)
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(decoder(features).flatten(), target)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    if start % 64 == 0:
                        update('Training Attention1D', 10+30*(member+(epoch+start/len(order))/cfg['epochs'])/ensemble_size,
                               f'Ensemble {member+1}/{ensemble_size}; epoch {epoch+1}/{cfg["epochs"]}; {start}/{len(order)} sequences')
                update('Training Attention1D', 10+30*(member+(epoch+1)/cfg['epochs'])/ensemble_size,
                       f'Ensemble {member+1}/{ensemble_size}, epoch {epoch+1}: training MSE {np.mean(losses):.4f}')
            decoder.eval()
            decoders.append(decoder)
        torch.save({'decoders': [d.state_dict() for d in decoders], 'model': cfg['model'],
                    'hidden_dim': 128, 'ensemble_size': ensemble_size}, folder/'decoder.pt')
        def predict_distribution(seqs):
            member_values = []
            for decoder in decoders:
                values = []
                for start in range(0, len(seqs), 4):
                    check()
                    with torch.no_grad():
                        values.extend(decoder(encode(seqs[start:start+4])).flatten().cpu().tolist())
                member_values.append(values)
            values = np.asarray(member_values)
            return values.mean(axis=0), values.std(axis=0)
        def predict(seqs):
            return predict_distribution(seqs)[0]
        mlm = None
        def propose(seq, position):
            nonlocal mlm
            if mlm is None:
                mlm = EsmForMaskedLM.from_pretrained(cfg['model'], local_files_only=local_files_only).to(device).eval()
            tokens = tokenizer(seq, return_tensors='pt').to(device)
            tokens['input_ids'][0, position+1] = tokenizer.mask_token_id
            with torch.no_grad():
                logits = mlm(**tokens).logits[0, position+1]
            allowed = [a for a in AA if a != seq[position]]
            ids = tokenizer.convert_tokens_to_ids(allowed)
            aa = allowed[int(logits[ids].argmax())]
            return seq[:position]+aa+seq[position+1:]
    check()
    predictions = predict([r['sequence'] for r in test])
    targets = np.array([r['fitness'] for r in test])
    correlation = float(spearmanr(targets, predictions).statistic)
    metrics = dict(rmse=float(np.sqrt(mean_squared_error(targets, predictions))),
                   mae=float(mean_absolute_error(targets, predictions)),
                   spearman=correlation if np.isfinite(correlation) else None,
                   train_count=len(train), test_count=len(test))
    with (folder/'holdout.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence','measured_fitness','predicted_fitness'])
        writer.writerows((r['sequence'], r['fitness'], float(y)) for r,y in zip(test,predictions))
    update('Searching variants', 50, 'Held-out evaluation saved. Search uses predicted fitness only.', metrics=metrics)
    # Start from a training sequence only; held-out labels never guide selection.
    parent = min(train, key=lambda r: abs(r['fitness']-np.median([r['fitness'] for r in train])))['sequence']
    pool = [parent]
    scores = predict(pool)
    uncertainties = predict_distribution(pool)[1]
    history = []
    if cfg['engine'] == 'esm':
        from de.samplers.maskers import ImportanceMasker2
        importance = ImportanceMasker2(k=1, max_subs=1)
    for step in range(cfg['steps']):
        check()
        candidates = list(pool)
        for i in range(cfg['population']*2):
            check()
            seq = rng.choice(pool)
            position = rng.randrange(len(seq))
            if propose and i % 2:
                _, poses = importance.run([seq])
                position = poses[0][0]
            if propose:
                candidate = propose(seq, position)
            else:
                aa = rng.choice([a for a in AA if a != seq[position]])
                candidate = seq[:position]+aa+seq[position+1:]
            candidates.append(candidate)
        candidates = list(dict.fromkeys(candidates))
        values, candidate_uncertainties = predict_distribution(candidates)
        if cfg['selection'] == 'uncertainty_diversity':
            acquisition = values + 0.5 * candidate_uncertainties
            distances = [[0] * len(candidates) for _ in candidates]
            for i in range(len(candidates)):
                for j in range(i):
                    distance = sum(a != b for a,b in zip(candidates[i], candidates[j]))
                    distances[i][j] = distances[j][i] = distance
            chosen = []
            available = set(range(len(candidates)))
            while available and len(chosen) < cfg['population']:
                def selection_score(index):
                    if not chosen:
                        diversity = 0.0
                    else:
                        min_distance = min(distances[index][j] for j in chosen)
                        diversity = 0.5 * min_distance / len(candidates[index])
                    return acquisition[index] + diversity
                index = max(available, key=lambda i: (selection_score(i), -i))
                chosen.append(index)
                available.remove(index)
            indices = np.asarray(chosen, dtype=int)
            selected_scores = np.asarray([values[i] + 0.5*candidate_uncertainties[i] +
                                          (0.5 * min(distances[i][j] for j in chosen[:k]) /
                                           len(candidates[i]) if k else 0.0)
                                          for k,i in enumerate(indices)])
        else:
            indices = np.argsort(-values, kind='stable')[:cfg['population']]
            selected_scores = values[indices]
        pool = [candidates[int(i)] for i in indices]
        scores = values[indices]
        uncertainties = candidate_uncertainties[indices]
        pairwise = [sum(a != b for a,b in zip(pool[i], pool[j])) for i in range(len(pool)) for j in range(i+1,len(pool))]
        history.append(dict(step=step+1, best=float(max(scores)), mean=float(np.mean(scores)), unique=len(pool),
                            mean_uncertainty=float(np.mean(uncertainties)),
                            mean_pairwise_distance=float(np.mean(pairwise)) if pairwise else 0.0))
        update('Searching variants', 50+49*(step+1)/cfg['steps'],
               f'Round {step+1}/{cfg["steps"]}: best predicted fitness {max(scores):.4f}', history=history)
    candidates = [dict(sequence=s, predicted_fitness=float(y), uncertainty=float(u), selection_score=float(sel),
                       mutations='; '.join(f'{a}{i+1}{b}' for i,(a,b) in enumerate(zip(parent,s)) if a != b) or 'Parent',
                       distance=sum(a != b for a,b in zip(parent,s)))
                  for s,y,u,sel in zip(pool,scores,uncertainties,selected_scores)]
    with (folder/'candidates.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    result = dict(metrics=metrics, history=history, candidates=candidates, parent=parent, manifest=manifest,
                  scatter=[dict(measured=float(y),predicted=float(p)) for y,p in zip(targets[:300],predictions[:300])])
    (folder/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
