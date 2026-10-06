"""16-sample windows -> fixed engineered features -> normal-only scaling.
Run from any directory: python preprocess.py
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SENSORS = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
F1 = ['AI2_period8', 'AI2_kurt', 'ratio_AI0_AI2', 'AI0_rms',
      'ratio_AI1_AI2', 'AI1_rms', 'AI1_skew', 'AI2_rms']
FEATURE_SETS = {'F1': F1, 'F2': [c for c in F1 if c not in ['AI2_period8', 'AI2_kurt']]}
EPS = 1e-12


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_raw(path, source, expected_label=None):
    d = pd.read_csv(path)
    required = ['TimeStamp'] + SENSORS
    if not set(required).issubset(d.columns):
        raise ValueError(f'Missing columns: {set(required) - set(d.columns)}')
    if expected_label is not None:
        if 'Equipment_state' not in d or not d.Equipment_state.eq(expected_label).all():
            raise ValueError(f'{source}: Equipment_state must be {expected_label}')
    d['raw_row'] = np.arange(len(d))  # zero-based CSV data row, before dedup
    n_raw = len(d)
    subset = required + (['Equipment_state'] if 'Equipment_state' in d else [])
    d = d.drop_duplicates(subset=subset).copy()  # ignore exported Unnamed index
    d['TimeStamp'] = pd.to_datetime(d.TimeStamp, errors='raise')
    d[SENSORS] = d[SENSORS].apply(pd.to_numeric, errors='raise')
    if d.TimeStamp.isna().any() or not np.isfinite(d[SENSORS].to_numpy()).all():
        raise ValueError('Missing/nonfinite input: fix upstream; no silent imputation.')
    # Keep chronological input order. Conflicting timestamps or reversal must be reviewed.
    dt = d.TimeStamp.diff().dt.total_seconds()
    if (dt.dropna() <= 0).any():
        raise ValueError('Conflicting timestamp or reversed order after exact deduplication.')
    boundary = ~np.isclose(dt.to_numpy(), 0.1, atol=1e-6, rtol=0)
    d['segment'] = np.cumsum(boundary) - 1
    d['source'] = source
    d = d.reset_index(drop=True)
    audit = {'raw_rows': n_raw, 'deduplicated_rows': len(d),
             'duplicates_removed': n_raw-len(d), 'segments': int(d.segment.nunique())}
    return d, audit


def correlation(x, y):
    a, b = x-x.mean(), y-y.mean()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.dot(a,b)/norm, -1, 1)) if norm > EPS else 0.0


def features(w):
    """std: ddof=0; skew/kurt: unbiased scipy estimators; kurt: Fisher excess.
    Constant signals have skew/kurt/correlation=0; no NaN reaches the model.
    """
    f = {}
    for i in range(3):
        x = w[:, i]; name = f'AI{i}'
        rms = float(np.sqrt(np.mean(x*x)))
        constant = np.ptp(x) <= EPS
        f.update({f'{name}_mean': float(x.mean()), f'{name}_std': float(x.std(ddof=0)),
                  f'{name}_rms': rms, f'{name}_peak': float(np.abs(x).max()),
                  f'{name}_p2p': float(np.ptp(x)),
                  f'{name}_skew': 0.0 if constant else float(skew(x, bias=False)),
                  f'{name}_kurt': 0.0 if constant else float(kurtosis(x, fisher=True, bias=False)),
                  f'{name}_crest': float(np.abs(x).max()/max(rms, EPS))})
    for i,j in [(0,1),(0,2),(1,2)]:
        f[f'corr_AI{i}_AI{j}'] = correlation(w[:,i], w[:,j])
    f['AI2_period8'] = correlation(w[:-8,2], w[8:,2])
    for i,j in [(0,2),(1,2),(0,1)]:
        f[f'ratio_AI{i}_AI{j}'] = f[f'AI{i}_rms']/max(f[f'AI{j}_rms'], EPS)
    return f


def make_windows(d, source, label=-1, quantum=1.192093, annotate_state=False):
    d = d.copy()
    if quantum > 0:
        # Same deterministic measurement-resolution correction for both sources.
        d['AI2_Current'] = np.rint(d.AI2_Current/quantum)*quantum
    rows, raw_windows, row_members = [], [], []
    for seg, g in d.groupby('segment', sort=True):
        for offset in range(0, len(g)-15, 16):
            part = g.iloc[offset:offset+16]
            if not np.allclose(part.TimeStamp.diff().dt.total_seconds().iloc[1:], .1, atol=1e-6, rtol=0):
                raise AssertionError('Window crossed a discontinuity')
            w = part[SENSORS].to_numpy(dtype=np.float64)
            state = ('B' if 472 <= seg <= 559 else 'A') if annotate_state else ('anom' if label == 1 else 'unknown')
            meta = {'window_id': f'{source}_{int(seg):04d}_{offset:02d}', 'source': source,
                    'label': label, 'state': state, 'segment': int(seg),
                    'group_id': f'{source}_{int(seg):04d}', 'offset': offset,
                    'start_raw_row': int(part.raw_row.iloc[0]), 'end_raw_row': int(part.raw_row.iloc[-1]),
                    'start_time': str(part.TimeStamp.iloc[0]), 'end_time': str(part.TimeStamp.iloc[-1])}
            rows.append({**meta, **features(w)})
            raw_windows.append(w); row_members.append(part.raw_row.to_numpy())
    if not rows:
        raise ValueError('No continuous 16-sample windows in input.')
    return pd.DataFrame(rows), np.stack(raw_windows), np.stack(row_members)


def split_groups(windows, seed):
    """Known A/B intervals are for stratification/audit only; never model inputs."""
    rng = np.random.default_rng(seed)
    assignment = {}
    normal = windows[windows.label.eq(0)]
    for state in ['A', 'B']:
        groups = normal.loc[normal.state.eq(state), 'group_id'].unique()
        if len(groups) < 10:
            raise ValueError(f'Too few {state} groups for 4-way split.')
        groups = rng.permutation(groups)
        n = len(groups); sizes = [int(n*.60), int(n*.15), int(n*.10)]
        sizes += [n-sum(sizes)]
        start = 0
        for name, size in zip(['train','val','cal','test'], sizes):
            assignment.update({g: name for g in groups[start:start+size]}); start += size
    result = windows.copy()
    result['split'] = result.group_id.map(assignment).fillna('test')
    if not result.groupby('group_id').split.nunique().eq(1).all():
        raise AssertionError('Segment leakage')
    if not result.loc[result.split.ne('test'), 'label'].eq(0).all():
        raise AssertionError('Anomaly entered model development')
    return result


def run(normal_path, anomaly_path, out, seed=42, quantum=1.192093):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    normal, an = read_raw(normal_path, 'normal', 0)
    anomaly, aa = read_raw(anomaly_path, 'anomaly', 1)
    # A/B annotations from the supplied project document are dataset-specific.
    if len(normal) != 19999 or normal.segment.nunique() != 599:
        raise ValueError('A/B annotation requires the supplied 19999-row, 599-segment normal dataset.')
    wn, xn, rn = make_windows(normal, 'normal', 0, quantum, True)
    wa, xa, ra = make_windows(anomaly, 'anomaly', 1, quantum)
    windows = split_groups(pd.concat([wn, wa], ignore_index=True), seed)
    raw = np.concatenate([xn, xa]); members = np.concatenate([rn, ra])
    # Verify exact raw-row memberships rather than just endpoint intervals.
    used = {}; overlap = 0
    for i, row in windows.iterrows():
        for raw_row in members[i]:
            key = (row.source, int(raw_row))
            if key in used:
                overlap += 1
            used[key] = row.split
    if overlap:
        raise AssertionError('Raw samples shared between windows')
    windows.to_csv(out/'windows_features.csv', index=False, encoding='utf-8-sig')
    meta_cols = ['window_id','source','label','state','segment','group_id','offset',
                 'start_raw_row','end_raw_row','start_time','end_time','split']
    windows[meta_cols].to_csv(out/'metadata.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(out/'raw_windows.npz', X=raw, raw_rows=members,
                        window_id=windows.window_id.to_numpy(dtype=str))
    for name, cols in FEATURE_SETS.items():
        train_mask = windows.split.eq('train')
        scaler = StandardScaler().fit(windows.loc[train_mask, cols].to_numpy())
        if (scaler.var_ < 1e-20).any():
            raise ValueError('Constant training feature; review feature specification.')
        arrays = {'feature_names': np.array(cols, dtype=str)}
        for split in ['train','val','cal','test']:
            subset = windows.loc[windows.split.eq(split)]
            x = scaler.transform(subset[cols].to_numpy()).astype(np.float64)
            if not np.isfinite(x).all(): raise AssertionError('Nonfinite features')
            arrays[f'X_{split}'] = x
            arrays[f'y_{split}'] = subset.label.to_numpy(dtype=np.int64)
            arrays[f'id_{split}'] = subset.window_id.to_numpy(dtype=str)
        np.savez_compressed(out/f'{name}.npz', **arrays)
        save_json(out/f'{name}_scaler.json', {'features': cols, 'mean': scaler.mean_.tolist(),
                  'scale': scaler.scale_.tolist(), 'variance': scaler.var_.tolist(),
                  'fit_split': 'train', 'train_count': int(train_mask.sum())})
    for a, d, w in [(an,normal,wn),(aa,anomaly,wa)]:
        a['windows'] = len(w); a['used_rows'] = len(w)*16
        a['unused_rows'] = len(d)-len(w)*16
        a['unused_fraction'] = 1-len(w)*16/len(d)
    summary = windows.groupby(['split','state','label']).size().rename('windows').reset_index()
    summary.to_csv(out/'split_summary.csv', index=False, encoding='utf-8-sig')
    audit = {'normal': an, 'anomaly': aa, 'raw_sample_overlap': overlap,
             'segment_split_violation': 0, 'boundary_violation': 0,
             'finite_arrays': True, 'seed': seed, 'window': 16, 'stride': 16,
             'sample_interval_seconds': .1, 'current_quantum': quantum,
             'group_split_target': {'train': .60,'val': .15,'cal': .10,'test': .15},
             'state_B_segment_range_zero_based_inclusive': [472,559],
             'sha256': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in [normal_path, anomaly_path]},
             'evaluation_status': 'exploratory: supplied feature selection used all anomaly labels',
             'caveats': ['Existing CNN split manifest was not supplied; this is a new group split.',
                         'Known A/B interval is retrospective and dataset-specific.',
                         'No independent day/equipment holdout; groups can remain correlated.',
                         'Fixed quantum, window and feature sets inherit full-data exploration.']}
    save_json(out/'audit.json', audit)
    print(summary.to_string(index=False))
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--normal', type=Path, default=ROOT/'data/raw/press_data_normal.csv')
    p.add_argument('--anomaly', type=Path, default=ROOT/'data/raw/outlier_data.csv')
    p.add_argument('--out', type=Path, default=ROOT/'data/processed')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--current-quantum', type=float, default=1.192093, help='0 disables rounding')
    args = p.parse_args()
    if args.current_quantum < 0: p.error('current-quantum must be nonnegative')
    run(args.normal, args.anomaly, args.out, args.seed, args.current_quantum)
