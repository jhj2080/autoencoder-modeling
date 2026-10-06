"""정상 학습 데이터만으로 표준화하는 1D-CNN AE 전처리. Python 3.10+."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

CHANNELS = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
SPLITS = ['train', 'val', 'cal', 'test_normal']

def load_segments(path, expected_label):
    d = pd.read_csv(path)
    required = ['TimeStamp', *CHANNELS, 'Equipment_state']
    if not set(required).issubset(d.columns):
        raise ValueError(f'{path}: 필수 열 누락')
    d['source_row'] = np.arange(len(d))
    d['TimeStamp'] = pd.to_datetime(d['TimeStamp'], errors='raise')
    for c in CHANNELS:
        d[c] = pd.to_numeric(d[c], errors='raise')
    if not np.isfinite(d[CHANNELS].to_numpy()).all():
        raise ValueError('결측/무한값 존재: 임의 보간하지 말고 확인하세요.')
    if not d.Equipment_state.eq(expected_label).all():
        raise ValueError('파일의 정상/이상 라벨이 예상과 다릅니다.')
    n_before = len(d)
    # 인덱스 열은 서로 달라도 같은 시각/센서/라벨이면 중복 측정입니다.
    d = d.drop_duplicates(required).sort_values('TimeStamp', kind='stable').reset_index(drop=True)
    if d.TimeStamp.duplicated().any():
        raise ValueError('동일 시각에 서로 다른 측정값이 있습니다: 수동 확인 필요')
    dt = d.TimeStamp.diff().dt.total_seconds().to_numpy()
    # 0.1초 간격이 아닌 곳에서 끊습니다. 긴 공백을 보간/연결하지 않습니다.
    boundary = ~np.isclose(dt, 0.1, rtol=0, atol=0.001)
    d['segment'] = np.cumsum(boundary) - 1
    return d, {'original_rows': n_before, 'deduplicated_rows': len(d),
               'removed_duplicates': n_before-len(d), 'segments': int(d.segment.nunique())}

def assign_splits(d, length, seed):
    """시간대별 10개 층에서 구간 단위 분할: 서로 다른 정상 운전 조건의 편중 완화.
    미래 예측 검증은 아닙니다. IF에도 동일 구간 할당을 사용하세요.
    """
    sizes = d.groupby('segment', sort=True).size()
    usable = sizes.index[sizes >= length].to_numpy()
    if len(usable) < 40:
        raise ValueError('4분할에 충분한 연속 구간이 없습니다.')
    rng = np.random.default_rng(seed)
    allocation = {}
    for block in np.array_split(usable, 10):
        ids = rng.permutation(block)
        n = len(ids)
        nv, nc, nt = max(1, round(n*.15)), max(1, round(n*.10)), max(1, round(n*.15))
        counts = [n-nv-nc-nt, nv, nc, nt]
        offset = 0
        for split, count in zip(SPLITS, counts):
            for segment in ids[offset:offset+count]:
                allocation[int(segment)] = split
            offset += count
    return allocation

def windows(d, ids, length, stride, source, split):
    arrays, meta, used = [], [], set()
    for sid, group in d[d.segment.isin(ids)].groupby('segment', sort=True):
        for start in range(0, len(group)-length+1, stride):
            part = group.iloc[start:start+length]
            arrays.append(part[CHANNELS].to_numpy(dtype=np.float64))
            used.update(part.index.tolist())
            meta.append({'source': source, 'split': split, 'segment': int(sid),
                         'source_row_start': int(part.source_row.iloc[0]),
                         'source_row_end': int(part.source_row.iloc[-1]),
                         'start_time': str(part.TimeStamp.iloc[0]),
                         'end_time': str(part.TimeStamp.iloc[-1]),
                         'label': int(part.Equipment_state.iloc[0])})
    x = np.stack(arrays) if arrays else np.empty((0, length, len(CHANNELS)))
    return x, meta, used

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--normal', required=True)
    p.add_argument('--outlier', required=True)
    p.add_argument('--out', default='prepared')
    p.add_argument('--length', type=int, default=16)
    p.add_argument('--stride', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    # 문서의 측정 해상도 가설은 기본 설정에 적용하지 않습니다.
    p.add_argument('--current-quantum', type=float, default=None,
                   help='선택 비교 실험: 양쪽 파일의 전류에 같은 양자화 적용')
    args = p.parse_args()
    stride = args.stride if args.stride is not None else args.length
    if args.length < 2 or stride < 1 or stride > args.length:
        p.error('length >= 2, 1 <= stride <= length 조건 필요')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    normal, ns = load_segments(args.normal, 0)
    anomaly, ads = load_segments(args.outlier, 1)
    if args.current_quantum is not None:
        q = args.current_quantum
        if not np.isfinite(q) or q <= 0:
            p.error('current-quantum must be positive and finite')
        for d in [normal, anomaly]:
            d['AI2_Current'] = np.round(d.AI2_Current / q) * q
    allocation = assign_splits(normal, args.length, args.seed)
    normal['split'] = normal.segment.map(allocation).fillna('excluded_short')
    raw, metadata, coverage = {}, [], {}
    train_used = None
    for split in SPLITS:
        ids = [sid for sid, target in allocation.items() if target == split]
        x, m, used = windows(normal, ids, args.length, stride, 'normal', split)
        raw[split] = x; metadata.extend(m)
        coverage[split] = {'windows': len(x), 'unique_used_rows': len(used),
                           'assigned_rows': int(normal.segment.isin(ids).sum())}
        if split == 'train': train_used = sorted(used)
    x, m, used = windows(anomaly, anomaly.segment.unique(), args.length, stride,
                         'outlier', 'test_anomaly')
    raw['test_anomaly'] = x; metadata.extend(m)
    coverage['test_anomaly'] = {'windows': len(x), 'unique_used_rows': len(used),
                                 'assigned_rows': len(anomaly)}
    if any(len(v) == 0 for v in raw.values()):
        raise ValueError('빈 분할이 생겼습니다. length와 분할 방법을 확인하세요.')
    # 重複する窓でも各測定行を一度だけ数えてfit。train以外を使いません。
    fit_values = normal.loc[train_used, CHANNELS].to_numpy(dtype=np.float64)
    mean, scale = fit_values.mean(axis=0), fit_values.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    scaled = {f'X_{k}': ((v-mean)/scale).astype(np.float32) for k,v in raw.items()}
    assert all(np.isfinite(v).all() for v in scaled.values())
    np.savez_compressed(out/'cnn_ae_data.npz', **scaled)
    pd.DataFrame(metadata).to_csv(out/'window_metadata.csv', index=False)
    normal.groupby('segment', sort=True).agg(start_time=('TimeStamp','min'),
        end_time=('TimeStamp','max'), rows=('TimeStamp','size'), split=('split','first'))\
        .to_csv(out/'normal_segment_split.csv')
    config = {'channels': CHANNELS, 'length': args.length, 'stride': stride,
              'sample_interval_seconds': .1, 'first_to_last_seconds': (args.length-1)*.1,
              'seed': args.seed, 'scaler_mean': mean.tolist(), 'scaler_scale': scale.tolist(),
              'scaler_fit_unique_train_rows': len(train_used),
              'current_quantum': args.current_quantum,
              'split_strategy': 'segment groups randomized within 10 chronological strata',
              'normal_audit': ns, 'anomaly_audit': ads, 'coverage': coverage,
              'excluded_short_normal_rows': int(normal['split'].eq('excluded_short').sum())}
    (out/'preprocessing_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: list(v.shape) for k,v in scaled.items()}, indent=2))

if __name__ == '__main__':
    main()
