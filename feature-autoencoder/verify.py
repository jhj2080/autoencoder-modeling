"""Verify prepared artifacts and saved-model inference consistency."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from preprocess import ROOT, features, read_raw
from predict import run as predict


def verify():
    data=ROOT/'data/processed'
    meta=pd.read_csv(data/'metadata.csv')
    table=pd.read_csv(data/'windows_features.csv').set_index('window_id')
    assert len(meta)==1071 and meta.window_id.is_unique
    assert meta.groupby('group_id').split.nunique().max()==1
    assert meta.loc[meta.split.ne('test'),'label'].eq(0).all()
    raw=np.load(data/'raw_windows.npz',allow_pickle=False)
    assert np.array_equal(raw['window_id'],meta.window_id.to_numpy())
    seen=set()
    for i,row in meta.iterrows():
        keys={(row.source,int(k)) for k in raw['raw_rows'][i]}
        assert len(keys)==16 and not keys.intersection(seen)
        seen.update(keys)
    assert all(np.isfinite(v) for v in features(np.zeros((16,3))).values())
    for name in ['F1','F2']:
        z=np.load(data/f'{name}.npz',allow_pickle=False)
        s=json.loads((data/f'{name}_scaler.json').read_text())
        cols=z['feature_names'].tolist()
        train=table.loc[z['id_train'],cols].to_numpy()
        np.testing.assert_allclose(train.mean(axis=0),s['mean'],atol=1e-12)
        np.testing.assert_allclose(train.std(axis=0),s['scale'],atol=1e-12)
        for split in ['train','val','cal','test']:
            ids=z[f'id_{split}']; expected=table.loc[ids]
            assert expected.split.eq(split).all()
            assert np.array_equal(expected.label,z[f'y_{split}'])
            np.testing.assert_allclose((expected[cols].to_numpy()-s['mean'])/s['scale'],z[f'X_{split}'],atol=1e-8)
        # Inference uses CSV -> feature extraction -> stored scaler -> restored model.
        # It must reproduce scores already saved by the training pipeline.
        with tempfile.TemporaryDirectory() as tmp:
            inferred=predict(ROOT/'data/raw/outlier_data.csv',ROOT/f'results/{name}/model.joblib',Path(tmp)/'pred.csv')
        saved=pd.read_csv(ROOT/f'results/{name}/test_predictions.csv').query('label == 1')
        np.testing.assert_allclose(inferred.score,saved.score,rtol=1e-9,atol=1e-9)
        assert np.array_equal(inferred.prediction,saved.prediction)
    # Verify malformed chronology fails rather than silently joining time-reversed rows.
    with tempfile.TemporaryDirectory() as tmp:
        d=pd.read_csv(ROOT/'data/raw/outlier_data.csv').iloc[::-1]
        path=Path(tmp)/'bad.csv';d.to_csv(path,index=False)
        try: read_raw(path,'bad')
        except ValueError: pass
        else: raise AssertionError('Reverse time was accepted')
    print('PASS: group/sample separation, finite features, train-only scaler, NPZ alignment, restored inference, invalid chronology.')


if __name__=='__main__': verify()
