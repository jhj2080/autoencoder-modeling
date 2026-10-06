"""Infer from a new raw CSV, without labels. Saved train scaler is reused."""
from pathlib import Path
import argparse
import joblib
import numpy as np
from preprocess import read_raw, make_windows


def run(csv, model_path, out):
    bundle=joblib.load(model_path)  # Load only your own/trusted joblib model.
    raw,_=read_raw(csv,'new')
    windows,_,_=make_windows(raw,'new',quantum=bundle['current_quantum'])
    cols=bundle['features']; scaler=bundle['scaler']
    x=(windows[cols].to_numpy()-np.array(scaler['mean']))/np.array(scaler['scale'])
    if not np.isfinite(x).all(): raise ValueError('Nonfinite features')
    error=(x-bundle['model'].predict(x))**2
    windows['score']=error.mean(axis=1)
    windows['threshold']=bundle['threshold']
    windows['prediction']=(windows.score>bundle['threshold']).astype(int)
    windows['top_error_feature']=np.array(cols)[error.argmax(axis=1)]
    for i,col in enumerate(cols): windows[f'error_{col}']=error[:,i]
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    windows.to_csv(out,index=False,encoding='utf-8-sig')
    print(f'{len(windows)} windows saved: {out}')
    return windows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--out',type=Path,default=Path('predictions.csv'))
    a=p.parse_args();run(a.csv,a.model,a.out)
