"""CPU feature autoencoder: multi-output MLP with reconstruction target X.
No anomaly labels are used for fitting, early stopping, or threshold selection.
"""
from pathlib import Path
import argparse
import copy
import json
import math
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocess import ROOT, save_json


def reconstruction_error(model, x):
    per_feature = (x-model.predict(x))**2
    return per_feature.mean(axis=1), per_feature


def run(data, out, feature_set='F1', seed=42, epochs=1000, patience=80, alpha=.05):
    if not 0 < alpha < 1 or epochs < 1 or patience < 1:
        raise ValueError('Require 0<alpha<1, epochs>=1, patience>=1')
    data, out = Path(data), Path(out); out.mkdir(parents=True, exist_ok=True)
    with np.load(data/f'{feature_set}.npz', allow_pickle=False) as f:
        z = {k: f[k] for k in f.files}
    train, val = z['X_train'], z['X_val']
    for split in ['train','val','cal']:
        if not np.all(z[f'y_{split}'] == 0): raise ValueError('Normal-only constraint violated')
    for split in ['train','val','cal','test']:
        if not np.isfinite(z[f'X_{split}']).all(): raise ValueError('Nonfinite inputs')
    d = train.shape[1]
    hidden = (6,3,6) if d == 8 else (4,2,4)
    model = MLPRegressor(hidden_layer_sizes=hidden, activation='tanh', solver='adam',
                         batch_size=min(32,len(train)), learning_rate_init=.001, alpha=1e-4,
                         shuffle=True, random_state=seed, early_stopping=False, max_iter=1)
    best_loss = float('inf'); best_model = None; best_epoch = 0; stale = 0; history = []
    # partial_fit is one epoch, preserving Adam optimizer state between calls.
    for epoch in range(1,epochs+1):
        model.partial_fit(train, train)
        tr = float(reconstruction_error(model,train)[0].mean())
        vl = float(reconstruction_error(model,val)[0].mean())
        if not np.isfinite(tr+vl): raise ValueError('Training diverged')
        history.append({'epoch':epoch,'train_mse':tr,'val_mse':vl})
        if vl < best_loss-1e-6:
            best_model = copy.deepcopy(model); best_loss=vl; best_epoch=epoch; stale=0
        else:
            stale += 1
        if stale >= patience: break
    model = best_model
    cal_scores,_ = reconstruction_error(model,z['X_cal'])
    # Finite-sample corrected upper quantile; this is empirical, not a field FPR guarantee.
    n = len(cal_scores); rank = math.ceil((n+1)*(1-alpha))
    if rank > n:
        raise ValueError(f'Too little calibration data for alpha={alpha}; need alpha >= {1/(n+1):.6f}')
    threshold = float(np.sort(cal_scores)[rank-1])
    scaler = json.loads((data/f'{feature_set}_scaler.json').read_text(encoding='utf-8'))
    audit = json.loads((data/'audit.json').read_text(encoding='utf-8'))
    bundle = {'model':model,'scaler':scaler,'threshold':threshold,'features':z['feature_names'].tolist(),
              'window':16,'stride':16,'current_quantum':audit['current_quantum'],
              'feature_set':feature_set,'sklearn_version':sklearn.__version__}
    joblib.dump(bundle,out/'model.joblib')
    metadata = pd.read_csv(data/'metadata.csv').set_index('window_id')
    outputs=[]
    for split in ['train','val','cal','test']:
        scores, contributions = reconstruction_error(model,z[f'X_{split}'])
        rows = metadata.loc[z[f'id_{split}']].reset_index().copy()
        if not np.array_equal(rows.label.to_numpy(),z[f'y_{split}']): raise AssertionError('Metadata mismatch')
        rows['score']=scores; rows['threshold']=threshold; rows['prediction']=(scores>threshold).astype(int)
        rows['top_error_feature']=np.array(bundle['features'])[contributions.argmax(axis=1)]
        for i, name in enumerate(bundle['features']): rows[f'error_{name}']=contributions[:,i]
        outputs.append(rows)
    predictions=pd.concat(outputs,ignore_index=True)
    predictions.to_csv(out/'all_scores.csv',index=False,encoding='utf-8-sig')
    test=predictions[predictions.split.eq('test')].copy()
    test.to_csv(out/'test_predictions.csv',index=False,encoding='utf-8-sig')
    wrong=test[test.label.ne(test.prediction)].copy()
    wrong['error_type']=np.where(wrong.label.eq(0),'false_alarm','miss')
    wrong.to_csv(out/'error_cases.csv',index=False,encoding='utf-8-sig')
    y=test.label.to_numpy(); pred=test.prediction.to_numpy()
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    metrics={'feature_set':feature_set,'evaluation_status':audit['evaluation_status'],
             'architecture':[d,*hidden,d],'seed':seed,'best_epoch':best_epoch,'epochs_run':len(history),
             'best_val_mse':best_loss,'threshold':threshold,'target_calibration_tail':alpha,
             'calibration_count':n,'calibration_rank':rank,
             'calibration_observed_fpr':float(np.mean(cal_scores>threshold)),
             'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
             'precision':float(precision_score(y,pred,zero_division=0)),
             'recall':float(recall_score(y,pred,zero_division=0)),
             'f1':float(f1_score(y,pred,zero_division=0)),
             'roc_auc':float(roc_auc_score(y,test.score)),
             'average_precision':float(average_precision_score(y,test.score)),
             'test_normal_fpr':float(fp/(tn+fp)),
             'normal_fpr_by_state':{state: {'n':len(g),'false_alarms':int(g.prediction.sum()),'fpr':float(g.prediction.mean())}
                for state,g in test[test.label.eq(0)].groupby('state')},
             'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}}
    save_json(out/'metrics.json',metrics)
    history=pd.DataFrame(history); history.to_csv(out/'history.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(history.epoch,history.train_mse,label='Train');ax[0].plot(history.epoch,history.val_mse,label='Validation')
    ax[0].axvline(best_epoch,ls='--',color='gray');ax[0].set(xlabel='Epoch',ylabel='Reconstruction MSE',yscale='log');ax[0].legend()
    for state,g in test.groupby('state'):
        ax[1].scatter(g.index,g.score,label=state,s=18,alpha=.7)
    ax[1].axhline(threshold,color='red',ls='--',label='Threshold')
    ax[1].set(xlabel='Test row (not continuous time)',ylabel='Score',yscale='log');ax[1].legend()
    fig.tight_layout();fig.savefig(out/'training_and_scores.png',dpi=160);plt.close(fig)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    return metrics


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/processed')
    p.add_argument('--out',type=Path)
    p.add_argument('--feature-set',choices=['F1','F2'],default='F1')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--patience',type=int,default=80)
    p.add_argument('--alpha',type=float,default=.05)
    a=p.parse_args(); run(a.data,a.out or ROOT/'results'/a.feature_set,a.feature_set,a.seed,a.epochs,a.patience,a.alpha)
