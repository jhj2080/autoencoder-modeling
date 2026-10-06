"""One command: preprocess, train F1/F2, write comparison."""
import pandas as pd
from preprocess import ROOT, run as preprocess
from train import run as train

if __name__=='__main__':
    preprocess(ROOT/'data/raw/press_data_normal.csv',ROOT/'data/raw/outlier_data.csv',ROOT/'data/processed')
    metrics=[train(ROOT/'data/processed',ROOT/'results'/name,name) for name in ['F1','F2']]
    cols=['feature_set','best_epoch','tn','fp','fn','tp','precision','recall','f1','roc_auc','average_precision','test_normal_fpr']
    pd.DataFrame(metrics)[cols].to_csv(ROOT/'results/comparison.csv',index=False,encoding='utf-8-sig')
    print('Complete. See data/processed and results. Evaluation is exploratory.')
