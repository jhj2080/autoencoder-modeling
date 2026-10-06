"""선택 사항: TensorFlow 설치 후 실행. 이 환경에서 모델 학습은 실행하지 않았습니다."""
import argparse
from pathlib import Path
import json
import numpy as np
from tensorflow import keras
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument('--data', default='prepared/cnn_ae_data.npz')
p.add_argument('--out', default='model_output')
p.add_argument('--quantile', type=float, default=.95)
a = p.parse_args()
if not 0 < a.quantile < 1: p.error('quantile must be between 0 and 1')
keras.utils.set_random_seed(42)
d = np.load(a.data)
x = d['X_train']
if x.shape[1] % 4: raise ValueError('이 예제 모델은 length가 4의 배수여야 합니다.')
model = keras.Sequential([
    keras.layers.Input(shape=x.shape[1:]),
    keras.layers.Conv1D(8, 3, padding='same', activation='relu'),
    keras.layers.AveragePooling1D(2),
    keras.layers.Conv1D(4, 3, padding='same', activation='relu'),
    keras.layers.AveragePooling1D(2),
    keras.layers.Conv1D(2, 3, padding='same', activation='relu'),
    keras.layers.UpSampling1D(2),
    keras.layers.Conv1D(4, 3, padding='same', activation='relu'),
    keras.layers.UpSampling1D(2),
    keras.layers.Conv1D(8, 3, padding='same', activation='relu'),
    keras.layers.Conv1D(x.shape[2], 3, padding='same', activation='linear'),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
model.fit(x, x, validation_data=(d['X_val'], d['X_val']), epochs=100,
          batch_size=32, shuffle=True, callbacks=[keras.callbacks.EarlyStopping(
              monitor='val_loss', patience=10, restore_best_weights=True)])

def score(v):
    error = (v-model.predict(v, verbose=0))**2
    return error.mean(axis=(1,2)), error.mean(axis=1)

cal, _ = score(d['X_cal'])
threshold = float(np.quantile(cal, a.quantile))
normal, nc = score(d['X_test_normal'])
anomaly, ac = score(d['X_test_anomaly'])
y = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)]
s = np.r_[normal, anomaly]
result = {'threshold': threshold, 'normal_calibration_quantile': a.quantile,
          'auroc': float(roc_auc_score(y,s)), 'average_precision': float(average_precision_score(y,s)),
          'confusion_matrix_rows_true_cols_pred': confusion_matrix(y,s>threshold).tolist(),
          'normal_test_false_positive_rate': float((normal>threshold).mean()),
          'anomaly_window_recall': float((anomaly>threshold).mean())}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model.save(out/'cnn_autoencoder.keras')
(out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
np.savez_compressed(out/'test_scores.npz', labels=y, scores=s, channel_mse=np.concatenate([nc,ac]))
print(json.dumps(result, indent=2))
