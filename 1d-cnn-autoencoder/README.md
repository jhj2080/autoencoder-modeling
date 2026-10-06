# 소성가공 데이터: 1D-CNN Autoencoder 시작 코드

## 파일
- preprocess.py: 원본 CSV → 중복 제거 → 연속 구간 → 구간별 분할 → 16×3 윈도 → 정상 train 기준 표준화.
- train_example.py: 작은 CNN AE 학습 및 정상 calibration 기준 임계값 설정 예제. TensorFlow 미설치 환경이므로 모델 학습은 검증하지 않았습니다.
- prepared/: 제공된 원본으로 실제 실행한 전처리 결과.

## 내 PC에서 실행
Python 3.10 이상에서 압축을 풀고 이 폴더를 터미널의 현재 폴더로 설정하세요. 원본 CSV 두 개를 이 폴더에 복사한 뒤 실행합니다.

```bash
python -m pip install numpy pandas scikit-learn
python preprocess.py --normal press_data_normal.csv --outlier outlier_data.csv --out prepared
```

TensorFlow 설치가 가능한 별도 환경에서 학습 예제를 실행합니다.

```bash
python -m pip install tensorflow
python train_example.py --data prepared/cnn_ae_data.npz --out model_output
```

전처리 데이터 불러오기:

```python
import numpy as np
with np.load('prepared/cnn_ae_data.npz') as d:
    X_train = d['X_train']
    X_val = d['X_val']
    X_cal = d['X_cal']
    X_test_normal = d['X_test_normal']
    X_test_anomaly = d['X_test_anomaly']
print(X_train.shape)  # (610, 16, 3)
# Keras: model.fit(X_train, X_train, validation_data=(X_val, X_val))
# PyTorch Conv1d: torch.from_numpy(X_train).permute(0, 2, 1)
```

## 입력과 목표
3개 채널 순서: AI0_Vibration(상부), AI1_Vibration(하부), AI2_Current(전류).
16은 시간축, 3은 센서축입니다. RMS·첨도 등 8개 특징을 시간축으로 나열하지 않습니다.
AE는 정상 입력 자체를 복원하는 모델이며, Equipment_state는 학습 입력/목표가 아닙니다.
평가는 표준화 공간에서 원본과 복원의 시간×센서 평균제곱오차(MSE)로 합니다.
센서별 MSE도 저장하지만 실제 고장 원인을 확정하는 근거로 볼 수는 없습니다.
이상이 항상 복원 오차가 크다는 보장은 없으므로 실제 평가가 필요합니다.

## 분할과 누수 방지
동일 연속 구간의 윈도는 한 분할에만 속합니다. 정상 구간을 시간순 10개 층으로 나누고 각 층 안에서 구간 단위로 약 60/15/10/15% 분할합니다. 윈도 수 비율은 구간 길이에 따라 달라집니다.
train: 학습/표준화; val: 조기 종료; cal: 정상 오차 분위수로 임계값; test_normal/test_anomaly: 최종 평가.
정상 A/B 상태를 자동 확정하지 않습니다. 시간대별 분할은 특정 시간대 편중을 완화할 뿐 A/B 균형을 보장하지 않습니다. 조원의 상태 정의를 사용하여 분할별 상태 비율과 상태별 오탐률을 확인하세요.
이 분할은 동일 수집 데이터 내 비교 실험이며 미래 운전 일반화 검증이 아닙니다. 미래 검증은 시간순 분할과 추가 수집 데이터가 필요합니다.
IF와 공정 비교를 하려면 normal_segment_split.csv와 window_metadata.csv로 동일 분할·윈도를 공유하고 IF 표준화/변수 선택도 train 데이터에서만 시행하세요.
전체 이상 데이터로 모델/변수/윈도/임계값을 선택하면 최종 평가가 낙관적이 됩니다. 이미 전체 이상을 확인한 상황이므로 최종 성능은 탐색적 결과로 표시하고 새 이상 데이터로 재검증하세요.
기본 임계값은 정상 cal의 95% 분위수입니다. 5%는 설정 목표이며 실제 정상 오탐률을 보장하지 않습니다. cal 116개로 99% 분위수를 사용하면 꼬리 추정이 특히 불안정합니다.

## 전처리 판단
- 전체 행 인덱스가 다르더라도 시각/센서/라벨이 같은 중복 측정만 제거합니다. 값만 같은 서로 다른 시점은 제거하지 않습니다.
- 0.1초±0.001초 밖 간격은 구간 경계로 취급합니다. 긴 공백 보간 없이 구간 내부에서만 윈도를 만듭니다.
- 표준화 평균과 표준편차는 train 윈도에 실제 사용된 정상 고유 행만으로 계산합니다. 다른 모든 분할에는 같은 값을 적용합니다.
- 윈도별 표준화, 정상/이상 별도 표준화, 이상 극단값 삭제, 기본 클리핑은 하지 않습니다. 이상 진폭 정보를 보존합니다.
- 기본 길이 16, stride 16은 조원 IF와 비교하는 시작점입니다. 최적값으로 확정하지 않습니다. 첫~마지막 샘플 간격은 1.5초, 명목상 16샘플 길이는 1.6초입니다.
- 16미만 짧은 구간/구간 끝 잔여 데이터는 제외하고 coverage에 기록합니다. 짧은 구간만으로 탐지되는 이상을 놓칠 수 있습니다.
- 겹치지 않는 윈도도 같은 구간에서는 통계적으로 독립적이지 않습니다. 윈도 수와 함께 이상 구간 수도 보고해야 합니다.
- 10Hz 데이터로 5Hz를 넘는 원래 진동 성분을 복원할 수 없습니다. FFT 주파수 특징은 주 분석에 넣지 않았습니다.
- 진폭/시간 순서/저주파 파형은 원시 채널에 남습니다. 변동 크기, 첨도, lag8 상관, RMS 비율은 초기 입력에 추가하지 않고 이상 사례 설명용으로 사용합니다.

## 측정 해상도 비교 실험
조원 문서는 이상 전류의 약 1.192093 단위 양자화 차이를 보고했습니다. 측정 장비 차이라는 원인은 확정되지 않았으므로 기본 전처리는 원본을 유지합니다. 선택 실험은 두 파일 모두에 같은 연산을 적용합니다.

```bash
python preprocess.py --normal press_data_normal.csv --outlier outlier_data.csv --out prepared_quantized --current-quantum 1.192093
```

진동 2채널만 사용하는 비교와 원본 3채널 비교도 고려하세요. 전체 이상 test 성능을 보고 반복 선택하면 독립 평가가 사라집니다. 개발용 이상 구간과 최종 평가용 이상 구간을 나누거나 새 데이터를 확보하세요.

## Codex에 이어서 요청할 문장
이 폴더의 README.md와 preprocess.py를 읽고 train_example.py를 점검해줘. prepared/cnn_ae_data.npz의 (N,16,3) 입력으로 작은 1D-CNN Autoencoder를 학습해줘. 정상 train만으로 모델을 학습하고 val로 조기 종료, cal의 정상 복원 MSE로 임계값을 설정해줘. test 데이터는 선택에 사용하지 말고 최종 평가에만 사용해줘. AUROC, Average Precision, 정상 오탐률, 이상 재현율을 계산하고, window_metadata.csv에 점수를 붙여 구간 단위 결과와 센서별 복원 오차도 확인해줘. 모델 저장 시 preprocessing_config.json도 함께 보존해줘.

## 확인된 실행 결과
정상 20000→19999행, 599구간, 1043윈도. 이상 600행, 21구간, 28윈도(14구간).
train 610, val 160, cal 116, test_normal 157, test_anomaly 28.
전처리 실데이터 실행, 표준화 기준/시간 간격/구간 분할 격리/누락 없는 윈도 수 검증 완료.
모델 학습 및 탐지 성능은 아직 확인하지 않았습니다.
