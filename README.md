# Learning to Rank(LTR) Pipeline for Search

검색 랭킹 시스템을 위한 Pairwise 및 Listwise Learning to Rank 모델의 구현입니다. 본 프로젝트는 산업계에서 널리 사용되는 최신 LTR 기법들의 실무적 구현을 시연합니다.

## 프로젝트 개요

본 프로젝트는 프로덕션 수준의 코드, 포괄적인 평가 메트릭, 확장 가능한 아키텍처에 중점을 둔 최신 Learning to Rank 알고리즘 구현체입니다.

### 구현된 모델

**Pairwise 방식:**
- **RankNet**: 경사하강법을 사용한 신경망 기반 pairwise 랭킹
- **LambdaRank**: NDCG를 고려한 gradient weighting을 적용한 RankNet

**Listwise 방식:**
- **ListNet**: 확률 분포와 cross-entropy loss를 사용한 listwise 랭킹
- **ListMLE**: 직접적인 listwise 최적화를 위한 Maximum Likelihood Estimation
- **ApproxNDCG**: NDCG의 미분 가능한 근사를 통한 직접 최적화

## 프로젝트 구조

```
learning_to_rank/
├── config.yaml                 # 설정 파일
├── requirements.txt            # Python 의존성
│
├── data/                       # 데이터 처리 모듈
│   ├── dataset.py             # 데이터셋 클래스 (Pairwise, Listwise)
│   ├── preprocessing.py       # 특징 전처리 유틸리티
│   └── __init__.py
│
├── models/                     # 모델 구현
│   ├── base.py                # 베이스 랭킹 모델
│   ├── pairwise.py            # RankNet, LambdaRank
│   ├── listwise.py            # ListNet, ListMLE, ApproxNDCG
│   └── __init__.py
│
├── training/                   # 학습 파이프라인
│   ├── trainer.py             # Pairwise 모델 트레이너
│   ├── listwise_trainer.py    # Listwise 모델 트레이너
│   └── __init__.py
│
├── evaluation/                 # 평가 메트릭
│   ├── metrics.py             # NDCG, MAP, MRR, Precision, Recall
│   └── __init__.py
│
├── inference/                  # 추론 파이프라인
│   ├── predictor.py           # 모델 예측기
│   └── __init__.py
│
├── utils/                      # 유틸리티
│   ├── config.py              # 설정 관리
│   ├── logger.py              # 로깅 유틸리티
│   └── __init__.py
│
├── scripts/                    # 실행 스크립트
│   ├── train_pairwise.py      # Pairwise 모델 학습
│   ├── train_listwise.py      # Listwise 모델 학습
│   ├── inference.py           # 추론 실행
│   ├── generate_sample_data.py # 샘플 데이터 생성
│   └── quickstart.sh          # 빠른 시작 스크립트
│
├── checkpoints/                # 저장된 모델 체크포인트
├── logs/                       # 학습 로그
├── results/                    # 예측 결과
└── data/                       # 데이터 디렉토리
    ├── raw/                   # 원본 데이터 파일
    └── processed/             # 전처리된 데이터 파일
```

## 주요 기능

### 데이터 처리
- LETOR 포맷 데이터 로딩
- 특징 정규화 (StandardScaler, MinMaxScaler)
- Pairwise 및 Listwise 데이터 생성
- 쿼리 단위 Train/Validation 분할

### 모델 아키텍처
- 설정 가능한 hidden layer를 가진 유연한 신경망 구조
- 정규화를 위한 Batch normalization과 Dropout
- 다양한 활성화 함수 지원 (ReLU, Tanh, ELU)
- 효율적인 배치 처리

### 학습 파이프라인
- Patience를 적용한 Early stopping
- Learning rate scheduling (Cosine, StepLR)
- Gradient clipping
- TensorBoard 통합
- 자동 체크포인팅
- 포괄적인 로깅

### 평가 메트릭
- NDCG@k (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- Precision@k 및 Recall@k

### 추론
- 배치 예측
- 랭킹 생성
- JSON 출력 포맷
- Preprocessor 통합

## 설치

```bash
# 리포지토리 클론
cd learning_to_rank

# 의존성 설치
pip install -r requirements.txt
```

## 데이터 포맷

본 프로젝트는 LETOR (Learning to Rank) 데이터 포맷을 사용합니다:

```
<label> qid:<qid> <feature_id>:<value> ... <feature_id>:<value>
```

예시:
```
2 qid:1 1:0.5 2:0.8 3:0.3 ...
0 qid:1 1:0.2 2:0.5 3:0.1 ...
1 qid:2 1:0.7 2:0.9 3:0.6 ...
```

호환 가능한 공개 데이터셋:
- MSLR-WEB10K
- MSLR-WEB30K
- Yahoo! Learning to Rank Challenge
- Microsoft Learning to Rank Datasets

## 사용 방법

### 1. 빠른 시작

```bash
# 설정 및 샘플 데이터 자동 생성
./scripts/quickstart.sh
```

### 2. 설정 조정

`config.yaml`을 수정하여 하이퍼파라미터 조정:

```yaml
data:
  raw_data_path: "data/raw"
  train_file: "train.txt"
  val_file: "val.txt"
  num_features: 136

model:
  pairwise:
    name: "ranknet"
    hidden_dims: [256, 128, 64]
    dropout: 0.2

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10
```

### 3. Pairwise 모델 학습

```bash
# RankNet 학습
python scripts/train_pairwise.py --model ranknet --device cuda

# LambdaRank 학습
python scripts/train_pairwise.py --model lambdarank --device cuda

# 커스텀 데이터 경로 사용
python scripts/train_pairwise.py \
    --model ranknet \
    --train_file data/raw/train.txt \
    --val_file data/raw/val.txt \
    --device cuda
```

### 4. Listwise 모델 학습

```bash
# ListNet 학습
python scripts/train_listwise.py --model listnet --device cuda

# ListMLE 학습
python scripts/train_listwise.py --model listmle --device cuda

# ApproxNDCG 학습
python scripts/train_listwise.py --model approxndcg --device cuda
```

### 5. 추론 실행

```bash
# 테스트 데이터에 대한 추론
python scripts/inference.py \
    --model_path checkpoints/best_model.pt \
    --data_path data/raw/test.txt \
    --output_dir results/predictions \
    --device cuda \
    --top_k 10
```

### 6. Python API 사용

```python
from data.dataset import LTRDataset, PairwiseDataset
from models.pairwise import RankNet
from training.trainer import PairwiseTrainer
from inference.predictor import LTRPredictor
from utils.config import Config

# 설정 로드
config = Config('config.yaml')

# 데이터 로드
train_dataset = LTRDataset('data/raw/train.txt')
pairwise_dataset = PairwiseDataset(train_dataset, num_pairs_per_query=10)

# 모델 생성
model = RankNet(num_features=136, hidden_dims=[256, 128, 64])

# 모델 학습
trainer = PairwiseTrainer(model, train_loader, val_dataset, config)
history = trainer.train()

# 추론
predictor = LTRPredictor('checkpoints/best_model.pt', device='cuda')
ranking = predictor.rank_documents(features, top_k=10)
```

## 모델 설명

### RankNet
- Pairwise 랭킹 방식
- 예측된 문서 쌍과 이상적인 문서 쌍 간의 cross-entropy loss 최소화
- 효율적인 gradient 계산
- **최적 사용처**: 명확한 pairwise 선호도가 있는 일반적인 랭킹 작업

### LambdaRank
- NDCG를 고려한 gradient를 적용한 RankNet의 확장
- NDCG 개선 가능성에 따라 gradient에 가중치 부여
- 상위 랭크 결과에 집중
- **최적 사용처**: 상위 k개 정확도가 중요한 작업 (예: 검색 엔진)

### ListNet
- 확률 분포를 사용하는 listwise 방식
- 예측 분포와 ground truth 분포 간의 KL divergence 최소화
- 전체 쿼리 컨텍스트 고려
- **최적 사용처**: 전역적 리스트 최적화가 필요한 작업

### ListMLE
- 순열에 대한 Maximum likelihood estimation
- 랭킹 우도의 직접 최적화
- 계산적으로 효율적
- **최적 사용처**: 안정적인 학습이 필요한 대규모 랭킹

### ApproxNDCG
- NDCG 메트릭의 직접 최적화
- 랭킹의 미분 가능한 근사 사용
- End-to-end 메트릭 최적화
- **최적 사용처**: NDCG가 주요 평가 메트릭인 경우

## 성능 최적화

### 학습 팁
1. **Learning Rate**: 0.001에서 시작하여 cosine annealing 사용으로 더 나은 수렴
2. **Batch Size**: GPU의 경우 32-64, CPU의 경우 8-16 사용
3. **Early Stopping**: 10-15 epoch의 patience가 적절
4. **Gradient Clipping**: 5.0으로 설정하여 exploding gradient 방지
5. **Data Augmentation**: Pairwise 모델의 경우 `num_pairs_per_query` 증가

### 하드웨어 권장사항
- **GPU**: 효율적인 학습을 위해 8GB+ VRAM을 가진 NVIDIA GPU
- **CPU**: 데이터 로딩을 위한 4개 이상의 코어 (num_workers=4)
- **RAM**: 대규모 데이터셋(MSLR-WEB30K)을 위한 16GB 이상

## 평가 결과

모델은 산업 표준 메트릭을 사용하여 평가됩니다:

| 메트릭 | 설명 | 중요도 |
|--------|------|--------|
| NDCG@k | 위치를 고려한 랭킹 품질 | 검색의 주요 메트릭 |
| MAP | 쿼리 간 평균 정밀도 | 전체 랭킹 품질 |
| MRR | 첫 번째 관련 결과의 위치 | 사용자 만족도 |
| Precision@k | 상위 k개 결과의 관련성 | 상위 결과 품질 |

## 프로덕션 고려사항

본 구현은 프로덕션 베스트 프랙티스를 따릅니다:

1. **모듈성**: 데이터, 모델, 학습, 추론의 명확한 분리
2. **설정 관리**: 쉬운 실험을 위한 중앙화된 설정 관리
3. **로깅**: TensorBoard 지원을 포함한 포괄적인 로깅
4. **체크포인팅**: 최고 성능 모델의 자동 저장
5. **오류 처리**: 견고한 데이터 검증 및 오류 메시지
6. **확장성**: 배치 처리 및 효율적인 메모리 사용
7. **재현성**: 랜덤 시드 제어 및 결정론적 연산


## 향후 개선 사항

- [ ] 분산 학습 지원 (DDP)
- [ ] 프로덕션 배포를 위한 ONNX 내보내기
- [ ] A/B 테스팅 프레임워크
- [ ] 실시간 추론 최적화
- [ ] Feature importance 분석
- [ ] 하이퍼파라미터 튜닝을 위한 AutoML
- [ ] 추가 데이터 포맷 지원 (CSV, Parquet)
- [ ] 모델 앙상블 기법

## 참고문헌

1. Burges, C., et al. (2005). "Learning to Rank using Gradient Descent" (RankNet)
2. Burges, C., et al. (2007). "Learning to Rank with Nonsmooth Cost Functions" (LambdaRank)
3. Cao, Z., et al. (2007). "Learning to Rank: From Pairwise Approach to Listwise Approach" (ListNet)
4. Xia, F., et al. (2008). "Listwise Approach to Learning to Rank" (ListMLE)
5. Microsoft Learning to Rank Datasets (MSLR-WEB10K, MSLR-WEB30K)
