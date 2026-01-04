# Learning to Rank Portfolio - Project Context

## 프로젝트 개요

이 프로젝트는 검색 도메인 4년차 ML 엔지니어의 포트폴리오로, **실무에서 사용되는 Learning to Rank (LTR) 시스템**을 구현한 것입니다.

### 목표
- Pairwise 및 Listwise LTR 모델 구현
- 데이터 전처리 → 모델 학습/검증 → 추론의 전체 파이프라인 구축
- 프로덕션 환경에서 사용 가능한 코드 품질과 아키텍처

### 기술 스택
- **Framework**: PyTorch
- **Data Processing**: NumPy, Pandas, scikit-learn
- **Logging**: Python logging, TensorBoard
- **Configuration**: YAML
- **Infrastructure**: 일반적인 GPU 서버 환경 (AWS/SageMaker 같은 SaaS 최소화)

## 구현된 모델

### Pairwise Models
1. **RankNet**
   - 문서 쌍의 상대적 순위를 학습
   - Cross-entropy loss 사용
   - 구현 위치: `models/pairwise.py`

2. **LambdaRank**
   - RankNet의 확장으로 NDCG를 고려한 gradient weighting
   - 상위 랭킹 결과에 더 집중
   - 구현 위치: `models/pairwise.py`

### Listwise Models
1. **ListNet**
   - 전체 리스트의 확률 분포를 학습
   - KL divergence 최소화
   - 구현 위치: `models/listwise.py`

2. **ListMLE**
   - Maximum Likelihood Estimation
   - 순열의 우도를 직접 최적화
   - 구현 위치: `models/listwise.py`

3. **ApproxNDCG**
   - NDCG를 직접 최적화하는 미분 가능한 근사
   - 구현 위치: `models/listwise.py`

## 프로젝트 구조

```
learning_to_rank/
│
├── config.yaml                 # 중앙화된 설정 관리
├── requirements.txt            # 의존성
│
├── data/                       # 데이터 처리
│   ├── dataset.py             # LETOR 포맷 로딩, Pairwise/Listwise 데이터셋
│   └── preprocessing.py       # Feature scaling, train/val split
│
├── models/                     # 모델 구현
│   ├── base.py                # 공통 base 모델
│   ├── pairwise.py            # RankNet, LambdaRank
│   └── listwise.py            # ListNet, ListMLE, ApproxNDCG
│
├── training/                   # 학습 파이프라인
│   ├── trainer.py             # Pairwise 학습
│   └── listwise_trainer.py    # Listwise 학습
│
├── evaluation/                 # 평가 메트릭
│   └── metrics.py             # NDCG, MAP, MRR, Precision, Recall
│
├── inference/                  # 추론 파이프라인
│   └── predictor.py           # 모델 로딩 및 예측
│
├── utils/                      # 유틸리티
│   ├── config.py              # 설정 관리
│   └── logger.py              # 로깅
│
└── scripts/                    # 실행 스크립트
    ├── train_pairwise.py      # Pairwise 모델 학습
    ├── train_listwise.py      # Listwise 모델 학습
    ├── inference.py           # 추론 실행
    └── generate_sample_data.py # 샘플 데이터 생성
```

## 주요 기능

### 1. 데이터 처리
- **LETOR 포맷 지원**: 표준 Learning to Rank 데이터 포맷 (MSLR-WEB10K, Yahoo! LTR 등)
- **Feature Normalization**: StandardScaler, MinMaxScaler
- **Query-level Split**: 쿼리 단위로 train/val 분할
- **Pairwise Generation**: 쿼리당 N개의 문서 쌍 생성
- **Listwise Batching**: 가변 길이 리스트를 고정 크기로 패딩 및 마스킹

### 2. 모델 아키텍처
- **유연한 신경망**: 설정 가능한 hidden layers
- **정규화**: Batch normalization, Dropout
- **활성화 함수**: ReLU, Tanh, ELU
- **효율적인 배치 처리**: GPU 최적화

### 3. 학습 파이프라인
- **Early Stopping**: Validation NDCG 기반
- **Learning Rate Scheduling**: Cosine annealing, StepLR
- **Gradient Clipping**: 안정적인 학습
- **TensorBoard**: 실시간 메트릭 모니터링
- **Checkpointing**: 최고 성능 모델 자동 저장
- **Comprehensive Logging**: 파일 및 콘솔 로깅

### 4. 평가 메트릭
- **NDCG@k**: 랭킹 품질의 표준 메트릭
- **MAP**: 전체 쿼리에 대한 평균 정확도
- **MRR**: 첫 번째 관련 문서의 위치
- **Precision@k, Recall@k**: 상위 k개 결과의 품질

### 5. 추론
- **배치 예측**: 효율적인 대량 예측
- **Top-k Ranking**: 상위 k개 문서 선택
- **JSON 출력**: 구조화된 결과 저장
- **Preprocessor 통합**: 학습 시 사용한 전처리 재사용

## 사용 방법

### 빠른 시작

1. **샘플 데이터 생성**
```bash
python scripts/generate_sample_data.py \
    --num_queries 1000 \
    --num_docs 50 \
    --output data/raw/train.txt

python scripts/generate_sample_data.py \
    --num_queries 200 \
    --num_docs 50 \
    --output data/raw/val.txt
```

2. **모델 학습**
```bash
# RankNet 학습
python scripts/train_pairwise.py --model ranknet --device cuda

# ListNet 학습
python scripts/train_listwise.py --model listnet --device cuda
```

3. **추론 실행**
```bash
python scripts/inference.py \
    --model_path checkpoints/best_model.pt \
    --data_path data/raw/test.txt \
    --output_dir results/predictions
```

### 설정 커스터마이징

`config.yaml` 파일을 수정하여 하이퍼파라미터 조정:

```yaml
model:
  pairwise:
    name: "ranknet"
    hidden_dims: [256, 128, 64]  # 네트워크 크기
    dropout: 0.2                  # 드롭아웃 비율

training:
  batch_size: 32                  # 배치 크기
  num_epochs: 50                  # 에폭 수
  learning_rate: 0.001            # 학습률
  early_stopping_patience: 10     # Early stopping patience
```

## 실무 적용 고려사항

### 프로덕션 베스트 프랙티스
1. **모듈화**: 데이터, 모델, 학습, 추론의 명확한 분리
2. **설정 관리**: 중앙화된 YAML 설정
3. **로깅**: 디버깅 및 모니터링을 위한 상세 로그
4. **체크포인팅**: 학습 중단 시 재개 가능
5. **에러 핸들링**: 견고한 데이터 검증
6. **확장성**: 대용량 데이터 처리 가능
7. **재현성**: 랜덤 시드 제어

### 성능 최적화
- **GPU 활용**: CUDA 지원으로 학습 가속
- **Data Loading**: Multi-worker를 통한 병렬 로딩
- **Batch Processing**: 효율적인 메모리 사용
- **Mixed Precision**: 선택적으로 FP16 학습 가능 (확장 가능)

### 실제 응용 분야
- E-commerce 상품 검색
- 콘텐츠 플랫폼 추천
- 검색 엔진 웹페이지 랭킹
- 구인구직 플랫폼
- 소셜 미디어 피드 랭킹

## 기술적 하이라이트

### 1. Pairwise vs Listwise
- **Pairwise**: 문서 쌍 비교, 확장성 좋음, 로컬 최적화
- **Listwise**: 전체 리스트 고려, 글로벌 최적화, NDCG 직접 최적화

### 2. Loss Functions
- **RankNet**: Binary cross-entropy on pairs
- **LambdaRank**: Weighted by NDCG change
- **ListNet**: KL divergence on distributions
- **ListMLE**: Negative log-likelihood of permutations
- **ApproxNDCG**: Soft ranking with NDCG approximation

### 3. Evaluation Strategy
- Query-level 평가
- K-fold cross-validation 지원 가능
- 다양한 k 값에 대한 메트릭 계산
- Business metric 연동 가능

## 향후 개선 사항

### 단기
- [ ] 실제 MSLR-WEB10K 데이터셋 다운로드 스크립트
- [ ] 하이퍼파라미터 튜닝 예제
- [ ] 모델 비교 분석 노트북

### 중기
- [ ] ONNX 내보내기 (프로덕션 배포)
- [ ] Serving API (FastAPI/Flask)
- [ ] Docker 컨테이너화
- [ ] Feature importance 분석

### 장기
- [ ] Distributed training (DDP)
- [ ] A/B 테스팅 프레임워크
- [ ] AutoML 하이퍼파라미터 최적화
- [ ] Model ensemble
- [ ] BERT 기반 semantic features 통합

## 기술 선택 이유

### PyTorch
- 동적 그래프로 복잡한 loss function 구현 용이
- 산업계 표준
- 풍부한 커뮤니티 및 라이브러리

### LETOR 포맷
- 학계 및 산업계 표준
- 공개 데이터셋 호환성
- 간단하고 명확한 구조

### TensorBoard
- 실시간 학습 모니터링
- 메트릭 시각화
- 무료 및 오픈소스

### YAML Configuration
- 가독성 좋음
- 계층 구조 지원
- 버전 관리 용이

## 성능 벤치마크 (예상)

MSLR-WEB10K 데이터셋 기준:

| Model | NDCG@1 | NDCG@5 | NDCG@10 | Training Time |
|-------|--------|--------|---------|---------------|
| RankNet | ~0.45 | ~0.48 | ~0.50 | ~2h (GPU) |
| LambdaRank | ~0.47 | ~0.50 | ~0.52 | ~2.5h (GPU) |
| ListNet | ~0.46 | ~0.49 | ~0.51 | ~3h (GPU) |
| ListMLE | ~0.48 | ~0.51 | ~0.53 | ~3.5h (GPU) |

*실제 성능은 데이터셋과 하이퍼파라미터에 따라 달라집니다.*

## 참고 자료

### 논문
1. Burges et al. (2005) - RankNet
2. Burges et al. (2007) - LambdaRank
3. Cao et al. (2007) - ListNet
4. Xia et al. (2008) - ListMLE

### 데이터셋
- MSLR-WEB10K: https://www.microsoft.com/en-us/research/project/mslr/
- MSLR-WEB30K: https://www.microsoft.com/en-us/research/project/mslr/
- Yahoo! Learning to Rank Challenge

### 관련 기술
- Information Retrieval
- Neural Ranking Models
- Learning to Rank
- Search Engine Optimization

---

**프로젝트 완성일**: 2026년 1월
**작성자**: 검색 도메인 ML 엔지니어 (4년차)
**목적**: 포트폴리오 및 실무 적용 가능한 LTR 시스템 구현
