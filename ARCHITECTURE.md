# Learning to Rank - 레포 아키텍처 및 개발 전략

## 목차
- [레포 포지셔닝](#레포-포지셔닝)
- [전체 시스템 아키텍처](#전체-시스템-아키텍처)
- [현재 레포의 역할](#현재-레포의-역할)
- [기술 스택](#기술-스택)
- [디렉토리 구조](#디렉토리-구조)
- [개발 워크플로우](#개발-워크플로우)
- [모델 라이프사이클](#모델-라이프사이클)
- [고도화 로드맵](#고도화-로드맵)

---

## 레포 포지셔닝

### 본 레포지토리: `learning_to_rank`

```
┌─────────────────────────────────────────┐
│  Learning to Rank (Research)            │
│  - 모델 개발 및 실험                      │
│  - Offline 평가                          │
│  - 모델 아티팩트 생성                     │
└─────────────────────────────────────────┘
              ↓ (모델 배포)
┌─────────────────────────────────────────┐
│  ltr-serving (Production)               │
│  - 실시간 API 서빙                        │
│  - Online 모니터링                        │
│  - A/B 테스팅                            │
└─────────────────────────────────────────┘
```

**정체성**: Model Development & Experimentation Repository

**책임 범위**:
- ✅ Pairwise vs Listwise 모델 개발
- ✅ 하이퍼파라미터 튜닝 및 실험
- ✅ 오프라인 평가 (NDCG, MAP, MRR)
- ✅ 모델 비교 및 검증
- ✅ 프로덕션용 모델 아티팩트 생성
- ❌ 실시간 서빙 (별도 레포)
- ❌ 프로덕션 배포 (별도 레포)

---

## 전체 시스템 아키텍처

### Multi-Repository 구조

```
┌────────────────────────────────────────────────────────────┐
│                    ML System Architecture                   │
└────────────────────────────────────────────────────────────┘

┌─────────────────────┐     ┌─────────────────────┐
│ learning_to_rank    │────▶│   ltr-serving       │
│ (Research Repo)     │     │   (Production Repo) │
│                     │     │                     │
│ - 모델 개발         │     │ - FastAPI 서버      │
│ - 실험 추적         │     │ - ONNX Runtime      │
│ - 오프라인 평가     │     │ - Redis 캐싱        │
│ - 모델 검증         │     │ - K8s 배포          │
└─────────────────────┘     └─────────────────────┘
         │                           │
         │                           │
         ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│ Model Registry      │     │ Monitoring          │
│ (Optional)          │     │ (Prometheus/Grafana)│
│                     │     │                     │
│ - MLflow            │     │ - 지연시간 추적      │
│ - Model Versioning  │     │ - 모델 성능 모니터링 │
│ - Metadata Store    │     │ - 알람              │
└─────────────────────┘     └─────────────────────┘
```

### 레포 간 데이터 흐름

```
1. 모델 개발 (learning_to_rank)
   ↓
   [실험] → [검증] → [best_model.pt]
   ↓
2. 모델 내보내기
   ↓
   [ONNX 변환] → [model_v1.0.0.onnx]
   ↓
3. 프로덕션 배포 (ltr-serving)
   ↓
   [API 서빙] → [실시간 추론]
   ↓
4. 온라인 평가
   ↓
   [A/B 테스트] → [성능 피드백]
   ↓
5. 다음 iteration (learning_to_rank)
```

---

## 현재 레포의 역할

### 1. Model Development (모델 개발)

```python
# Pairwise 모델 개발
python scripts/train_pairwise.py \
    --model ranknet \
    --config experiments/exp001.yaml

# Listwise 모델 개발
python scripts/train_listwise.py \
    --model listnet \
    --config experiments/exp002.yaml
```

**산출물**:
- `checkpoints/ranknet_exp001.pt`
- `checkpoints/listnet_exp002.pt`

### 2. Experimentation (실험 및 비교)

```bash
# 실험 비교
experiments/
├── exp001_ranknet_baseline/
│   ├── config.yaml
│   ├── training_history.json
│   └── metrics.json (NDCG@10: 0.512)
├── exp002_listnet_baseline/
│   ├── config.yaml
│   ├── training_history.json
│   └── metrics.json (NDCG@10: 0.518)
└── exp003_lambdarank_tuned/
    ├── config.yaml
    ├── training_history.json
    └── metrics.json (NDCG@10: 0.525) ⭐ BEST
```

### 3. Offline Evaluation (오프라인 평가)

```python
# 테스트 데이터셋 평가
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --data data/raw/test.txt

# 결과
{
  "ndcg@1": 0.482,
  "ndcg@5": 0.501,
  "ndcg@10": 0.525,
  "map": 0.456,
  "mrr": 0.521
}
```

### 4. Model Artifact Generation (모델 아티팩트 생성)

```bash
# 프로덕션용 모델 내보내기
production/
├── model_v1.0.0.onnx           # ONNX 변환 모델
├── model_metadata.json         # 메타데이터
├── preprocessor.pkl            # 전처리기
└── model_card.md              # 모델 카드
```

---

## 기술 스택

### Core Technologies

| 계층 | 기술 | 용도 |
|------|------|------|
| **Deep Learning** | PyTorch 2.0+ | 모델 개발 및 학습 |
| **Data Processing** | NumPy, Pandas | 데이터 처리 |
| **Preprocessing** | scikit-learn | Feature scaling |
| **Configuration** | YAML | 실험 설정 관리 |
| **Logging** | Python logging, TensorBoard | 학습 추적 |
| **Evaluation** | Custom metrics | NDCG, MAP, MRR |

### Development Tools

| 도구 | 용도 |
|------|------|
| **Git** | 버전 관리 |
| **Python 3.8+** | 개발 언어 |
| **pytest** | 단위 테스트 (향후) |
| **pre-commit** | 코드 품질 (향후) |

---

## 디렉토리 구조

```
learning_to_rank/
│
├── README.md                   # 프로젝트 개요
├── ARCHITECTURE.md            # 본 문서
├── CONTEXT.md                 # 프로젝트 컨텍스트
├── config.yaml                # 기본 설정
├── requirements.txt           # Python 의존성
│
├── data/                      # 데이터 모듈
│   ├── dataset.py            # LETOR 포맷, Pairwise/Listwise 데이터셋
│   ├── preprocessing.py      # Feature normalization
│   └── __init__.py
│
├── models/                    # 모델 구현
│   ├── base.py               # BaseRankingModel
│   ├── pairwise.py           # RankNet, LambdaRank
│   ├── listwise.py           # ListNet, ListMLE, ApproxNDCG
│   └── __init__.py
│
├── training/                  # 학습 파이프라인
│   ├── trainer.py            # Pairwise 학습
│   ├── listwise_trainer.py   # Listwise 학습
│   └── __init__.py
│
├── evaluation/                # 평가 메트릭
│   ├── metrics.py            # NDCG, MAP, MRR, Precision, Recall
│   └── __init__.py
│
├── inference/                 # 추론 파이프라인
│   ├── predictor.py          # 오프라인 배치 추론
│   └── __init__.py
│
├── utils/                     # 유틸리티
│   ├── config.py             # 설정 관리
│   ├── logger.py             # 로깅
│   └── __init__.py
│
├── scripts/                   # 실행 스크립트
│   ├── train_pairwise.py     # Pairwise 모델 학습
│   ├── train_listwise.py     # Listwise 모델 학습
│   ├── inference.py          # 오프라인 추론
│   ├── evaluate.py           # 모델 평가 (향후)
│   ├── export_model.py       # ONNX 내보내기 (향후)
│   └── generate_sample_data.py
│
├── experiments/               # 실험 디렉토리 (향후)
│   ├── exp001_ranknet/
│   ├── exp002_listnet/
│   └── ...
│
├── checkpoints/               # 모델 체크포인트
│   ├── experiments/          # 실험 중인 모델
│   ├── validated/            # 검증 완료 (스테이징)
│   └── production/           # 프로덕션 배포용
│
├── logs/                      # 학습 로그
├── results/                   # 평가 결과
└── data/                      # 데이터
    ├── raw/                  # 원본 데이터
    └── processed/            # 전처리된 데이터
```

---

## 개발 워크플로우

### 1. 실험 설계

```yaml
# experiments/exp003/config.yaml
experiment:
  name: "exp003_lambdarank_tuned"
  description: "LambdaRank with tuned hyperparameters"

model:
  pairwise:
    name: "lambdarank"
    hidden_dims: [512, 256, 128]  # 더 큰 네트워크
    dropout: 0.3

training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.0005  # 낮은 learning rate
  early_stopping_patience: 15
```

### 2. 모델 학습

```bash
# 실험 실행
python scripts/train_pairwise.py \
    --config experiments/exp003/config.yaml \
    --model lambdarank \
    --device cuda

# 출력
Training exp003_lambdarank_tuned...
Epoch 1/100: loss=0.452, val_ndcg@10=0.498
Epoch 2/100: loss=0.421, val_ndcg@10=0.512
...
Epoch 45/100: loss=0.298, val_ndcg@10=0.525 ⭐ BEST
Early stopping at epoch 60
```

### 3. 모델 평가

```bash
# 테스트셋 평가
python scripts/evaluate.py \
    --model checkpoints/exp003_best.pt \
    --data data/raw/test.txt

# 실험 결과 기록
experiments/exp003/results.json
```

### 4. 모델 비교

```python
# 실험 비교 (향후 기능)
python scripts/compare_experiments.py

# 출력
┌─────────┬───────────┬───────────┬─────────┐
│ Exp ID  │ Model     │ NDCG@10   │ MAP     │
├─────────┼───────────┼───────────┼─────────┤
│ exp001  │ RankNet   │ 0.512     │ 0.445   │
│ exp002  │ ListNet   │ 0.518     │ 0.451   │
│ exp003  │ LambdaRank│ 0.525 ⭐  │ 0.456   │
└─────────┴───────────┴───────────┴─────────┘
```

### 5. 모델 검증 및 스테이징

```bash
# 검증 완료 후 스테이징으로 이동
mv checkpoints/exp003_best.pt \
   checkpoints/validated/lambdarank_v1.0.0.pt

# 메타데이터 생성
{
  "model_id": "lambdarank_v1.0.0",
  "experiment_id": "exp003",
  "metrics": {
    "ndcg@10": 0.525,
    "map": 0.456
  },
  "validated_at": "2026-01-04",
  "status": "ready_for_production"
}
```

### 6. 프로덕션 내보내기

```bash
# ONNX 변환 (향후 구현)
python scripts/export_model.py \
    --model checkpoints/validated/lambdarank_v1.0.0.pt \
    --output production/model_v1.0.0.onnx

# 프로덕션 레포로 전달
production/
├── model_v1.0.0.onnx
├── model_metadata.json
├── preprocessor.pkl
└── model_card.md
```

---

## 모델 라이프사이클

### Stage 1: Development (개발)

```
Location: checkpoints/experiments/
Status: 개발 중
Purpose: 다양한 아키텍처 시도
Example: exp001_ranknet.pt, exp002_listnet.pt
```

### Stage 2: Validation (검증)

```
Location: checkpoints/validated/
Status: 검증 완료
Purpose: 테스트셋 평가 통과
Example: lambdarank_v1.0.0.pt
Criteria: NDCG@10 > 0.52
```

### Stage 3: Production (프로덕션)

```
Location: production/
Status: 배포 준비 완료
Purpose: 프로덕션 서빙
Example: model_v1.0.0.onnx
Format: ONNX (최적화)
```

### Stage 4: Deployment (배포)

```
Location: ltr-serving 레포
Status: 실시간 서빙 중
Purpose: API 엔드포인트 제공
Monitoring: Prometheus + Grafana
```

### Stage 5: Monitoring & Feedback (모니터링)

```
Metrics:
- Online NDCG (실시간 사용자 반응)
- Latency (응답 시간)
- Throughput (처리량)
- Error Rate (에러율)

Feedback Loop:
온라인 성능 저하 → 새 실험 (Stage 1)
```

---

## 고도화 로드맵

### Phase 1: 현재 (Research Foundation) ✅

**목표**: 모델 개발 및 실험 환경 구축

**완료된 것**:
- ✅ Pairwise 모델 (RankNet, LambdaRank)
- ✅ Listwise 모델 (ListNet, ListMLE, ApproxNDCG)
- ✅ 데이터 처리 파이프라인
- ✅ 학습 파이프라인 (Early stopping, LR scheduling)
- ✅ 오프라인 평가 메트릭
- ✅ 오프라인 배치 추론

**현재 레벨**: Junior → Mid-level

---

### Phase 2: 실험 인프라 (Experiment Tracking) 🔄

**목표**: 체계적인 실험 관리

**추가 기능**:
```python
# 1. 실험 추적
experiments/
├── experiment_tracker.py
└── compare_experiments.py

# 2. 하이퍼파라미터 최적화
scripts/
└── hyperparameter_search.py  # Optuna 통합

# 3. 실험 대시보드
notebooks/
└── experiment_analysis.ipynb  # Jupyter Notebook
```

**기술 스택**:
- MLflow (실험 추적)
- Optuna (하이퍼파라미터 튜닝)
- Weights & Biases (대시보드)

**소요 시간**: 2-3주

**현재 레벨**: Mid-level → Senior

---

### Phase 3: 프로덕션 연결 (Production Bridge) 🔄

**목표**: 프로덕션 레포와의 통합

**추가 기능**:
```python
# 1. ONNX 변환
scripts/export_model.py
- PyTorch → ONNX 변환
- 추론 속도 10배 개선

# 2. 모델 검증
scripts/validate_model.py
- ONNX 모델 검증
- Latency 벤치마크

# 3. CI/CD 파이프라인
.github/workflows/
├── train.yml          # 자동 학습
├── evaluate.yml       # 자동 평가
└── export.yml         # ONNX 변환
```

**기술 스택**:
- ONNX Runtime
- GitHub Actions
- pytest (모델 테스트)

**소요 시간**: 2주

**현재 레벨**: Senior

---

### Phase 4: 프로덕션 레포 (Production Serving) 📋

**목표**: 실시간 API 서빙 시스템 구축

**새 레포**: `ltr-serving`

```
ltr-serving/
├── app/
│   ├── main.py              # FastAPI 서버
│   ├── models.py            # Pydantic 모델
│   └── inference.py         # ONNX 추론
├── deployment/
│   ├── Dockerfile
│   ├── kubernetes/
│   └── docker-compose.yml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana_dashboard.json
└── tests/
    ├── test_api.py
    └── load_test.py
```

**기술 스택**:
- FastAPI (API 서버)
- ONNX Runtime (추론)
- Redis (캐싱)
- Docker + Kubernetes (배포)
- Prometheus + Grafana (모니터링)

**소요 시간**: 3-4주

**현재 레벨**: Senior → Staff

---

### Phase 5: MLOps 플랫폼 (Full MLOps) 📋

**목표**: End-to-End ML 시스템

**새 레포**: `ml-platform`

```
ml-platform/
├── feature-store/
│   └── online_features.py
├── model-registry/
│   └── registry_service.py
├── ab-testing/
│   └── experiment_framework.py
└── monitoring/
    └── model_performance.py
```

**기술 스택**:
- Feature Store (Feast)
- Model Registry (MLflow)
- A/B Testing (자체 구현)
- Data Pipeline (Airflow)

**소요 시간**: 2-3개월

**현재 레벨**: Staff → Principal

---

## 레포 간 역할 분담

### learning_to_rank (본 레포)

```yaml
역할: Model Development & Research
책임:
  - 모델 아키텍처 개발
  - 실험 및 하이퍼파라미터 튜닝
  - 오프라인 평가
  - 모델 아티팩트 생성

소유: Data Science / ML Research 팀
배포: 없음 (개발 환경)
업데이트 주기: 주 1-2회
```

### ltr-serving (프로덕션 레포)

```yaml
역할: Production Serving
책임:
  - 실시간 API 서빙
  - 추론 최적화 (ONNX, 배치)
  - 모니터링 및 알람
  - A/B 테스트

소유: ML Platform / Engineering 팀
배포: Kubernetes
업데이트 주기: 월 1-2회 (모델 업데이트)
```

### ml-platform (MLOps 레포)

```yaml
역할: ML Infrastructure
책임:
  - Feature Store 관리
  - Model Registry 운영
  - 실험 플랫폼 제공
  - 통합 모니터링

소유: ML Platform 팀
배포: Cloud Infrastructure
업데이트 주기: 지속적
```

---

## 다음 단계

### Immediate (즉시 진행)

1. **실험 디렉토리 구조화**
   ```bash
   mkdir -p experiments/{exp001,exp002,exp003}
   ```

2. **모델 스테이징 프로세스**
   ```bash
   mkdir -p checkpoints/{experiments,validated,production}
   ```

3. **문서화 개선**
   - Model Card 템플릿 작성
   - 실험 가이드 작성

### Short-term (1개월 내)

1. **ONNX 변환 스크립트** 작성
2. **모델 평가 스크립트** 개선
3. **CI/CD 파이프라인** 구축

### Mid-term (3개월 내)

1. **ltr-serving 레포** 구축
2. **프로덕션 배포** 테스트
3. **A/B 테스트** 프레임워크

### Long-term (6개월+)

1. **MLOps 플랫폼** 구축
2. **AutoML** 파이프라인
3. **Feature Store** 통합

---

## 참고 자료

### 내부 문서
- [README.md](README.md) - 프로젝트 개요
- [CONTEXT.md](CONTEXT.md) - 프로젝트 배경
- [config.yaml](config.yaml) - 설정 파일

### 외부 참고
- [MLOps Best Practices](https://ml-ops.org/)
- [ONNX Runtime Optimization](https://onnxruntime.ai/)
- [FastAPI Production Guide](https://fastapi.tiangolo.com/deployment/)

---

**최종 업데이트**: 2026-01-04
**작성자**: ML Engineering Team
**버전**: 1.0.0
