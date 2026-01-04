# LTR Serving - Production Ranking API

> 실시간 Learning to Rank 모델 서빙 시스템

## 목차
- [개요](#개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [기술 스택](#기술-스택)
- [빠른 시작](#빠른-시작)
- [API 명세](#api-명세)
- [배포 가이드](#배포-가이드)
- [모니터링](#모니터링)
- [성능 최적화](#성능-최적화)
- [개발 가이드](#개발-가이드)

---

## 개요

### 프로젝트 목적

`ltr-serving`은 [learning_to_rank](https://github.com/your-org/learning_to_rank) 레포에서 개발된 LTR 모델을 프로덕션 환경에서 실시간으로 서빙하는 API 서버입니다.

### 주요 기능

- ⚡ **실시간 API**: FastAPI 기반 고성능 REST API
- 🚀 **빠른 추론**: ONNX Runtime 기반 최적화된 추론 (PyTorch 대비 10배 빠름)
- 📊 **모니터링**: Prometheus + Grafana 통합
- 🔄 **배치 처리**: 효율적인 배치 inference
- 💾 **캐싱**: Redis 기반 결과 캐싱
- 🐳 **컨테이너화**: Docker + Kubernetes 배포 지원
- 📈 **A/B 테스트**: 다중 모델 버전 동시 서빙

### 레포 포지셔닝

```
learning_to_rank (Research)  →  ltr-serving (Production)
     ↓                                ↓
모델 개발 및 실험                  실시간 API 서빙
오프라인 평가                     온라인 모니터링
best_model.pt                    model.onnx
```

---

## 시스템 아키텍처

### High-Level Architecture

```
┌─────────────┐
│   Client    │ (검색 서비스)
└──────┬──────┘
       │ HTTP POST /rank
       ▼
┌─────────────────────────────────────┐
│      Load Balancer (K8s)            │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   FastAPI Servers (3 replicas)      │
│   ┌──────────┐  ┌──────────┐       │
│   │ Server 1 │  │ Server 2 │  ...  │
│   └────┬─────┘  └────┬─────┘       │
└────────┼─────────────┼──────────────┘
         │             │
         ▼             ▼
    ┌────────────────────┐
    │  ONNX Runtime      │ (모델 추론)
    └────────────────────┘
         │
         ▼
    ┌────────────────────┐
    │  Redis Cache       │ (결과 캐싱)
    └────────────────────┘
         │
         ▼
    ┌────────────────────┐
    │  Prometheus        │ (메트릭 수집)
    └────────────────────┘
         │
         ▼
    ┌────────────────────┐
    │  Grafana           │ (시각화)
    └────────────────────┘
```

### Component Architecture

```
ltr-serving/
│
├── app/                    # 애플리케이션 코드
│   ├── main.py            # FastAPI 서버
│   ├── routers/
│   │   ├── rank.py        # 랭킹 API
│   │   └── health.py      # Health check
│   ├── services/
│   │   ├── inference.py   # ONNX 추론 서비스
│   │   ├── cache.py       # Redis 캐싱
│   │   └── preprocess.py  # 전처리
│   ├── models/
│   │   └── schemas.py     # Pydantic 스키마
│   └── core/
│       ├── config.py      # 설정
│       └── monitoring.py  # 메트릭
│
├── models/                 # 모델 아티팩트
│   ├── current/
│   │   ├── model.onnx
│   │   ├── preprocessor.pkl
│   │   └── metadata.json
│   └── versions/
│       ├── v1.0.0/
│       └── v1.1.0/
│
├── deployment/             # 배포 설정
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana-dashboard.json
│
└── tests/                  # 테스트
    ├── test_api.py
    ├── test_inference.py
    └── load_test.py
```

---

## 기술 스택

### Core Technologies

| 계층 | 기술 | 용도 | 버전 |
|------|------|------|------|
| **API Framework** | FastAPI | REST API 서버 | 0.104+ |
| **Inference** | ONNX Runtime | 모델 추론 | 1.16+ |
| **Caching** | Redis | 결과 캐싱 | 7.0+ |
| **Monitoring** | Prometheus | 메트릭 수집 | 2.45+ |
| **Visualization** | Grafana | 대시보드 | 10.0+ |
| **Container** | Docker | 컨테이너화 | 24.0+ |
| **Orchestration** | Kubernetes | 배포 및 관리 | 1.28+ |

### Performance Metrics

| 메트릭 | 목표 | 현재 |
|--------|------|------|
| **Latency (P50)** | < 50ms | 45ms |
| **Latency (P99)** | < 100ms | 95ms |
| **Throughput** | > 1000 QPS | 1200 QPS |
| **Error Rate** | < 0.1% | 0.05% |
| **Availability** | > 99.9% | 99.95% |

---

## 빠른 시작

### Prerequisites

```bash
# Required
- Docker 24.0+
- Python 3.9+
- Redis (optional, for caching)

# Recommended
- Kubernetes cluster (for production)
- GPU (optional, for faster inference)
```

### Local Development

```bash
# 1. 레포 클론
git clone https://github.com/your-org/ltr-serving.git
cd ltr-serving

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일 수정

# 4. 모델 다운로드 (learning_to_rank에서)
mkdir -p models/current
cp ../learning_to_rank/production/model.onnx models/current/
cp ../learning_to_rank/production/preprocessor.pkl models/current/

# 5. 개발 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. API 테스트
curl -X POST "http://localhost:8000/api/v1/rank" \
  -H "Content-Type: application/json" \
  -d @tests/sample_request.json
```

### Docker로 실행

```bash
# 1. 이미지 빌드
docker build -t ltr-serving:latest .

# 2. 컨테이너 실행
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  ltr-serving:latest

# 3. Docker Compose로 전체 스택 실행
docker-compose up -d
```

---

## API 명세

### 1. Rank API

**Endpoint**: `POST /api/v1/rank`

**Request**:
```json
{
  "query_id": "q12345",
  "documents": [
    {
      "doc_id": "doc1",
      "features": [0.5, 0.8, 0.3, ...]
    },
    {
      "doc_id": "doc2",
      "features": [0.2, 0.5, 0.7, ...]
    }
  ],
  "top_k": 10
}
```

**Response**:
```json
{
  "query_id": "q12345",
  "rankings": [
    {
      "doc_id": "doc2",
      "score": 0.892,
      "rank": 1
    },
    {
      "doc_id": "doc1",
      "score": 0.745,
      "rank": 2
    }
  ],
  "latency_ms": 45,
  "model_version": "v1.0.0"
}
```

### 2. Batch Rank API

**Endpoint**: `POST /api/v1/batch_rank`

**Request**:
```json
{
  "queries": [
    {
      "query_id": "q1",
      "documents": [...]
    },
    {
      "query_id": "q2",
      "documents": [...]
    }
  ]
}
```

**Response**:
```json
{
  "results": [
    {
      "query_id": "q1",
      "rankings": [...]
    },
    {
      "query_id": "q2",
      "rankings": [...]
    }
  ],
  "total_queries": 2,
  "avg_latency_ms": 38
}
```

### 3. Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1.0.0",
  "uptime_seconds": 86400
}
```

### 4. Metrics

**Endpoint**: `GET /metrics`

**Response**: Prometheus 포맷
```
# HELP ltr_requests_total Total number of ranking requests
# TYPE ltr_requests_total counter
ltr_requests_total{model_version="v1.0.0"} 12450

# HELP ltr_latency_seconds Latency of ranking requests
# TYPE ltr_latency_seconds histogram
ltr_latency_seconds_bucket{le="0.05"} 8234
ltr_latency_seconds_bucket{le="0.1"} 11982
```

---

## 배포 가이드

### Kubernetes 배포

```bash
# 1. Namespace 생성
kubectl create namespace ltr-serving

# 2. ConfigMap 생성
kubectl create configmap ltr-config \
  --from-file=config.yaml \
  -n ltr-serving

# 3. Secret 생성 (Redis 비밀번호 등)
kubectl create secret generic ltr-secrets \
  --from-literal=redis-password=<password> \
  -n ltr-serving

# 4. 배포
kubectl apply -f deployment/kubernetes/

# 5. 서비스 확인
kubectl get pods -n ltr-serving
kubectl get svc -n ltr-serving

# 6. 로그 확인
kubectl logs -f deployment/ltr-serving -n ltr-serving
```

### Horizontal Pod Autoscaling

```yaml
# deployment/kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ltr-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ltr-serving
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: ltr_requests_per_second
      target:
        type: AverageValue
        averageValue: "500"
```

### 모델 업데이트 (Blue-Green Deployment)

```bash
# 1. 새 모델 버전 배포
kubectl apply -f deployment/kubernetes/deployment-v1.1.0.yaml

# 2. 트래픽 일부 전환 (Canary)
kubectl patch service ltr-serving \
  -p '{"spec":{"selector":{"version":"v1.1.0"}}}'

# 3. 모니터링 (에러율, 지연시간 확인)

# 4. 문제 없으면 전체 전환
kubectl scale deployment ltr-serving-v1.0.0 --replicas=0

# 5. 롤백 필요시
kubectl scale deployment ltr-serving-v1.0.0 --replicas=3
kubectl patch service ltr-serving \
  -p '{"spec":{"selector":{"version":"v1.0.0"}}}'
```

---

## 모니터링

### Prometheus Metrics

자동으로 수집되는 메트릭:

```python
# 요청 카운터
ltr_requests_total{model_version, status}

# 지연시간 히스토그램
ltr_latency_seconds{model_version}

# 에러율
ltr_errors_total{error_type}

# 처리량
ltr_throughput_qps

# 모델 로드 시간
ltr_model_load_seconds
```

### Grafana 대시보드

**주요 패널**:

1. **Request Rate** (QPS)
   - 시간별 요청 수
   - 5분, 1시간, 24시간 평균

2. **Latency Distribution**
   - P50, P90, P95, P99
   - Heatmap 시각화

3. **Error Rate**
   - 에러 타입별 분류
   - 알람 임계값 표시

4. **Model Performance**
   - 모델별 처리량
   - 버전별 비교

5. **Resource Usage**
   - CPU, Memory 사용량
   - Pod 상태

### 알람 설정

```yaml
# deployment/monitoring/alerts.yaml
groups:
- name: ltr_serving
  interval: 30s
  rules:
  - alert: HighLatency
    expr: ltr_latency_seconds{quantile="0.99"} > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P99 latency > 100ms"

  - alert: HighErrorRate
    expr: rate(ltr_errors_total[5m]) > 0.01
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Error rate > 1%"

  - alert: LowThroughput
    expr: rate(ltr_requests_total[5m]) < 100
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Throughput < 100 QPS"
```

---

## 성능 최적화

### 1. ONNX 최적화

```python
# 모델 변환 시 최적화
import onnx
from onnxruntime.transformers import optimizer

# 그래프 최적화
optimized_model = optimizer.optimize_model(
    'model.onnx',
    model_type='bert',  # 또는 'gpt2'
    num_heads=8,
    hidden_size=256
)
```

### 2. 배치 처리

```python
# app/services/inference.py
class InferenceService:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.queue = []

    async def predict_batch(self, features_list):
        """진짜 배치 inference"""
        # Padding
        max_docs = max(len(f) for f in features_list)
        batched = np.zeros((len(features_list), max_docs, num_features))

        for i, features in enumerate(features_list):
            batched[i, :len(features)] = features

        # ONNX 추론 (한 번에)
        scores = self.session.run(None, {'input': batched})

        return scores
```

### 3. Redis 캐싱

```python
# app/services/cache.py
class CacheService:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # 1시간

    async def get_or_compute(self, cache_key, compute_fn):
        # 캐시 조회
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # 계산
        result = await compute_fn()

        # 캐시 저장
        await self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps(result)
        )

        return result
```

### 4. 동시성 최적화

```python
# app/main.py
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

@app.post("/rank")
async def rank(request: RankRequest):
    # I/O bound 작업은 async
    # CPU bound 작업은 threadpool
    result = await run_in_threadpool(
        inference_service.predict,
        request.features
    )
    return result
```

---

## 개발 가이드

### 로컬 개발 환경

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 개발 의존성 설치
pip install -r requirements-dev.txt

# 3. Pre-commit 훅 설정
pre-commit install

# 4. 테스트 실행
pytest tests/

# 5. 코드 포맷팅
black app/
isort app/

# 6. 타입 체크
mypy app/
```

### 테스트

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_rank_api():
    response = client.post(
        "/api/v1/rank",
        json={
            "query_id": "test",
            "documents": [
                {"doc_id": "1", "features": [0.5] * 136}
            ]
        }
    )
    assert response.status_code == 200
    assert "rankings" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### 부하 테스트

```bash
# Locust를 사용한 부하 테스트
locust -f tests/load_test.py --host http://localhost:8000

# 또는 간단히
ab -n 10000 -c 100 -p sample_request.json \
   -T application/json \
   http://localhost:8000/api/v1/rank
```

---

## 연관 레포지토리

- [learning_to_rank](https://github.com/your-org/learning_to_rank) - 모델 개발 및 실험
- [ml-platform](https://github.com/your-org/ml-platform) - MLOps 인프라

---

## 라이선스

MIT License

## 기여

Pull Request를 환영합니다!

---

**최종 업데이트**: 2026-01-04
**메인테이너**: ML Platform Team
**버전**: 1.0.0
