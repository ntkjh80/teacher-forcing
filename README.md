# H200과 Ironwood의 티처 포싱 비교

NVIDIA H200과 Google Cloud TPU Ironwood에서 같은 응답 토큰을 한 개씩 채점합니다. 두 장비가 선택한 다음 토큰이 처음 달라지는 위치를 찾고, 그 위치의 전체 어휘 로짓과 Transformer 블록별 출력을 비교합니다.

대상은 **Qwen/Qwen2.5-0.5B-Instruct**, 단일 장치, 요청 1개, BF16 추론입니다. PyTorch와 Transformers에서 직접 실행합니다. 기존 H200·TPU 프로젝트 없이 이 저장소만으로 동작합니다.

## 왜 이렇게 테스트하나요?

**같은 질문에서 GPU와 TPU의 답이 달라졌다면, 같은 문맥에서도 계산이 달랐을까요? 아니면 앞서 고른 단어가 달라져 이후 문맥까지 바뀌었을까요?** 이 실험은 두 장비에 같은 토큰 이력을 주고, 다음 토큰을 선택하는 순간의 점수와 그 점수를 만든 중간값을 비교합니다. [합성 예시 바로 실행](#1-분석-pc에서-예시-실행)

### 티처 포싱으로 같은 조건의 예측을 비교합니다

자기회귀 모델은 매 단계에서 **지금까지의 문맥을 조건으로** 다음 토큰의 확률 $p(y_t \mid x, y_{<t})$를 계산합니다. 여기서 $x$는 질문, $y_{<t}$는 앞서 입력한 응답 토큰입니다. 자유 생성 중 GPU가 `A B`, TPU가 `A X`를 선택했다면, 세 번째 토큰의 점수에는 이미 서로 다른 문맥의 효과가 들어갑니다. [Vaswani 외, 2017, §3.1](https://arxiv.org/html/1706.03762v7) · [Bengio 외, 2015](https://proceedings.nips.cc/paper_files/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)

여기서는 티처 포싱을 **학습 없이 비교 실험에** 적용합니다. 두 번째 토큰을 예측할 때는 양쪽에 `질문 + A`를 넣고 각 장비가 스스로 매긴 1위와 전체 로짓을 기록합니다. TPU의 1위가 `X`여도 다음 단계에는 양쪽 모두 기준 토큰 `B`를 넣습니다. 따라서 세 번째 위치에서도 같은 입력에 대한 차이를 잴 수 있습니다. H200의 응답은 문맥을 고정하는 기준열입니다. 업무 정답률은 별도의 정답 데이터로 평가해야 합니다.

### 같은 BF16에서도 연산 순서가 결과에 영향을 줍니다

부동소수점은 중간값을 표현 가능한 수로 반올림합니다. 예를 들어 **덧셈할 때마다 BF16으로 반올림**하면 `(256 + 1) − 256 = 0`이고 `(256 − 256) + 1 = 1`입니다. 256과 258 사이의 257이 가장 가까운 짝수 방식에서 256으로 반올림되기 때문입니다. 실제 TPU 행렬곱은 통상 BF16 입력의 곱을 FP32로 누산합니다. 다만 부분합을 묶는 순서, 커널 경계에서 중간값을 변환하는 시점은 실행 구현에 따라 달라질 수 있습니다. 그러므로 양쪽의 BF16 설정만으로 비트 단위 일치를 보장하기는 어렵습니다. [Goldberg, 1991](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) · [PyTorch 수치 정확도](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) · [Google Cloud TPU의 BF16 연산](https://docs.cloud.google.com/tpu/docs/bfloat16)

같은 체크포인트와 토큰 ID를 쓰고, 첫 실험을 장치 1개·요청 1개로 제한하면 가중치·토큰화·배치·분산 합산 차이를 줄일 수 있습니다. 장비별 3회 반복은 **같은 장비에서도 관찰되는 편차**를 확인하기 위한 기본값입니다. 3회 모두 일치해도 다른 입력이나 배치의 재현성까지 보장하지는 않습니다. 레이어 값을 꺼내는 훅을 끈 실행과 켠 첫 실행의 로짓도 비교해, 계측 과정이 출력에 미친 영향을 확인합니다. [수집 코드](src/teacher_forcing/capture.py)

### 최종 로짓과 레이어 활성값은 서로 다른 질문에 답합니다

| 측정값 | 왜 측정하는가 | 결과에서 볼 항목 |
|---|---|---|
| **최종 로짓** | 같은 토큰을 골라도 후보별 원점수는 다를 수 있습니다. 반대로 1·2위가 접전이면 작은 차이로 선택이 바뀝니다. | `tokens.csv`에서 같은 응답 위치의 1위 ID, 1·2위 점수 차이, 전체 어휘의 최대 절대차를 함께 봅니다. |
| **블록별 활성값** | 최종 점수만 보면 차이가 어느 단계에서 커졌는지 알기 어렵습니다. 두 모델의 같은 연산 경계에서 중간 벡터를 비교하면 조사할 블록을 좁힐 수 있습니다. | `layers.csv`에서 같은 토큰 위치의 잔차 합산 후 블록 출력을 순서대로 봅니다. 상대 L2(%)는 GPU 벡터 크기에 비례한 오차, 최대 절대차는 한 성분에서 관찰한 가장 큰 오차입니다. |

**어느 정도의 로짓 차이가 토큰 선택을 바꿀까요?** GPU의 1·2위 점수 차이를 $m$, 같은 위치에서 전체 어휘에 걸쳐 측정한 GPU·TPU 로짓 최대 절대차를 $\varepsilon$이라고 두겠습니다. GPU의 1위 점수는 TPU에서 최대 $\varepsilon$만큼 내려가고 경쟁 후보는 최대 $\varepsilon$만큼 올라갈 수 있습니다. 따라서 **$m > 2\varepsilon$이면 1위가 유지됩니다.** 예를 들어 점수 차이가 0.10이고 최대 절대차가 0.02라면 GPU의 1위가 TPU에서도 1위입니다. 점수 차이가 0.01이면 실제 1위 ID를 확인해야 합니다. 이는 해당 위치의 그리디 선택에 대한 충분조건이며, 서비스 품질 합격선은 별도로 정합니다.

활성값은 **코사인 유사도 하나로 판정하기 어렵습니다.** 벡터 `[1, 2]`와 `[2, 4]`의 코사인 유사도는 1이지만 크기는 두 배입니다. 또 12번 블록의 오차가 크더라도 11번 블록에서 받은 입력이 이미 달랐을 수 있습니다. 원인을 확인하려면 후속 실험에서 같은 블록 입력과 문맥 상태를 주고 재실행합니다. 대응하는 중간 텐서를 비교해 오차 위치를 찾는 방식은 ONNX Runtime의 양자화 디버깅에도 쓰입니다. [ONNX Runtime의 활성값 비교 방법](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantization-debugging)

### 프리필과 KV 디코드를 나눠야 캐시 경로의 영향을 좁힐 수 있습니다

인과적 어텐션에서는 뒤에 토큰을 붙여도 이전 위치가 참조할 수 있는 문맥은 그대로입니다. 따라서 같은 가중치·위치·마스크에 대해, 정확한 산술로 계산하면 이전 Key와 Value를 다시 계산한 결과와 저장해 두었다가 읽은 결과가 같습니다. 이 성질을 이용해 같은 위치를 **전체 문맥으로 한 번에 계산한 프리필**과 **캐시를 유지하며 한 토큰씩 계산한 디코드**로 각각 채점합니다. 첫 응답의 점수는 두 경로 모두 프롬프트 처리에서 나옵니다. 이전 단계의 KV를 재사용한 비교는 **두 번째 응답 위치부터** 시작합니다. [Hugging Face의 KV 캐시 원리](https://huggingface.co/docs/transformers/en/cache_explanation)

두 경로에서 모두 같은 위치부터 오차가 크면 입력·가중치와 공통 블록 연산부터 조사합니다. 디코드에서만 오차가 커지면 캐시 저장·복원과 단일 토큰용 커널을 우선 봅니다. **이 결과만으로 KV 양자화를 원인으로 확정할 수는 없습니다.** 원인 확인에는 저장 직전 값, 복원 직후 값, 어텐션 출력의 추가 계측이 필요합니다. 최종 이식 판단에는 실제 서빙 엔진에서 자유 생성한 응답과 사내 평가셋 점수를 함께 사용합니다.

## 1. 분석 PC에서 예시 실행

저장소 루트에서 실행합니다. 분석에는 NumPy만 필요합니다. 예시 입력과 점수는 합성 데이터입니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
teacher-parity demo --out runs/demo
```

Windows PowerShell에서는 가상환경을 `.venv\Scripts\Activate.ps1`로 활성화합니다. 결과는 `runs/demo/summary.md`에 저장됩니다. 설치 후 `python -m teacher_forcing`도 `teacher-parity`와 같은 명령을 제공합니다. [예시 결과표](examples/demo_summary.md)

## 2. H200과 Ironwood에서 실측

두 장비에는 **똑같은 safetensors 체크포인트와 tokenizer/config 파일**을 준비합니다. 가속기에 맞는 PyTorch가 동작하는 환경에서 이 저장소를 설치합니다. Python 3.12를 권장합니다. TPU에서는 네이티브 TorchTPU가 필요합니다. 기존 서빙 환경과 별도의 진단용 가상환경 또는 컨테이너를 사용합니다. [장비 환경 및 체크포인트 준비](docs/runbook_ko.md)

```bash
# 각 장비의 가속기용 가상환경에서 실행
python -m pip install -e '.[capture]'
```

| 순서 | 실행 장소 | 작업 |
|---|---|---|
| 1 | H200 | 아래 GPU 명령을 실행해 `runs/case01/case.json`과 `gpu/` 생성 |
| 2 | Ironwood | GPU의 `case.json`을 `runs/case01/`에 복사한 뒤 TPU 명령 실행 |
| 3 | 분석 PC | `runs/case01/`에 같은 `case.json`, `gpu/`, `tpu/`를 모아 비교 명령 실행 |

```bash
# H200
teacher-parity capture --device gpu --model-dir /models/qwen-parity --root runs/case01

# Ironwood
teacher-parity capture --device tpu --model-dir /models/qwen-parity --root runs/case01

# 분석 PC
teacher-parity compare --root runs/case01
```

기본값은 기준 응답 최대 8토큰, 장비별 3회 반복입니다. GPU에서 새 실험을 시작할 때 `--question`, `--max-reference-tokens`를 지정할 수 있습니다. 여러 질문은 질문별로 다른 `--root`에 저장합니다. [노트북으로 실행](notebooks/H200_Ironwood_Teacher_Forcing.ipynb)

## 3. 결과 확인

| 파일 | 확인할 내용 |
|---|---|
| `report/summary.md` | 경로별 Top-1 일치 수, 최초 불일치 응답 위치, 해당 위치의 블록별 오차, 장비 자체 반복과 계측 영향 |
| `report/tokens.csv` | 모든 응답 위치의 기준 토큰, 양쪽 1위 ID, 1·2위 점수 차이, 전체 로짓 오차 |
| `report/layers.csv` | 같은 응답 위치·같은 블록에서 GPU 기준 상대 L2(%)와 최대 절대차 |
| `report/repeats.csv` | 장비별 첫 실행과 뒤의 반복 사이에서 발견한 최대 절대차 |

프리필은 전체 문맥을 한 번에 계산합니다. 캐시 디코드는 같은 기준 토큰을 한 개씩 넣습니다. 캐시 디코드의 첫 응답 점수는 프롬프트 처리에서 얻으므로, 요약표에는 **두 번째 응답부터의 KV 재사용 결과**도 따로 표시합니다. 서비스 이식 판정에는 실제 서빙 엔진의 재현 결과와 업무 정답률이 추가로 필요합니다. [지표와 해석 예시](docs/runbook_ko.md)

## 저장소 구성

| 경로 | 역할 |
|---|---|
| `src/teacher_forcing/` | CLI, 장비별 계측, 비교와 보고서 생성 |
| `notebooks/` | CLI와 같은 Python 함수를 사용하는 실습 노트북 |
| `tests/` | 토큰 정렬, 캐시 재사용, 파일 검증과 결과표 회귀 검사 |
| `examples/` | 공개 가능한 합성 예시 결과표 |
| `runs/` | 로컬 실험 입력·원시 텐서·보고서. Git 추적 제외 |

개발 검사는 `python -m pip install -e '.[dev]'`와 `python -m pytest -q`로 실행합니다. PyTorch가 설치돼 있으면 CPU에서 캐시 재사용과 레이어 훅도 검사합니다. Git에는 실행 코드를 보관하고 실측 데이터와 모델은 각 장비 또는 별도 저장소에서 관리합니다.