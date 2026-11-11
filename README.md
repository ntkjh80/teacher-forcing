# H200과 Ironwood의 티처 포싱 비교

NVIDIA H200과 Google Cloud TPU Ironwood에서 같은 응답 토큰을 한 개씩 채점합니다. 두 장비가 선택한 다음 토큰이 처음 달라지는 위치를 찾고, 그 위치의 전체 어휘 로짓과 Transformer 블록별 출력을 비교합니다.

대상은 **Qwen/Qwen2.5-0.5B-Instruct**, 단일 장치, 요청 1개, BF16 추론입니다. PyTorch와 Transformers에서 직접 실행합니다. 기존 H200·TPU 프로젝트 없이 이 저장소만으로 동작합니다.

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

## 새 원격 저장소에 올리기

빈 GitHub 또는 GitLab 저장소를 만든 뒤, 이 로컬 저장소에서 `git remote -v`로 연결 상태를 확인합니다. 아래 URL은 본인의 원격 저장소 URL로 바꿉니다.

```bash
# 원격 연결이 없는 로컬 저장소
git remote add origin <새-원격-저장소-URL>
git push -u origin main
```

Git 번들을 `git clone`으로 열었다면 `origin`이 번들 파일을 가리킵니다. 그 경우 첫 줄은 `git remote set-url origin <새-원격-저장소-URL>`을 사용합니다.
