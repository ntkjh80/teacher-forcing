# 실행 환경과 결과 해석

## 장비와 모델 준비

분석 PC는 Python과 NumPy로 합성 예시 실행과 실측 결과 비교를 수행합니다. H200 호스트에서는 CUDA용 PyTorch, Ironwood 호스트에서는 네이티브 TorchTPU를 준비합니다. 각 가속기 환경에서 아래 확인 명령을 먼저 실행하세요. 장치 초기화에 실패하면 해당 환경의 설치를 확인합니다.

```bash
# H200
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'

# Ironwood
python -c 'import torch, torch_tpu; print(torch.ones(1, device="tpu:0").cpu())'
```

각 환경에서 저장소 루트를 열고 `python -m pip install -e '.[capture]'`로 설치합니다. 장비 런타임과 호환되는 **동일한 Transformers 버전**을 선택하고, `python -m pip show torch transformers`로 실제 설치 버전을 확인합니다. 기존 장비 런타임이 Transformers 버전을 지정했다면 해당 버전을 유지합니다. [TorchTPU 플랫폼 안내](https://huggingface.co/docs/transformers/main/google_tpu) · [PyTorch 설치](https://pytorch.org/get-started/locally/)

체크포인트는 한 곳에서 다운로드하고 디렉터리 전체를 양쪽에 배포합니다. 아래는 Hugging Face CLI가 있는 환경의 예시입니다.

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --revision eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2 --local-dir /models/qwen-parity
```

GPU가 기준 질문을 템플릿에 넣고 실제 토큰 ID로 바꾼 뒤 그리디로 응답합니다. TPU에는 GPU가 만든 `case.json` 원본 파일을 복사합니다. 줄바꿈이나 공백을 변경하면 파일 해시가 달라집니다. 양쪽 모두 같은 체크포인트로 실행했는지 확인하기 위해 가중치와 모델 폴더의 설정 JSON을 해싱합니다. 수집은 **순수 BF16 Qwen2 계열용 단일 장치 진단**입니다. 모델 파일은 로컬에서 읽습니다.

```bash
# H200에서 다른 질문으로 시작할 때는 새 폴더 지정
teacher-parity capture --device gpu --model-dir /models/qwen-parity --root runs/case02 \
  --question "대한민국의 수도를 한 문장으로 답해주세요." --max-reference-tokens 16 --repeats 3

# GPU에서 만든 runs/case02/case.json을 TPU에 복사한 뒤 실행
teacher-parity capture --device tpu --model-dir /models/qwen-parity --root runs/case02 --repeats 3

# 분석 PC에서 case.json, gpu/, tpu/가 같은 폴더에 있는지 확인한 뒤 실행
teacher-parity compare --root runs/case02
```

수집이 완료되면 장비 하위 폴더에 `meta.json`과 `prefill_0.npz`, `cached_0.npz` 등의 파일이 생깁니다. 완료된 결과를 재실행하려면 새 실험 폴더나 새 보고서 경로(`compare --out …`)를 사용합니다. 기존 단독 노트북이 쓴 `case.json`과 두 장비 폴더를 그대로 같은 `--root`에 모아 비교할 수도 있습니다.

## 같은 위치의 점수를 읽는 법

| 기준 응답 | 양쪽에 넣는 문맥 | 점수를 읽는 위치 |
|---|---|---|
| A | 프롬프트 | 프롬프트의 마지막 토큰 |
| B | 프롬프트 + A | A |
| C | 프롬프트 + A + B | B |

TPU가 B 대신 X를 1위로 평가해도 다음 입력은 두 장비 모두 B입니다. 다음 토큰의 일치 여부는 **강제 입력 전 전체 어휘 로짓의 1위**로 판정합니다. 응답 위치 번호는 1부터, 블록 번호는 0부터 시작합니다. `block_...`은 잔차 합산을 마친 블록 출력, `final_norm`은 최종 정규화 출력입니다.

프리필은 causal mask를 적용해 전체 문맥을 한 번에 처리합니다. 캐시 디코드는 프롬프트를 한 번 처리한 뒤 기준 토큰을 차례로 넣으며 이전 KV를 다시 씁니다. 이 경로에서도 첫 응답 점수는 프리필에서 나옵니다. 기준 응답이 한 토큰이면 KV 재사용 위치 수는 0입니다. `layers.csv`와 `tokens.csv`의 장비 간 비교는 첫 반복을 사용하고 `repeats.csv`는 같은 장비의 각 반복을 첫 반복과 대조합니다.

## 최종 결과 정리

1. **같은 장비의 변동부터 확인합니다.** `summary.md`의 자체 반복과 계측 ON/OFF 로짓 최대 절대차를 봅니다. 계측 자체가 출력을 바꾸면 그 크기를 기록합니다.
2. **처음 선택이 바뀐 토큰을 찾습니다.** `tokens.csv`에서 해당 위치의 1위 ID와 양쪽 1·2위 점수 차이를 봅니다. 후보 간 점수 차이가 작으면 작은 오차로도 1위가 바뀝니다.
3. **그 위치에서 레이어를 대조합니다.** `layers.csv`의 `relative_l2_pct`는 GPU 기준 `100 × ‖TPU−GPU‖₂ / ‖GPU‖₂`입니다. GPU 벡터가 0이면 절대차를 사용합니다. 처음 오차가 커진 블록은 별도 재실행에서 같은 블록 입력을 넣어 원인을 좁힐 출발점입니다.
4. **KV 효과를 확인합니다.** 프리필에서는 비슷하지만 디코드에서만 오차가 커지면, 운영 엔진에서 같은 위치의 캐시 저장 직전 → 복원 직후 → Attention 출력을 추가 측정합니다. 이 세 값은 본 도구의 수집 범위 밖입니다. 디코드 커널과 캐시 형식의 영향을 함께 검토합니다.
5. **서비스 품질을 판정합니다.** 업무 표본에서 자유 생성과 사내 정답 채점을 추가합니다. 모델·토큰 위치별 수치 재현성, 그리디 토큰 일치율, 업무 정답률 차이(%p)를 각각 보고합니다. 실제 vLLM 결과는 별도 실행에서 확보합니다.

합성 예시에서는 프리필 3/3, 캐시 경로 전체 2/3 위치의 1위가 일치합니다. 디코드 두 번째 응답의 1위가 다르며 `block_001`에서 차이가 관찰됩니다. 이 값으로 H200·Ironwood의 실측 합격 기준을 정할 수는 없습니다. [예시 결과표](../examples/demo_summary.md)

## 같은 BF16에서도 달라지는 이유

BF16은 표현할 수 있는 수가 제한됩니다. 덧셈을 할 때마다 BF16으로 반올림한다고 가정하면 `(256 + 1) − 256`은 0, `(256 − 256) + 1`은 1이 됩니다. 첫 번째 식에서 257은 256과 258의 중간이고 가장 가까운 짝수 방식에서는 256으로 반올림됩니다. 실제 행렬곱은 커널에 따라 FP32 누산을 사용할 수 있습니다.

GPU와 TPU의 커널은 행렬을 나누는 크기, 부분합을 더하는 순서, 중간값을 반올림하는 시점이 달라질 수 있습니다. KV 저장·복원 형식, 배치 구성, 분산 합산 경로도 결과에 영향을 줄 수 있습니다. 그래서 같은 입력·자료형에서도 비트 단위 동일성은 보장되지 않습니다. [PyTorch 수치 정확도](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) · [Google Cloud BF16 설명](https://docs.cloud.google.com/tpu/docs/bfloat16) · [vLLM 재현성](https://docs.vllm.ai/en/stable/usage/reproducibility/)

## 노트북 실행

선택한 가상환경에서 `python -m pip install -e '.[notebook]'` 후 VS Code 또는 Jupyter에서 그 환경을 커널로 선택합니다. 수집 모드에는 앞의 capture 의존성도 필요합니다. [실습 노트북](../notebooks/H200_Ironwood_Teacher_Forcing.ipynb)을 열고 `MODE`를 바꾼 뒤 전체 실행합니다. 코드와 결과는 CLI와 같습니다. Git에 올릴 때는 노트북 셀 출력을 비우고 실제 질문이나 로컬 계정 경로를 제거합니다.
