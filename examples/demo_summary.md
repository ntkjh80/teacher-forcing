# 합성 예시 결과 · 실측 아님

## 같은 입력에서의 다음 토큰

| 경로 | 비교 위치 | Top-1 일치 | 첫 불일치 위치 | 로짓 최대 절대차 |
| --- | --- | --- | --- | --- |
| prefill | 3 | 3 | — | 0 |
| cached | 3 | 2 | 2 | 0.02 |
| cached KV 재사용 위치만 | 2 | 1 | 2 | 0.02 |

## 상세 사례 · cached, 응답 2번째

| 기준 토큰 ID | GPU 1위 ID | TPU 1위 ID | GPU 1·2위 차이 | TPU 1·2위 차이 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1 | 0.01 | 0.03 |

| 측정 경계 | 상대 L2(%) | 최대 절대차 |
| --- | --- | --- |
| block_000 | 0 | 0 |
| block_001 | 1.22859 | 0.02 |
| final_norm | 1.22859 | 0.02 |

## 장비 자체 반복

| side | path | boundary | max_abs_vs_first_repeat |
| --- | --- | --- | --- |
| gpu | prefill | logits | 0 |
| tpu | prefill | logits | 0 |
| gpu | cached | logits | 0 |
| tpu | cached | logits | 0 |

## 판정

로짓·토큰 표는 고정 문맥의 수치 비교다. 서비스 이식 판정에는 자유 생성과 사내 정답 채점이 필요하다.
