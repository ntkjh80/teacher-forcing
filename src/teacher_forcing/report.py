"""동일한 응답 위치의 원본 로짓, 블록 경계, 장비 자체 반복을 비교한다."""

import csv

import numpy as np

from .storage import PATHS, errors, output_directory, read_json, sha256


def top_two(vector):
    # 동점은 작은 token ID를 먼저 고른다. 동점 여부는 margin=0으로 확인한다.
    first = int(np.argmax(vector))
    temp = vector.copy()
    temp[first] = -np.inf
    second = int(np.argmax(temp))
    return first, float(vector[first]) - float(vector[second])


def make_demo():
    # 사람이 결과표를 읽어볼 수 있도록 직접 정한 단위 없는 점수다.
    golden = np.array([[3.0, 2.0, 0.1], [1.01, 1.0, 0.0], [0.2, 0.3, 1.7]], dtype=float)
    states = np.array([[1.0, 1.1], [1.2, 1.1], [1.0, 1.3]], dtype=float)
    values = {}
    for side in ("gpu", "tpu"):
        for path in PATHS:
            group = []
            for rep in range(3):
                change = side == "tpu" and path == "cached"
                logits = golden.copy()
                tail = states.copy()
                if change:
                    logits[1] = [0.99, 1.02, 0.0]
                    tail[1, 0] += 0.02
                group.append(
                    {
                        "block_000": states.copy(),
                        "block_001": tail,
                        "final_norm": tail.copy(),
                        "logits": logits,
                    }
                )
            values[side, path] = group
    return {"sample": "가상 예시", "reference_ids": [0, 0, 2]}, values


def load_captures(folder):
    case = read_json(folder / "case.json")
    if not case["reference_ids"] or not case["prompt_ids"]:
        raise ValueError("기준 응답 또는 프롬프트가 비어 있습니다.")
    values = {}
    configs = {s: read_json(folder / s / "meta.json") for s in ("gpu", "tpu")}
    for field in (
        "case_sha256",
        "checkpoint_sha256",
        "repeats",
        "stack",
        "transformers",
        "attention",
        "parameter_dtype",
    ):
        if configs["gpu"][field] != configs["tpu"][field]:
            raise ValueError(f"비교 조건 불일치: {field}. 두 meta.json을 확인하세요.")
    for side, meta in configs.items():
        if (
            meta["side"] != side
            or meta["case_sha256"] != sha256(folder / "case.json")
            or meta["checkpoint_sha256"] != case["checkpoint_sha256"]
        ):
            raise ValueError(f"{side} 실행과 기준 입력이 일치하지 않습니다.")
        if meta["repeats"] < 2:
            raise ValueError("자체 반복 편차 계산에는 2회 이상 측정이 필요합니다.")
        print(f"{side} 환경:", meta)
        if any(value > 0 for value in meta["hook_logit_max_abs"].values()):
            print(f"{side}: 계측 ON/OFF 출력에 차이가 있습니다. 원인 해석 전에 확인하세요.")
        for path in PATHS:
            group = []
            for rep in range(meta["repeats"]):
                with np.load(folder / side / f"{path}_{rep}.npz", allow_pickle=False) as data:
                    group.append({key: data[key] for key in data.files})
            values[side, path] = group
    return case, values, configs


def compare_case(case, values):
    tokens, layers, repeats, summary = [], [], [], []
    for path in PATHS:
        gpu, tpu = values["gpu", path][0], values["tpu", path][0]
        if not {"logits", "block_000", "final_norm"}.issubset(gpu) or set(gpu) != set(tpu):
            raise ValueError("양쪽 수집 경계가 다릅니다.")
        for name in gpu:
            if (
                gpu[name].ndim != 2
                or gpu[name].shape != tpu[name].shape
                or gpu[name].shape[0] != len(case["reference_ids"])
            ):
                raise ValueError(f"위치 또는 모양 불일치: {path}/{name}")
        for side in ("gpu", "tpu"):
            group = values[side, path]
            if len(group) < 2 or any(run.keys() != gpu.keys() for run in group):
                raise ValueError(f"반복 수 또는 계측 경계가 일치하지 않습니다: {side}/{path}")
            for name, tensor in group[0].items():
                delta = max(errors(tensor, run[name])[1] for run in group[1:])
                repeats.append(
                    {"side": side, "path": path, "boundary": name, "max_abs_vs_first_repeat": delta}
                )
        for i, ref in enumerate(case["reference_ids"]):
            if not 0 <= ref < gpu["logits"].shape[-1]:
                raise ValueError(f"유효 어휘 밖의 기준 토큰 ID: {ref}")
            gpu_logits, tpu_logits = gpu["logits"][i], tpu["logits"][i]
            rel, maximum = errors(gpu_logits, tpu_logits)
            gpu_first, gpu_margin = top_two(gpu_logits)
            tpu_first, tpu_margin = top_two(tpu_logits)
            tokens.append(
                {
                    "sample": case["sample"],
                    "path": path,
                    "position": i + 1,
                    "kv_reused": path == "cached" and i > 0,
                    "reference_id": ref,
                    "gpu_top1_id": gpu_first,
                    "tpu_top1_id": tpu_first,
                    "same_top1": gpu_first == tpu_first,
                    "gpu_margin": gpu_margin,
                    "tpu_margin": tpu_margin,
                    "logit_max_abs": maximum,
                    "logit_rel_l2_pct": rel,
                    "reference_logit_delta_tpu_minus_gpu": float(tpu_logits[ref])
                    - float(gpu_logits[ref]),
                }
            )
            for name in sorted(set(gpu) - {"logits"}):
                rel, maximum = errors(gpu[name][i], tpu[name][i])
                layers.append(
                    {
                        "sample": case["sample"],
                        "path": path,
                        "position": i + 1,
                        "boundary": name,
                        "relative_l2_pct": rel,
                        "max_abs": maximum,
                    }
                )
        rows = [r for r in tokens if r["path"] == path]
        selections = [(path, rows)]
        if path == "cached":
            selections.append(("cached KV 재사용 위치만", rows[1:]))
        for label, subset in selections:
            misses = [r["position"] for r in subset if not r["same_top1"]]
            summary.append(
                {
                    "경로": label,
                    "비교 위치": len(subset),
                    "Top-1 일치": sum(r["same_top1"] for r in subset),
                    "첫 불일치 위치": misses[0] if misses else "—",
                    "로짓 최대 절대차": max((r["logit_max_abs"] for r in subset), default=None),
                }
            )
    return tokens, layers, repeats, summary


def as_table(rows):
    if not rows:
        return "비교할 위치가 없습니다."

    def fmt(x):
        return "—" if x is None else (f"{x:.6g}" if isinstance(x, float) else str(x))

    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    lines.extend("| " + " | ".join(fmt(row[k]) for k in keys) + " |" for row in rows)
    return "\n".join(lines)


def result_markdown(case, tokens, layers, repeats, summary, demo):
    title = "합성 예시 결과 · 실측 아님" if demo else "Transformers 진단 결과 · " + case["sample"]
    parts = ["# " + title, "## 같은 입력에서의 다음 토큰", as_table(summary)]
    changes = [r for r in tokens if not r["same_top1"]]
    chosen = changes[0] if changes else max(tokens, key=lambda r: r["logit_max_abs"])
    at = chosen["position"]
    path = chosen["path"]
    parts += [
        f"## 상세 사례 · {path}, 응답 {at}번째",
        as_table(
            [
                {
                    "기준 토큰 ID": chosen["reference_id"],
                    "GPU 1위 ID": chosen["gpu_top1_id"],
                    "TPU 1위 ID": chosen["tpu_top1_id"],
                    "GPU 1·2위 차이": chosen["gpu_margin"],
                    "TPU 1·2위 차이": chosen["tpu_margin"],
                }
            ]
        ),
    ]
    detail = [r for r in layers if r["path"] == path and r["position"] == at]
    parts += [
        as_table(
            [
                {
                    "측정 경계": r["boundary"],
                    "상대 L2(%)": r["relative_l2_pct"],
                    "최대 절대차": r["max_abs"],
                }
                for r in detail
            ]
        )
    ]
    brief = [r for r in repeats if r["boundary"] == "logits"]
    parts += [
        "## 장비 자체 반복",
        as_table(brief),
        "## 판정",
        "로짓·토큰 표는 고정 문맥의 수치 비교다. 서비스 이식 판정에는 자유 생성과 사내 정답 채점이 필요하다.",
    ]
    return "\n\n".join(parts) + "\n"


def build_report(root=None, out=None):
    """root가 없으면 합성 예제를 계산한다. out이 지정되면 Markdown·CSV를 저장한다."""
    if root is None:
        case, values = make_demo()
    else:
        case, values, configs = load_captures(root)
    tokens, layers, repeats, summary = compare_case(case, values)
    markdown = result_markdown(case, tokens, layers, repeats, summary, root is None)
    if root is not None:
        env = []
        for side, meta in configs.items():
            env.append(
                {
                    "장비": side,
                    "장치": meta.get("accelerator", meta.get("device")),
                    "torch": meta.get("torch"),
                    "Transformers": meta["transformers"],
                    "정밀도": meta["parameter_dtype"],
                    "반복": meta["repeats"],
                    "계측 전후 로짓 차이(프리필)": meta["hook_logit_max_abs"]["prefill"],
                    "계측 전후 로짓 차이(디코드)": meta["hook_logit_max_abs"]["cached"],
                }
            )
        markdown += "\n## 실행 환경\n\n" + as_table(env) + "\n"
    if out is not None:
        with output_directory(out) as folder:
            for name, rows in [("tokens", tokens), ("layers", layers), ("repeats", repeats)]:
                with (folder / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            (folder / "summary.md").write_text(markdown, encoding="utf-8")
    return markdown
