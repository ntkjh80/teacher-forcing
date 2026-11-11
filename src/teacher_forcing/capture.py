"""고정 토큰으로 Qwen의 프리필 및 캐시 디코드를 계측한다."""

import hashlib
import os
import tempfile
from importlib import metadata
from pathlib import Path

import numpy as np

from .storage import PATHS, errors, output_directory, read_json, sha256, write_json


def checkpoint_hash(folder):
    files = sorted(folder.glob("*.safetensors"))
    if not files or not (folder / "config.json").is_file():
        raise ValueError("config.json과 safetensors 가중치가 있는 원본 모델 폴더를 지정하세요.")
    files += [p for p in sorted(folder.glob("*.json")) if p.name != "download_identity.json"]
    h = hashlib.sha256()
    for path in files:
        h.update((path.name + ":" + sha256(path) + "\n").encode())
    return h.hexdigest()


def package_ver(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_path(torch, model, case, device, path, inspect_layers):
    """같은 응답 위치를 원패스와 캐시 재사용 경로에서 각각 추출한다."""
    if path not in PATHS:
        raise ValueError(f"지원하지 않는 실행 경로: {path}")
    prompt, reference = case["prompt_ids"], case["reference_ids"]
    selected = slice(None)
    buffers = {}
    handles = []

    def keep(name, tensor):
        value = tensor[0, selected, :].detach().float().cpu().numpy().copy()
        buffers.setdefault(name, []).append(value)

    def attach(name):
        def capture(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            keep(name, value)

        return capture

    try:
        if inspect_layers:
            for number, block in enumerate(model.model.layers):
                handles.append(block.register_forward_hook(attach(f"block_{number:03d}")))
            handles.append(model.model.norm.register_forward_hook(attach("final_norm")))
        with torch.inference_mode():
            if path == "prefill":
                ids = torch.tensor([prompt + reference[:-1]], dtype=torch.long, device=device)
                # 첫 기준 토큰의 로짓은 프롬프트 마지막 위치에서 읽는다.
                selected = slice(len(prompt) - 1, None)
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                keep("logits", output.logits)
            else:
                selected = slice(-1, None)
                cache = None
                # 예측이 달라져도 다음 입력은 case.json의 기준 토큰을 사용한다.
                chunks = [prompt] + [[token] for token in reference[:-1]]
                total = 0
                for chunk in chunks:
                    ids = torch.tensor([chunk], dtype=torch.long, device=device)
                    total += len(chunk)
                    mask = torch.ones((1, total), dtype=torch.long, device=device)
                    output = model(
                        input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True
                    )
                    keep("logits", output.logits)
                    cache = output.past_key_values
                    if cache is None:
                        raise RuntimeError("모델이 KV 캐시를 반환하지 않았습니다.")
    finally:
        for handle in handles:
            handle.remove()
    return {name: np.concatenate(parts, axis=0) for name, parts in buffers.items()}


def capture_device(side, root, model_dir, *, question=None, max_reference_tokens=None, repeats=3):
    """GPU가 기준 응답을 준비하고, 두 가속기가 독립적으로 같은 토큰열을 채점한다."""
    root, model_dir = Path(root), Path(model_dir)
    if side not in {"gpu", "tpu"}:
        raise ValueError("장비는 gpu 또는 tpu여야 합니다.")
    if repeats < 2 or (max_reference_tokens is not None and max_reference_tokens < 1):
        raise ValueError("반복은 2 이상, 기준 응답 토큰 수는 1 이상이어야 합니다.")
    casefile = root / "case.json"
    create_case = not casefile.exists()
    if create_case and side == "tpu":
        raise FileNotFoundError("H200에서 만든 case.json을 실험 폴더로 복사하세요.")
    if not create_case and (question is not None or max_reference_tokens is not None):
        raise ValueError(
            "기존 case.json을 사용합니다. 새 질문이나 길이는 새 실험 폴더에 지정하세요."
        )
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if side == "tpu":
        import torch_tpu  # noqa: F401  # 네이티브 TPU 장치 등록
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA 장치를 찾지 못했습니다.")
    device = torch.device("cuda:0" if side == "gpu" else "tpu:0")
    root.mkdir(parents=True, exist_ok=True)
    if (root / side).exists():
        raise FileExistsError(f"{root / side}가 이미 있습니다. 새 실험 폴더를 지정하세요.")
    config = read_json(model_dir / "config.json")
    if config.get("model_type") != "qwen2" or config.get("quantization_config"):
        raise ValueError("수집기는 비양자화 Qwen2/Qwen2.5 dense 체크포인트용입니다.")
    identity = checkpoint_hash(model_dir)
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    if model.config.model_type != "qwen2" or next(model.parameters()).device.type != device.type:
        raise ValueError("이 예제는 Qwen2 계열의 지정된 단일 장비 진단용입니다.")
    if side == "gpu" and create_case:
        question = question if question is not None else "대한민국의 수도는 어디인가요?"
        max_reference_tokens = max_reference_tokens or 8
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        text = tok.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            seq = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=max_reference_tokens,
            )
        case = {
            "sample": "case01",
            "checkpoint_sha256": identity,
            "question": question,
            "prompt_ids": ids[0].tolist(),
            "reference_ids": seq[0, ids.shape[1] :].tolist(),
        }
        # 동시 실행에서도 한 기준 응답만 공개한다. 기존 입력은 덮어쓰지 않는다.
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            source = Path(tmp) / "case.json"
            write_json(source, case)
            os.link(source, casefile)
    case = read_json(casefile)
    if identity != case["checkpoint_sha256"] or not case["prompt_ids"] or not case["reference_ids"]:
        raise ValueError("체크포인트 불일치 또는 빈 토큰열")
    with output_directory(root / side) as out:
        hook_delta = {}
        for path in PATHS:
            plain = collect_path(torch, model, case, device, path, False)["logits"]
            for rep in range(repeats):
                data = collect_path(torch, model, case, device, path, True)
                if rep == 0:
                    hook_delta[path] = errors(plain, data["logits"])[1]
                np.savez(out / f"{path}_{rep}.npz", **data)
        meta = {
            "side": side,
            "case_sha256": sha256(casefile),
            "checkpoint_sha256": identity,
            "repeats": repeats,
            "stack": "Transformers",
            "transformers": transformers.__version__,
            "torch": str(torch.__version__),
            "torch_tpu": package_ver("torch-tpu"),
            "libtpu": package_ver("libtpu"),
            "device": str(device),
            "accelerator": torch.cuda.get_device_name(0) if side == "gpu" else str(device),
            "attention": model.config._attn_implementation,
            "parameter_dtype": str(next(model.parameters()).dtype),
            "hook_logit_max_abs": hook_delta,
        }
        write_json(out / "meta.json", meta)
    print(f"{side} 수집 완료. 샘플 {case['sample']}, 응답 토큰 {len(case['reference_ids'])}개.")
    print("계측 OFF/ON 로짓 최대 절대차:", hook_delta)
