"""수치 비교와 완료된 실행 결과의 저장."""

import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PATHS = ("prefill", "cached")


def errors(gpu, tpu):
    """GPU 기준 상대 L2(%)와 최대 절대차를 계산한다. 영벡터는 절대차만 유효하다."""
    if gpu.shape != tpu.shape or not gpu.size:
        raise ValueError(f"텐서 모양 불일치 또는 빈 텐서: {gpu.shape} / {tpu.shape}")
    gpu = gpu.astype(np.float64, copy=False)
    tpu = tpu.astype(np.float64, copy=False)
    if not (np.isfinite(gpu).all() and np.isfinite(tpu).all()):
        raise ValueError("NaN/Inf가 있습니다. 원본 텐서를 확인하세요.")
    delta = tpu - gpu
    norm = np.linalg.norm(gpu)
    relative = float(np.linalg.norm(delta) / norm * 100) if norm else None
    return relative, float(np.abs(delta).max())


@contextmanager
def output_directory(destination):
    """모든 파일을 작성한 후 결과 폴더를 공개해 중단된 실행과 정상 결과를 구분한다."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"결과가 이미 있습니다: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staging = Path(tmp) / "output"
        staging.mkdir()
        yield staging
        staging.rename(destination)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
