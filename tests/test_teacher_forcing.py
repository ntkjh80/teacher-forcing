"""실측 없이도 확인할 수 있는 위치 정렬, 보고서 및 이전 노트북 파일 호환성."""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teacher_forcing.__main__ import main
from teacher_forcing.capture import collect_path
from teacher_forcing.report import build_report, make_demo
from teacher_forcing.storage import PATHS, errors, output_directory, sha256, write_json


def test_demo_matches_published_report(tmp_path):
    output = tmp_path / "report"
    main(["demo", "--out", str(output)])
    expected = Path(__file__).resolve().parents[1] / "examples" / "demo_summary.md"
    assert (output / "summary.md").read_text().strip() == expected.read_text().strip()
    with (output / "tokens.csv").open(encoding="utf-8-sig") as f:
        tokens = list(csv.DictReader(f))
    assert len(tokens) == 6
    different = [r for r in tokens if r["same_top1"] == "False"]
    assert [(r["path"], r["position"]) for r in different] == [("cached", "2")]
    with pytest.raises(FileExistsError):
        build_report(out=output)


def legacy_files(root):
    # 원래 노트북의 case.json, meta.json, npz 계약을 그대로 사용한다.
    case, runs = make_demo()
    case.update(prompt_ids=[1], checkpoint_sha256="synthetic")
    write_json(root / "case.json", case)
    for side in ("gpu", "tpu"):
        folder = root / side
        folder.mkdir()
        metadata = dict(
            side=side,
            case_sha256=sha256(root / "case.json"),
            checkpoint_sha256="synthetic",
            repeats=3,
            stack="Transformers",
            transformers="fixture",
            attention="eager",
            parameter_dtype="bf16",
            hook_logit_max_abs={"prefill": 0, "cached": 0},
        )
        write_json(folder / "meta.json", metadata)
        for path in PATHS:
            for i, tensors in enumerate(runs[side, path]):
                np.savez(folder / f"{path}_{i}.npz", **tensors)


def test_previous_notebook_files_and_corrupt_input(tmp_path):
    legacy_files(tmp_path)
    main(["compare", "--root", str(tmp_path)])
    assert {p.name for p in (tmp_path / "report").iterdir()} == {
        "summary.md",
        "layers.csv",
        "tokens.csv",
        "repeats.csv",
    }
    case = tmp_path / "case.json"
    case.write_text(case.read_text() + "\n")
    with pytest.raises(ValueError, match="기준 입력"):
        build_report(tmp_path, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_no_partial_results(tmp_path):
    destination = tmp_path / "interrupted"
    with pytest.raises(RuntimeError):
        with output_directory(destination) as work:
            (work / "first").write_text("partial")
            raise RuntimeError()
    assert not destination.exists()


@pytest.mark.parametrize("reference", [[2, 3, 4], [2]])
def test_cached_and_prefill_predict_same_positions(reference):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
            self.model.norm = torch.nn.Identity()

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=False):
            # 누적 one-hot 벡터는 한 번에 계산하거나 캐시에 이어 계산할 때 같은 결과를 낸다.
            values = torch.nn.functional.one_hot(input_ids, num_classes=5).float()
            if past_key_values is None:
                past = torch.zeros_like(values[:, :1])
            else:
                past = past_key_values
                assert input_ids.shape[1] == 1
            values = values.cumsum(1) + past
            assert int(values[0, -1].sum()) == attention_mask.shape[1]
            for block in self.model.layers:
                values = block(values)
            logits = self.model.norm(values)
            return SimpleNamespace(
                logits=logits, past_key_values=values[:, -1:] if use_cache else None
            )

    case = {"prompt_ids": [0, 1], "reference_ids": reference}
    model = Model()
    prefill = collect_path(torch, model, case, "cpu", "prefill", True)
    cached = collect_path(torch, model, case, "cpu", "cached", True)
    plain = collect_path(torch, model, case, "cpu", "prefill", False)
    np.testing.assert_array_equal(plain["logits"], prefill["logits"])
    assert prefill.keys() == cached.keys() == {"block_000", "block_001", "final_norm", "logits"}
    for name in prefill:
        assert prefill[name].shape == (len(reference), 5)
        np.testing.assert_array_equal(prefill[name], cached[name])
    np.testing.assert_array_equal(prefill["logits"][0], [1, 1, 0, 0, 0])


def test_zero_norm_and_nonfinite():
    assert errors(np.array([0.0]), np.array([1.0])) == (None, 1.0)
    with pytest.raises(ValueError, match="NaN"):
        errors(np.array([np.nan]), np.array([0.0]))
