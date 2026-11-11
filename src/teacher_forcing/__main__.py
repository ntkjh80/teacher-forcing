"""티처 포싱 명령줄 진입점. 분석 PC에서는 NumPy만 사용한다."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="H200·Ironwood 티처 포싱 수치 비교")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="합성 예제에서 결과표 읽기")
    demo.add_argument("--out", type=Path, help="새 보고서 폴더")
    default_root = Path("runs/teacher-forcing")
    capture = commands.add_parser("capture", help="가속기 한 대에서 기준 응답 위치별 텐서 수집")
    capture.add_argument("--device", choices=("gpu", "tpu"), required=True)
    capture.add_argument("--model-dir", type=Path, required=True, help="원본 체크포인트")
    capture.add_argument("--root", type=Path, default=default_root)
    capture.add_argument("--question", help="GPU에서 새 case.json을 만들 질문")
    capture.add_argument("--max-reference-tokens", type=int, help="새 기준 응답 길이 상한 (기본 8)")
    capture.add_argument("--repeats", type=int, default=3)
    compare = commands.add_parser("compare", help="case.json, gpu/, tpu/에서 보고서 생성")
    compare.add_argument("--root", type=Path, default=default_root)
    compare.add_argument("--out", type=Path, help="새 보고서 폴더 (기본 ROOT/report)")
    args = parser.parse_args(argv)
    if args.command == "capture":
        if args.device == "tpu" and (
            args.question is not None or args.max_reference_tokens is not None
        ):
            parser.error("질문과 기준 응답 길이는 GPU에서 입력을 만들 때 지정하세요.")
        from .capture import capture_device

        capture_device(
            args.device,
            args.root,
            args.model_dir,
            question=args.question,
            max_reference_tokens=args.max_reference_tokens,
            repeats=args.repeats,
        )
    else:
        from .report import build_report

        if args.command == "demo":
            print(build_report(out=args.out), end="")
        else:
            destination = args.out if args.out is not None else args.root / "report"
            print(build_report(args.root, destination), end="")
            print(f"보고서: {destination / 'summary.md'}")


if __name__ == "__main__":
    main()
