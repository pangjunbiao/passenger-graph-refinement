"""Sole command-line entry point for the clean SC-HTM V6 project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fail-closed stage of the clean SC-HTM V6 pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=(
            "step1",
            "step2",
            "step3",
            "step4",
            "step4r2",
            "step5r2",
            "step6",
            "step6r2",
            "step7",
            "step8",
            "step9",
            "step10",
            "step11",
            "step12",
            "step13",
            "step11b",
            "step14",
        ),
        default="step1",
        help="Implemented V6 stage (default: step1).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="V6 project root (default: directory containing main_v6.py).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Step configuration (default: the selected stage's frozen YAML).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Read-only sibling SC-HTM data/source project for Steps 2–3, 9–10, "
            "12–13 "
            "(default: source_project_root in the selected stage YAML)."
        ),
    )
    parser.add_argument(
        "--encot-source",
        type=Path,
        default=None,
        help=(
            "Existing clean official EnCOT checkout for Step 3. If omitted, "
            "V6 first searches the sibling SC-HTM project and then clones the "
            "locked commit into V6 without changing either Python environment."
        ),
    )
    parser.add_argument(
        "--encot-python",
        type=Path,
        default=None,
        help=(
            "Existing exact EnCOT Python interpreter for Step 3. If omitted, "
            "V6 searches the prior SC-HTM CUDA cache. No package is installed."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    if args.stage == "step1":
        config = (
            args.config or root / "configs" / "v6" / "step01.yaml"
        ).expanduser().resolve()
        from src.v6.step1 import run_v6_step1

        run_v6_step1(project_root=root, config_path=config)
        return 0
    if args.stage == "step2":
        config = (
            args.config or root / "configs" / "v6" / "step02.yaml"
        ).expanduser().resolve()
        from src.v6.step2 import run_v6_step2

        run_v6_step2(
            project_root=root,
            config_path=config,
            source_root=args.source_root,
        )
        return 0
    if args.stage == "step3":
        config = (
            args.config or root / "configs" / "v6" / "step03.yaml"
        ).expanduser().resolve()
        from src.v6.step3 import run_v6_step3

        run_v6_step3(
            project_root=root,
            config_path=config,
            source_root=args.source_root,
            encot_source=args.encot_source,
            encot_python=args.encot_python,
        )
        return 0
    if args.stage == "step4":
        config = (
            args.config or root / "configs" / "v6" / "step04.yaml"
        ).expanduser().resolve()
        from src.v6.development_stage import run_v6_step4

        run_v6_step4(project_root=root, config_path=config)
        return 0
    if args.stage == "step4r2":
        config = (
            args.config or root / "configs" / "v6" / "step04r2.yaml"
        ).expanduser().resolve()
        from src.v6.step4r2 import run_v6_step4r2

        run_v6_step4r2(project_root=root, config_path=config)
        return 0
    if args.stage == "step5r2":
        config = (
            args.config or root / "configs" / "v6" / "step05r2.yaml"
        ).expanduser().resolve()
        from src.v6.step5r2 import run_v6_step5r2

        run_v6_step5r2(project_root=root, config_path=config)
        return 0
    if args.stage == "step6":
        config = (
            args.config or root / "configs" / "v6" / "step06.yaml"
        ).expanduser().resolve()
        from src.v6.step6 import run_v6_step6

        run_v6_step6(project_root=root, config_path=config)
        return 0
    if args.stage == "step6r2":
        config = (
            args.config or root / "configs" / "v6" / "step06r2.yaml"
        ).expanduser().resolve()
        from src.v6.step6r2 import run_v6_step6r2

        run_v6_step6r2(project_root=root, config_path=config)
        return 0
    if args.stage == "step7":
        config = (
            args.config or root / "configs" / "v6" / "step07.yaml"
        ).expanduser().resolve()
        from src.v6.step7 import run_v6_step7

        run_v6_step7(project_root=root, config_path=config)
        return 0
    if args.stage == "step8":
        config = (
            args.config or root / "configs" / "v6" / "step08.yaml"
        ).expanduser().resolve()
        from src.v6.step8 import run_v6_step8

        run_v6_step8(project_root=root, config_path=config)
        return 0
    if args.stage == "step9":
        config = (
            args.config or root / "configs" / "v6" / "step09.yaml"
        ).expanduser().resolve()
        from src.v6.step9 import run_v6_step9

        run_v6_step9(
            project_root=root,
            config_path=config,
            source_root=args.source_root,
        )
        return 0
    if args.stage == "step10":
        config = (
            args.config or root / "configs" / "v6" / "step10.yaml"
        ).expanduser().resolve()
        from src.v6.step10 import run_v6_step10

        run_v6_step10(
            project_root=root,
            config_path=config,
            source_root=args.source_root,
        )
        return 0
    if args.stage == "step11":
        config = (
            args.config or root / "configs" / "v6" / "step11.yaml"
        ).expanduser().resolve()
        from src.v6.step11 import run_v6_step11

        run_v6_step11(project_root=root, config_path=config)
        return 0
    if args.stage == "step12":
        config = (
            args.config or root / "configs" / "v6" / "step12.yaml"
        ).expanduser().resolve()
        from src.v6.step12 import run_v6_step12

        run_v6_step12(
            project_root=root,
            config_path=config,
            source_root=args.source_root,
        )
        return 0
    if args.stage == "step13":
        config = (
            args.config or root / "configs" / "v6" / "step13.yaml"
        ).expanduser().resolve()
        from src.v6.step13 import run_v6_step13

        run_v6_step13(
            project_root=root,
            config_path=config,
            source_root=args.source_root,
        )
        return 0
    if args.stage == "step11b":
        config = (
            args.config or root / "configs" / "v6" / "step11b.yaml"
        ).expanduser().resolve()
        from src.v6.step11b import run_v6_step11b

        run_v6_step11b(project_root=root, config_path=config)
        return 0
    if args.stage == "step14":
        config = (
            args.config or root / "configs" / "v6" / "step14.yaml"
        ).expanduser().resolve()
        from src.v6.step14 import run_v6_step14

        run_v6_step14(project_root=root, config_path=config)
        return 0
    raise AssertionError("Unreachable stage")


if __name__ == "__main__":
    raise SystemExit(main())
