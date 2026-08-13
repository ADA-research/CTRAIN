"""Install CTRAIN dependencies that cannot be declared on PyPI."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys


REQUIRED_MODULES = ("auto_LiRPA", "onnx2pytorch")
GIT_DEPENDENCIES = (
    "git+https://github.com/KaidiXu/onnx2pytorch@"
    "8447c42c3192dad383e5598edc74dddac5706ee2",
    "git+https://github.com/Verified-Intelligence/auto_LiRPA.git@"
    "cf0169ce6bfb4fddd82cfff5c259c162a23ad03c",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install CTRAIN dependencies that are hosted on GitHub."
    )
    parser.add_argument(
        "--with-deps",
        action="store_true",
        help="Allow pip to resolve transitive dependencies for the Git packages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pip commands without running them.",
    )
    return parser


def install_git_dependencies(*, with_deps: bool = False, dry_run: bool = False) -> None:
    base_command = [sys.executable, "-m", "pip", "install"]
    if not with_deps:
        base_command.append("--no-deps")

    for dependency in GIT_DEPENDENCIES:
        command = [*base_command, dependency]
        print(" ".join(command))
        if not dry_run:
            subprocess.check_call(command)


def missing_git_dependency_modules() -> list[str]:
    return [
        module_name
        for module_name in REQUIRED_MODULES
        if importlib.util.find_spec(module_name) is None
    ]


def ensure_git_dependencies(*, with_deps: bool = False, dry_run: bool = False) -> None:
    missing_modules = missing_git_dependency_modules()
    if not missing_modules:
        return

    print(
        "Missing CTRAIN Git-hosted dependencies: "
        + ", ".join(missing_modules)
        + ". Installing now."
    )
    install_git_dependencies(with_deps=with_deps, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    install_git_dependencies(with_deps=args.with_deps, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
