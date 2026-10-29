"""Install CTRAIN dependencies that cannot be declared on PyPI."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys


REQUIRED_MODULES = ("auto_LiRPA", "onnx2pytorch", "autoattack")
GIT_DEPENDENCIES = (
    "git+https://github.com/KaidiXu/onnx2pytorch@"
    "8447c42c3192dad383e5598edc74dddac5706ee2",
    "git+https://github.com/Verified-Intelligence/auto_LiRPA.git@"
    "ca767f1d8c0a6b125a292ba165adb2319bbaf615",
    "git+https://github.com/fra31/auto-attack.git@"
    "a39220048b3c9f2cca9a4d3a54604793c68eca7e",
)


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line options for installing CTRAIN's pinned Git dependencies."""
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
    """Install the pinned Git-hosted packages in the active Python environment.

    Args:
        with_deps (bool): Whether pip may resolve transitive dependencies; False adds
            --no-deps.
        dry_run (bool): Whether to print commands without running them.
    Installation invokes pip through the current interpreter and propagates
    subprocess failures to the caller.
    """
    base_command = [sys.executable, "-m", "pip", "install"]
    if not with_deps:
        base_command.append("--no-deps")

    for dependency in GIT_DEPENDENCIES:
        command = [*base_command, dependency]
        print(" ".join(command))
        if not dry_run:
            subprocess.check_call(command)


def missing_git_dependency_modules() -> list[str]:
    """Return names of Git-hosted dependencies that cannot be found by the active interpreter."""
    return [
        module_name
        for module_name in REQUIRED_MODULES
        if importlib.util.find_spec(module_name) is None
    ]


def ensure_git_dependencies(*, with_deps: bool = False, dry_run: bool = False) -> None:
    """Install pinned Git packages when any required module is missing.

    Module discovery checks import availability, not the installed revision.
    with_deps and dry_run are forwarded to install_git_dependencies.
    """
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
    """Run the Git-dependency installer from command-line arguments and return zero on success."""
    args = build_parser().parse_args(argv)
    install_git_dependencies(with_deps=args.with_deps, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
