"""Run script entry points without leaking their global state into other tests."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = {"test_neuralsat_bundled.py"}
SCRIPTS = sorted(path for path in (ROOT / "tests").glob("test_*.py")
                 if path.name not in OPTIONAL | {Path(__file__).name})


def run_script(path, *args):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    # Assertions are the legacy scripts' test framework, even under python -O.
    env.pop("PYTHONOPTIMIZE", None)
    try:
        result = subprocess.run(
            [sys.executable, str(path), *args], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"{path.name} exceeded 300 seconds\n{error.stdout}")
    assert result.returncode == 0, result.stdout


@pytest.mark.regression
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_cpu_regression(script):
    run_script(script)


@pytest.mark.verifier
def test_bundled_neuralsat():
    run_script(ROOT / "tests/test_neuralsat_bundled.py")
