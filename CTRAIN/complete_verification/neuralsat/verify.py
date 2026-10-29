"""Process adapter for NeuralSAT's ONNX/VNN-LIB CLI (develop/src/main.py)."""
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def validate_neuralsat_command(command):
    """Resolve and validate the NeuralSAT executable, using bundled source by default."""
    if command is None:
        if sys.version_info < (3, 10):
            raise RuntimeError("Bundled NeuralSAT requires Python 3.10+; supply neuralsat_command for a newer interpreter")
        main = Path(__file__).resolve().parents[2] / 'verification_systems' / 'neuralsat' / 'src' / 'main.py'
        if not main.is_file():
            raise FileNotFoundError(f"Bundled NeuralSAT source is missing: {main}")
        command = [sys.executable, str(main)]
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("neuralsat_command must be a nonempty argv list, e.g. "
                         "['/path/venv/bin/python', '/path/neuralsat/src/main.py']")
    command = [os.fspath(part) for part in command]
    if shutil.which(command[0]) is None:
        raise FileNotFoundError(f"NeuralSAT executable not found: {command[0]}")
    return command


def validate_neuralsat_options(timeout, device, batch_size):
    """Validate the positive solver budget, supported device and parallel branch count."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('neuralsat_batch_size must be a positive integer')
    if str(device) not in ('cpu', 'cuda') and not str(device).startswith('cuda:'):
        raise ValueError('NeuralSAT supports cpu or cuda devices')


def neuralsat_eval(model_onnx_path, instance, command=None, timeout=1000,
                   device='cpu', batch_size=1000, log_path=None):
    """Evaluate an ONNX network and VNNLIB query using the NeuralSAT command line.

    The solver runs in a separate process and writes a structured result file.
    Failed processes or missing results remain unknown; log text is not treated
    as a certificate. Returns elapsed wall time and sat, unsat, unknown or timeout.

    NeuralSAT runs in its own environment. Failed processes and invalid/missing
    result files are unknown, even if stdout contains a proof-looking string.
    """
    command = validate_neuralsat_command(command)
    validate_neuralsat_options(timeout, device, batch_size)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='ctrain-neuralsat-') as work_dir:
        result_path = Path(work_dir) / 'result.txt'
        argv = command + [
            '--net', str(Path(model_onnx_path).resolve()),
            '--spec', str(Path(instance).resolve()),
            '--timeout', str(timeout), '--device', str(device),
            '--batch', str(batch_size), '--result_file', str(result_path),
        ]
        with open(log_path or os.devnull, 'w') as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # NeuralSAT may launch workers; stop the whole process group.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                return time.monotonic() - start, 'timeout'
        status = 'unknown'
        if process.returncode == 0 and result_path.is_file():
            lines = result_path.read_text().splitlines()
            token = lines[0].split(',')[0].strip() if lines else ''
            if token in ('sat', 'unsat', 'unknown', 'timeout'):
                status = token
        return time.monotonic() - start, status
