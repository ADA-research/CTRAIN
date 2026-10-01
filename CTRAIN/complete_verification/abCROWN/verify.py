import json
import os
import time
import pickle
import subprocess
import traceback
import sys
import signal
import tempfile
from pathlib import Path

import yaml

import torch
from CTRAIN.verification_systems.abCROWN.complete_verifier.abcrown import ABCROWN
from CTRAIN.verification_systems.abCROWN.complete_verifier.read_vnnlib import read_vnnlib
from CTRAIN.complete_verification.abCROWN.util import get_abcrown_standard_conf

MAX_LOSS = 10 ** 10

# TODO: automatically point to correct runner path inside of CTRAIN
def limited_abcrown_eval(work_dir=None, runner_path=None, *args, **kwargs):
    """Run abCROWN in this interpreter; distinguish outer timeouts from failures."""
    timeout = kwargs['timeout'] * 1.2
    runner_path = runner_path or str(Path(__file__).with_name('runner.py'))
    start = time.monotonic()
    with tempfile.TemporaryDirectory(dir=work_dir, prefix='ctrain-abcrown-') as directory:
        args_path = Path(directory) / 'args.pkl'
        result_path = Path(directory) / 'result.pkl'
        with args_path.open('wb') as file:
            pickle.dump((args, kwargs), file)
        try:
            process = subprocess.Popen([sys.executable, runner_path, str(args_path), str(result_path)], start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                return time.monotonic() - start, 'timeout'
            if process.returncode != 0:
                return time.monotonic() - start, 'unknown'
            with result_path.open('rb') as file:
                return pickle.load(file)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            return time.monotonic() - start, 'unknown'


def abcrown_eval(*args, **kwargs):
    """Run with isolated temporary files; failures never become certificates."""
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='ctrain-abcrown-instance-') as directory:
        try:
            return _abcrown_eval(*args, **kwargs, _work_dir=directory)
        except TimeoutError:
            return time.monotonic() - start, 'timeout'
        except Exception:
            traceback.print_exc()
            return time.monotonic() - start, 'unknown'


def _abcrown_eval(config, seed, instance, vnnlib_path='../../vnnlib/', model_name='mnist_6_100', model_path='./abCROWN/complete_verifier/models/eran/mnist_6_100_nat.pth', model_onnx_path=None, input_shape=[-1, 1, 28, 28], timeout=600, no_cores=28, par_factor=10, _work_dir=None):
    """
    Runs the abCROWN verification process with the given configuration.
    abCROWN is invoked from inside the program code, so a crash/freeze can only be handled partially.

    Args:
        config (dict): Configuration dictionary for the verification process.
        seed (int): Seed for random number generation.
        instance (str): Path to the VNN-LIB instance file.
        vnnlib_path (str, optional): Path prefix for VNN-LIB files. Defaults to '../../vnnlib/'.
        model_name (str, optional): Name of the model to be verified. Defaults to 'mnist_6_100'.
        model_path (str, optional): Path to the model file. Defaults to './abCROWN/complete_verifier/models/eran/mnist_6_100_nat.pth'.
        model_onnx_path (str, optional): Path to the ONNX model file. Defaults to None.
        input_shape (list, optional): Shape of the input tensor. Defaults to [-1, 1, 28, 28].
        timeout (int, optional): Timeout for the verification process in seconds. Defaults to 600.
        no_cores (int, optional): Number of CPU cores to use for parallel solvers, when abCROWN is configured to use MIP Solvers. Defaults to 28.
        par_factor (int, optional): Penalty factor for running time in case of timeout. Defaults to 10.

    Returns:
        (tuple): Running time of the verification process and the result of the verification (sat/unsat or timeout/unknown).
    """
    print(config, seed, instance)
    import copy
    std_conf = copy.deepcopy(config)
    std_conf.setdefault('bab', {})['hugetensor_allocator'] = False

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    timestamp = time.time()

    std_conf['model']['name'] = model_name
    std_conf['model']['path'] = f'/tmp/{model_name}.pth' if model_name is not None else None
    std_conf['model']['onnx_path'] = model_onnx_path if model_onnx_path is not None else None
    std_conf['model']['input_shape'] = input_shape

    std_conf['general']['device'] = device

    std_conf['bab']['timeout'] = timeout

    if not std_conf['solver'].get('mip'):
        std_conf['solver']['mip'] = get_abcrown_standard_conf(timeout=timeout, no_cores=no_cores)['solver']['mip']
    std_conf['solver']['mip']['parallel_solvers'] = no_cores

    std_conf['specification']['vnnlib_path_prefix'] = vnnlib_path
    std_conf['specification']['vnnlib_path'] = instance
    std_conf['general']['output_file'] = os.path.join(_work_dir, 'out.pkl')
    std_conf['general']['results_file'] = os.path.join(_work_dir, 'status.txt')
    std_conf['general']['save_output'] = True

    print(json.dumps(config, indent=2))

    with open(os.path.join(_work_dir, 'config.yaml'), "w", encoding='u8') as f:
        yaml.dump(std_conf, f)

    abcrown_instance = ABCROWN(
        ['--config', os.path.join(_work_dir, 'config.yaml')]
    )

    # Precompile VNN-LIB s.t. each run can access the cache
    _ = read_vnnlib(instance)

    start_time = time.time()
    try:
        verification_res = abcrown_instance.main()
    except TimeoutError:
        raise
    except Exception as e:
        print(type(e), e)
        print(traceback.format_exc())
        return time.time() - start_time, 'unknown'
    end_time = time.time()


    with open(os.path.join(_work_dir, 'out.pkl'), 'rb') as f:
        result_dict = pickle.load(f)

    result = result_dict['results']

    elapsed = end_time - start_time
    # abCROWN's logger maps all unresolved statuses to timeout. Only retain
    # that label when its recorded solver time reached the configured budget.
    solver_time = result_dict.get('time', elapsed)
    budget = getattr(getattr(abcrown_instance, 'logger', None), 'timeout_threshold', timeout)
    if result in ('unknown', 'timeout'):
        result = 'timeout' if solver_time >= budget else 'unknown'
    running_time = elapsed * par_factor if result == 'timeout' else elapsed
    return running_time, result
