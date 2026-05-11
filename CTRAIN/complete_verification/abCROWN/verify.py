import json
import os
import time
import pickle
import subprocess
import traceback

import yaml

import torch
from CTRAIN.verification_systems.abCROWN.complete_verifier.abcrown import ABCROWN
from CTRAIN.verification_systems.abCROWN.complete_verifier.read_vnnlib import read_vnnlib
from CTRAIN.complete_verification.abCROWN.util import get_abcrown_standard_conf

MAX_LOSS = 10 ** 10

# TODO: automatically point to correct runner path inside of CTRAIN
def limited_abcrown_eval(work_dir, runner_path='CTRAIN/complete_verification/abCROWN/runner.py', *args, **kwargs):
    """
    Executes the abCROWN verification using
    a specified verification process with a specified timeout and handles the results.
    This increases robustness, since a crash of abCROWN does not result in a crash of the
    python script.
    This function serializes the provided arguments and keyword arguments, runs a
    separate Python script to perform the verification, and handles the process
    execution including timeout management. The results are deserialized and returned
    if the process completes successfully within the timeout period.

    Args:
        work_dir (str): The working directory where temporary files will be stored.
        runner_path (str, optional): The path to the runner script that performs the
            verification. Defaults to 'src/complete_verification/abCROWN/runner.py'.
        *args: Additional positional arguments to be passed to the runner script.
        **kwargs: Additional keyword arguments to be passed to the runner script.
            Must include 'timeout' (float) which specifies the timeout period in seconds.

    Returns:
        (tuple): A tuple containing the running time and the result (sat/unsat or timeout/unknown) if the verification
                    completes successfully. If the verification fails or times out, returns
                    (MAX_LOSS, 'unknown').

    Raises:
        (Exception): If there is an error running the process.
    """
    outer_timeout = kwargs['timeout'] * 1.2

    timestamp = time.time()

    args_pkl_path = f'{work_dir}/args_{timestamp}.pkl'
    result_path = f"{work_dir}/result_{timestamp}.pkl"

    with open(f'{work_dir}/args_{timestamp}.pkl', "wb") as f:
        pickle.dump((args, kwargs), f)

    verification_ok = False

    runner_args = [args_pkl_path, result_path]

    try:
        print(f"Running {['python3', runner_path] + runner_args}")
        process = subprocess.Popen(
            ["python3", runner_path] + runner_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(timeout=outer_timeout)
            print("Function finished successfully.")
            print("Output:", stdout.decode())
            print("Error Output:", stderr.decode())
            verification_ok = True

        except subprocess.TimeoutExpired:
            print(f"Function exceeded timeout of {outer_timeout} seconds. Terminating...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Function did not terminate after SIGTERM. Killing...")
                process.kill()

    except Exception as e:
        print(f"Error running the process: {e}")

    if verification_ok:
        with open(result_path, 'rb') as f:
            running_time, result = pickle.load(f)

        return running_time, result

    return MAX_LOSS, 'unknown'


def abcrown_eval(config, seed, instance, vnnlib_path='../../vnnlib/', model_name='mnist_6_100', model_path='./abCROWN/complete_verifier/models/eran/mnist_6_100_nat.pth', model_onnx_path=None, input_shape=[-1, 1, 28, 28], timeout=600, no_cores=28, par_factor=10):
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
    std_conf = config

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
    std_conf['general']['output_file'] = f'/tmp/out_{timestamp}.pkl'
    std_conf['general']['save_output'] = True

    print(json.dumps(config, indent=2))

    with open(f"/tmp/conf_{timestamp}.yaml", "w", encoding='u8') as f:
        yaml.dump(std_conf, f)

    abcrown_instance = ABCROWN(
        ['--config', f'/tmp/conf_{timestamp}.yaml']
    )

    # Precompile VNN-LIB s.t. each run can access the cache
    _ = read_vnnlib(instance)

    start_time = time.time()
    try:
        verification_res = abcrown_instance.main()
    except Exception as e:
        print(type(e), e)
        print(traceback.format_exc())
        return MAX_LOSS, 'unknown'
    end_time = time.time()

    os.system(f'rm /tmp/conf_{timestamp}.yaml')

    with open(f'/tmp/out_{timestamp}.pkl', 'rb') as f:
        result_dict = pickle.load(f)

    result = result_dict['results']

    if result == 'unknown':
        print("PENALISING RUNNING TIME DUE TO TIMEOUT!")
        running_time = timeout * par_factor if timeout > (end_time - start_time) else (end_time - start_time) * par_factor
    else:
        running_time = end_time - start_time

    return running_time, result
