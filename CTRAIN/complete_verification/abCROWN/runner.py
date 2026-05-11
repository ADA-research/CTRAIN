import json
import pickle
import sys
import time
import traceback
import torch
import yaml
import os

from CTRAIN.verification_systems.abCROWN.complete_verifier.abcrown import ABCROWN
from CTRAIN.verification_systems.abCROWN.complete_verifier.read_vnnlib import read_vnnlib
from CTRAIN.complete_verification.abCROWN.util import get_abcrown_standard_conf

MAX_LOSS = 2**25


def run_abcrown_eval(config, seed, instance, vnnlib_path='../../vnnlib/', model_name='mnist_6_100', model_path='./abCROWN/complete_verifier/models/eran/mnist_6_100_nat.pth', model_onnx_path=None, input_shape=[-1, 1, 28, 28], timeout=600, no_cores=28, par_factor=10):
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


if __name__ == "__main__":
    args_path = sys.argv[1]
    result_path = sys.argv[2]

    with open(args_path, 'rb') as f:
        args, kwargs = pickle.load(f)

    running_time, result = run_abcrown_eval(*args, **kwargs)

    with open(result_path, 'wb') as f:
        pickle.dump((running_time, result), f)

    sys.exit(0)
