"""Check suite case selection, option forwarding, cover reuse and output safety."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark = root / 'benchmark'
        (benchmark / 'onnx').mkdir(parents=True)
        (benchmark / 'vnnlib').mkdir()
        rows = []
        for network in ('2_1', '3_9', '4_5', '5_9'):
            model = f'onnx/ACASXU_run2a_{network}_batch_2000.onnx'
            (benchmark / model).touch()  # Dry-run validates presence, not ONNX conversion.
            for prop in ([1, 2, 3, 4, 10] if network == '4_5' else [1, 2, 3, 4]):
                (benchmark / f'vnnlib/prop_{prop}.vnnlib').touch()
                rows.append(f'{model},vnnlib/prop_{prop}.vnnlib,116')
        (benchmark / 'instances.csv').write_text('\n'.join(rows) + '\n')
        output = root / 'experiments'
        command = [sys.executable, '-m', 'case_studies.acasxu.suite',
                   '--benchmark', str(benchmark), '--output', str(output), '--dry-run']
        defaults = subprocess.run(command, check=True, capture_output=True, text=True)
        default_commands = json.loads(defaults.stdout)['commands']
        assert len(default_commands) == 12 and not output.exists()
        for train in default_commands[::3]:
            model_name = Path(train[train.index('--model') + 1]).name
            specs = {Path(train[train.index('--specification') + 1]).name,
                     *[Path(path).name for path in train[train.index('--additional-specifications') + 1:]]}
            expected = {'prop_1.vnnlib', 'prop_2.vnnlib', 'prop_3.vnnlib', 'prop_4.vnnlib'}
            if '4_5_batch' in model_name:
                expected.add('prop_10.vnnlib')
            assert specs == expected
            assert '_all_properties_seed1' in train[train.index('--output') + 1]
        assert all('ALPHA-CROWN' not in invocation for invocation in default_commands)
        assert {invocation[invocation.index('--method') + 1] for invocation in default_commands} == {'CROWN-IBP', 'CROWN'}
        result = subprocess.run(command + ['--epochs', '7', '--seed', '3', '--device', 'cuda',
            '--samples', '16', '--max-cells', '8', '--alpha-iterations', '4',
            '--evaluation-methods', 'CROWN-IBP', 'CROWN', 'ALPHA-CROWN'],
            check=True, capture_output=True, text=True)
        plan = json.loads(result.stdout)
        assert len(plan['commands']) == 16 and not output.exists()
        for offset in range(0, 16, 4):
            train, ibp, crown, alpha = plan['commands'][offset:offset + 4]
            assert train[train.index('--epochs') + 1] == '7'
            for invocation in (train, ibp, crown, alpha):
                assert invocation[invocation.index('--seed') + 1] == '3'
                assert invocation[invocation.index('--device') + 1] == 'cuda'
            assert '--assigned-only' in alpha
            assert alpha[alpha.index('--alpha-iterations') + 1] == '4'
            assert alpha[alpha.index('--samples') + 1] == '16'
            assert alpha[alpha.index('--max-cells') + 1] == '8'
            assert alpha[alpha.index('--cover-from') + 1] == str(
                Path(crown[crown.index('--output') + 1]) / 'results.json')
        single = subprocess.run(command + ['--cases', '4_5:2', '--evaluation-methods', 'ALPHA-CROWN'],
                                check=True, capture_output=True, text=True)
        invocations = json.loads(single.stdout)['commands']
        assert len(invocations) == 2 and '--cover-from' not in invocations[-1]
        for extra in (['--cases', '4_5:9'], ['--cases', '4_5:2', '4_5:2'],
                      ['--cases', '4_5:2', '4_5:10'], ['--epochs', '0']):
            assert subprocess.run(command + extra, capture_output=True).returncode != 0
        output.mkdir()
        assert subprocess.run(command, capture_output=True).returncode != 0
    print('ACAS suite selection, propagation, cover reuse and output safety checks passed')


if __name__ == '__main__':
    main()
