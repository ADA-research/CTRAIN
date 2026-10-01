"""Subprocess entry point; keep verification behaviour in the shared adapter."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from CTRAIN.complete_verification.abCROWN.verify import abcrown_eval

if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as file:
        args, kwargs = pickle.load(file)
    result = abcrown_eval(*args, **kwargs)
    with open(sys.argv[2], 'wb') as file:
        pickle.dump(result, file)
