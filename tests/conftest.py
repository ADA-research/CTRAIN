"""Keep synthetic numerical checks reproducible and inexpensive on CPU."""
import os

# Set these before importing torch, including in regression subprocesses.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ctrain-mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pytest
import torch


@pytest.fixture(autouse=True)
def numerical_state():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        try:
            yield
        finally:
            torch.set_num_threads(threads)
