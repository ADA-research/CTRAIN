import pytest
import torch

from CTRAIN.train.certified.eps_scheduler import LinearScheduler, SmoothedScheduler


@pytest.fixture(params=[LinearScheduler, SmoothedScheduler], ids=["linear", "smoothed"])
def scheduler_type(request):
    return request.param


def options(**changes):
    return dict(dict(num_epochs=10, eps=.4, start_eps=.1, mean=0.,
                     std=torch.tensor([.5, 2.]), batches_per_epoch=2,
                     eps_schedule=(1, 4), eps_schedule_unit="epoch"), **changes)


def test_schedule_warmup_monotonicity_endpoints_and_normalisation(scheduler_type):
    scheduler = scheduler_type(**options())
    values = []
    for _ in range(24):
        values.append(scheduler.get_cur_eps(False))
        torch.testing.assert_close(scheduler.get_cur_eps(),
                                   torch.tensor(values[-1]) / torch.tensor([.5, 2.]))
        assert 0 <= scheduler.get_cur_kappa() <= 1
        assert 0 <= scheduler.get_cur_beta() <= 1
        scheduler.batch_step()
    assert values[:3] == [.1] * 3
    assert all(a <= b for a, b in zip(values, values[1:]))
    assert values[-1] == .4
    assert scheduler.get_cur_kappa() == scheduler.get_cur_beta() == 0


def test_epoch_and_batch_units_agree(scheduler_type):
    epoch = scheduler_type(**options())
    batch = scheduler_type(**options(eps_schedule=(2, 8), eps_schedule_unit="batch"))
    for _ in range(24):
        assert epoch.get_cur_eps(False) == batch.get_cur_eps(False)
        epoch.batch_step()
        batch.batch_step()


@pytest.mark.parametrize("changes", [
    {"eps": -.1}, {"eps": float("nan")}, {"start_eps": .5},
    {"std": 0.}, {"std": float("inf")}, {"batches_per_epoch": 0},
    {"eps_schedule": (-1, 4)}, {"eps_schedule": (1, 20)},
    {"eps_schedule_unit": "step"}, {"start_beta": 1.1},
])
def test_invalid_schedule_rejected(scheduler_type, changes):
    with pytest.raises(ValueError):
        scheduler_type(**options(**changes))
