"""Neutral Shi initialization extensions preserve existing weights and RNG behavior."""
import copy
import math
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from CTRAIN.train.certified.initialisation import ibp_init_shi


def initialization():
    original = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2)).double()
    with torch.no_grad():
        original[1].bias.fill_(.3)
    dummy = SimpleNamespace(_modules={})
    for last, zero in ((False, False), (True, False), (False, True), (True, True)):
        model = copy.deepcopy(original)
        torch.manual_seed(17)
        with patch('torch.nn.init.normal_', wraps=nn.init.normal_) as normal:
            ibp_init_shi(model, dummy, init_last_layer=last, zero_bias=zero)
        assert normal.call_count == (2 if last else 1)
        for call in normal.call_args_list:
            fan_in = call.args[0].shape[1]
            assert call.kwargs['std'] == math.sqrt(2 * math.pi / fan_in**2)
        if not last:
            torch.testing.assert_close(model[2].weight, original[2].weight, rtol=0, atol=0)
        torch.testing.assert_close(model[1].weight, original[1].weight, rtol=0, atol=0)
        for (name, value), (_, old) in zip(model.named_parameters(), original.named_parameters()):
            if 'bias' in name:
                torch.testing.assert_close(value, torch.zeros_like(value) if zero else old, rtol=0, atol=0)
    # Legacy call and RNG consumption stay exactly unchanged.
    model, expected = copy.deepcopy(original), copy.deepcopy(original)
    torch.manual_seed(31)
    ibp_init_shi(model, dummy)
    next_actual = torch.rand(1)
    torch.manual_seed(31)
    nn.init.normal_(expected[0].weight, mean=0, std=math.sqrt(2 * math.pi / 3**2))
    next_expected = torch.rand(1)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(next_actual, next_expected, rtol=0, atol=0)
    for options in ({'init_last_layer': 1}, {'zero_bias': 'yes'}):
        model = copy.deepcopy(original)
        try:
            ibp_init_shi(model, dummy, **options)
            raise AssertionError('Expected boolean validation')
        except ValueError:
            pass
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    initialization()
    print('Shi initialization checks passed')
