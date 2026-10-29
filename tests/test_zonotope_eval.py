"""Full-box LiRPA certification: sound margins, BN isolation, and fresh bounds."""
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import BoundedModule
from CTRAIN.eval.zonotope import eval_zonotope
from CTRAIN.bound.set_based_zono import box_zonotope, bound_zonotope, zonotope_margins


def check(model, data, eps):
    labels = torch.arange(len(data)) % 2
    loader = DataLoader(TensorDataset(data, labels), batch_size=3)
    loader.min, loader.max, loader.normalised = 0., 1., False
    model.train()
    modes = [m.training for m in model.modules()]
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    captured = []
    compute = BoundedModule.compute_bounds

    def record(self, *args, **kwargs):
        assert kwargs['method'] == 'Forward'
        result = compute(self, *args, **kwargs)
        captured.append((kwargs['x'][0].detach().clone(), kwargs['C'], result[0]))
        return result

    with patch.object(BoundedModule, 'compute_bounds', record):
        result = eval_zonotope(model, loader, eps, propagation_batch_size=2, device='cpu')
    assert len(captured) == 3  # chunks 2+1, then 1; different batch shapes
    assert modes == [m.training for m in model.modules()]
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    model.eval()
    for chunk, C, margins in captured:
        lower, upper = (chunk - eps).clamp(0, 1), (chunk + eps).clamp(0, 1)
        for _ in range(20):
            point = lower + torch.rand_like(lower) * (upper - lower)
            actual = torch.bmm(C, model(point).unsqueeze(-1)).squeeze(-1)
            assert (actual >= margins - 1e-9).all()
    assert result == eval_zonotope(model, loader, eps, propagation_batch_size=1, device='cpu')
    # Ordinary point bounds at zero radius agree with the clean classifier.
    clean, certified = eval_zonotope(model, loader, 0., device='cpu')
    assert clean == certified
    model.train()
    with patch.object(BoundedModule, 'compute_bounds', side_effect=RuntimeError('injected')):
        try:
            eval_zonotope(model, loader, eps, device='cpu')
        except RuntimeError as error:
            assert str(error) == 'injected'
        else:
            raise AssertionError('Expected injected failure')
    assert all(m.training for m in model.modules())
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    for eps in (0., .2):
        check(nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2)).double(),
              torch.tensor([[0., 1.], [.3, .4], [.8, .2], [1., 0.]], dtype=torch.double), eps)
        check(nn.Sequential(nn.Conv2d(1, 2, 2), nn.BatchNorm2d(2), nn.ReLU(),
                            nn.Flatten(), nn.Linear(8, 2)).double(), torch.rand(4, 1, 3, 3, dtype=torch.double), eps)
    # Shared final rows cancel when C is merged; output-box errors cannot cancel.
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 2)).double().eval()
    with torch.no_grad():
        model[0].weight.fill_(1); model[0].bias.fill_(-.5)
        model[2].weight.fill_(1); model[2].bias.copy_(torch.tensor([.1, 0.]))
    data = torch.tensor([[.5]], dtype=torch.double)
    loader = DataLoader(TensorDataset(data, torch.tensor([0])))
    loader.min, loader.max = 0., 1.
    old = bound_zonotope(model, box_zonotope(data - .5, data + .5))
    assert zonotope_margins(old, torch.tensor([0]))[0, 1] < 0
    assert eval_zonotope(model, loader, .5, device='cpu') == (1., 1.)
    assert eval_zonotope(model, loader, .5, error_propagation='generators', device='cpu') == (1., 1.)
    print('LiRPA full-box certification checks passed')


if __name__ == '__main__':
    main()
