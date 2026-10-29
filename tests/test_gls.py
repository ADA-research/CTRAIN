"""Run directly: PYTHONPATH=. ../venv/bin/python tests/test_gls.py."""
import copy
import tempfile
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch
from CTRAIN.train.certified.gls import GLSOptimizer
from CTRAIN import GLSModelWrapper


def estimators():
    for method in ('rgs', 'pgpe'):
        model = torch.nn.Linear(1, 1, bias=False).double()
        model.weight.data.fill_(2.)
        opt = GLSOptimizer(model, torch.optim.SGD, lr=.1, estimator=method,
                           samples=2, sigma=.5, sigma_lr=0.)
        with patch('torch.randn_like', side_effect=lambda x: torch.ones_like(x)):
            opt.step(lambda: model.weight.square().sum())
        expected = 1.5 if method == 'rgs' else 1.8
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, expected))
        state = opt.state_dict()
        opt.load_state_dict(state)
        before = model.weight.clone()
        model.weight.grad = torch.ones_like(model.weight)
        def failure():
            raise RuntimeError('failure')
        try:
            opt.step(failure)
        except RuntimeError:
            pass
        else:
            raise AssertionError('exception not raised')
        torch.testing.assert_close(model.weight, before)
        torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight))



def sampled_clipping():
    model = torch.nn.Linear(1, 1, bias=False).double()
    model.weight.data.fill_(2.)
    opt = GLSOptimizer(model, torch.optim.SGD, lr=.1, estimator='rgs',
                       samples=2, sigma=.5, gradient_clip=4.)
    noise = iter([torch.ones_like(model.weight), -torch.ones_like(model.weight)])
    with patch('torch.randn_like', side_effect=lambda _: next(noise)):
        opt.step(lambda: model.weight.square().sum())
    # Sample gradients are 5 and 3; clipping individually gives (4+3)/2.
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 1.65))


def bounded_estimator():
    """Compare sampled auto_LiRPA gradients with exact linear-box margins."""
    from auto_LiRPA import BoundedModule, PerturbationLpNorm
    from CTRAIN.bound import bound_crown
    from CTRAIN.util import construct_c
    from CTRAIN.train.certified.losses.util import get_loss_from_bounds
    torch.manual_seed(31)
    original = torch.nn.Linear(2, 3)
    data, target = torch.tensor([[.2, .8], [.7, .4]]), torch.tensor([2, 0])
    model = BoundedModule(original, data, device='cpu')
    model.train()
    params = list(model.parameters())
    centers = [p.detach().clone() for p in params]
    reference = copy.deepcopy(original)
    expected = [torch.zeros_like(p) for p in params]
    c = construct_c(data, target, 3)
    criterion = torch.nn.CrossEntropyLoss()
    for direction in (1., -.7):
        with torch.no_grad():
            for p, center in zip(reference.parameters(), centers):
                p.copy_(center + direction * .03)
        margin_weight = c @ reference.weight
        lb = (margin_weight * data.unsqueeze(1)).sum(-1) + c @ reference.bias
        lb = lb - .1 * margin_weight.abs().sum(-1)
        loss = get_loss_from_bounds(lb, criterion)
        for grad, value in zip(expected, torch.autograd.grad(loss, tuple(reference.parameters()))):
            grad.add_(value / 2)
    region = PerturbationLpNorm(eps=.1, x_L=data-.1, x_U=data+.1)
    def closure():
        model(data)
        lb, _ = bound_crown(model, region, data, target, 3)
        return get_loss_from_bounds(lb, criterion)
    directions = iter([torch.full_like(p, direction) for direction in (1., -.7) for p in params])
    opt = GLSOptimizer(model, torch.optim.SGD, lr=.01, estimator='rgs', samples=2, sigma=.03)
    with patch('torch.randn_like', side_effect=lambda _: next(directions)):
        opt.step(closure)
    for p, center, grad in zip(params, centers, expected):
        torch.testing.assert_close(p, center - .01 * grad)


def state_and_scales():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    model.train()
    x = torch.rand(4, 2)
    opt = GLSOptimizer(model, estimator='rgs', samples=2)
    buffers = [b.clone() for b in model.buffers()]
    opt.step(lambda: model(x).square().mean())
    for old, current in zip(buffers, model.buffers()):
        torch.testing.assert_close(old, current)
    assert model.training and all(p.requires_grad for p in model.parameters())
    model = torch.nn.Linear(1,1,bias=False).double()
    model.weight.data.fill_(2.)
    opt = GLSOptimizer(model, torch.optim.SGD, estimator='pgpe', samples=2,
                       sigma=.5, sigma_lr=.1)
    with patch('torch.randn_like', side_effect=lambda x: torch.full_like(x,2.)):
        opt.step(lambda: model.weight.square().sum())
    torch.testing.assert_close(opt.scales[0],torch.full_like(opt.scales[0],.49))
    clone=copy.deepcopy(model)
    restored=GLSOptimizer(clone,torch.optim.SGD,estimator='pgpe',samples=2,sigma=.5,sigma_lr=.1)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    rng=torch.get_rng_state()
    opt.step(lambda: model.weight.square().sum())
    torch.set_rng_state(rng)
    restored.step(lambda: clone.weight.square().sum())
    torch.testing.assert_close(model.weight,clone.weight)
    torch.testing.assert_close(opt.scales[0],restored.scales[0])


def hpo():
    from CTRAIN.model_wrappers.configs import get_config_space
    from pathlib import Path
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    wrapper = GLSModelWrapper(model, (2,), .1, 1, device='cpu', shi_init=False,
                              warm_up_epochs=0, ramp_up_epochs=0)
    space = get_config_space(wrapper, wrapper.num_epochs, wrapper.eps)
    assert 'sigma' in space
    config = space.get_default_configuration()
    with tempfile.TemporaryDirectory() as root:
        Path(root,'nets').mkdir()
        with patch.object(GLSModelWrapper,'train_model') as train, patch.object(
                wrapper,'_evaluate_hpo_model',return_value=(.8,.4,.3)):
            value,metrics=wrapper._hpo_runner(config,1,wrapper.num_epochs,None,None,root)
        assert train.called and abs(value+1.5)<1e-6
        assert metrics['cert_acc']==.4


def training():
    loader = DataLoader(TensorDataset(torch.rand(4, 1, 2, 2), torch.tensor([0,1,0,1])),batch_size=2)
    loader.mean, loader.std = torch.tensor([0.]), torch.tensor([1.])
    loader.min, loader.max, loader.normalised = torch.tensor(0.), torch.tensor(1.), False
    for method in ('rgs', 'pgpe'):
        for bound in ('IBP', 'CROWN-IBP', 'CROWN'):
            model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,3),torch.nn.ReLU(),torch.nn.Linear(3,2))
            wrapper = GLSModelWrapper(model,(1,2,2),.1,1,warm_up_epochs=0,
                ramp_up_epochs=0,device='cpu',estimator=method,bound_method=bound,
                shi_init=False,shi_reg_weight=.1)
            old = [p.clone() for p in wrapper.bounded_model.parameters()]
            wrapper.train_model(loader)
            assert any(not torch.equal(a,b) for a,b in zip(old,wrapper.bounded_model.parameters()))
            assert all(torch.isfinite(p).all() for p in wrapper.bounded_model.parameters())
            wrapper.eval()
            output = wrapper(torch.rand(2,1,2,2))
            assert output.shape == (2,2)
    # Actual bounded BatchNorm training and final-radius validation.
    bn_model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,3),
                                  torch.nn.BatchNorm1d(3),torch.nn.ReLU(),torch.nn.Linear(3,2))
    bn_wrapper = GLSModelWrapper(bn_model,(1,2,2),.1,2,warm_up_epochs=0,
        ramp_up_epochs=0,device='cpu',shi_init=False,population_bn=True)
    with patch('CTRAIN.eval.eval_certified',side_effect=[(.8,{}),(.2,{})]) as evaluate:
        bn_wrapper.train_model(loader,loader)
    assert evaluate.call_count==2
    for call in evaluate.call_args_list:
        torch.testing.assert_close(call.args[2],torch.tensor([[[.1]]]))
    assert all(torch.isfinite(b).all() for b in bn_wrapper.bounded_model.buffers())
    # RGS decay and resumable optimizer/scheduler state.
    with tempfile.TemporaryDirectory() as root:
        wrapper = GLSModelWrapper(model,(1,2,2),.1,1,warm_up_epochs=0,
            ramp_up_epochs=1,device='cpu',shi_init=False,shi_reg_weight=.1,
            sigma_decay_milestones=(1,),checkpoint_save_path=root,checkpoint_save_interval=1)
        wrapper.train_model(loader)
        torch.testing.assert_close(wrapper.optimizer.scales[0], torch.full_like(wrapper.optimizer.scales[0],.0004))
        from pathlib import Path
        saved=torch.load(next(Path(root).glob('*.pt')),weights_only=False)
        assert 'optimizer_state_dict' in saved
        assert 'lr_scheduler_state_dict' in saved
        fresh = GLSModelWrapper(copy.deepcopy(model),(1,2,2),.1,1,warm_up_epochs=0,
            ramp_up_epochs=1,device='cpu',shi_init=False,shi_reg_weight=.1,
            sigma_decay_milestones=(1,))
        fresh.resume_from_checkpoint(str(next(Path(root).glob('*.pt'))),loader)
        torch.testing.assert_close(fresh.optimizer.scales[0],wrapper.optimizer.scales[0])
        for a,b in zip(fresh.parameters(),wrapper.parameters()):
            torch.testing.assert_close(a,b)


if __name__ == '__main__':
    estimators()
    sampled_clipping()
    bounded_estimator()
    state_and_scales()
    training()
    hpo()
    print('GLS checks passed')
