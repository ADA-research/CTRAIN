"""Run directly: PYTHONPATH=. ../venv/bin/python tests/test_cc_dist.py."""
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from auto_LiRPA import PerturbationLpNorm
from auto_LiRPA.operators.normalization import BoundBatchNormalization

from CTRAIN import CCDistModelWrapper, PGDModelWrapper
from CTRAIN.bound import bound_ibp
from CTRAIN.model_wrappers.configs import get_config_space
from CTRAIN.train.certified.losses import get_cc_ibp_loss
from CTRAIN.train.certified.losses.cc_dist import feature_distillation_loss
from CTRAIN.train.adversarial.losses import get_pgd_loss
from CTRAIN.util import construct_c


def loader_for(x, y, batch_size=2):
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    loader.normalised = False
    loader.std, loader.mean = torch.ones(x.shape[1]), torch.zeros(x.shape[1])
    loader.min, loader.max = torch.tensor(0.), torch.tensor(1.)
    return loader


def check():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    x = torch.tensor([[[[.2]], [[.6]]], [[[.7]], [[.4]]]])
    y = torch.tensor([0, 1])
    student = nn.Sequential(nn.Flatten(), nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        student[1].weight.copy_(torch.tensor([[.4, -.2], [-.3, .6], [.2, .3]]))
        student[1].bias.copy_(torch.tensor([.5, .3, .4]))
    wrapper = CCDistModelWrapper(student, (2, 1, 1), .1, 2, teacher=teacher,
        device='cpu', warm_up_epochs=0, ramp_up_epochs=0, shi_reg_weight=0,
        bound_opts={'param': {'auto_requires_grad': False}, 'forward_before_compute_bounds': True})
    wrapper.eval()
    for parameter in wrapper.parameters():
        parameter.requires_grad_(True)
    criterion = nn.CrossEntropyLoss(reduction='none')
    adv = x + .03
    c = construct_c(x, y, 2)
    features = student[:-1](adv)
    center = nn.functional.linear(x.flatten(1), student[1].weight, student[1].bias)
    deviation = .1 * student[1].weight.abs().sum(1)
    lower, upper = (center-deviation).relu(), (center+deviation).relu()
    target_features = teacher[:-1](x).detach()
    head_weight, head_bias = student[-1].weight, student[-1].bias
    margin_weight = c @ head_weight
    exact_margins = (margin_weight.clamp_min(0) * lower.unsqueeze(1)
                     + margin_weight.clamp_max(0) * upper.unsqueeze(1)).sum(-1) + c @ head_bias
    adv_margins = (c @ student(adv).unsqueeze(-1)).squeeze(-1)
    for alpha in (0., .37, 1.):
        ptb = PerturbationLpNorm(eps=.1, x_L=x-.1, x_U=x+.1)
        with patch('CTRAIN.train.certified.losses.convex_combinations.pgd_attack', return_value=adv):
            actual, = wrapper._loss_func(wrapper.bounded_model, student, ptb, x, y, 2,
                                         criterion, alpha, n_steps=1, device='cpu')
        mixed_margins = alpha*exact_margins + (1-alpha)*adv_margins
        ce = torch.logsumexp(torch.cat([torch.zeros(2, 1), -mixed_margins], 1), 1).mean()
        a, b = (1-alpha)*features+alpha*lower, (1-alpha)*features+alpha*upper
        expected = ce + 5/3 * torch.maximum((a-target_features)**2, (b-target_features)**2).sum(1).mean()
        torch.testing.assert_close(actual, expected)
        parameters = list(student.parameters())
        for left, right in zip(torch.autograd.grad(actual, parameters, retain_graph=True),
                               torch.autograd.grad(expected, parameters, retain_graph=True)):
            torch.testing.assert_close(left, right)
    # beta=0 is precisely CC-IBP, with no teacher pass and identical gradients.
    with patch('CTRAIN.train.certified.losses.convex_combinations.pgd_attack', return_value=adv), \
         patch.object(wrapper.teacher_features, 'forward', side_effect=AssertionError('unused teacher')):
        a, = wrapper._loss_func(wrapper.bounded_model, student, ptb, x, y, 2,
                                criterion, .37, device='cpu', distillation_weight=0.)
        b, = get_cc_ibp_loss(wrapper.bounded_model, student, ptb, x, y, 2,
                             criterion, .37, device='cpu')
        torch.testing.assert_close(a, b)
    # A missing square or a feature mean instead of sum fails these closed-form checks.
    a = torch.tensor([[1., 2.]], requires_grad=True)
    ref = torch.tensor([[.5, 3.]], requires_grad=True)
    value = feature_distillation_loss(a, a-1, a+1, ref, .5)
    torch.testing.assert_close(value, torch.tensor(8.125))
    value.backward()
    assert ref.grad is None and torch.isfinite(a.grad).all()

    images = torch.rand(4, 2, 1, 1)
    labels = torch.tensor([0, 1, 0, 1])
    loader = loader_for(images, labels)
    # Warm-up reports actual predictions; a zero margin is not certified.
    for bias, target, accuracy in (([1., -1.], 1, 0.), ([0., 0.], 0, 1.)):
        model = copy.deepcopy(student)
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.copy_(torch.tensor(bias))
        warm = CCDistModelWrapper(model, (2,1,1), .1, 2, teacher=copy.deepcopy(model),
            device='cpu', warm_up_epochs=1, ramp_up_epochs=0,
            shi_reg_weight=0., l1_reg_weight=None)
        for group in warm.optimizer.param_groups:
            group['lr'] = 0.
        warm_loader = loader_for(images, torch.full((4,), target))
        with patch('CTRAIN.train.certified.expressive_ibp.ibp_init_shi'), \
             patch('CTRAIN.train.certified.expressive_ibp.log_epoch_summary') as summary:
            warm.train_model(warm_loader, end_epoch=1)
        assert summary.call_args.kwargs['adv_acc'] == accuracy
        assert summary.call_args.kwargs['cert_acc'] == 0.
        if target == 0:
            ptb_zero = PerturbationLpNorm(eps=0., x_L=images, x_U=images)
            with patch('CTRAIN.train.certified.losses.convex_combinations.pgd_attack', return_value=images):
                _, robust_err, adv_err = warm._loss_func(warm.bounded_model, model,
                    ptb_zero, images, warm_loader.dataset.tensors[1], 2, criterion, .5,
                    return_stats=True, distillation_weight=0., device='cpu')
            assert robust_err == 1. and adv_err == 0.
    bn_model = nn.Sequential(nn.Flatten(), nn.Linear(2, 3), nn.BatchNorm1d(3),
                             nn.ReLU(), nn.Linear(3, 2))
    teacher = copy.deepcopy(bn_model).train()
    for options in ({'forward_before_compute_bounds': True}, {'bn': {'mode': 'ibp'}}):
        with patch('CTRAIN.model_wrappers.cc_ibp_model_wrapper.CCIBPModelWrapper.__init__',
                   side_effect=AssertionError('converted before validation')):
            try:
                CCDistModelWrapper(bn_model, (2,1,1), .1, 2, teacher=teacher,
                    warm_up_epochs=0, ramp_up_epochs=0, bound_opts=options)
            except ValueError as error:
                assert 'BatchNorm' in str(error)
            else:
                raise AssertionError('incompatible BN options accepted')
    teacher[2].eval()
    teacher_state = copy.deepcopy(teacher.state_dict())
    teacher_modes = [m.training for m in teacher.modules()]
    wrapper = CCDistModelWrapper(bn_model, (2, 1, 1), .1, 2, teacher=teacher,
        device='cpu', warm_up_epochs=0, ramp_up_epochs=0, shi_reg_weight=0,
        pgd_steps=1, population_bn=False, relu_upper_retention=.8, pgd_eps_factor=2.,
        bound_opts=None)
    bn = next(m for m in wrapper.bounded_model.modules() if isinstance(m, BoundBatchNormalization))
    # Direct losses/trainers also reject changed options before attacks or warm-up updates.
    from CTRAIN.train.certified.cc_dist import cc_dist_train_model
    wrapper.bounded_model.bound_opts['forward_before_compute_bounds'] = True
    with patch('CTRAIN.train.certified.losses.convex_combinations.pgd_attack',
               side_effect=AssertionError('attack before validation')):
        for call in (lambda: wrapper._loss_func(wrapper.bounded_model, bn_model, ptb, x, y, 2,
                                                criterion, .5),
                     lambda: cc_dist_train_model(bn_model, wrapper.bounded_model, loader,
                                                 teacher_features=wrapper.teacher_features)):
            try:
                call()
            except ValueError as error:
                assert 'BatchNorm' in str(error)
            else:
                raise AssertionError('incompatible BN options accepted')
    wrapper.bounded_model.bound_opts['forward_before_compute_bounds'] = False
    # No clean pass after warmup: BN running means reflect exactly two adversarial batches.
    expected_running = bn.inputs[3].buffer.detach().clone()
    seen = []
    def attack(**kwargs):
        assert not kwargs['model'].training
        assert kwargs['device'] == images.device or kwargs['device'] == torch.device('cpu')
        torch.testing.assert_close(kwargs['x_L'], (kwargs['data']-.2).clamp(0, 1))
        point = (kwargs['data']+.04).clamp(0, 1)
        seen.append(point)
        return point
    with patch('CTRAIN.train.certified.losses.convex_combinations.pgd_attack', side_effect=attack):
        # Skip initialization so independent BN expectations use the unchanged initial weights.
        with patch('CTRAIN.train.certified.expressive_ibp.ibp_init_shi'):
            before = [p.detach().clone() for p in wrapper.parameters()]
            # Freeze optimizer updates solely for the independent running-statistic check.
            for group in wrapper.optimizer.param_groups:
                group['lr'] = 0.
            wrapper.train_model(loader, end_epoch=1)
    for point in seen:
        expected_running = (1-bn.momentum)*expected_running + bn.momentum*bn_model[1](point.flatten(1)).mean(0)
    torch.testing.assert_close(bn.inputs[3].buffer, expected_running)
    assert len(seen) == 2
    assert all(torch.equal(teacher.state_dict()[key], value) for key, value in teacher_state.items())
    assert [m.training for m in teacher.modules()] == teacher_modes
    assert all(p.grad is None for p in teacher.parameters())
    assert all(not p.requires_grad and p.grad is None for p in wrapper.teacher_features.parameters())
    assert not any(key.startswith('teacher') for key in wrapper.state_dict())
    wrapper.eval()
    sound, _ = bound_ibp(wrapper.bounded_model, ptb, x, y, 2)
    explicit, _ = bound_ibp(wrapper.bounded_model, ptb, x, y, 2, relu_upper_retention=1.)
    torch.testing.assert_close(sound, explicit)
    space = get_config_space(wrapper, 2, .1)
    beta_name = 'cc_dist:distillation_weight'
    hp = space[beta_name]
    width = bn_model[-1].in_features
    assert hp.log and abs(hp.lower - .1 / width) < 1e-12
    assert abs(hp.upper - 50 / width) < 1e-12
    config = space.get_default_configuration()
    assert abs(config[beta_name] - 5 / width) < 1e-12
    override = get_config_space(wrapper, 2, .1, defaults={'distillation_weight': 2 / width})
    assert abs(override.get_default_configuration()[beta_name] - 2 / width) < 1e-12
    other_space = get_config_space(warm, 2, .1)
    assert abs(other_space[beta_name].lower - .1 / model[-1].in_features) < 1e-12
    with tempfile.TemporaryDirectory() as root:
        Path(root, 'nets').mkdir()
        with patch.object(CCDistModelWrapper, 'train_model', autospec=True) as train, \
             patch.object(CCDistModelWrapper, '_evaluate_hpo_model', return_value=(.5,.4,.3)):
            legacy = dict(config)
            del legacy[beta_name]
            trials = [dict(config, **{beta_name: beta}) for beta in (hp.lower, hp.upper, 0.)]
            for trial in [config, *trials, legacy]:
                wrapper._hpo_runner(trial, 1, 2, loader, loader, root)
                recreated = train.call_args.args[0]
                assert recreated.pgd_eps_factor == config['cc_ibp:pgd_eps_factor']
                assert recreated.relu_upper_retention == .8
                beta = trial.get(beta_name, wrapper.distillation_weight)
                assert recreated.distillation_weight == beta
                assert recreated._loss_func.keywords['distillation_weight'] == beta
                torch.testing.assert_close(recreated.teacher_features(images), wrapper.teacher_features(images))
    # CPU device/mixed modes and parameter gradients are preserved on attack failure.
    bn_model.train(); bn_model[2].eval()
    flags = [m.training for m in bn_model.modules()]
    for p in bn_model.parameters():
        p.grad = torch.ones_like(p)
    def fail(**kwargs):
        assert not bn_model.training
        raise RuntimeError('attack failure')
    with patch('CTRAIN.train.adversarial.losses.pgd.pgd_attack', side_effect=fail):
        try:
            get_pgd_loss(bn_model, ptb, x, y, 2, criterion)
        except RuntimeError:
            pass
        else:
            raise AssertionError('attack exception lost')
    assert [m.training for m in bn_model.modules()] == flags
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in bn_model.parameters())

    # Native teacher training, normalized boxes, triangular LR and trajectory continuation.
    loader.normalised = True
    loader.std = torch.tensor([2., 4.])
    with tempfile.TemporaryDirectory() as root:
        def build():
            return PGDModelWrapper(copy.deepcopy(bn_model).requires_grad_(True), (2,1,1), .1, 2,
                device='cpu', shi_init=False, pgd_steps=2, checkpoint_save_path=root,
                checkpoint_save_interval=1, l1_reg_weight=1e-5)
        full = build()
        with patch('CTRAIN.train.adversarial.pgd.save_checkpoint'):
            torch.manual_seed(12)
            full.train_model(loader)
        split = build()
        torch.manual_seed(12)
        split.train_model(loader, end_epoch=1)
        rng = torch.get_rng_state()
        resumed = build()
        torch.set_rng_state(rng)
        resumed.resume_from_checkpoint(Path(root, '1_checkpoint.pt'), loader)
        for a, b in zip(full.original_model.state_dict().values(), resumed.original_model.state_dict().values()):
            torch.testing.assert_close(a, b)
        assert full.optimizer.param_groups[0]['lr'] == 0
        assert full.optimizer.param_groups[0]['momentum'] == .9
        assert not hasattr(full, 'bounded_model')
    # None is the shared API's spelling for disabled L1 regularization.
    no_l1 = PGDModelWrapper(copy.deepcopy(student), (2,1,1), .1, 1,
                           device='cpu', pgd_steps=1, l1_reg_weight=None)
    no_l1.train_model(loader)

    # Real nonzero-radius student updates and checkpoint continuation with matched attack RNG.
    with tempfile.TemporaryDirectory() as root:
        def student_wrapper():
            return CCDistModelWrapper(copy.deepcopy(student).requires_grad_(True), (2,1,1), .1, 2,
                teacher=teacher, warm_up_epochs=0, ramp_up_epochs=0, shi_reg_weight=0,
                lr_decay_kwargs=dict(milestones=(1,), gamma=.2), device='cpu', pgd_steps=1,
                checkpoint_save_path=root, checkpoint_save_interval=1, l1_reg_weight=None)
        full = student_wrapper()
        with patch('CTRAIN.train.certified.expressive_ibp.save_checkpoint'):
            torch.manual_seed(15)
            full.train_model(loader_for(images,labels))
        split = student_wrapper()
        torch.manual_seed(15)
        split.train_model(loader_for(images,labels), end_epoch=1)
        rng = torch.get_rng_state()
        resumed = student_wrapper()
        torch.set_rng_state(rng)
        resumed.resume_from_checkpoint(Path(root,'1_checkpoint.pt'), loader_for(images,labels))
        for a,b in zip(full.state_dict().values(), resumed.state_dict().values()):
            torch.testing.assert_close(a,b)
        assert full.lr_scheduler.state_dict() == resumed.lr_scheduler.state_dict()
    for weight in (-1., float('nan')):
        try:
            CCDistModelWrapper(student, (2,1,1), .1, 2, teacher=teacher,
                               warm_up_epochs=0, ramp_up_epochs=0, distillation_weight=weight)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid beta accepted')
    print('CC-Dist formula, gradients, BN, frozen teacher, HPO and PGD checkpoint checks passed')


def check_cuda():
    """Actual CNN7s: native teacher updates, full-radius distillation and sound IBP."""
    from CTRAIN.model_definitions import CNN7_Shi
    assert torch.cuda.is_available(), 'A CUDA device is required'
    torch.set_num_threads(1)
    for size, classes, eps in ((32, 10, 2/255), (32, 10, 8/255), (64, 200, 1/255)):
        images = torch.rand(4, 3, size, size)
        labels = torch.arange(4) % classes
        loader = loader_for(images, labels)
        teacher = PGDModelWrapper(CNN7_Shi((3,size,size), n_classes=classes,
            downsample_final=size==64), (3,size,size), eps, 1, device='cuda', pgd_steps=10)
        before = [p.detach().clone() for p in teacher.parameters()]
        teacher.train_model(loader)
        assert any(not torch.equal(a,b) for a,b in zip(before, teacher.parameters()))
        student = CCDistModelWrapper(CNN7_Shi((3,size,size), n_classes=classes,
            downsample_final=size==64), (3,size,size), eps, 1, teacher=teacher.original_model,
            device='cuda', warm_up_epochs=0, ramp_up_epochs=0, pgd_steps=8 if eps==2/255 else 1,
            pgd_alpha=.25 if eps==2/255 else 10., pgd_eps_factor=2.1 if eps==2/255 else 1.,
            cc_ibp_alpha=.01 if eps==2/255 else .5 if size==32 else .005)
        before = [p.detach().clone() for p in student.parameters()]
        student.train_model(loader)
        assert any(not torch.equal(a,b) for a,b in zip(before, student.parameters()))
        assert all(torch.isfinite(p).all() and torch.isfinite(p.grad).all()
                   for p in student.parameters() if p.grad is not None)
        student.eval()
        x, y = images[:2].cuda(), labels[:2].cuda()
        ptb = PerturbationLpNorm(eps=eps, x_L=(x-eps).clamp(0,1), x_U=(x+eps).clamp(0,1))
        lower, _ = bound_ibp(student.bounded_model, ptb, x, y, classes)
        assert torch.isfinite(lower).all()
        print(f'CUDA teacher/distillation/IBP passed: {size}px, {classes} classes, eps={eps}')


def check_verifier():
    """Train/export a known robust toy student, then run the actual bundled solver."""
    import json
    torch.set_num_threads(1)
    torch.manual_seed(27)
    x = torch.tensor([[[[.3]],[[.4]]], [[[.6]],[[.7]]],
                      [[[.5]],[[.4]]], [[[.3]],[[.7]]]])
    loader = loader_for(x, torch.zeros(4, dtype=torch.long))
    model = nn.Sequential(nn.Flatten(), nn.Linear(2,3), nn.ReLU(), nn.Linear(3,2))
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.copy_(torch.tensor([2., -2.]))
    wrapper = CCDistModelWrapper(model, (2,1,1), .1, 1, teacher=copy.deepcopy(model),
        device='cpu', warm_up_epochs=0, ramp_up_epochs=0, shi_reg_weight=0, pgd_steps=1)
    wrapper.train_model(loader)
    wrapper.eval()
    # Bypass screening only, to exercise the solver on the trained/exported network.
    def unresolved(**kwargs):
        return 0, 1, torch.zeros(1, dtype=torch.bool), {0: {'method': 'CROWN', 'running_time': 0.}}
    with tempfile.TemporaryDirectory() as root, \
         patch('CTRAIN.eval.eval.eval_adaptive', side_effect=unresolved):
        result = wrapper.evaluate_complete(loader, results_path=root, start_idx=0,
                                           end_idx=1, timeout=30, no_cores=1)
        records = json.loads(Path(root, 'results.json').read_text())
        assert records['0']['method'] == 'abCROWN' and records['0']['result'] == 'unsat', records
        assert result == (1., 1., 1.), result
    print('Real CC-Dist training -> ONNX -> VNNLIB -> abCROWN passed')


if __name__ == '__main__':
    import sys
    if '--cuda' in sys.argv:
        check_cuda()
    elif '--verifier' in sys.argv:
        check_verifier()
    else:
        check()
