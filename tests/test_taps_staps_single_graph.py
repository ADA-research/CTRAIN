"""Run with PYTHONPATH=. python tests/test_taps_staps_single_graph.py."""
from unittest.mock import patch

import torch
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.operators.normalization import BoundBatchNormalization

from CTRAIN.bound import bound_ibp
from CTRAIN.bound.taps import bound_taps, _get_bound_estimation_from_pts
from CTRAIN.model_definitions import CNN7_Shi
from CTRAIN.model_wrappers import TAPSModelWrapper, STAPSModelWrapper
from CTRAIN.train.certified.losses.taps import get_taps_loss
from CTRAIN.train.certified.util import synchronise_bn
from CTRAIN.util import construct_c


def check():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    data, target = torch.rand(4, 3, 8, 8), torch.tensor([0, 1, 2, 0])
    ptb = PerturbationLpNorm(x_L=data - .1, x_U=data + .1)
    for wrapper_type in (TAPSModelWrapper, STAPSModelWrapper):
        for split in (3, 13, 16):
            model = CNN7_Shi(in_shape=(3, 8, 8), width=2, linear_size=8, n_classes=3)
            wrapper = wrapper_type(model, (3, 8, 8), .1, 4, warm_up_epochs=0,
                                   ramp_up_epochs=2, block_sizes=(split, 20 - split),
                                   pgd_steps=2, device='cpu')
            graph = wrapper.bounded_model
            feature, classifier = graph.bounded_blocks
            graph_nodes = set(graph.nodes())
            assert feature.graph is classifier.graph is graph
            assert set(feature.operators) | set(classifier.operators) == graph_nodes
            assert sum(isinstance(m, BoundedModule) for m in graph.modules()) == 1
            # A fresh full forward updates exactly the BN objects used by IBP.
            first_bn = next(m for m in graph.nodes() if isinstance(m, BoundBatchNormalization))
            assert first_bn in feature.operators
            means = []
            for shift in (0., 1.):
                graph.train()
                graph(data + shift)
                expected = model.layers[0](data + shift).mean((0, 2, 3))
                assert torch.allclose(first_bn.current_mean, expected)
                cached = first_bn.current_mean
                shifted_ptb = PerturbationLpNorm(x_L=data + shift - .1, x_U=data + shift + .1)
                lower, upper = bound_ibp(feature, shifted_ptb, data + shift, None, bound_upper=True)
                assert first_bn.current_mean is cached
                means.append(cached.detach().clone())
            assert not torch.allclose(*means)

            # A classifier-only forward must neither execute the prefix nor
            # update inference running buffers; training BN still uses points.
            before_buffers = {name: b.clone() for name, b in graph.named_buffers()}
            momenta = {m: m.momentum for m in classifier.operators if isinstance(m, BoundBatchNormalization)}
            latent = ((lower + upper) / 2).detach()
            points = latent[:, None].expand(-1, 2, *latent.shape[1:]).clone()
            c = construct_c(data, target, 3)
            with patch.object(first_bn, 'forward', side_effect=AssertionError('prefix rerun')):
                estimated = _get_bound_estimation_from_pts(classifier, points, None, c)
            assert torch.isfinite(estimated).all()
            assert all(torch.equal(before_buffers[name], b) for name, b in graph.named_buffers())
            assert all(m.momentum == value for m, value in momenta.items())

            # Compare the cut-node IBP to an independently converted suffix
            # with the same active BN statistics, including gradients at the cut.
            suffix = nn.Sequential(model.layers[split:]).eval()
            reference = BoundedModule(suffix, latent, device='cpu')
            reference.train()
            current = {id(m.inputs[1].param): m for m in classifier.operators
                       if isinstance(m, BoundBatchNormalization)}
            for m in reference.modules():
                if isinstance(m, BoundBatchNormalization):
                    source = current[id(m.inputs[1].param)]
                    m.current_mean, m.current_var = source.current_mean, source.current_var
            latent_ptb = PerturbationLpNorm(x_L=lower, x_U=upper)
            kwargs = dict(ptb=latent_ptb, data=latent, target=target,
                          n_classes=3, bound_upper=True, relu_upper_retention=.8)
            actual, _ = bound_ibp(classifier, **kwargs)
            expected, _ = bound_ibp(reference, **kwargs)
            assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-5)
            actual_grads = torch.autograd.grad(actual.sum(), (lower, upper), retain_graph=True)
            expected_grads = torch.autograd.grad(expected.sum(), (lower, upper), retain_graph=True)
            assert all(torch.allclose(a, b, atol=1e-4, rtol=1e-5)
                       for a, b in zip(actual_grads, expected_grads))
            assert any(g.abs().max() > 0 for g in actual_grads)

            graph.train()
            for block in graph.bounded_blocks:
                block.train()
            graph(data)
            propagation = {} if wrapper_type is TAPSModelWrapper else dict(
                propagation='SABR', sabr_args=dict(hardened_model=graph,
                    original_model=model, data=data, target=target, eps=.1,
                    subselection_ratio=.7, data_min=0., data_max=1.,
                    n_classes=3, device='cpu', n_steps=2, early_stopping=False))
            loss, = get_taps_loss(model, graph, graph.bounded_blocks,
                                  nn.CrossEntropyLoss(reduction='none'), data, target,
                                  3, ptb, device='cpu', pgd_steps=2,
                                  relu_upper_retention=.8, **propagation)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert model.layers[0].weight.grad.abs().max() > 0

            with patch.object(graph, 'get_forward_value', side_effect=RuntimeError('failed')):
                try:
                    classifier(latent)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('expected failure')
            assert all(m.momentum == value for m, value in momenta.items())
            modes = {m: m.training for parent in (model, graph, feature, classifier)
                     for m in parent.modules()}
            gradients = {p: p.requires_grad for p in model.parameters()}
            with patch('CTRAIN.bound.taps._get_pivotal_points', side_effect=RuntimeError('attack failed')):
                try:
                    bound_taps(model, graph, graph.bounded_blocks, data, target,
                               3, ptb, device='cpu', **propagation)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('expected attack failure')
            assert all(m.training == value for m, value in modes.items())
            assert all(p.requires_grad == value for p, value in gradients.items())
            for m in (model, graph, feature, classifier):
                m.eval()
            before_buffers = {name: b.clone() for name, b in graph.named_buffers()}
            bound_taps(model, graph, graph.bounded_blocks, data, target, 3, ptb,
                       device='cpu', pgd_steps=2, **propagation)
            assert all(torch.equal(before_buffers[name], b) for name, b in graph.named_buffers())
            # Ordinary full forwards and sound IBP clear the injected cut cache.
            synchronise_bn(model, graph, None, 'cpu')
            output = graph(data)
            assert torch.allclose(output, model(data), atol=1e-5, rtol=1e-4)
            bound_ibp(graph, ptb, data, target, n_classes=3)
            assert all('interval_propagate' not in m.__dict__ for m in graph.nodes())
            fresh_graph = BoundedModule(model, data[:1], bound_opts=graph.bound_opts, device='cpu')
            fresh_graph.eval()
            with torch.no_grad():
                kwargs = dict(x=(BoundedTensor(data, ptb),), C=c, method='CROWN')
                actual, _ = graph.compute_bounds(**kwargs)
                expected, _ = fresh_graph.compute_bounds(**kwargs)
            assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    print('Single-graph TAPS/STAPS checks passed')


if __name__ == '__main__':
    check()
