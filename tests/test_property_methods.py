"""Numerical and API checks for property strategies and wrapper package aliases."""
import copy
import importlib
import inspect
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from CTRAIN import PropertyModelWrapper
from CTRAIN.model_wrappers.certified import ShiIBPModelWrapper, SABRModelWrapper
from CTRAIN.model_wrappers.adversarial import PGDModelWrapper
from CTRAIN.properties import Outputs, And, Or, Box, PropertySpec, obligations
from CTRAIN.train.certified.losses.property_methods import TRAINING_METHODS, get_property_training_loss
from CTRAIN.train.certified.losses.properties import property_loss
from CTRAIN.attacks import pgd_attack
from CTRAIN.bound.properties import box_endpoints


def data():
    return DataLoader(TensorDataset(torch.tensor([[.1], [.3], [.7], [.9]], dtype=torch.double),
                                   torch.full((4, 1), .5, dtype=torch.double)), batch_size=2,
                      generator=torch.Generator().manual_seed(9))


def make(method='IBP', options=None, *, spec=None, model=None, epochs=2, **kwargs):
    model = model or nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 1)).double()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return PropertyModelWrapper(model, (1,), properties=[spec or PropertySpec('positive',
            Outputs()[0] >= .2, [Box([0.], [1.])])], task_loss=nn.MSELoss(), num_epochs=epochs,
            training_method=method, training_options=options, property_batch_size=2, lr=.01, **kwargs)


def rejects(function, text):
    try:
        function()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('Invalid method/options did not fail')


def imports_and_validation():
    from CTRAIN.model_wrappers import _EXPORTS
    for name, module in _EXPORTS.items():
        canonical = importlib.import_module('CTRAIN.model_wrappers'+module)
        legacy = importlib.import_module('CTRAIN.model_wrappers.'+module.rsplit('.', 1)[1])
        assert canonical is legacy
        assert getattr(canonical, name) is getattr(importlib.import_module('CTRAIN.model_wrappers'), name)
    for cls in (ShiIBPModelWrapper, SABRModelWrapper, PGDModelWrapper):
        sig = inspect.signature(cls)
        assert 'eps' in sig.parameters and 'objectives' not in sig.parameters
    # Actual constructor and trainer retain the simple robustness API.
    ordinary = ShiIBPModelWrapper(nn.Sequential(nn.Flatten(), nn.Linear(1, 2)), eps=.1, input_shape=(1, 1, 1),
        num_epochs=1, warm_up_epochs=0, ramp_up_epochs=0, device='cpu', shi_reg_weight=0.)
    ordinary.train_model(loader_for_classification())
    for method in ('TAPS', 'STAPS', 'SET-BASED', 'CC-DIST'):
        with patch('CTRAIN.model_wrappers.certified.property_model_wrapper.PropertyBounds',
                   side_effect=AssertionError('Unsupported method converted a graph')):
            rejects(lambda: make(method), 'not supported')
    for method,options in [('SABR', {'alpha': .1}), ('MTL-IBP', {'alpha': float('nan')}),
                           ('PGD', {'pgd_restarts': 0}), ('GLS', {'estimator': 'pgpe', 'samples': 3}),
                           ('IBP-R', {'regularisation_mask': 1}), ('IBP', {'relu_upper_retention': 0.}),
                           ('MTL-IBP', {'alpha': None}), ('CROWN-IBP', {'beta': 2.}), ('IBP', {1: 1})]:
        rejects(lambda: make(method, options), '')


def loader_for_classification():
    loader = DataLoader(TensorDataset(torch.tensor([[[[.1]]], [[[.9]]]]),
                                      torch.tensor([0, 1])), batch_size=2)
    loader.normalised=False;loader.mean=torch.zeros(1);loader.std=torch.ones(1)
    loader.min=torch.tensor(0.);loader.max=torch.tensor(1.)
    return loader


def training_and_selection_api():
    signature = inspect.signature(PropertyModelWrapper)
    assert signature.parameters['training_method'].default == 'CROWN-IBP'
    assert 'bound_method' not in signature.parameters and 'selection_bound_method' not in signature.parameters
    assert make('GLS', {'bound_method': 'CROWN-IBP'})._selection_bound_method == 'CROWN-IBP'

    # Selection and refinement must use fresh full-box bounds, independent of SABR's surrogate.
    model = nn.Linear(1, 1).double()
    with torch.no_grad():
        model.weight.fill_(.1); model.bias.fill_(-.3)
    w = make('SABR', {'pgd_steps': 1, 'relu_upper_retention': .8}, model=model,
             refinement=True, max_cells=2, split_count=1)
    calls = []
    original = type(w.bounds).__call__
    def capture(bounds, index, boxes, method, **kwargs):
        calls.append((method, kwargs.get('relu_upper_retention', 1.)))
        return original(bounds, index, boxes, method, **kwargs)
    with patch.object(type(w.bounds), '__call__', capture):
        w.train_model(data(), data(), end_epoch=1)
    assert calls == [('IBP', .8), ('IBP', .8), ('IBP', 1.), ('IBP', 1.)], calls
    assert len(w.partition) == 2 and w.inference_selected
    assert w.certify_properties(refine=False)['method'] == 'CROWN'

    # Lazy datasets have a separate epoch-evaluation path and use sound full-box IBP too.
    class Boxes(list):
        def to_dict(self):
            return dict(boxes=[box.to_dict() for box in self])
    pool = Boxes([Box([0.], [1.]), Box([.2], [.8])])
    lazy = make('PGD', {'pgd_steps': 1}, property_dataset=pool)
    from CTRAIN.eval.properties import evaluate_property_dataset
    with patch('CTRAIN.train.certified.properties.evaluate_property_dataset',
               wraps=evaluate_property_dataset) as evaluate:
        lazy.train_model(data(), end_epoch=1)
    assert evaluate.call_args.args[2] == 'IBP'

    # Migrate compatible checkpoints and reject changed selection policies before loading weights.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'old.pt'
        lazy.save_checkpoint(path)
        saved = torch.load(path, weights_only=False)
        config = saved['configuration']
        config['bound_method'] = config.pop('selection_bound_method')
        torch.save(saved, path)
        resumed = make('PGD', {'pgd_steps': 1}, property_dataset=pool)
        resumed.load_checkpoint(path)
        assert resumed.history == lazy.history and resumed.epoch == lazy.epoch
        config['bound_method'] = 'CROWN'
        torch.save(saved, path)
        incompatible = make('PGD', {'pgd_steps': 1}, property_dataset=pool)
        before = copy.deepcopy(incompatible.original_model.state_dict())
        rejects(lambda: incompatible.load_checkpoint(path), 'configuration differs')
        for name, value in before.items():
            torch.testing.assert_close(value, incompatible.original_model.state_dict()[name])


def methods_and_resume():
    y=Outputs()
    spec=PropertySpec('logical', And(y[0] >= .2, Or(y[0] >= .3, y[0] <= -.3)), [Box([0.], [1.])])
    for method in TRAINING_METHODS:
        for loss in ('hinge', 'cross_entropy'):
            opts=dict(loss=loss)
            if method in ('MTL-IBP', 'CC-IBP', 'EXP-IBP', 'SABR', 'PGD', 'IBP-R'):
                opts.update(pgd_steps=2, pgd_region_factor=1.5)
            for estimator in (('rgs','pgpe') if method=='GLS' else (None,)):
                if estimator:opts.update(estimator=estimator, samples=2)
                w=make(method,opts,spec=spec)
                expected = method if method in ('IBP', 'CROWN-IBP', 'CROWN') else ('CROWN' if method == 'GLS' else 'IBP')
                assert w._selection_bound_method == expected
                initial=copy.deepcopy(w.original_model)
                torch.manual_seed(31);w.train_model(data())
                assert w.original_model.training
                assert all(torch.isfinite(p).all() for p in w.original_model.parameters())
                assert any(not torch.equal(a,b) for a,b in zip(initial.parameters(),w.original_model.parameters()))
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in w.original_model.parameters())
                assert w.certify_properties(refine=False)['method']=='CROWN'
                # Every method's optimizer/attack state resumes identically.
                interrupted=make(method,opts,spec=spec,model=copy.deepcopy(initial))
                torch.manual_seed(31);training=data();interrupted.train_model(training,end_epoch=1)
                with tempfile.TemporaryDirectory(prefix='ctrain-property-method-') as folder:
                    p=Path(folder)/'epoch.pt';interrupted.save_checkpoint(p,training)
                    resumed=make(method,opts,spec=spec,model=copy.deepcopy(initial))
                    resumed.load_checkpoint(p);resumed.train_model(data())
                    for a,b in zip(w.original_model.parameters(),resumed.original_model.parameters()):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Checkpoints produced before method selection remain usable for original bound training.
    w=make('CROWN-IBP');w.train_model(data(),end_epoch=1)
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'old.pt';w.save_checkpoint(p)
        saved=torch.load(p,weights_only=False)
        saved['configuration']['bound_method'] = saved['configuration'].pop('selection_bound_method')
        saved['configuration'].pop('training_method');saved['configuration'].pop('training_options')
        torch.save(saved,p)
        make('CROWN-IBP').load_checkpoint(p)
        rejects(lambda: make('PGD').load_checkpoint(p), 'configuration differs')


def geometry_and_components():
    # The compiled attack formula must retain both strict/non-strict ties and
    # point gradients while sharing affine leaves across repeated predicates.
    y = Outputs()
    formula = And(Or(y[0] > 0, y[1] >= 0), Or(y[0] >= 0, y[1] < 0), y[2] > 0)
    spec = PropertySpec('ties', formula, [Box([0.], [1.])])
    compiled = make('PGD', spec=spec, model=nn.Linear(1, 3).double()).bounds
    points = torch.tensor([[0., 0., 1.], [0., -.1, 1.], [.2, -.1, .4], [-.3, .2, .5]],
                          dtype=torch.double, requires_grad=True)
    expected = torch.cat([f.margin(points) for f in obligations(formula)], dim=1)
    actual = compiled.point_margins(0, points)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(compiled.point_margins(0, points, boolean=True).all(1),
                       formula.evaluate(points).flatten())
    expected_grad = torch.autograd.grad(expected[2:].sum(), points, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual[2:].sum(), points)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    model=nn.Linear(2, 1).double()
    with torch.no_grad():model.weight.copy_(torch.tensor([[1., 0.]],dtype=torch.double));model.bias.zero_()
    spec=PropertySpec('outside', Outputs()[0] >= 1.5, [Box([0., .3],[1., .3])])
    w=PropertyModelWrapper(model,(2,),properties=[spec],task_loss=nn.MSELoss(),num_epochs=1,
        training_method='SABR',training_options={'subselection_ratio':.4,'pgd_region_factor':3.,'pgd_steps':2})
    seen=[];original=w.bounds.__class__.__call__
    def capture(bounds,index,boxes,*args,**kwargs):
        seen.extend(boxes)
        return original(bounds,index,boxes,*args,**kwargs)
    with patch.object(w.bounds.__class__,'__call__',capture):get_property_training_loss(w,0,spec.domains)
    assert seen and seen[0].lower[0]>=0 and seen[0].upper[0]<=1
    torch.testing.assert_close(seen[0].upper-seen[0].lower,torch.tensor([.4,0.],dtype=torch.double))
    torch.testing.assert_close(seen[0].lower[1],torch.tensor(.3,dtype=torch.double))
    for method in ('MTL-IBP','CC-IBP','EXP-IBP'):
        full=make('IBP',{'loss':'cross_entropy'});full.eval()
        hybrid=make(method,{'alpha':1.,'loss':'cross_entropy'},model=copy.deepcopy(full.original_model));hybrid.eval()
        with patch('CTRAIN.train.certified.losses.property_methods.pgd_attack',side_effect=AssertionError('alpha=1 attacked')):
            loss=get_property_training_loss(hybrid,0,hybrid.properties[0].domains)
        expected=get_property_training_loss(full,0,full.properties[0].domains)
        torch.testing.assert_close(loss,expected)
        left=torch.autograd.grad(loss,list(hybrid.original_model.parameters()))
        right=torch.autograd.grad(expected,list(full.original_model.parameters()))
        for a,b in zip(left,right):torch.testing.assert_close(a,b)
    crown=make('CROWN-IBP',{'beta':0.,'loss':'cross_entropy'},model=copy.deepcopy(full.original_model));crown.eval()
    torch.testing.assert_close(get_property_training_loss(crown,0,crown.properties[0].domains),expected)
    # Training shrinkage must never leak into full-box certification.
    net=nn.Sequential(nn.Linear(1,1),nn.ReLU(),nn.Linear(1,1)).double()
    with torch.no_grad():
        net[0].weight.fill_(1);net[0].bias.zero_();net[2].weight.fill_(1);net[2].bias.zero_()
    s=PropertySpec('upper',Outputs()[0]<=1.8,[Box([-1.],[2.])])
    shrink=make('IBP',{'relu_upper_retention':.8},model=net,spec=s)
    assert get_property_training_loss(shrink,0,s.domains)==0
    assert not shrink.certify_properties(method='IBP',refine=False)['certified']
    # Hinge attack must move even from a satisfied point; parameter gradients stay untouched.
    net=nn.Linear(1,1).double()
    with torch.no_grad():net.weight.fill_(1);net.bias.zero_()
    for p in net.parameters():p.grad=torch.full_like(p,7.)
    lo=torch.tensor([[0.]],dtype=torch.double);hi=torch.tensor([[1.]],dtype=torch.double)
    point=pgd_attack(net,(lo+hi)/2,None,lo,hi,n_steps=4,step_size=1.,early_stopping=False,device='cpu',
        loss_fn=lambda output,_:-output[:,0],is_success=lambda output,_:output[:,0]<.1)
    assert point.item()<.1
    assert all(torch.equal(p.grad,torch.full_like(p,7.)) for p in net.parameters())
    from CTRAIN.train.certified.gls import GLSOptimizer
    for estimator in ('rgs','pgpe'):
        original=[p.detach().clone() for p in net.parameters()]
        optimizer=GLSOptimizer(net,estimator=estimator,samples=2)
        try:
            optimizer.step(lambda: net.weight.sum()*float('nan'))
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Non-finite GLS objective was accepted')
        for p,saved in zip(net.parameters(),original):
            torch.testing.assert_close(p,saved)
            torch.testing.assert_close(p.grad,torch.full_like(p,7.))
    margins=torch.tensor([[.2,-.3],[1.,2.]],dtype=torch.double,requires_grad=True)
    ce=property_loss(margins,loss='cross_entropy',reduction='none')
    torch.testing.assert_close(ce,torch.logsumexp(torch.cat((margins.new_zeros(2,1),-margins),1),1))


if __name__=='__main__':
    torch.set_num_threads(1)
    imports_and_validation();training_and_selection_api();methods_and_resume();geometry_and_components()
    print('Property methods, package aliases, unchanged robustness API, geometry, gradients and resumption passed')
