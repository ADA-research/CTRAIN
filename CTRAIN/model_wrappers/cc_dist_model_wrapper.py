"""CC-Dist student wrapper; the teacher is never part of the certified network."""
import copy
import math
from functools import partial

from torch import nn
from auto_LiRPA.operators import BoundLinear

from CTRAIN.model_wrappers.cc_ibp_model_wrapper import CCIBPModelWrapper
from CTRAIN.train.certified.cc_dist import cc_dist_train_model
from CTRAIN.train.certified.losses.cc_dist import get_cc_dist_loss, validate_cc_dist_bn_options
from CTRAIN.bound.zonotope import sequential_layers
from CTRAIN.train.adversarial.pgd import validate_pgd_training_options


class CCDistModelWrapper(CCIBPModelWrapper):
    """CC-Dist from arXiv:2602.02626, Algorithm 1.

    ``teacher`` is a pretrained native classifier with ``layers`` (or a
    Sequential), ending in Linear. Its feature width must match the student's
    final Linear input. A private frozen copy supplies clean teacher features.
    All CC-IBP training/evaluation/HPO options remain available; beta defaults
    to 5 divided by the feature width. Set beta=0 for the paper's CC-IBP control.
    """
    _trainer = staticmethod(cc_dist_train_model)
    _clean_forward = False

    def __init__(self, model, input_shape, eps, num_epochs, *, teacher,
                 distillation_weight=None, **kwargs):
        warm, ramp = kwargs.get('warm_up_epochs', 1), kwargs.get('ramp_up_epochs', 70)
        if (type(num_epochs) is not int or num_epochs < 1 or type(warm) is not int
                or type(ramp) is not int or warm < 0 or ramp < 0 or warm+ramp > num_epochs):
            raise ValueError('Invalid epoch budget or warm-up/ramp-up schedule')
        validate_pgd_training_options(eps, kwargs.get('pgd_steps', 1),
            kwargs.get('pgd_alpha', 10), kwargs.get('pgd_restarts', 1),
            kwargs.get('gradient_clip', 10), kwargs.get('l1_reg_weight', 1e-6))
        validate_cc_dist_bn_options(model, kwargs.get('bound_opts', {}))
        if distillation_weight is not None and (
                not math.isfinite(distillation_weight) or distillation_weight < 0):
            raise ValueError('distillation_weight must be finite and nonnegative')
        student_layers = model if isinstance(model, nn.Sequential) else getattr(model, 'layers', None)
        teacher_layers = teacher if isinstance(teacher, nn.Sequential) else getattr(teacher, 'layers', None)
        if any(not isinstance(layers, nn.Sequential) or len(layers) < 2
               or not isinstance(layers[-1], nn.Linear)
               for layers in (student_layers, teacher_layers)):
            raise ValueError('CC-Dist requires Sequential classifiers ending in Linear')
        if (student_layers[-1].in_features != teacher_layers[-1].in_features
                or student_layers[-1].out_features != teacher_layers[-1].out_features):
            raise ValueError('Teacher and student feature widths and class counts must match')
        for classifier, layers in ((model, student_layers), (teacher, teacher_layers)):
            if sequential_layers(classifier) != list(layers):
                raise ValueError('CC-Dist feature extraction must match the classifier forward graph')
        private_features = copy.deepcopy(teacher_layers[:-1])
        super().__init__(model, input_shape, eps, num_epochs, **kwargs)
        self.cert_train_method = 'cc_dist'
        if not isinstance(self.bounded_model[self.bounded_model.output_name[0]], BoundLinear):
            raise ValueError('CC-Dist requires an affine classifier in the LiRPA graph')
        private_features.to(device=self.device, dtype=next(model.parameters()).dtype)
        private_features.eval().requires_grad_(False)
        # Teacher weights are training inputs, not inference weights or optimizer parameters.
        object.__setattr__(self, 'teacher_features', private_features)
        self.distillation_weight = distillation_weight
        self._loss_func = partial(get_cc_dist_loss, teacher_features=private_features,
                                  distillation_weight=distillation_weight)

    def _hpo_extra_options(self, config):
        teacher = nn.Sequential(*copy.deepcopy(list(self.teacher_features)),
                                copy.deepcopy(self.original_model.layers[-1]
                                              if hasattr(self.original_model, 'layers')
                                              else self.original_model[-1]))
        return dict(teacher=teacher, distillation_weight=config.get(
            'cc_dist:distillation_weight', self.distillation_weight))
