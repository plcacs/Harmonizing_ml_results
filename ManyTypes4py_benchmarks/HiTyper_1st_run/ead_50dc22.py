from typing import Union, Tuple, Any, Optional
from typing_extensions import Literal
import math
import eagerpy as ep
from ..models import Model
from ..criteria import Misclassification, TargetedMisclassification
from ..distances import l1
from ..devutils import atleast_kd, flatten
from .base import MinimizationAttack
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs
from .base import verify_input_bounds

class EADAttack(MinimizationAttack):
    """Implementation of the EAD Attack with EN Decision Rule. [#Chen18]_

    Args:
        binary_search_steps : Number of steps to perform in the binary search
            over the const c.
        steps : Number of optimization steps within each binary search step.
        initial_stepsize : Initial stepsize to update the examples.
        confidence : Confidence required for an example to be marked as adversarial.
            Controls the gap between example and decision boundary.
        initial_const : Initial value of the const c with which the binary search starts.
        regularization : Controls the L1 regularization.
        decision_rule : Rule according to which the best adversarial examples are selected.
            They either minimize the L1 or ElasticNet distance.
        abort_early : Stop inner search as soons as an adversarial example has been found.
            Does not affect the binary search over the const c.

    References:
        .. [#Chen18] Pin-Yu Chen, Yash Sharma, Huan Zhang, Jinfeng Yi, Cho-Jui Hsieh,
        "EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples",
        https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16893
    """
    distance = l1

    def __init__(self, binary_search_steps: int=9, steps: int=10000, initial_stepsize: float=0.01, confidence: float=0.0, initial_const: float=0.001, regularization: float=0.01, decision_rule: typing.Text='EN', abort_early: bool=True) -> None:
        if decision_rule not in ('EN', 'L1'):
            raise ValueError('invalid decision rule')
        self.binary_search_steps = binary_search_steps
        self.steps = steps
        self.confidence = confidence
        self.initial_stepsize = initial_stepsize
        self.regularization = regularization
        self.initial_const = initial_const
        self.abort_early = abort_early
        self.decision_rule = decision_rule

    def run(self, model: Union[models.Model, criteria.Criterion, base.T], inputs: Union[float, None, base.T], criterion: Union[float, None, base.T], *, early_stop: Union[None, float, base.T]=None, **kwargs):
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs
        verify_input_bounds(x, model)
        N = len(x)
        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
            change_classes_logits = self.confidence
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
            change_classes_logits = -self.confidence
        else:
            raise ValueError('unsupported criterion')

        def is_adversarial(perturbed: Any, logits: Any):
            if change_classes_logits != 0:
                logits += ep.onehot_like(logits, classes, value=change_classes_logits)
            return criterion_(perturbed, logits)
        if classes.shape != (N,):
            name = 'target_classes' if targeted else 'labels'
            raise ValueError(f'expected {name} to have shape ({N},), got {classes.shape}')
        min_, max_ = model.bounds
        rows = range(N)

        def loss_fun(y_k: Any, consts: Any) -> tuple:
            assert y_k.shape == x.shape
            assert consts.shape == (N,)
            logits = model(y_k)
            if targeted:
                c_minimize = _best_other_classes(logits, classes)
                c_maximize = classes
            else:
                c_minimize = classes
                c_maximize = _best_other_classes(logits, classes)
            is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)
            is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts
            squared_norms = flatten(y_k - x).square().sum(axis=-1)
            loss = is_adv_loss.sum() + squared_norms.sum()
            return (loss, logits)
        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)
        consts = self.initial_const * ep.ones(x, (N,))
        lower_bounds = ep.zeros(x, (N,))
        upper_bounds = ep.inf * ep.ones(x, (N,))
        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.ones(x, (N,)) * ep.inf
        for binary_search_step in range(self.binary_search_steps):
            if binary_search_step == self.binary_search_steps - 1 and self.binary_search_steps >= 10:
                consts = ep.minimum(upper_bounds, 10000000000.0)
            x_k = x
            y_k = x
            found_advs = ep.full(x, (N,), value=False).bool()
            loss_at_previous_check = ep.inf
            for iteration in range(self.steps):
                stepsize = self.initial_stepsize * (1.0 - iteration / self.steps) ** 0.5
                loss, logits, gradient = loss_aux_and_grad(y_k, consts)
                x_k_old = x_k
                x_k = _project_shrinkage_thresholding(y_k - stepsize * gradient, x, self.regularization, min_, max_)
                y_k = x_k + iteration / (iteration + 3.0) * (x_k - x_k_old)
                if self.abort_early and iteration % math.ceil(self.steps / 10) == 0:
                    if not loss.item() <= 0.9999 * loss_at_previous_check:
                        break
                    loss_at_previous_check = loss.item()
                found_advs_iter = is_adversarial(x_k, model(x_k))
                best_advs, best_advs_norms = _apply_decision_rule(self.decision_rule, self.regularization, best_advs, best_advs_norms, x_k, x, found_advs_iter)
                found_advs = ep.logical_or(found_advs, found_advs_iter)
            upper_bounds = ep.where(found_advs, consts, upper_bounds)
            lower_bounds = ep.where(found_advs, lower_bounds, consts)
            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = ep.where(ep.isinf(upper_bounds), consts_exponential_search, consts_binary_search)
        return restore_type(best_advs)

def _best_other_classes(logits: Union[eagerpy.Tensor, float, tuple[bool]], exclude: Union[eagerpy.Tensor, float, tuple[bool]]) -> Union[bool, set[typing.Match], float]:
    other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)

def _apply_decision_rule(decision_rule: Union[float, list, int], beta: float, best_advs: Union[int, float], best_advs_norms: float, x_k: float, x: float, found_advs: Union[float, bool]) -> tuple:
    if decision_rule == 'EN':
        norms = beta * flatten(x_k - x).abs().sum(axis=-1) + flatten(x_k - x).square().sum(axis=-1)
    else:
        norms = flatten(x_k - x).abs().sum(axis=-1)
    new_best = ep.logical_and(norms < best_advs_norms, found_advs)
    new_best_kd = atleast_kd(new_best, best_advs.ndim)
    best_advs = ep.where(new_best_kd, x_k, best_advs)
    best_advs_norms = ep.where(new_best, norms, best_advs_norms)
    return (best_advs, best_advs_norms)

def _project_shrinkage_thresholding(z: Union[float, eagerpy.Tensor], x0: Union[float, eagerpy.Tensor], regularization: Union[float, eagerpy.Tensor], min_: Union[float, str], max_: Union[float, eagerpy.Tensor, str]):
    """Performs the element-wise projected shrinkage-thresholding
    operation"""
    upper_mask = z - x0 > regularization
    lower_mask = z - x0 < -regularization
    projection = ep.where(upper_mask, ep.minimum(z - regularization, max_), x0)
    projection = ep.where(lower_mask, ep.maximum(z + regularization, min_), projection)
    return projection