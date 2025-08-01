from typing import Union, Tuple, Any, Optional, Callable
from functools import partial
import numpy as np
import eagerpy as ep
from ..devutils import flatten
from ..devutils import atleast_kd
from ..types import Bounds
from ..models import Model
from ..distances import l2
from ..criteria import Misclassification
from ..criteria import TargetedMisclassification
from .base import MinimizationAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs
from .base import verify_input_bounds
from .gradient_descent_base import AdamOptimizer

class L2CarliniWagnerAttack(MinimizationAttack):
    """Implementation of the Carlini & Wagner L2 Attack. [#Carl16]_

    Args:
        binary_search_steps : Number of steps to perform in the binary search
            over the const c.
        steps : Number of optimization steps within each binary search step.
        stepsize : Stepsize to update the examples.
        confidence : Confidence required for an example to be marked as adversarial.
            Controls the gap between example and decision boundary.
        initial_const : Initial value of the const c with which the binary search starts.
        abort_early : Stop inner search as soons as an adversarial example has been found.
            Does not affect the binary search over the const c.

    References:
        .. [#Carl16] Nicholas Carlini, David Wagner, "Towards evaluating the robustness of
            neural networks. In 2017 ieee symposium on security and privacy"
            https://arxiv.org/abs/1608.04644
    """
    distance = l2

    def __init__(self, binary_search_steps: int = 9, steps: int = 10000, stepsize: float = 0.01, confidence: float = 0, initial_const: float = 0.001, abort_early: bool = True) -> None:
        self.binary_search_steps = binary_search_steps
        self.steps = steps
        self.stepsize = stepsize
        self.confidence = confidence
        self.initial_const = initial_const
        self.abort_early = abort_early

    def run(self, model: Model, inputs: Any, criterion: Any, *, early_stop: Optional[Callable] = None, **kwargs: Any) -> Any:
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

        def is_adversarial(perturbed: Any, logits: Any) -> Any:
            if change_classes_logits != 0:
                logits += ep.onehot_like(logits, classes, value=change_classes_logits)
            return criterion_(perturbed, logits)
        if classes.shape != (N,):
            name = 'target_classes' if targeted else 'labels'
            raise ValueError(f'expected {name} to have shape ({N},), got {classes.shape}')
        bounds = model.bounds
        to_attack_space = partial(_to_attack_space, bounds=bounds)
        to_model_space = partial(_to_model_space, bounds=bounds)
        x_attack = to_attack_space(x)
        reconstsructed_x = to_model_space(x_attack)
        rows = range(N)

        def loss_fun(delta: Any, consts: Any) -> Tuple[Any, Tuple[Any, Any]]:
            assert delta.shape == x_attack.shape
            assert consts.shape == (N,)
            x = to_model_space(x_attack + delta)
            logits = model(x)
            if targeted:
                c_minimize = best_other_classes(logits, classes)
                c_maximize = classes
            else:
                c_minimize = classes
                c_maximize = best_other_classes(logits, classes)
            is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)
            is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts
            squared_norms = flatten(x - reconstsructed_x).square().sum(axis=-1)
            loss = is_adv_loss.sum() + squared_norms.sum()
            return (loss, (x, logits))
        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)
        consts = self.initial_const * np.ones((N,))
        lower_bounds = np.zeros((N,))
        upper_bounds = np.inf * np.ones((N,))
        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.full(x, (N,), ep.inf)
        for binary_search_step in range(self.binary_search_steps):
            if binary_search_step == self.binary_search_steps - 1 and self.binary_search_steps >= 10:
                consts = np.minimum(upper_bounds, 10000000000.0)
            delta = ep.zeros_like(x_attack)
            optimizer = AdamOptimizer(delta, self.stepsize)
            found_advs = np.full((N,), fill_value=False)
            loss_at_previous_check = np.inf
            consts_ = ep.from_numpy(x, consts.astype(np.float32))
            for step in range(self.steps):
                loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_)
                delta -= optimizer(gradient)
                if self.abort_early and step % np.ceil(self.steps / 10) == 0:
                    if not loss <= 0.9999 * loss_at_previous_check:
                        break
                    loss_at_previous_check = loss.item()
                found_advs_iter = is_adversarial(perturbed, logits)
                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())
                norms = flatten(perturbed - x).norms.l2(axis=-1)
                closer = norms < best_advs_norms
                new_best = ep.logical_and(closer, found_advs_iter)
                new_best_ = atleast_kd(new_best, best_advs.ndim)
                best_advs = ep.where(new_best_, perturbed, best_advs)
                best_advs_norms = ep.where(new_best, norms, best_advs_norms)
            upper_bounds = np.where(found_advs, consts, upper_bounds)
            lower_bounds = np.where(found_advs, lower_bounds, consts)
            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = np.where(np.isinf(upper_bounds), consts_exponential_search, consts_binary_search)
        return restore_type(best_advs)

def best_other_classes(logits: Any, exclude: Any) -> Any:
    other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)

def _to_attack_space(x: Any, *, bounds: Bounds) -> Any:
    min_, max_ = bounds
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = (x - a) / b
    x = x * 0.999999
    x = x.arctanh()
    return x

def _to_model_space(x: Any, *, bounds: Bounds) -> Any:
    min_, max_ = bounds
    x = x.tanh()
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = x * b + a
    return x
