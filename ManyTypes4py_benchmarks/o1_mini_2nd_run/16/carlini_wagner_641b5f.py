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
from .base import (
    MinimizationAttack,
    T,
    get_criterion,
    raise_if_kwargs,
    verify_input_bounds,
)
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
        abort_early : Stop inner search as soon as an adversarial example has been found.
            Does not affect the binary search over the const c.

    References:
        .. [#Carl16] Nicholas Carlini, David Wagner, "Towards evaluating the robustness of
            neural networks. In 2017 IEEE Symposium on Security and Privacy"
            https://arxiv.org/abs/1608.04644
    """

    distance = l2

    def __init__(
        self,
        binary_search_steps: int = 9,
        steps: int = 10000,
        stepsize: float = 0.01,
        confidence: float = 0.0,
        initial_const: float = 0.001,
        abort_early: bool = True,
    ) -> None:
        self.binary_search_steps = binary_search_steps
        self.steps = steps
        self.stepsize = stepsize
        self.confidence = confidence
        self.initial_const = initial_const
        self.abort_early = abort_early

    def run(
        self,
        model: Model,
        inputs: ep.Tensor,
        criterion: Union[Misclassification, TargetedMisclassification],
        *,
        early_stop: Optional[Callable[[Any], bool]] = None,
        **kwargs: Any
    ) -> T:
        raise_if_kwargs(kwargs)
        x: ep.Tensor
        restore_type: Callable[[ep.Tensor], T]
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs
        verify_input_bounds(x, model)
        N: int = len(x)
        if isinstance(criterion_, Misclassification):
            targeted = False
            classes: ep.Tensor = criterion_.labels
            change_classes_logits: float = self.confidence
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes: ep.Tensor = criterion_.target_classes
            change_classes_logits: float = -self.confidence
        else:
            raise ValueError("unsupported criterion")

        def is_adversarial(perturbed: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
            if change_classes_logits != 0:
                logits += ep.onehot_like(logits, classes, value=change_classes_logits)
            return criterion_(perturbed, logits)

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )
        bounds: Bounds = model.bounds
        to_attack_space = partial(_to_attack_space, bounds=bounds)
        to_model_space = partial(_to_model_space, bounds=bounds)
        x_attack: ep.Tensor = to_attack_space(x)
        reconstructed_x: ep.Tensor = to_model_space(x_attack)
        rows = range(N)

        def loss_fun(
            delta: ep.Tensor, consts: ep.Tensor
        ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
            assert delta.shape == x_attack.shape
            assert consts.shape == (N,)
            x_modified: ep.Tensor = to_model_space(x_attack + delta)
            logits: ep.Tensor = model(x_modified)
            if targeted:
                c_minimize: ep.Tensor = best_other_classes(logits, classes)
                c_maximize: ep.Tensor = classes
            else:
                c_minimize: ep.Tensor = classes
                c_maximize: ep.Tensor = best_other_classes(logits, classes)
            is_adv_loss: ep.Tensor = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)
            is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts
            squared_norms: ep.Tensor = flatten(x_modified - reconstructed_x).square().sum(axis=-1)
            loss: ep.Tensor = is_adv_loss.sum() + squared_norms.sum()
            return loss, (x_modified, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)
        consts: np.ndarray = self.initial_const * np.ones((N,), dtype=np.float32)
        lower_bounds: np.ndarray = np.zeros((N,), dtype=np.float32)
        upper_bounds: np.ndarray = np.full((N,), np.inf, dtype=np.float32)
        best_advs: ep.Tensor = ep.zeros_like(x)
        best_advs_norms: ep.Tensor = ep.full(x, (N,), ep.inf)
        for binary_search_step in range(self.binary_search_steps):
            if (
                binary_search_step == self.binary_search_steps - 1
                and self.binary_search_steps >= 10
            ):
                consts = np.minimum(upper_bounds, 10000000000.0)
            delta: ep.Tensor = ep.zeros_like(x_attack)
            optimizer = AdamOptimizer(delta, self.stepsize)
            found_advs: np.ndarray = np.full((N,), fill_value=False, dtype=bool)
            loss_at_previous_check: float = float("inf")
            consts_: ep.Tensor = ep.from_numpy(x, consts.astype(np.float32))
            for step in range(self.steps):
                loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_)
                delta = delta - optimizer(gradient)
                if self.abort_early and step % int(np.ceil(self.steps / 10)) == 0:
                    if loss > 0.9999 * loss_at_previous_check:
                        break
                    loss_at_previous_check = loss.item()
                found_advs_iter: ep.Tensor = is_adversarial(perturbed, logits)
                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())
                norms: ep.Tensor = flatten(perturbed - x).norms.l2(axis=-1)
                closer: ep.Tensor = norms < best_advs_norms
                new_best: ep.Tensor = ep.logical_and(closer, found_advs_iter)
                new_best_: ep.Tensor = atleast_kd(new_best, best_advs.ndim)
                best_advs = ep.where(new_best_, perturbed, best_advs)
                best_advs_norms = ep.where(closer, norms, best_advs_norms)
            upper_bounds = np.where(found_advs, consts, upper_bounds)
            lower_bounds = np.where(found_advs, lower_bounds, consts)
            consts_exponential_search: np.ndarray = consts * 10
            consts_binary_search: np.ndarray = (lower_bounds + upper_bounds) / 2
            consts = np.where(
                np.isinf(upper_bounds), consts_exponential_search, consts_binary_search
            )
        return restore_type(best_advs)


def best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    other_logits: ep.Tensor = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)


def _to_attack_space(x: ep.Tensor, *, bounds: Bounds) -> ep.Tensor:
    min_, max_ = bounds
    a: ep.Tensor = (min_ + max_) / 2
    b: ep.Tensor = (max_ - min_) / 2
    x = (x - a) / b
    x = x * 0.999999
    x = x.arctanh()
    return x


def _to_model_space(x: ep.Tensor, *, bounds: Bounds) -> ep.Tensor:
    min_, max_ = bounds
    x = x.tanh()
    a: ep.Tensor = (min_ + max_) / 2
    b: ep.Tensor = (max_ - min_) / 2
    x = x * b + a
    return x
