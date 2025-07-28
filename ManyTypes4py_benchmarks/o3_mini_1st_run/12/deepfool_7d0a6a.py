from typing import Union, Optional, Tuple, Any, Callable
from typing_extensions import Literal
import eagerpy as ep
import logging
from abc import ABC, abstractmethod
from ..devutils import flatten
from ..devutils import atleast_kd
from ..models import Model
from ..criteria import Criterion
from ..distances import l2, linf
from .base import MinimizationAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs
from .base import verify_input_bounds

# Type for the loss function used in _get_loss_fn:
# It takes an input tensor and a class index integer, and returns a tuple of:
#   - a scalar loss of type ep.Tensor,
#   - an auxiliary tuple (losses: ep.Tensor, logits: ep.Tensor)
LossFnType = Callable[[ep.Tensor, int], Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]]

class DeepFoolAttack(MinimizationAttack, ABC):
    """A simple and fast gradient-based adversarial attack.

    Implements the `DeepFool`_ attack.

    Args:
        steps : Maximum number of steps to perform.
        candidates : Limit on the number of the most likely classes that should
            be considered. A small value is usually sufficient and much faster.
        overshoot : How much to overshoot the boundary.
        loss  Loss function to use inside the update function.

    .. _DeepFool:
            Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
            "DeepFool: a simple and accurate method to fool deep neural
            networks", https://arxiv.org/abs/1511.04599
    """

    def __init__(
        self,
        *,
        steps: int = 50,
        candidates: Optional[int] = 10,
        overshoot: float = 0.02,
        loss: Literal["logits", "crossentropy"] = "logits"
    ) -> None:
        self.steps = steps
        self.candidates = candidates
        self.overshoot = overshoot
        self.loss = loss

    def _get_loss_fn(self, model: Model, classes: ep.Tensor) -> LossFnType:
        N: int = len(classes)
        rows = range(N)
        i0 = classes[:, 0]
        if self.loss == "logits":

            def loss_fun(x: ep.Tensor, k: int) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
                logits: ep.Tensor = model(x)
                ik = classes[:, k]
                l0 = logits[rows, i0]
                lk = logits[rows, ik]
                loss = lk - l0
                return (loss.sum(), (loss, logits))
            return loss_fun

        elif self.loss == "crossentropy":

            def loss_fun(x: ep.Tensor, k: int) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
                logits: ep.Tensor = model(x)
                ik = classes[:, k]
                l0 = -ep.crossentropy(logits, i0)
                lk = -ep.crossentropy(logits, ik)
                loss = lk - l0
                return (loss.sum(), (loss, logits))
            return loss_fun

        else:
            raise ValueError(f"expected loss to be 'logits' or 'crossentropy', got '{self.loss}'")

    def run(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        early_stop: Optional[Callable[..., bool]] = None,
        **kwargs: Any
    ) -> Any:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs
        verify_input_bounds(x, model)
        criterion = get_criterion(criterion)
        min_, max_ = model.bounds
        logits: ep.Tensor = model(x)
        classes: ep.Tensor = logits.argsort(axis=-1).flip(axis=-1)
        if self.candidates is None:
            candidates: int = logits.shape[-1]
        else:
            candidates = min(self.candidates, logits.shape[-1])
            if not candidates >= 2:
                raise ValueError(
                    f'expected the model output to have atleast 2 classes, got {logits.shape[-1]}'
                )
            logging.info(f'Only testing the top-{candidates} classes')
            classes = classes[:, :candidates]
        N: int = len(x)
        rows = range(N)
        loss_fun: LossFnType = self._get_loss_fn(model, classes)
        # The value_and_grad_fn returns a function that given (x, k) returns:
        # (loss: ep.Tensor, aux: Tuple[ep.Tensor, ep.Tensor], grad: ep.Tensor)
        loss_aux_and_grad: Callable[[ep.Tensor, int], Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor], ep.Tensor]] = ep.value_and_grad_fn(x, loss_fun, has_aux=True)
        x0: ep.Tensor = x
        p_total: ep.Tensor = ep.zeros_like(x)
        for _ in range(self.steps):
            diffs = [loss_aux_and_grad(x, 1)]
            _, (_, logits), _ = diffs[0]
            is_adv: ep.Tensor = criterion(x, logits)
            if is_adv.all():
                break
            diffs += [loss_aux_and_grad(x, k) for k in range(2, candidates)]
            # diffs_: List of tuples (losses: ep.Tensor, grad: ep.Tensor)
            diffs_ = [(losses, grad) for _, (losses, _), grad in diffs]
            losses: ep.Tensor = ep.stack([lo for lo, _ in diffs_], axis=1)
            grads: ep.Tensor = ep.stack([g for _, g in diffs_], axis=1)
            assert losses.shape == (N, candidates - 1)
            assert grads.shape == (N, candidates - 1) + x0.shape[1:]
            distances: ep.Tensor = self.get_distances(losses, grads)
            assert distances.shape == (N, candidates - 1)
            best: ep.Tensor = distances.argmin(axis=1)
            distances = distances[tuple([slice(None), best])]
            losses = losses[tuple([slice(None), best])]
            grads = grads[tuple([slice(None), best])]
            assert distances.shape == (N,)
            assert losses.shape == (N,)
            assert grads.shape == x0.shape
            distances = distances + 0.0001
            p_step: ep.Tensor = self.get_perturbations(distances, grads)
            assert p_step.shape == x0.shape
            p_total = p_total + p_step
            x = ep.where(atleast_kd(is_adv, x.ndim), x, x0 + (1.0 + self.overshoot) * p_total)
            x = ep.clip(x, min_, max_)
        return restore_type(x)

    @abstractmethod
    def get_distances(self, losses: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:
        ...

    @abstractmethod
    def get_perturbations(self, distances: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:
        ...

class L2DeepFoolAttack(DeepFoolAttack):
    """A simple and fast gradient-based adversarial attack.

    Implements the DeepFool L2 attack. [#Moos15]_

    Args:
        steps : Maximum number of steps to perform.
        candidates : Limit on the number of the most likely classes that should
            be considered. A small value is usually sufficient and much faster.
        overshoot : How much to overshoot the boundary.
        loss  Loss function to use inside the update function.

    References:
        .. [#Moos15] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
            "DeepFool: a simple and accurate method to fool deep neural
            networks", https://arxiv.org/abs/1511.04599

    """
    distance = l2

    def get_distances(self, losses: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:
        return abs(losses) / (flatten(grads, keep=2).norms.l2(axis=-1) + 1e-08)

    def get_perturbations(self, distances: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:
        norm_factor: ep.Tensor = flatten(grads).norms.l2(axis=-1) + 1e-08
        scaled: ep.Tensor = distances / norm_factor
        return atleast_kd(scaled, grads.ndim) * grads

class LinfDeepFoolAttack(DeepFoolAttack):
    """A simple and fast gradient-based adversarial attack.

    Implements the `DeepFool`_ L-Infinity attack.

    Args:
        steps : Maximum number of steps to perform.
        candidates : Limit on the number of the most likely classes that should
            be considered. A small value is usually sufficient and much faster.
        overshoot : How much to overshoot the boundary.
        loss  Loss function to use inside the update function.

    .. _DeepFool:
            Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
            "DeepFool: a simple and accurate method to fool deep neural
            networks", https://arxiv.org/abs/1511.04599

    """
    distance = linf

    def get_distances(self, losses: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:
        return abs(losses) / (flatten(grads, keep=2).abs().sum(axis=-1) + 1e-08)

    def get_perturbations(self, distances: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:
        return atleast_kd(distances, grads.ndim) * grads.sign()