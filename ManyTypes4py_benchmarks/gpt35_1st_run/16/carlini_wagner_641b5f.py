from typing import Union, Tuple, Any, Optional
import numpy as np
import eagerpy as ep
from ..types import Bounds
from ..models import Model
from ..criteria import Misclassification, TargetedMisclassification
from .base import MinimizationAttack, T, get_criterion, raise_if_kwargs, verify_input_bounds
from .gradient_descent_base import AdamOptimizer

class L2CarliniWagnerAttack(MinimizationAttack):
    distance: Callable[[ep.TensorType, ep.TensorType], ep.TensorType] = l2

    def __init__(self, binary_search_steps: int = 9, steps: int = 10000, stepsize: float = 0.01, confidence: float = 0, initial_const: float = 0.001, abort_early: bool = True) -> None:
        self.binary_search_steps: int = binary_search_steps
        self.steps: int = steps
        self.stepsize: float = stepsize
        self.confidence: float = confidence
        self.initial_const: float = initial_const
        self.abort_early: bool = abort_early

    def run(self, model: Model, inputs: ep.TensorType, criterion: Union[Misclassification, TargetedMisclassification], *, early_stop: Optional[Any] = None) -> ep.TensorType:
        raise_if_kwargs({})
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion
        verify_input_bounds(x, model)
        N: int = len(x)
        if isinstance(criterion_, Misclassification):
            targeted: bool = False
            classes: ep.TensorType = criterion_.labels
            change_classes_logits: float = self.confidence
        elif isinstance(criterion_, TargetedMisclassification):
            targeted: bool = True
            classes: ep.TensorType = criterion_.target_classes
            change_classes_logits: float = -self.confidence
        else:
            raise ValueError('unsupported criterion')

        def is_adversarial(perturbed: ep.TensorType, logits: ep.TensorType) -> ep.TensorType:
            if change_classes_logits != 0:
                logits += ep.onehot_like(logits, classes, value=change_classes_logits)
            return criterion_(perturbed, logits)
        if classes.shape != (N,):
            name: str = 'target_classes' if targeted else 'labels'
            raise ValueError(f'expected {name} to have shape ({N},), got {classes.shape}')
        bounds: Bounds = model.bounds
        to_attack_space = partial(_to_attack_space, bounds=bounds)
        to_model_space = partial(_to_model_space, bounds=bounds)
        x_attack: ep.TensorType = to_attack_space(x)
        reconstsructed_x: ep.TensorType = to_model_space(x_attack)
        rows: range = range(N)

        def loss_fun(delta: ep.TensorType, consts: ep.TensorType) -> Tuple[ep.TensorType, Tuple[ep.TensorType, ep.TensorType]]:
            assert delta.shape == x_attack.shape
            assert consts.shape == (N,)
            x: ep.TensorType = to_model_space(x_attack + delta)
            logits: ep.TensorType = model(x)
            if targeted:
                c_minimize: ep.TensorType = best_other_classes(logits, classes)
                c_maximize: ep.TensorType = classes
            else:
                c_minimize: ep.TensorType = classes
                c_maximize: ep.TensorType = best_other_classes(logits, classes)
            is_adv_loss: ep.TensorType = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)
            is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts
            squared_norms: ep.TensorType = flatten(x - reconstsructed_x).square().sum(axis=-1)
            loss: ep.TensorType = is_adv_loss.sum() + squared_norms.sum()
            return (loss, (x, logits))
        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)
        consts: np.ndarray = self.initial_const * np.ones((N,))
        lower_bounds: np.ndarray = np.zeros((N,))
        upper_bounds: np.ndarray = np.inf * np.ones((N,))
        best_advs: ep.TensorType = ep.zeros_like(x)
        best_advs_norms: ep.TensorType = ep.full(x, (N,), ep.inf)
        for binary_search_step in range(self.binary_search_steps):
            if binary_search_step == self.binary_search_steps - 1 and self.binary_search_steps >= 10:
                consts = np.minimum(upper_bounds, 10000000000.0)
            delta: ep.TensorType = ep.zeros_like(x_attack)
            optimizer: AdamOptimizer = AdamOptimizer(delta, self.stepsize)
            found_advs: np.ndarray = np.full((N,), fill_value=False)
            loss_at_previous_check: float = np.inf
            consts_: ep.TensorType = ep.from_numpy(x, consts.astype(np.float32))
            for step in range(self.steps):
                loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_)
                delta -= optimizer(gradient)
                if self.abort_early and step % np.ceil(self.steps / 10) == 0:
                    if not loss <= 0.9999 * loss_at_previous_check:
                        break
                    loss_at_previous_check = loss.item()
                found_advs_iter: np.ndarray = is_adversarial(perturbed, logits)
                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())
                norms: ep.TensorType = flatten(perturbed - x).norms.l2(axis=-1)
                closer: ep.TensorType = norms < best_advs_norms
                new_best: ep.TensorType = ep.logical_and(closer, found_advs_iter)
                new_best_: ep.TensorType = atleast_kd(new_best, best_advs.ndim)
                best_advs = ep.where(new_best_, perturbed, best_advs)
                best_advs_norms = ep.where(new_best, norms, best_advs_norms)
            upper_bounds = np.where(found_advs, consts, upper_bounds)
            lower_bounds = np.where(found_advs, lower_bounds, consts)
            consts_exponential_search: np.ndarray = consts * 10
            consts_binary_search: np.ndarray = (lower_bounds + upper_bounds) / 2
            consts = np.where(np.isinf(upper_bounds), consts_exponential_search, consts_binary_search)
        return restore_type(best_advs)

def best_other_classes(logits: ep.TensorType, exclude: ep.TensorType) -> ep.TensorType:
    other_logits: ep.TensorType = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)

def _to_attack_space(x: ep.TensorType, *, bounds: Bounds) -> ep.TensorType:
    min_, max_ = bounds
    a: ep.TensorType = (min_ + max_) / 2
    b: ep.TensorType = (max_ - min_) / 2
    x = (x - a) / b
    x = x * 0.999999
    x = x.arctanh()
    return x

def _to_model_space(x: ep.TensorType, *, bounds: Bounds) -> ep.TensorType:
    min_, max_ = bounds
    x = x.tanh()
    a: ep.TensorType = (min_ + max_) / 2
    b: ep.TensorType = (max_ - min_) / 2
    x = x * b + a
    return x
