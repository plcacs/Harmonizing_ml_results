import logging
from typing import Union, Any, Optional, Callable, List, Tuple, Dict, TypeVar
from typing_extensions import Literal
import math
import eagerpy as ep
import numpy as np
from foolbox.attacks import LinearSearchBlendedUniformNoiseAttack
from foolbox.tensorboard import TensorBoard
from ..models import Model
from ..criteria import Criterion
from ..distances import l1
from ..devutils import atleast_kd, flatten
from .base import MinimizationAttack, get_is_adversarial
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs
from .base import verify_input_bounds
from ..distances import l2, linf

T = TypeVar('T')
TensorType = TypeVar('TensorType', bound=ep.Tensor)
StepSizeSearch = Literal['geometric_progression', 'grid_search']
Constraint = Literal['l2', 'linf']

class HopSkipJumpAttack(MinimizationAttack):
    distance = l1

    def __init__(
        self,
        init_attack: Optional[MinimizationAttack] = None,
        steps: int = 64,
        initial_gradient_eval_steps: int = 100,
        max_gradient_eval_steps: int = 10000,
        stepsize_search: StepSizeSearch = 'geometric_progression',
        gamma: float = 1.0,
        tensorboard: Union[bool, str, None] = False,
        constraint: Constraint = 'l2'
    ) -> None:
        if init_attack is not None and (not isinstance(init_attack, MinimizationAttack)):
            raise NotImplementedError
        self.init_attack = init_attack
        self.steps = steps
        self.initial_num_evals = initial_gradient_eval_steps
        self.max_num_evals = max_gradient_eval_steps
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.tensorboard = tensorboard
        self.constraint = constraint
        assert constraint in ('l2', 'linf')
        if constraint == 'l2':
            self.distance = l2
        else:
            self.distance = linf

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        **kwargs: Any
    ) -> T:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs
        verify_input_bounds(originals, model)
        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)
        if starting_points is None:
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
                logging.info(f'Neither starting_points nor init_attack given. Falling back to {init_attack!r} for initialization.')
            else:
                init_attack = self.init_attack
            x_advs = init_attack.run(model, originals, criterion, early_stop=early_stop)
        else:
            x_advs = ep.astensor(starting_points)
        is_adv = is_adversarial(x_advs)
        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            if starting_points is None:
                raise ValueError(f'init_attack failed for {failed} of {len(is_adv)} inputs')
            else:
                raise ValueError(f'{failed} of {len(is_adv)} starting_points are not adversarial')
        del starting_points
        tb = TensorBoard(logdir=self.tensorboard)
        x_advs = self._binary_search(is_adversarial, originals, x_advs)
        assert ep.all(is_adversarial(x_advs))
        distances = self.distance(originals, x_advs)
        for step in range(self.steps):
            delta = self.select_delta(originals, distances, step)
            num_gradient_estimation_steps = int(min([self.initial_num_evals * math.sqrt(step + 1), self.max_num_evals]))
            gradients = self.approximate_gradients(is_adversarial, x_advs, num_gradient_estimation_steps, delta)
            if self.constraint == 'linf':
                update = ep.sign(gradients)
            else:
                update = gradients
            if self.stepsize_search == 'geometric_progression':
                epsilons = distances / math.sqrt(step + 1)
                while True:
                    x_advs_proposals = ep.clip(x_advs + atleast_kd(epsilons, x_advs.ndim) * update, 0, 1)
                    success = is_adversarial(x_advs_proposals)
                    epsilons = ep.where(success, epsilons, epsilons / 2.0)
                    if ep.all(success):
                        break
                x_advs = ep.clip(x_advs + atleast_kd(epsilons, update.ndim) * update, 0, 1)
                assert ep.all(is_adversarial(x_advs))
                x_advs = self._binary_search(is_adversarial, originals, x_advs)
                assert ep.all(is_adversarial(x_advs))
            elif self.stepsize_search == 'grid_search':
                epsilons_grid = ep.expand_dims(ep.from_numpy(distances, np.logspace(-4, 0, num=20, endpoint=True, dtype=np.float32)), 1) * ep.expand_dims(distances, 0)
                proposals_list = []
                for epsilons in epsilons_grid:
                    x_advs_proposals = x_advs + atleast_kd(epsilons, update.ndim) * update
                    x_advs_proposals = ep.clip(x_advs_proposals, 0, 1)
                    mask = is_adversarial(x_advs_proposals)
                    x_advs_proposals = self._binary_search(is_adversarial, originals, x_advs_proposals)
                    x_advs_proposals = ep.where(atleast_kd(mask, x_advs.ndim), x_advs_proposals, x_advs)
                    proposals_list.append(x_advs_proposals)
                proposals = ep.stack(proposals_list, 0)
                proposals_distances = self.distance(ep.expand_dims(originals, 0), proposals)
                minimal_idx = ep.argmin(proposals_distances, 0)
                x_advs = proposals[minimal_idx]
            distances = self.distance(originals, x_advs)
            tb.histogram('norms', distances, step)
        return restore_type(x_advs)

    def approximate_gradients(
        self,
        is_adversarial: Callable[[TensorType], TensorType],
        x_advs: TensorType,
        steps: int,
        delta: TensorType
    ) -> TensorType:
        noise_shape = tuple([steps] + list(x_advs.shape))
        if self.constraint == 'l2':
            rv = ep.normal(x_advs, noise_shape)
        elif self.constraint == 'linf':
            rv = ep.uniform(x_advs, low=-1, high=1, shape=noise_shape)
        rv /= atleast_kd(ep.norms.l2(flatten(rv, keep=1), rv.ndim) + 1e-12
        scaled_rv = atleast_kd(ep.expand_dims(delta, 0), rv.ndim) * rv
        perturbed = ep.expand_dims(x_advs, 0) + scaled_rv
        perturbed = ep.clip(perturbed, 0, 1)
        rv = (perturbed - x_advs) / atleast_kd(ep.expand_dims(delta + 1e-08, 0), rv.ndim)
        multipliers_list = []
        for step in range(steps):
            decision = is_adversarial(perturbed[step])
            multipliers_list.append(ep.where(decision, ep.ones(x_advs, len(x_advs)), -ep.ones(x_advs, len(decision))))
        multipliers = ep.stack(multipliers_list, 0)
        vals = ep.where(ep.abs(ep.mean(multipliers, axis=0, keepdims=True)) == 1, multipliers, multipliers - ep.mean(multipliers, axis=0, keepdims=True))
        grad = ep.mean(atleast_kd(vals, rv.ndim) * rv, axis=0)
        grad /= ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12
        return grad

    def _project(
        self,
        originals: TensorType,
        perturbed: TensorType,
        epsilons: TensorType
    ) -> TensorType:
        epsilons = atleast_kd(epsilons, originals.ndim)
        if self.constraint == 'linf':
            perturbation = perturbed - originals
            clipped_perturbed = ep.where(perturbation > epsilons, originals + epsilons, perturbed)
            clipped_perturbed = ep.where(perturbation < -epsilons, originals - epsilons, clipped_perturbed)
            return clipped_perturbed
        else:
            return (1.0 - epsilons) * originals + epsilons * perturbed

    def _binary_search(
        self,
        is_adversarial: Callable[[TensorType], TensorType],
        originals: TensorType,
        perturbed: TensorType
    ) -> TensorType:
        d = int(np.prod(perturbed.shape[1:]))
        if self.constraint == 'linf':
            highs = linf(originals, perturbed)
            thresholds = highs * self.gamma / (d * d)
        else:
            highs = ep.ones(perturbed, len(perturbed))
            thresholds = highs * self.gamma / (d * math.sqrt(d))
        lows = ep.zeros_like(highs)
        old_mids = highs
        while ep.any(highs - lows > thresholds):
            mids = (lows + highs) / 2
            mids_perturbed = self._project(originals, perturbed, mids)
            is_adversarial_ = is_adversarial(mids_perturbed)
            highs = ep.where(is_adversarial_, mids, highs)
            lows = ep.where(is_adversarial_, lows, mids)
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids
            if reached_numerical_precision:
                break
        res = self._project(originals, perturbed, highs)
        return res

    def select_delta(
        self,
        originals: TensorType,
        distances: TensorType,
        step: int
    ) -> TensorType:
        if step == 0:
            result = 0.1 * ep.ones_like(distances)
        else:
            d = int(np.prod(originals.shape[1:]))
            if self.constraint == 'linf':
                theta = self.gamma / (d * d)
                result = d * theta * distances
            else:
                theta = self.gamma / (d * np.sqrt(d))
                result = np.sqrt(d) * theta * distances
        return result
