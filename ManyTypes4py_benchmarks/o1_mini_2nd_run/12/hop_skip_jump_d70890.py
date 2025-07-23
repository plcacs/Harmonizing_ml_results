import logging
from typing import Union, Any, Optional, Callable, List, TypeVar
from typing_extensions import Literal
import math
import eagerpy as ep
import numpy as np
from foolbox.attacks import LinearSearchBlendedUniformNoiseAttack
from foolbox.tensorboard import TensorBoard
from ..models import Model
from ..criteria import Criterion
from ..distances import l1, l2, linf
from ..devutils import atleast_kd, flatten
from .base import MinimizationAttack, get_is_adversarial, get_criterion, T, raise_if_kwargs, verify_input_bounds

class HopSkipJumpAttack(MinimizationAttack):
    """A powerful adversarial attack that requires neither gradients
    nor probabilities [#Chen19].

    Args:
        init_attack : Attack to use to find a starting points. Defaults to
            LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
        steps : Number of optimization steps within each binary search step.
        initial_gradient_eval_steps: Initial number of evaluations for gradient estimation.
            Larger initial_num_evals increases time efficiency, but
            may decrease query efficiency.
        max_gradient_eval_steps : Maximum number of evaluations for gradient estimation.
        stepsize_search : How to search for stepsize; choices are 'geometric_progression',
            'grid_search'. 'geometric progression' initializes the stepsize
            by ||x_t - x||_p / sqrt(iteration), and keep decreasing by half
            until reaching the target side of the boundary. 'grid_search'
            chooses the optimal epsilon over a grid, in the scale of
            ||x_t - x||_p.
        gamma : The binary search threshold theta is gamma / d^1.5 for
                   l2 attack and gamma / d^2 for linf attack.
        tensorboard : The log directory for TensorBoard summaries. If False, TensorBoard
            summaries will be disabled (default). If None, the logdir will be
            runs/CURRENT_DATETIME_HOSTNAME.
        constraint : Norm to minimize, either "l2" or "linf"

    References:
        .. [#Chen19] Jianbo Chen, Michael I. Jordan, Martin J. Wainwright,
        "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack",
        https://arxiv.org/abs/1904.02144
    """
    distance: Callable[[ep.Tensor, ep.Tensor], ep.Tensor] = l1

    def __init__(
        self,
        init_attack: Optional[MinimizationAttack] = None,
        steps: int = 64,
        initial_gradient_eval_steps: int = 100,
        max_gradient_eval_steps: int = 10000,
        stepsize_search: Literal['geometric_progression', 'grid_search'] = 'geometric_progression',
        gamma: float = 1.0,
        tensorboard: Optional[str] = False,
        constraint: Literal['l2', 'linf'] = 'l2'
    ) -> None:
        if init_attack is not None and not isinstance(init_attack, MinimizationAttack):
            raise NotImplementedError
        self.init_attack: Optional[MinimizationAttack] = init_attack
        self.steps: int = steps
        self.initial_num_evals: int = initial_gradient_eval_steps
        self.max_num_evals: int = max_gradient_eval_steps
        self.stepsize_search: Literal['geometric_progression', 'grid_search'] = stepsize_search
        self.gamma: float = gamma
        self.tensorboard: Optional[str] = tensorboard
        self.constraint: Literal['l2', 'linf'] = constraint
        assert constraint in ('l2', 'linf')
        if constraint == 'l2':
            self.distance = l2
        else:
            self.distance = linf

    def run(
        self,
        model: Model,
        inputs: Any,
        criterion: Criterion,
        *,
        early_stop: Optional[Callable[[ep.Tensor], bool]] = None,
        starting_points: Optional[Any] = None,
        **kwargs: Any
    ) -> Any:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs
        verify_input_bounds(originals, model)
        criterion_obj: Criterion = get_criterion(criterion)
        is_adversarial: Callable[[ep.Tensor], ep.Tensor] = get_is_adversarial(criterion_obj, model)
        if starting_points is None:
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
                logging.info(f'Neither starting_points nor init_attack given. Falling back to {init_attack!r} for initialization.')
            else:
                init_attack = self.init_attack
            x_advs: ep.Tensor = init_attack.run(model, originals, criterion_obj, early_stop=early_stop)
        else:
            x_advs = ep.astensor(starting_points)
        is_adv: ep.Tensor = is_adversarial(x_advs)
        if not is_adv.all():
            failed: float = is_adv.logical_not().float32().sum().item()
            if starting_points is None:
                raise ValueError(f'init_attack failed for {failed} of {len(is_adv)} inputs')
            else:
                raise ValueError(f'{failed} of {len(is_adv)} starting_points are not adversarial')
        del starting_points
        tb: TensorBoard = TensorBoard(logdir=self.tensorboard)
        x_advs = self._binary_search(is_adversarial, originals, x_advs)
        assert ep.all(is_adversarial(x_advs))
        distances: ep.Tensor = self.distance(originals, x_advs)
        for step in range(self.steps):
            delta: ep.Tensor = self.select_delta(originals, distances, step)
            num_gradient_estimation_steps: int = int(min([self.initial_num_evals * math.sqrt(step + 1), self.max_num_evals]))
            gradients: ep.Tensor = self.approximate_gradients(is_adversarial, x_advs, num_gradient_estimation_steps, delta)
            if self.constraint == 'linf':
                update: ep.Tensor = ep.sign(gradients)
            else:
                update = gradients
            if self.stepsize_search == 'geometric_progression':
                epsilons: ep.Tensor = distances / math.sqrt(step + 1)
                while True:
                    x_advs_proposals: ep.Tensor = ep.clip(x_advs + atleast_kd(epsilons, x_advs.ndim) * update, 0, 1)
                    success: ep.Tensor = is_adversarial(x_advs_proposals)
                    epsilons = ep.where(success, epsilons, epsilons / 2.0)
                    if ep.all(success):
                        break
                x_advs = ep.clip(x_advs + atleast_kd(epsilons, update.ndim) * update, 0, 1)
                assert ep.all(is_adversarial(x_advs))
                x_advs = self._binary_search(is_adversarial, originals, x_advs)
                assert ep.all(is_adversarial(x_advs))
            elif self.stepsize_search == 'grid_search':
                epsilons_grid: ep.Tensor = ep.expand_dims(
                    ep.from_numpy(
                        distances.numpy(),  # Assuming distances can be converted to numpy
                        np.logspace(-4, 0, num=20, endpoint=True, dtype=np.float32)
                    ),
                    1
                ) * ep.expand_dims(distances, 0)
                proposals_list: List[ep.Tensor] = []
                for epsilons in epsilons_grid:
                    x_advs_proposals: ep.Tensor = x_advs + atleast_kd(epsilons, update.ndim) * update
                    x_advs_proposals = ep.clip(x_advs_proposals, 0, 1)
                    mask: ep.Tensor = is_adversarial(x_advs_proposals)
                    x_advs_proposals = self._binary_search(is_adversarial, originals, x_advs_proposals)
                    x_advs_proposals = ep.where(atleast_kd(mask, x_advs.ndim), x_advs_proposals, x_advs)
                    proposals_list.append(x_advs_proposals)
                proposals: ep.Tensor = ep.stack(proposals_list, 0)
                proposals_distances: ep.Tensor = self.distance(ep.expand_dims(originals, 0), proposals)
                minimal_idx: ep.Tensor = ep.argmin(proposals_distances, 0)
                x_advs = proposals[minimal_idx]
            distances = self.distance(originals, x_advs)
            tb.histogram('norms', distances, step)
        return restore_type(x_advs)

    def approximate_gradients(
        self,
        is_adversarial: Callable[[ep.Tensor], ep.Tensor],
        x_advs: ep.Tensor,
        steps: int,
        delta: ep.Tensor
    ) -> ep.Tensor:
        noise_shape: tuple = (steps,) + tuple(x_advs.shape)
        if self.constraint == 'l2':
            rv: ep.Tensor = ep.normal(x_advs, noise_shape)
        elif self.constraint == 'linf':
            rv = ep.uniform(x_advs, low=-1, high=1, shape=noise_shape)
        rv = rv / (atleast_kd(ep.norms.l2(flatten(rv, keep=1), -1), rv.ndim) + 1e-12)
        scaled_rv: ep.Tensor = atleast_kd(ep.expand_dims(delta, 0), rv.ndim) * rv
        perturbed: ep.Tensor = ep.expand_dims(x_advs, 0) + scaled_rv
        perturbed = ep.clip(perturbed, 0, 1)
        rv = (perturbed - x_advs) / (atleast_kd(ep.expand_dims(delta + 1e-08, 0), rv.ndim))
        multipliers_list: List[ep.Tensor] = []
        for step_idx in range(steps):
            decision: ep.Tensor = is_adversarial(perturbed[step_idx])
            multipliers_list.append(ep.where(decision, ep.ones(x_advs, len(x_advs)), -ep.ones(x_advs, len(decision))))
        multipliers: ep.Tensor = ep.stack(multipliers_list, 0)
        vals: ep.Tensor = ep.where(
            ep.abs(ep.mean(multipliers, axis=0, keepdims=True)) == 1,
            multipliers,
            multipliers - ep.mean(multipliers, axis=0, keepdims=True)
        )
        grad: ep.Tensor = ep.mean(atleast_kd(vals, rv.ndim) * rv, axis=0)
        grad = grad / (ep.norms.l2(atleast_kd(flatten(grad), grad.ndim)) + 1e-12)
        return grad

    def _project(
        self,
        originals: ep.Tensor,
        perturbed: ep.Tensor,
        epsilons: ep.Tensor
    ) -> ep.Tensor:
        """Clips the perturbations to epsilon and returns the new perturbed

        Args:
            originals: A batch of reference inputs.
            perturbed: A batch of perturbed inputs.
            epsilons: A batch of norm values to project to.
        Returns:
            A tensor like perturbed but with the perturbation clipped to epsilon.
        """
        epsilons = atleast_kd(epsilons, originals.ndim)
        if self.constraint == 'linf':
            perturbation: ep.Tensor = perturbed - originals
            clipped_perturbed: ep.Tensor = ep.where(perturbation > epsilons, originals + epsilons, perturbed)
            clipped_perturbed = ep.where(perturbation < -epsilons, originals - epsilons, clipped_perturbed)
            return clipped_perturbed
        else:
            return (1.0 - epsilons) * originals + epsilons * perturbed

    def _binary_search(
        self,
        is_adversarial: Callable[[ep.Tensor], ep.Tensor],
        originals: ep.Tensor,
        perturbed: ep.Tensor
    ) -> ep.Tensor:
        d: int = int(np.prod(perturbed.shape[1:]))
        if self.constraint == 'linf':
            highs: ep.Tensor = linf(originals, perturbed)
            thresholds: ep.Tensor = highs * self.gamma / (d * d)
        else:
            highs = ep.ones(perturbed, len(perturbed))
            thresholds = highs * self.gamma / (d * math.sqrt(d))
        lows: ep.Tensor = ep.zeros_like(highs)
        old_mids: ep.Tensor = highs
        while ep.any(highs - lows > thresholds):
            mids: ep.Tensor = (lows + highs) / 2
            mids_perturbed: ep.Tensor = self._project(originals, perturbed, mids)
            is_adversarial_: ep.Tensor = is_adversarial(mids_perturbed)
            highs = ep.where(is_adversarial_, mids, highs)
            lows = ep.where(is_adversarial_, lows, mids)
            reached_numerical_precision: bool = (old_mids == mids).all()
            old_mids = mids
            if reached_numerical_precision:
                break
        res: ep.Tensor = self._project(originals, perturbed, highs)
        return res

    def select_delta(
        self,
        originals: ep.Tensor,
        distances: ep.Tensor,
        step: int
    ) -> ep.Tensor:
        if step == 0:
            result: ep.Tensor = 0.1 * ep.ones_like(distances)
        else:
            d: int = int(np.prod(originals.shape[1:]))
            if self.constraint == 'linf':
                theta: float = self.gamma / (d * d)
                result = d * theta * distances
            else:
                theta = self.gamma / (d * np.sqrt(d))
                result = np.sqrt(d) * theta * distances
        return result
