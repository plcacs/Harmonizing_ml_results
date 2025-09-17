from typing import Union, Tuple, Optional, Any, Callable
from typing_extensions import Literal
import numpy as np
import eagerpy as ep
import logging
from ..devutils import flatten
from ..devutils import atleast_kd
from ..types import Bounds
from ..models import Model
from ..criteria import Criterion
from ..distances import l2
from ..tensorboard import TensorBoard
from .blended_noise import LinearSearchBlendedUniformNoiseAttack
from .base import MinimizationAttack
from .base import T
from .base import get_criterion
from .base import get_is_adversarial
from .base import raise_if_kwargs
from .base import verify_input_bounds

class BoundaryAttack(MinimizationAttack):
    """A powerful adversarial attack that requires neither gradients
    nor probabilities.

    This is the reference implementation for the attack. [#Bren18]_

    Notes:
        Differences to the original reference implementation:
        * We do not perform internal operations with float64
        * The samples within a batch can currently influence each other a bit
        * We don't perform the additional convergence confirmation
        * The success rate tracking changed a bit
        * Some other changes due to batching and merged loops

    Args:
        init_attack : Attack to use to find a starting points. Defaults to
            LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
        steps : Maximum number of steps to run. Might converge and stop before that.
        spherical_step : Initial step size for the orthogonal (spherical) step.
        source_step : Initial step size for the step towards the target.
        source_step_convergance : Sets the threshold of the stop criterion:
            if source_step becomes smaller than this value during the attack,
            the attack has converged and will stop.
        step_adaptation : Factor by which the step sizes are multiplied or divided.
        tensorboard : The log directory for TensorBoard summaries. If False, TensorBoard
            summaries will be disabled (default). If None, the logdir will be
            runs/CURRENT_DATETIME_HOSTNAME.
        update_stats_every_k : Controls how frequently stats are updated.
    """
    distance = l2

    def __init__(
        self,
        init_attack: Optional[MinimizationAttack] = None,
        steps: int = 25000,
        spherical_step: float = 0.01,
        source_step: float = 0.01,
        source_step_convergance: float = 1e-07,
        step_adaptation: float = 1.5,
        tensorboard: Union[str, bool, None] = False,
        update_stats_every_k: int = 10,
    ) -> None:
        if init_attack is not None and (not isinstance(init_attack, MinimizationAttack)):
            raise NotImplementedError
        self.init_attack = init_attack
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.source_step_convergance = source_step_convergance
        self.step_adaptation = step_adaptation
        self.tensorboard = tensorboard
        self.update_stats_every_k = update_stats_every_k

    def run(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        early_stop: Optional[Any] = None,
        starting_points: Optional[ep.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)  # restore_type: Callable[[ep.Tensor], Any]
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
            best_advs = init_attack.run(model, originals, criterion, early_stop=early_stop)
        else:
            best_advs = ep.astensor(starting_points)
        is_adv = is_adversarial(best_advs)
        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            if starting_points is None:
                raise ValueError(f'init_attack failed for {failed} of {len(is_adv)} inputs')
            else:
                raise ValueError(f'{failed} of {len(is_adv)} starting_points are not adversarial')
        del starting_points
        tb = TensorBoard(logdir=self.tensorboard)
        N = len(originals)
        ndim = originals.ndim
        spherical_steps: ep.Tensor = ep.ones(originals, N) * self.spherical_step
        source_steps: ep.Tensor = ep.ones(originals, N) * self.source_step
        tb.scalar('batchsize', N, 0)
        stats_spherical_adversarial = ArrayQueue(maxlen=100, N=N)
        stats_step_adversarial = ArrayQueue(maxlen=30, N=N)
        bounds: Bounds = model.bounds
        for step in range(1, self.steps + 1):
            converged: ep.Tensor = source_steps < self.source_step_convergance
            if converged.all():
                break
            converged = atleast_kd(converged, ndim)
            unnormalized_source_directions: ep.Tensor = originals - best_advs
            source_norms: ep.Tensor = ep.norms.l2(flatten(unnormalized_source_directions), axis=-1)
            source_directions: ep.Tensor = unnormalized_source_directions / atleast_kd(source_norms, ndim)
            check_spherical_and_update_stats: bool = step % self.update_stats_every_k == 0
            candidates, spherical_candidates = draw_proposals(
                bounds,
                originals,
                best_advs,
                unnormalized_source_directions,
                source_directions,
                source_norms,
                spherical_steps,
                source_steps,
            )
            candidates.dtype == originals.dtype
            spherical_candidates.dtype == spherical_candidates.dtype
            is_adv = is_adversarial(candidates)
            if check_spherical_and_update_stats:
                spherical_is_adv = is_adversarial(spherical_candidates)
                stats_spherical_adversarial.append(spherical_is_adv)
                stats_step_adversarial.append(is_adv)
            else:
                spherical_is_adv = None
            distances: ep.Tensor = ep.norms.l2(flatten(originals - candidates), axis=-1)
            closer: ep.Tensor = distances < source_norms
            is_best_adv: ep.Tensor = ep.logical_and(is_adv, closer)
            is_best_adv = atleast_kd(is_best_adv, ndim)
            cond: ep.Tensor = converged.logical_not().logical_and(is_best_adv)
            best_advs = ep.where(cond, candidates, best_advs)
            tb.probability('converged', converged, step)
            tb.scalar('updated_stats', check_spherical_and_update_stats, step)
            tb.histogram('norms', source_norms, step)
            tb.probability('is_adv', is_adv, step)
            if spherical_is_adv is not None:
                tb.probability('spherical_is_adv', spherical_is_adv, step)
            tb.histogram('candidates/distances', distances, step)
            tb.probability('candidates/closer', closer, step)
            tb.probability('candidates/is_best_adv', is_best_adv, step)
            tb.probability('new_best_adv_including_converged', is_best_adv, step)
            tb.probability('new_best_adv', cond, step)
            if check_spherical_and_update_stats:
                full = stats_spherical_adversarial.isfull()
                tb.probability('spherical_stats/full', full, step)
                if full.any():
                    probs = stats_spherical_adversarial.mean()
                    cond1 = ep.logical_and(probs > 0.5, full)
                    spherical_steps = ep.where(cond1, spherical_steps * self.step_adaptation, spherical_steps)
                    source_steps = ep.where(cond1, source_steps * self.step_adaptation, source_steps)
                    cond2 = ep.logical_and(probs < 0.2, full)
                    spherical_steps = ep.where(cond2, spherical_steps / self.step_adaptation, spherical_steps)
                    source_steps = ep.where(cond2, source_steps / self.step_adaptation, source_steps)
                    stats_spherical_adversarial.clear(ep.logical_or(cond1, cond2))
                    tb.conditional_mean('spherical_stats/isfull/success_rate/mean', probs, full, step)
                    tb.probability_ratio('spherical_stats/isfull/too_linear', cond1, full, step)
                    tb.probability_ratio('spherical_stats/isfull/too_nonlinear', cond2, full, step)
                full = stats_step_adversarial.isfull()
                tb.probability('step_stats/full', full, step)
                if full.any():
                    probs = stats_step_adversarial.mean()
                    cond1 = ep.logical_and(probs > 0.25, full)
                    source_steps = ep.where(cond1, source_steps * self.step_adaptation, source_steps)
                    cond2 = ep.logical_and(probs < 0.1, full)
                    source_steps = ep.where(cond2, source_steps / self.step_adaptation, source_steps)
                    stats_step_adversarial.clear(ep.logical_or(cond1, cond2))
                    tb.conditional_mean('step_stats/isfull/success_rate/mean', probs, full, step)
                    tb.probability_ratio('step_stats/isfull/success_rate_too_high', cond1, full, step)
                    tb.probability_ratio('step_stats/isfull/success_rate_too_low', cond2, full, step)
            tb.histogram('spherical_step', spherical_steps, step)
            tb.histogram('source_step', source_steps, step)
        tb.close()
        return restore_type(best_advs)

class ArrayQueue:
    def __init__(self, maxlen: int, N: int) -> None:
        self.data: np.ndarray = np.full((maxlen, N), np.nan)
        self.next: int = 0
        self.tensor: Optional[ep.Tensor] = None

    @property
    def maxlen(self) -> int:
        return int(self.data.shape[0])

    @property
    def N(self) -> int:
        return int(self.data.shape[1])

    def append(self, x: ep.Tensor) -> None:
        if self.tensor is None:
            self.tensor = x
        x_np: np.ndarray = x.numpy()
        assert x_np.shape == (self.N,)
        self.data[self.next] = x_np
        self.next = (self.next + 1) % self.maxlen

    def clear(self, dims: ep.Tensor) -> None:
        if self.tensor is None:
            self.tensor = dims
        dims_np: np.ndarray = dims.numpy()
        assert dims_np.shape == (self.N,)
        assert dims_np.dtype == np.bool_
        self.data[:, dims_np] = np.nan

    def mean(self) -> ep.Tensor:
        assert self.tensor is not None
        result: np.ndarray = np.nanmean(self.data, axis=0)
        return ep.from_numpy(self.tensor, result)

    def isfull(self) -> ep.Tensor:
        assert self.tensor is not None
        result: np.ndarray = ~np.isnan(self.data).any(axis=0)
        return ep.from_numpy(self.tensor, result)

def draw_proposals(
    bounds: Bounds,
    originals: ep.Tensor,
    perturbed: ep.Tensor,
    unnormalized_source_directions: ep.Tensor,
    source_directions: ep.Tensor,
    source_norms: ep.Tensor,
    spherical_steps: ep.Tensor,
    source_steps: ep.Tensor,
) -> Tuple[ep.Tensor, ep.Tensor]:
    shape: Tuple[int, ...] = originals.shape
    assert perturbed.shape == shape
    assert unnormalized_source_directions.shape == shape
    assert source_directions.shape == shape
    originals_flat = flatten(originals)
    perturbed_flat = flatten(perturbed)
    unnormalized_source_directions_flat = flatten(unnormalized_source_directions)
    source_directions_flat = flatten(source_directions)
    N, D = originals_flat.shape
    assert source_norms.shape == (N,)
    assert spherical_steps.shape == (N,)
    assert source_steps.shape == (N,)
    eta: ep.Tensor = ep.normal(perturbed, (D, 1))
    eta = eta.T - ep.matmul(source_directions_flat, eta) * source_directions_flat
    assert eta.shape == (N, D)
    norms: ep.Tensor = ep.norms.l2(eta, axis=-1)
    assert norms.shape == (N,)
    eta = eta * atleast_kd(spherical_steps * source_norms / norms, eta.ndim)
    distances: ep.Tensor = atleast_kd((spherical_steps.square() + 1).sqrt(), eta.ndim)
    directions: ep.Tensor = eta - unnormalized_source_directions_flat
    spherical_candidates_flat: ep.Tensor = originals_flat + directions / distances
    min_, max_ = bounds
    spherical_candidates_flat = spherical_candidates_flat.clip(min_, max_)
    new_source_directions: ep.Tensor = originals_flat - spherical_candidates_flat
    assert new_source_directions.ndim == 2
    new_source_directions_norms: ep.Tensor = ep.norms.l2(flatten(new_source_directions), axis=-1)
    lengths: ep.Tensor = source_steps * source_norms
    lengths = lengths + new_source_directions_norms - source_norms
    lengths = ep.maximum(lengths, 0)
    lengths = lengths / new_source_directions_norms
    lengths = atleast_kd(lengths, new_source_directions.ndim)
    candidates_flat: ep.Tensor = spherical_candidates_flat + lengths * new_source_directions
    candidates_flat = candidates_flat.clip(min_, max_)
    candidates: ep.Tensor = candidates_flat.reshape(shape)
    spherical_candidates: ep.Tensor = spherical_candidates_flat.reshape(shape)
    return (candidates, spherical_candidates)