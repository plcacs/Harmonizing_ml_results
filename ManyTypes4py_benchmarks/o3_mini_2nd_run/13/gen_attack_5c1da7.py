from typing import Optional, Any, Tuple, Union, Callable
import numpy as np
import eagerpy as ep
from ..devutils import atleast_kd
from ..models import Model
from ..criteria import TargetedMisclassification
from ..distances import linf
from .base import FixedEpsilonAttack
from .base import T
from .base import get_channel_axis
from .base import raise_if_kwargs
from .base import verify_input_bounds
import math
from .gen_attack_utils import rescale_images


class GenAttack(FixedEpsilonAttack):
    """A black-box algorithm for L-infinity adversarials. [#Alz18]_

    This attack performs a genetic search in order to find an adversarial
    perturbation in a black-box scenario in as few queries as possible.

    References:
        .. [#Alz18] Moustafa Alzantot, Yash Sharma, Supriyo Chakraborty, Huan Zhang,
           Cho-Jui Hsieh, Mani Srivastava,
           "GenAttack: Practical Black-box Attacks with Gradient-Free
           Optimization",
           https://arxiv.org/abs/1805.11090

    """

    distance = linf

    def __init__(
        self,
        *,
        steps: int = 1000,
        population: int = 10,
        mutation_probability: float = 0.1,
        mutation_range: float = 0.15,
        sampling_temperature: float = 0.3,
        channel_axis: Optional[int] = None,
        reduced_dims: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.steps: int = steps
        self.population: int = population
        self.min_mutation_probability: float = mutation_probability
        self.min_mutation_range: float = mutation_range
        self.sampling_temperature: float = sampling_temperature
        self.channel_axis: Optional[int] = channel_axis
        self.reduced_dims: Optional[Tuple[int, int]] = reduced_dims

    def apply_noise(
        self, x: ep.Tensor, noise: ep.Tensor, epsilon: float, channel_axis: Optional[int]
    ) -> ep.Tensor:
        if noise.shape != x.shape and channel_axis is not None:
            noise = rescale_images(noise, x.shape, channel_axis)
        noise = ep.clip(noise, -epsilon, +epsilon)
        return ep.clip(x + noise, 0.0, 1.0)

    def choice(self, a: int, size: int, replace: bool, p: ep.Tensor) -> np.ndarray:
        p_np: np.ndarray = p.numpy()
        x: np.ndarray = np.random.choice(a, size, replace, p_np)
        return x

    def run(
        self,
        model: Model,
        inputs: Any,
        criterion: TargetedMisclassification,
        *,
        epsilon: float,
        **kwargs: Any
    ) -> Any:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)  # x is ep.Tensor; restore_type: Callable[[ep.Tensor], Any]
        del inputs, kwargs
        verify_input_bounds(x, model)
        N: int = len(x)
        if isinstance(criterion, TargetedMisclassification):
            classes = criterion.target_classes
        else:
            raise ValueError("unsupported criterion")
        if classes.shape != (N,):
            raise ValueError(f"expected target_classes to have shape ({N},), got {classes.shape}")
        channel_axis: Optional[int] = None
        if self.reduced_dims is not None:
            if x.ndim != 4:
                raise NotImplementedError(
                    "only implemented for inputs with two spatial dimensions " +
                    "(and one channel and one batch dimension)"
                )
            if self.channel_axis is None:
                maybe_axis: Optional[int] = get_channel_axis(model, x.ndim)
                if maybe_axis is None:
                    raise ValueError(
                        "cannot infer the data_format from the model, " +
                        "please specify channel_axis when initializing the attack"
                    )
                else:
                    channel_axis = maybe_axis
            else:
                channel_axis = self.channel_axis % x.ndim
            if channel_axis == 1:
                noise_shape = (x.shape[1], *self.reduced_dims)
            elif channel_axis == 3:
                noise_shape = (*self.reduced_dims, x.shape[3])
            else:
                raise ValueError(f"expected 'channel_axis' to be 1 or 3, got {channel_axis}")
        else:
            noise_shape = x.shape[1:]

        def is_adversarial(logits: ep.Tensor) -> ep.Tensor:
            return ep.argmax(logits, 1) == classes

        num_plateaus: ep.Tensor = ep.zeros(x, len(x))
        mutation_probability: ep.Tensor = ep.ones_like(num_plateaus) * self.min_mutation_probability
        mutation_range: ep.Tensor = ep.ones_like(num_plateaus) * self.min_mutation_range
        noise_pops: ep.Tensor = ep.uniform(
            x, (N, self.population, *noise_shape), -epsilon, epsilon
        )

        def calculate_fitness(logits: ep.Tensor) -> ep.Tensor:
            first: ep.Tensor = logits[range(N), classes]
            second: ep.Tensor = ep.log(ep.exp(logits).sum(1) - first)
            return first - second

        n_its_wo_change: ep.Tensor = ep.zeros(x, (N,))
        for step in range(self.steps):
            fitness_l: list[ep.Tensor] = []
            is_adv_l: list[ep.Tensor] = []
            for i in range(self.population):
                it: ep.Tensor = self.apply_noise(x, noise_pops[:, i], epsilon, channel_axis)
                logits: ep.Tensor = model(it)
                f: ep.Tensor = calculate_fitness(logits)
                a: ep.Tensor = is_adversarial(logits)
                fitness_l.append(f)
                is_adv_l.append(a)
            fitness: ep.Tensor = ep.stack(fitness_l)
            is_adv: ep.Tensor = ep.stack(is_adv_l, 1)
            elite_idxs: ep.Tensor = ep.argmax(fitness, 0)
            elite_noise: ep.Tensor = noise_pops[range(N), elite_idxs]
            is_adv = is_adv[range(N), elite_idxs]
            if is_adv.all():
                return restore_type(self.apply_noise(x, elite_noise, epsilon, channel_axis))
            probs: ep.Tensor = ep.softmax(fitness / self.sampling_temperature, 0)
            parents_idxs = np.stack(
                [
                    self.choice(self.population, 2 * self.population - 2, replace=True, p=probs[:, i])
                    for i in range(N)
                ],
                1,
            )
            new_noise_pops: list[ep.Tensor] = [elite_noise]
            for i in range(self.population - 1):
                parents_1: ep.Tensor = noise_pops[range(N), parents_idxs[2 * i]]
                parents_2: ep.Tensor = noise_pops[range(N), parents_idxs[2 * i + 1]]
                p: ep.Tensor = probs[parents_idxs[2 * i], range(N)] / (
                    probs[parents_idxs[2 * i], range(N)] + probs[parents_idxs[2 * i + 1], range(N)]
                )
                p = atleast_kd(p, x.ndim)
                p = ep.tile(p, (1, *noise_shape))
                crossover_mask: ep.Tensor = ep.uniform(p, p.shape, 0, 1) < p
                children: ep.Tensor = ep.where(crossover_mask, parents_1, parents_2)
                mutations: ep.Tensor = ep.stack(
                    [
                        ep.uniform(
                            x,
                            noise_shape,
                            -mutation_range[i].item() * epsilon,
                            mutation_range[i].item() * epsilon,
                        )
                        for i in range(N)
                    ],
                    0,
                )
                mutation_mask: ep.Tensor = ep.uniform(children, children.shape)
                mutation_mask = mutation_mask <= atleast_kd(mutation_probability, children.ndim)
                children = ep.where(mutation_mask, children + mutations, children)
                children = ep.clip(children, -epsilon, epsilon)
                new_noise_pops.append(children)
            noise_pops = ep.stack(new_noise_pops, 1)
            n_its_wo_change = ep.where(elite_idxs == 0, n_its_wo_change + 1, ep.zeros_like(n_its_wo_change))
            num_plateaus = ep.where(n_its_wo_change >= 100, num_plateaus + 1, num_plateaus)
            n_its_wo_change = ep.where(n_its_wo_change >= 100, ep.zeros_like(n_its_wo_change), n_its_wo_change)
            mutation_probability = ep.maximum(
                self.min_mutation_probability,
                0.5 * ep.exp(math.log(0.9) * ep.ones_like(num_plateaus) * num_plateaus),
            )
            mutation_range = ep.maximum(
                self.min_mutation_range,
                0.4 * ep.exp(math.log(0.9) * ep.ones_like(num_plateaus) * num_plateaus),
            )
        return restore_type(self.apply_noise(x, elite_noise, epsilon, channel_axis))