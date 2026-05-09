from typing import Union, Any, cast, TypeVar, Literal
from abc import ABC
from abc import abstractmethod
import eagerpy as ep
from ..devutils import flatten
from ..devutils import atleast_kd
from ..distances import l2, linf, Distance
from .base import FixedEpsilonAttack
from .base import Criterion
from .base import Model
from .base import T
from .base import get_criterion
from .base import get_is_adversarial
from .base import raise_if_kwargs
from ..external.clipping_aware_rescaling import l2_clipping_aware_rescaling
from .base import verify_input_bounds

class BaseAdditiveNoiseAttack(FixedEpsilonAttack, ABC):
    """Base class for additive noise attacks."""

    def run(self, model: Model, inputs: T, criterion: Criterion = None, *, epsilon: float, **kwargs: Any) -> T:
        """Run the attack.

        Args:
            model: The model to attack.
            inputs: The input to attack.
            criterion: The criterion to use for adversarial examples.
            epsilon: The maximum allowed L-infinity norm of the perturbation.
            **kwargs: Additional keyword arguments.

        Returns:
            The adversarial example.
        """
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, criterion, kwargs
        verify_input_bounds(x, model)
        min_, max_ = model.bounds
        p = self.sample_noise(x)
        epsilons = self.get_epsilons(x, p, epsilon, min_=min_, max_=max_)
        x = x + epsilons * p
        x = x.clip(min_, max_)
        return restore_type(x)

    @abstractmethod
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        """Sample noise.

        Args:
            x: The input tensor.

        Returns:
            The noise tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_epsilons(self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float) -> ep.Tensor:
        """Get the epsilons.

        Args:
            x: The input tensor.
            p: The noise tensor.
            epsilon: The maximum allowed L-infinity norm of the perturbation.
            min_: The minimum value of the input.
            max_: The maximum value of the input.

        Returns:
            The epsilons tensor.
        """
        raise NotImplementedError

class L2Mixin:
    """L2 mixin."""

    @property
    def distance(self) -> Distance:
        return l2

    def get_epsilons(self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float) -> ep.Tensor:
        """Get the epsilons.

        Args:
            x: The input tensor.
            p: The noise tensor.
            epsilon: The maximum allowed L-infinity norm of the perturbation.
            min_: The minimum value of the input.
            max_: The maximum value of the input.

        Returns:
            The epsilons tensor.
        """
        norms = flatten(p).norms.l2(axis=-1)
        return epsilon / atleast_kd(norms, p.ndim)

class L2ClippingAwareMixin:
    """L2 clipping-aware mixin."""

    @property
    def distance(self) -> Distance:
        return l2

    def get_epsilons(self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float) -> ep.Tensor:
        """Get the epsilons.

        Args:
            x: The input tensor.
            p: The noise tensor.
            epsilon: The maximum allowed L-infinity norm of the perturbation.
            min_: The minimum value of the input.
            max_: The maximum value of the input.

        Returns:
            The epsilons tensor.
        """
        return cast(ep.Tensor, l2_clipping_aware_rescaling(x, p, epsilon, a=min_, b=max_))

class LinfMixin:
    """L-infinity mixin."""

    @property
    def distance(self) -> Distance:
        return linf

    def get_epsilons(self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float) -> ep.Tensor:
        """Get the epsilons.

        Args:
            x: The input tensor.
            p: The noise tensor.
            epsilon: The maximum allowed L-infinity norm of the perturbation.
            min_: The minimum value of the input.
            max_: The maximum value of the input.

        Returns:
            The epsilons tensor.
        """
        norms = flatten(p).max(axis=-1)
        return epsilon / atleast_kd(norms, p.ndim)

class GaussianMixin:
    """Gaussian mixin."""

    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        """Sample noise.

        Args:
            x: The input tensor.

        Returns:
            The noise tensor.
        """
        return x.normal(x.shape)

class UniformMixin:
    """Uniform mixin."""

    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        """Sample noise.

        Args:
            x: The input tensor.

        Returns:
            The noise tensor.
        """
        return x.uniform(x.shape, -1, 1)

class L2AdditiveGaussianNoiseAttack(L2Mixin, GaussianMixin, BaseAdditiveNoiseAttack):
    """Samples Gaussian noise with a fixed L2 size."""

class L2AdditiveUniformNoiseAttack(L2Mixin, UniformMixin, BaseAdditiveNoiseAttack):
    """Samples uniform noise with a fixed L2 size."""

class L2ClippingAwareAdditiveGaussianNoiseAttack(L2ClippingAwareMixin, GaussianMixin, BaseAdditiveNoiseAttack):
    """Samples Gaussian noise with a fixed L2 size after clipping.

    The implementation is based on [Rauber20]_.
    """

class L2ClippingAwareAdditiveUniformNoiseAttack(L2ClippingAwareMixin, UniformMixin, BaseAdditiveNoiseAttack):
    """Samples uniform noise with a fixed L2 size after clipping.

    The implementation is based on [Rauber20]_.

    References:
        .. [Rauber20] Jonas Rauber, Matthias Bethge
            "Fast Differentiable Clipping-Aware Normalization and Rescaling"
            https://arxiv.org/abs/2007.07677

    """

class LinfAdditiveUniformNoiseAttack(LinfMixin, UniformMixin, BaseAdditiveNoiseAttack):
    """Samples uniform noise with a fixed L-infinity size"""

class BaseRepeatedAdditiveNoiseAttack(FixedEpsilonAttack, ABC):
    """Base class for repeated additive noise attacks."""

    def __init__(self, *, repeats: int = 100, check_trivial: bool = True):
        """Initialize the attack.

        Args:
            repeats: The number of times to repeat the attack.
            check_trivial: Whether to check whether the original sample is already adversarial.
        """
        self.repeats = repeats
        self.check_trivial = check_trivial

    def run(self, model: Model, inputs: T, criterion: Criterion = None, *, epsilon: float, **kwargs: Any) -> T:
        """Run the attack.

        Args:
            model: The model to attack.
            inputs: The input to attack.
            criterion: The criterion to use for adversarial examples.
            epsilon: The maximum allowed L-infinity norm of the perturbation.
            **kwargs: Additional keyword arguments.

        Returns:
            The adversarial example.
        """
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs
        verify_input_bounds(x0, model)
        is_adversarial = get_is_adversarial(criterion_, model)
        min_, max_ = model.bounds
        result = x0
        if self.check_trivial:
            found = is_adversarial(result)
        else:
            found = ep.zeros(x0, len(result)).bool()
        for _ in range(self.repeats):
            if found.all():
                break
            p = self.sample_noise(x0)
            epsilons = self.get_epsilons(x0, p, epsilon, min_=min_, max_=max_)
            x = x0 + epsilons * p
            x = x.clip(min_, max_)
            is_adv = is_adversarial(x)
            is_new_adv = ep.logical_and(is_adv, ep.logical_not(found))
            result = ep.where(atleast_kd(is_new_adv, x.ndim), x, result)
            found = ep.logical_or(found, is_adv)
        return restore_type(result)

    @abstractmethod
    def sample_noise(self, x: ep.Tensor) -> ep.Tensor:
        """Sample noise.

        Args:
            x: The input tensor.

        Returns:
            The noise tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_epsilons(self, x: ep.Tensor, p: ep.Tensor, epsilon: float, min_: float, max_: float) -> ep.Tensor:
        """Get the epsilons.

        Args:
            x: The input tensor.
            p: The noise tensor.
            epsilon: The maximum allowed L-infinity norm of the perturbation.
            min_: The minimum value of the input.
            max_: The maximum value of the input.

        Returns:
            The epsilons tensor.
        """
        raise NotImplementedError

class L2RepeatedAdditiveGaussianNoiseAttack(L2Mixin, GaussianMixin, BaseRepeatedAdditiveNoiseAttack):
    """Repeatedly samples Gaussian noise with a fixed L2 size.

    Args:
        repeats: The number of times to repeat the attack.
        check_trivial: Whether to check whether the original sample is already adversarial.
    """

class L2RepeatedAdditiveUniformNoiseAttack(L2Mixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack):
    """Repeatedly samples uniform noise with a fixed L2 size.

    Args:
        repeats: The number of times to repeat the attack.
        check_trivial: Whether to check whether the original sample is already adversarial.
    """

class L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(L2ClippingAwareMixin, GaussianMixin, BaseRepeatedAdditiveNoiseAttack):
    """Repeatedly samples Gaussian noise with a fixed L2 size after clipping.

    The implementation is based on [Rauber20]_.

    Args:
        repeats: The number of times to repeat the attack.
        check_trivial: Whether to check whether the original sample is already adversarial.
    """

class L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(L2ClippingAwareMixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack):
    """Repeatedly samples uniform noise with a fixed L2 size after clipping.

    The implementation is based on [Rauber20]_.

    Args:
        repeats: The number of times to repeat the attack.
        check_trivial: Whether to check whether the original sample is already adversarial.
    """

class LinfRepeatedAdditiveUniformNoiseAttack(LinfMixin, UniformMixin, BaseRepeatedAdditiveNoiseAttack):
    """Repeatedly samples uniform noise with a fixed L-infinity size.

    Args:
        repeats: The number of times to repeat the attack.
        check_trivial: Whether to check whether the original sample is already adversarial.
    """
