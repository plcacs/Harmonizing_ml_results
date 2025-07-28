from typing import Callable, TypeVar, Any, Union, Optional, Sequence, List, Tuple, Dict, Iterable
from typing_extensions import final, overload
from abc import ABC, abstractmethod
import eagerpy as ep
from ..models import Model
from ..criteria import Criterion
from ..criteria import Misclassification
from ..devutils import atleast_kd
from ..distances import Distance

T = TypeVar('T')
CriterionType = TypeVar('CriterionType', bound=Criterion)


class Attack(ABC):
    @overload
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Any:
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Any:
        ...

    @abstractmethod
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Any:
        ...

    @abstractmethod
    def repeat(self, times: int) -> "Attack":
        ...

    def __repr__(self) -> str:
        args = ", ".join((f"{k.strip('_')}={v}" for k, v in vars(self).items()))
        return f"{self.__class__.__name__}({args})"


class AttackWithDistance(Attack):
    @property
    @abstractmethod
    def distance(self) -> Distance:
        ...

    def repeat(self, times: int) -> "Attack":
        return Repeated(self, times)


class Repeated(AttackWithDistance):
    """Repeats the wrapped attack and returns the best result"""

    def __init__(self, attack: Attack, times: int) -> None:
        if times < 1:
            raise ValueError(f"expected times >= 1, got {times}")
        self.attack = attack
        self.times = times

    @property
    def distance(self) -> Distance:
        return self.attack.distance

    @overload
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Any:
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Any:
        ...

    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Union[Tuple[List[Any], List[Any], Any], Tuple[Any, Any, Any]]:
        x, restore_type = ep.astensor_(inputs)
        del inputs
        verify_input_bounds(x, model)
        criterion = get_criterion(criterion)
        was_iterable: bool = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False
        N: int = len(x)
        K: int = len(epsilons)
        best_xps: List[Any] = []
        best_xpcs: List[Any] = []
        best_success: Any = None
        for i in range(self.times):
            xps, xpcs, success = self.attack(model, x, criterion, epsilons=epsilons, **kwargs)
            assert len(xps) == K
            assert len(xpcs) == K
            for xp in xps:
                assert xp.shape == x.shape
            for xpc in xpcs:
                assert xpc.shape == x.shape
            assert success.shape == (K, N)
            if i == 0:
                best_xps = xps
                best_xpcs = xpcs
                best_success = success
                continue
            for k, epsilon in enumerate(epsilons):
                first = best_success[k].logical_not()
                assert first.shape == (N,)
                if epsilon is None:
                    closer = self.distance(x, xps[k]) < self.distance(x, best_xps[k])
                    assert closer.shape == (N,)
                    new_best = ep.logical_and(success[k], ep.logical_or(closer, first))
                else:
                    new_best = ep.logical_and(success[k], first)
                new_best = atleast_kd(new_best, x.ndim)
                best_xps[k] = ep.where(new_best, xps[k], best_xps[k])
                best_xpcs[k] = ep.where(new_best, xpcs[k], best_xpcs[k])
            best_success = ep.logical_or(success, best_success)
        best_xps_ = [restore_type(xp) for xp in best_xps]
        best_xpcs_ = [restore_type(xpc) for xpc in best_xpcs]
        if was_iterable:
            return (best_xps_, best_xpcs_, restore_type(best_success))
        else:
            assert len(best_xps_) == 1
            assert len(best_xpcs_) == 1
            return (
                best_xps_[0],
                best_xpcs_[0],
                restore_type(best_success.squeeze(axis=0)),
            )

    def repeat(self, times: int) -> "Attack":
        return Repeated(self.attack, self.times * times)


class FixedEpsilonAttack(AttackWithDistance):
    """Fixed-epsilon attacks try to find adversarials whose perturbation sizes
    are limited by a fixed epsilon"""

    @abstractmethod
    def run(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> Any:
        """
        Runs the attack and returns perturbed inputs.

        The size of the perturbations should be at most epsilon, but this
        is not guaranteed and the caller should verify this or clip the result.
        """
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Any:
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Any:
        ...

    @final
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Union[Tuple[List[Any], List[Any], Any], Tuple[Any, Any, Any]]:
        x, restore_type = ep.astensor_(inputs)
        del inputs
        verify_input_bounds(x, model)
        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)
        was_iterable: bool = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False
        N: int = len(x)
        K: int = len(epsilons)
        if any((eps is None for eps in epsilons)):
            raise NotImplementedError("FixedEpsilonAttack subclasses do not yet support None in epsilons")
        real_epsilons: List[float] = [eps for eps in epsilons if eps is not None]  # type: ignore
        del epsilons
        xps: List[Any] = []
        xpcs: List[Any] = []
        success: List[Any] = []
        for epsilon in real_epsilons:
            xp = self.run(model, x, criterion, epsilon=epsilon, **kwargs)
            xpc = self.distance.clip_perturbation(x, xp, epsilon)
            is_adv = is_adversarial(xpc)
            xps.append(xp)
            xpcs.append(xpc)
            success.append(is_adv)
        success_ = ep.stack(success)
        assert success_.shape == (len(real_epsilons), N)
        xps_ = [restore_type(xp) for xp in xps]
        xpcs_ = [restore_type(xpc) for xpc in xpcs]
        if was_iterable:
            return (xps_, xpcs_, restore_type(success_))
        else:
            assert len(xps_) == 1
            assert len(xpcs_) == 1
            return (
                xps_[0],
                xpcs_[0],
                restore_type(success_.squeeze(axis=0)),
            )


class MinimizationAttack(AttackWithDistance):
    """Minimization attacks try to find adversarials with minimal perturbation sizes"""

    @abstractmethod
    def run(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Runs the attack and returns perturbed inputs.

        The size of the perturbations should be as small as possible such that
        the perturbed inputs are still adversarial. In general, this is not
        guaranteed and the caller has to verify this.
        """
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Any:
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Any:
        ...

    @final
    def __call__(
        self,
        model: Model,
        inputs: Any,
        criterion: Union[Criterion, Any],
        *,
        epsilons: Union[float, Iterable[Optional[float]]],
        **kwargs: Any,
    ) -> Union[Tuple[List[Any], List[Any], Any], Tuple[Any, Any, Any]]:
        x, restore_type = ep.astensor_(inputs)
        del inputs
        verify_input_bounds(x, model)
        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)
        was_iterable: bool = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False
        N: int = len(x)
        K: int = len(epsilons)
        if any((eps is None for eps in epsilons)):
            early_stop = None
        else:
            early_stop = min(epsilons)  # type: ignore
        xp = self.run(model, x, criterion, early_stop=early_stop, **kwargs)
        xpcs: List[Any] = []
        success: List[Any] = []
        for epsilon in epsilons:
            if epsilon is None:
                xpc = xp
            else:
                xpc = self.distance.clip_perturbation(x, xp, epsilon)
            is_adv = is_adversarial(xpc)
            xpcs.append(xpc)
            success.append(is_adv)
        success_ = ep.stack(success)
        assert success_.shape == (K, N)
        xp_ = restore_type(xp)
        xpcs_ = [restore_type(xpc) for xpc in xpcs]
        if was_iterable:
            return ([xp_] * K, xpcs_, restore_type(success_))
        else:
            assert len(xpcs_) == 1
            return (xp_, xpcs_[0], restore_type(success_.squeeze(axis=0)))


class FlexibleDistanceMinimizationAttack(MinimizationAttack):
    def __init__(self, *, distance: Optional[Distance] = None) -> None:
        self._distance = distance

    @property
    def distance(self) -> Distance:
        if self._distance is None:
            raise ValueError("unknown distance, please pass `distance` to the attack initializer")
        return self._distance


def get_is_adversarial(criterion: Criterion, model: Model) -> Callable[[Any], Any]:
    def is_adversarial(perturbed: Any) -> Any:
        outputs = model(perturbed)
        return criterion(perturbed, outputs)
    return is_adversarial


def get_criterion(criterion: Union[Criterion, Any]) -> Criterion:
    if isinstance(criterion, Criterion):
        return criterion
    else:
        return Misclassification(criterion)


def get_channel_axis(model: Model, ndim: int) -> Optional[int]:
    data_format = getattr(model, "data_format", None)
    if data_format is None:
        return None
    if data_format == "channels_first":
        return 1
    if data_format == "channels_last":
        return ndim - 1
    raise ValueError(f"unknown data_format, expected 'channels_first' or 'channels_last', got {data_format}")


def raise_if_kwargs(kwargs: Dict[str, Any]) -> None:
    if kwargs:
        raise TypeError(f"attack got an unexpected keyword argument '{next(iter(kwargs.keys()))}'")


def verify_input_bounds(input: Any, model: Model) -> None:
    assert input.min().item() >= model.bounds.lower
    assert input.max().item() <= model.bounds.upper