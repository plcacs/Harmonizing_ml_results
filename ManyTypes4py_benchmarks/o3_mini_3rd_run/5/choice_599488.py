from typing import Any, Optional, Union, Iterable, Tuple, TypeVar
import numpy as np
import nevergrad.common.typing as tp
from . import discretization
from . import core
from . import container
from . import _datalayers
from .data import Array

C = tp.TypeVar("C", bound="Choice")
T = tp.TypeVar("T", bound="TransitionChoice")


class BaseChoice(container.Container):
    def __init__(
        self,
        *,
        choices: Union[int, Iterable[Any]],
        repetitions: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        assert repetitions is None or isinstance(repetitions, int)
        self._repetitions: Optional[int] = repetitions
        lchoices = list(choices if not isinstance(choices, int) else range(choices))
        if not lchoices:
            raise ValueError(f"{self.__class__.__name__} received an empty list of options.")
        super().__init__(choices=container.Tuple(*lchoices), **kwargs)

    def __len__(self) -> int:
        """Number of choices"""
        return len(self.choices)

    def _get_parameters_str(self) -> str:
        params = sorted(
            ((k, p.name) for k, p in self._content.items() if p.name != self._ignore_in_repr.get(k, "#ignoredrepr#"))
        )
        return ",".join((f"{k}={n}" for k, n in params))

    @property
    def index(self) -> int:
        """Index of the chosen option"""
        inds = self.indices.value
        assert inds.size == 1
        return int(inds[0])

    @property
    def indices(self) -> Any:
        """Array of indices of the chosen option"""
        return self["indices"]

    @property
    def choices(self) -> Any:
        """The different options, as a Tuple Parameter"""
        return self["choices"]

    def _layered_get_value(self) -> Union[Any, Tuple[Any, ...]]:
        if self._repetitions is None:
            return core.as_parameter(self.choices[self.index]).value
        return tuple(core.as_parameter(self.choices[ind]).value for ind in self.indices.value)

    def _layered_set_value(self, value: Any) -> None:
        """Must be adapted to each class
        This handles a list of values, not just one
        """
        values = [value] if self._repetitions is None else value  # type: ignore
        self._check_frozen()
        indices = -1 * np.ones(len(values), dtype=int)
        for i, val in enumerate(values):
            for k, choice in enumerate(self.choices):
                try:
                    choice.value = val
                    indices[i] = k
                    break
                except Exception:
                    pass
            if indices[i] == -1:
                raise ValueError(f"Could not figure out where to put value {value}")
        self.indices.value = indices

    def get_value_hash(self) -> Union[int, Tuple[Any, ...]]:
        hashes = []
        for ind in self.indices.value:
            c = self.choices[int(ind)]
            const = isinstance(c, core.Constant) or not isinstance(c, core.Parameter)
            hashes.append(int(ind) if const else (int(ind), c.get_value_hash()))
        return tuple(hashes) if len(hashes) > 1 else hashes[0]


class Choice(BaseChoice):
    """Unordered categorical parameter, randomly choosing one of the provided choice options as a value.
    The choices can be Parameters, in which case there value will be returned instead.
    The chosen parameter is drawn randomly from the softmax of weights which are
    updated during the optimization.

    Parameters
    ----------
    choices: list or int
        a list of possible values or Parameters for the variable (or an integer as a shortcut for range(num))
    repetitions: None or int
        set to an integer :code:`n` if you want :code:`n` similar choices sampled independently (each with its own distribution)
        This is equivalent to :code:`Tuple(*[Choice(options) for _ in range(n)])` but can be
        30x faster for large :code:`n`.
    deterministic: bool
        whether to always draw the most likely choice (hence avoiding the stochastic behavior, but loosing
        continuity)
    """

    def __init__(
        self, choices: Union[int, Iterable[Any]], repetitions: Optional[int] = None, deterministic: bool = False
    ) -> None:
        lchoices = list(choices if not isinstance(choices, int) else range(choices))
        rep: int = 1 if repetitions is None else repetitions
        indices: Array = Array(shape=(rep, len(lchoices)), mutable_sigma=False)
        indices.add_layer(_datalayers.SoftmaxSampling(len(lchoices), deterministic=deterministic))
        super().__init__(choices=lchoices, repetitions=repetitions, indices=indices)
        self._indices: Optional[Any] = None

    def mutate(self) -> None:
        self.random_state  # Presumed to initialize or validate the random state.
        self.indices.mutate()
        for ind in self.indices.value:
            self.choices[ind].mutate()


class TransitionChoice(BaseChoice):
    """Categorical parameter, choosing one of the provided choice options as a value, with continuous transitions.
    By default, this is ordered, and most algorithms except discrete OnePlusOne algorithms will consider it as ordered.
    The choices can be Parameters, in which case there value will be returned instead.
    The chosen parameter is drawn using transitions between current choice and the next/previous ones.
    """

    def __init__(
        self,
        choices: Union[int, Iterable[Any]],
        transitions: Union[np.ndarray, Array] = (1.0, 1.0),
        repetitions: Optional[int] = None,
        ordered: bool = True,
    ) -> None:
        choices_list = list(choices if not isinstance(choices, int) else range(choices))
        rep: int = 1 if repetitions is None else repetitions
        init_value = (len(choices_list) / 2.0) * np.ones((rep,))
        indices: Array = Array(init=init_value)
        indices.set_bounds(0, len(choices_list), method="gaussian")
        indices = indices - 0.5
        intcasting = _datalayers.Int(deterministic=True)
        intcasting.arity = len(choices_list)
        indices.add_layer(intcasting)
        transitions_array: Union[np.ndarray, Array]
        if isinstance(transitions, Array):
            transitions_array = transitions
        else:
            transitions_array = np.asarray(transitions)
        super().__init__(choices=choices_list, repetitions=repetitions, indices=indices, transitions=transitions_array)
        assert self.transitions.value.ndim == 1
        self._ref: Optional[Any] = None
        if not ordered:
            self._ref = self.copy()

    def _internal_set_standardized_data(self, data: np.ndarray, reference: Any) -> None:
        ref = reference if self._ref is None else self._ref
        super()._internal_set_standardized_data(data, ref)
        super()._layered_set_value(super()._layered_get_value())

    def _internal_get_standardized_data(self, reference: Any) -> np.ndarray:
        ref = reference if self._ref is None else self._ref
        return super()._internal_get_standardized_data(ref)

    @property
    def transitions(self) -> Any:
        return self["transitions"]

    def mutate(self) -> None:
        if self._ref is not None:
            new: np.ndarray = self.random_state.normal(size=self.indices.value.size)
            self._internal_set_standardized_data(new, self._ref)
            return
        self.random_state  # Presumed to check or update random state.
        transitions_param = core.as_parameter(self.transitions)
        transitions_param.mutate()
        rep: int = 1 if self._repetitions is None else self._repetitions
        enc = discretization.Encoder(
            np.ones((rep, 1)) * np.log(self["transitions"].value), self.random_state
        )
        moves: np.ndarray = enc.encode()
        signs: np.ndarray = self.random_state.choice([-1, 1], size=rep)
        new_index: np.ndarray = np.clip(self.indices.value + signs * moves, 0, len(self) - 1)
        self.indices.value = new_index
        indices_set = set(self.indices.value)
        for ind in indices_set:
            self.choices[ind].mutate()