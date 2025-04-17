import numpy as np
import nevergrad.common.typing as tp
from . import discretization
from . import core
from . import container
from . import _datalayers
from .data import Array
C = tp.TypeVar('C', bound='Choice')
T = tp.TypeVar('T', bound='TransitionChoice')

class BaseChoice(container.Container):

    def __init__(self, *, choices: tp.Union[int, tp.Iterable[tp.Any]], repetitions: tp.Optional[int]=None, **kwargs: tp.Any):
        assert repetitions is None or isinstance(repetitions, int)
        self._repetitions: tp.Optional[int] = repetitions
        lchoices: tp.List[tp.Any] = list(choices if not isinstance(choices, int) else range(choices))
        if not lchoices:
            raise ValueError(f'{self._class__.__name__} received an empty list of options.')
        super().__init__(choices=container.Tuple(*lchoices), **kwargs)

    def __len__(self):
        """Number of choices"""
        return len(self.choices)

    def _get_parameters_str(self):
        params: tp.List[tp.Tuple[str, str]] = sorted(((k, p.name) for k, p in self._content.items() if p.name != self._ignore_in_repr.get(k, '#ignoredrepr#')))
        return ','.join((f'{k}={n}' for k, n in params))

    @property
    def index(self):
        """Index of the chosen option"""
        inds: np.ndarray = self.indices.value
        assert inds.size == 1
        return int(inds[0])

    @property
    def indices(self):
        """Array of indices of the chosen option"""
        return self['indices']

    @property
    def choices(self):
        """The different options, as a Tuple Parameter"""
        return self['choices']

    def _layered_get_value(self):
        if self._repetitions is None:
            return core.as_parameter(self.choices[self.index]).value
        return tuple((core.as_parameter(self.choices[int(ind)]).value for ind in self.indices.value))

    def _layered_set_value(self, value):
        """Must be adapted to each class
        This handles a list of values, not just one
        """
        values: tp.List[tp.Any] = [value] if self._repetitions is None else value
        self._check_frozen()
        indices: np.ndarray = -1 * np.ones(len(values), dtype=int)
        for i, val in enumerate(values):
            for k, choice in enumerate(self.choices):
                try:
                    choice.value = val
                    indices[i] = k
                    break
                except Exception:
                    pass
            if indices[i] == -1:
                raise ValueError(f'Could not figure out where to put value {value}')
        self.indices.value = indices

    def get_value_hash(self):
        hashes: tp.List[tp.Hashable] = []
        for ind in self.indices.value:
            c = self.choices[int(ind)]
            const = isinstance(c, core.Constant) or not isinstance(c, core.Parameter)
            hashes.append(int(ind) if const else (int(ind), c.get_value_hash()))
        return tuple(hashes) if len(hashes) > 1 else hashes[0]

class Choice(BaseChoice):
    """Unordered categorical parameter, randomly choosing one of the provided choice options as a value.
    The choices can be Parameters, in which case their value will be returned instead.
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
        whether to always draw the most likely choice (hence avoiding the stochastic behavior, but losing
        continuity)

    Note
    ----
    - Since the chosen value is drawn randomly, the use of this variable makes deterministic
      functions become stochastic, hence "adding noise"
    - the "mutate" method only mutates the weights and the chosen Parameter (if it is not constant),
      leaving others untouched

    Examples
    --------

    >>> print(Choice(["a", "b", "c", "e"]).value)
    "c"

    >>> print(Choice(["a", "b", "c", "e"], repetitions=3).value)
    ("b", "b", "c")
    """

    def __init__(self, choices, repetitions=None, deterministic=False):
        lchoices: tp.List[tp.Any] = list(choices if not isinstance(choices, int) else range(choices))
        rep: int = 1 if repetitions is None else repetitions
        indices: Array = Array(shape=(rep, len(lchoices)), mutable_sigma=False)
        indices.add_layer(_datalayers.SoftmaxSampling(len(lchoices), deterministic=deterministic))
        super().__init__(choices=lchoices, repetitions=repetitions, indices=indices)
        self._indices: tp.Optional[np.ndarray] = None

    def mutate(self):
        _ = self.random_state
        self.indices.mutate()
        for ind in self.indices.value:
            self.choices[int(ind)].mutate()

class TransitionChoice(BaseChoice):
    """Categorical parameter, choosing one of the provided choice options as a value, with continuous transitions.
    By default, this is ordered, and most algorithms except discrete OnePlusOne algorithms will consider it as ordered.
    The choices can be Parameters, in which case their value will be returned instead.
    The chosen parameter is drawn using transitions between current choice and the next/previous ones.

    Parameters
    ----------
    choices: list or int
        a list of possible values or Parameters for the variable (or an integer as a shortcut for range(num))
    transitions: np.ndarray or Array
        the transition weights. During transition, the direction (forward or backward will be drawn with
        equal probabilities), then the transitions weights are normalized through softmax, the 1st value gives
        the probability to remain in the same state, the second to move one step (backward or forward) and so on.
    ordered: bool
        if False, changes the default behavior to be unordered and sampled uniformly when setting the data to a
        normalized and centered Gaussian (used in DiscreteOnePlusOne only)

    Note
    ----
    - the "mutate" method only mutates the weights and the chosen Parameter (if it is not constant),
      leaving others untouched
    - in order to support export to standardized space, the index is encoded as a scalar. A normal distribution N(0,1)
      on this scalar yields a uniform choice of index. This may come to evolve for simplicity's sake.
    - currently, transitions are computed through softmax, this may evolve since this is somehow impractical
    """

    def __init__(self, choices, transitions=(1.0, 1.0), repetitions=None, ordered=True):
        choices_list: tp.List[tp.Any] = list(choices if not isinstance(choices, int) else range(choices))
        indices_init: np.ndarray = len(choices_list) / 2.0 * np.ones((repetitions if repetitions is not None else 1,), dtype=float)
        indices: Array = Array(init=indices_init)
        indices.set_bounds(0, len(choices_list), method='gaussian')
        indices.value = indices.value - 0.5
        intcasting: _datalayers.Int = _datalayers.Int(deterministic=True)
        intcasting.arity = len(choices_list)
        indices.add_layer(intcasting)
        transitions_array: Array = transitions if isinstance(transitions, Array) else Array(init=np.asarray(transitions))
        super().__init__(choices=choices_list, repetitions=repetitions, indices=indices, transitions=transitions_array)
        assert self.transitions.value.ndim == 1
        self._ref: tp.Optional['TransitionChoice'] = None
        if not ordered:
            self._ref = self.copy()

    def _internal_set_standardized_data(self, data, reference):
        ref: T = reference if self._ref is None else self._ref
        super()._internal_set_standardized_data(data, ref)
        super()._layered_set_value(super()._layered_get_value())

    def _internal_get_standardized_data(self, reference):
        ref: T = reference if self._ref is None else self._ref
        return super()._internal_get_standardized_data(ref)

    @property
    def transitions(self):
        return self['transitions']

    def mutate(self):
        if self._ref is not None:
            new_data: np.ndarray = self.random_state.normal(size=self.indices.value.size)
            self._internal_set_standardized_data(new_data, self._ref)
            return
        _ = self.random_state
        transitions_param: core.Parameter = core.as_parameter(self.transitions)
        transitions_param.mutate()
        rep: int = 1 if self._repetitions is None else self._repetitions
        enc: discretization.Encoder = discretization.Encoder(np.ones((rep, 1)) * np.log(self['transitions'].value), self.random_state)
        moves: np.ndarray = enc.encode()
        signs: np.ndarray = self.random_state.choice([-1, 1], size=rep)
        new_index: np.ndarray = np.clip(self.indices.value + signs * moves, 0, len(self) - 1)
        self.indices.value = new_index
        indices_set: tp.Set[int] = set((int(ind) for ind in self.indices.value))
        for ind in indices_set:
            self.choices[ind].mutate()