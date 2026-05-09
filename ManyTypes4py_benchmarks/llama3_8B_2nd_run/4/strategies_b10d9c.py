import sys
import warnings
from collections import abc, defaultdict
from collections.abc import Sequence
from functools import lru_cache
from random import shuffle
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Literal, Optional, TypeVar, Union, cast, overload
from hypothesis._settings import HealthCheck, Phase, Verbosity, settings
from hypothesis.control import _current_build_context, current_build_context
from hypothesis.errors import HypothesisException, HypothesisWarning, InvalidArgument, NonInteractiveExampleWarning, UnsatisfiedAssumption
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.conjecture.utils import calc_label_from_cls, calc_label_from_name, combine_labels
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import get_pretty_function_description, is_identity_function
from hypothesis.strategies._internal.utils import defines_strategy
from hypothesis.utils.conventions import UniqueIdentifier
if TYPE_CHECKING:
    from typing import TypeAlias
    Ex: TypeVar('Ex', covariant=True, default=Any)
else:
    Ex = TypeVar('Ex', covariant=True)

T: TypeVar('T')
T3: TypeVar('T3')
T4: TypeVar('T4')
T5: TypeVar('T5')
MappedFrom: TypeVar('MappedFrom')
MappedTo: TypeVar('MappedTo')
RecurT: Callable[[SearchStrategy], Any]
PackT: Callable[[T], T3]
PredicateT: Callable[[T], object]
TransformationsT: tuple[Union[tuple[Literal['filter'], PredicateT], tuple[Literal['map'], PackT]], ...]
calculating: UniqueIdentifier
MAPPED_SEARCH_STRATEGY_DO_DRAW_LABEL: str
FILTERED_SEARCH_STRATEGY_DO_DRAW_LABEL: str

class SearchStrategy(Generic[Ex]):
    # ... (same as before)

class SampledFromStrategy(SearchStrategy[Ex]):
    # ... (same as before)

class OneOfStrategy(SearchStrategy[Ex]):
    # ... (same as before)

class MappedStrategy(SearchStrategy[MappedTo], Generic[MappedFrom, MappedTo]):
    # ... (same as before)

class FilteredStrategy(SearchStrategy[Ex]):
    # ... (same as before)
