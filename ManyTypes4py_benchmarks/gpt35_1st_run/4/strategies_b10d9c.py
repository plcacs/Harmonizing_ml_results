from typing import TypeVar, Callable, Union, Literal, Any, Tuple, List

Ex = TypeVar('Ex', covariant=True, default=Any)
T = TypeVar('T')
T3 = TypeVar('T3')
T4 = TypeVar('T4')
T5 = TypeVar('T5')
MappedFrom = TypeVar('MappedFrom')
MappedTo = TypeVar('MappedTo')
RecurT = Callable[['SearchStrategy'], Any]
PackT = Callable[[T], T3]
PredicateT = Callable[[T], object]
TransformationsT = Tuple[Union[Tuple[Literal['filter'], PredicateT], Tuple[Literal['map'], PackT]], ...]

def recursive_property(strategy: 'SearchStrategy', name: str, default: Any) -> Any:
    ...

class SearchStrategy(Generic[Ex]):
    supports_find: ClassVar[bool] = True
    validate_called: bool = False
    __label: Optional[int] = None
    __module__: str = 'hypothesis.strategies'

    def available(self, data: Any) -> bool:
        ...

    @property
    def is_empty(self) -> bool:
        ...

    @property
    def has_reusable_values(self) -> bool:
        ...

    @property
    def is_cacheable(self) -> bool:
        ...

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        ...

    def calc_is_empty(self, recur: RecurT) -> bool:
        ...

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        ...

    def example(self) -> Any:
        ...

    def map(self, pack: PackT) -> 'MappedStrategy':
        ...

    def flatmap(self, expand: Callable[[T], Any]) -> 'FlatMapStrategy':
        ...

    def filter(self, condition: PredicateT) -> 'FilteredStrategy':
        ...

    def _filter_for_filtered_draw(self, condition: PredicateT) -> 'FilteredStrategy':
        ...

    @property
    def branches(self) -> List['SearchStrategy']:
        ...

    def __or__(self, other: 'SearchStrategy') -> 'OneOfStrategy':
        ...

    def __bool__(self) -> bool:
        ...

    def validate(self) -> None:
        ...

    @property
    def class_label(self) -> int:
        ...

    @property
    def label(self) -> int:
        ...

    def calc_label(self) -> int:
        ...

    def do_validate(self) -> None:
        ...

    def do_draw(self, data: Any) -> Any:
        ...

class SampledFromStrategy(SearchStrategy[Ex]):
    _MAX_FILTER_CALLS: int = 10000

    def __init__(self, elements: Any, repr_: Optional[str] = None, transformations: TransformationsT = ()) -> None:
        ...

    def map(self, pack: PackT) -> 'SearchStrategy[T]':
        ...

    def filter(self, condition: PredicateT) -> 'SampledFromStrategy':
        ...

    def __repr__(self) -> str:
        ...

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        ...

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        ...

    def _transform(self, element: Any) -> Any:
        ...

    def do_draw(self, data: Any) -> Any:
        ...

    def get_element(self, i: int) -> Any:
        ...

    def do_filtered_draw(self, data: Any) -> Any:
        ...

class OneOfStrategy(SearchStrategy[Ex]):
    def __init__(self, strategies: Tuple['SearchStrategy']) -> None:
        ...

    def calc_is_empty(self, recur: RecurT) -> bool:
        ...

    def calc_has_reusable_values(self, recur: RecurT) -> bool:
        ...

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        ...

    @property
    def element_strategies(self) -> List['SearchStrategy']:
        ...

    def calc_label(self) -> int:
        ...

    def do_draw(self, data: Any) -> Any:
        ...

    def __repr__(self) -> str:
        ...

    def do_validate(self) -> None:
        ...

    @property
    def branches(self) -> List['SearchStrategy']:
        ...

    def filter(self, condition: PredicateT) -> 'FilteredStrategy':
        ...

def one_of(*args: Union['SearchStrategy', Tuple['SearchStrategy']]) -> 'OneOfStrategy':
    ...

class MappedStrategy(SearchStrategy[MappedTo], Generic[MappedFrom, MappedTo]):
    def __init__(self, strategy: 'SearchStrategy', pack: PackT) -> None:
        ...

    def calc_is_empty(self, recur: RecurT) -> bool:
        ...

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def do_validate(self) -> None:
        ...

    def do_draw(self, data: Any) -> Any:
        ...

    @property
    def branches(self) -> List['MappedStrategy']:
        ...

    def filter(self, condition: PredicateT) -> 'FilteredStrategy':
        ...

class FilteredStrategy(SearchStrategy[Ex]):
    def __init__(self, strategy: 'SearchStrategy', conditions: Tuple[PredicateT]) -> None:
        ...

    def calc_is_empty(self, recur: RecurT) -> bool:
        ...

    def calc_is_cacheable(self, recur: RecurT) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def do_validate(self) -> None:
        ...

    def filter(self, condition: PredicateT) -> 'FilteredStrategy':
        ...

    @property
    def condition(self) -> PredicateT:
        ...

    def do_draw(self, data: Any) -> Any:
        ...

    def do_filtered_draw(self, data: Any) -> Any:
        ...

    @property
    def branches(self) -> List['FilteredStrategy']:
        ...
