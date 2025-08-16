from typing import TypeVar, Callable, Union, Literal, Any, Tuple, Sequence, List

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

def one_of(__args: Union[Tuple[SearchStrategy, ...], Sequence[SearchStrategy]]) -> SearchStrategy:
    ...

def check_strategy(arg: Any, name: str = '') -> None:
    ...
