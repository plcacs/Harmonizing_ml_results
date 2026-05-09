import re
from itertools import zip_longest
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    Any,
    Dict,
    Iterator,
    overload,
)
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, LazyTreeValue
from jedi.inference.names import ParamName, AnonymousParamName
from jedi.inference.value import iterable
from jedi.inference.utils import PushBackIterator
from jedi.inference.cache import inference_state_as_method_param_cache
from parso.python import tree

class ParamIssue(Exception):
    pass

def try_iter_content(types: Iterable[Any], depth: int = 0) -> None:
    ...

def repack_with_argument_clinic(clinic_string: str) -> Callable[[Callable], Callable]:
    ...

def iterate_argument_clinic(
    inference_state: Any,
    arguments: Any,
    clinic_string: str
) -> Generator[ValueSet, None, None]:
    ...

def _parse_argument_clinic(string: str) -> Generator[Tuple[str, bool, bool, int], None, None]:
    ...

class _AbstractArgumentsMixin:
    def unpack(self, funcdef: Any = None) -> Iterable[Any]:
        ...
    def get_calling_nodes(self) -> List[Any]:
        ...

class AbstractArguments(_AbstractArgumentsMixin):
    context: Any
    argument_node: Any
    trailer: Any

class TreeArguments(AbstractArguments):
    def __init__(
        self,
        inference_state: Any,
        context: Any,
        argument_node: Any,
        trailer: Optional[Any] = None
    ) -> None:
        ...
    
    @classmethod
    @inference_state_as_method_param_cache()
    def create_cached(cls, *args: Any, **kwargs: Any) -> Any:
        ...

    def unpack(self, funcdef: Any = None) -> Generator[Tuple[Optional[str], Any], None, None]:
        ...

    def _as_tree_tuple_objects(self) -> Generator[Tuple[Any, Any, int], None, None]:
        ...

    def iter_calling_names_with_star(self) -> Generator[Any, None, None]:
        ...

    def __repr__(self) -> str:
        ...

    def get_calling_nodes(self) -> List[Any]:
        ...

class ValuesArguments(AbstractArguments):
    def __init__(self, values_list: Iterable[Any]) -> None:
        ...
    def unpack(self, funcdef: Any = None) -> Generator[Tuple[None, Any], None, None]:
        ...
    def __repr__(self) -> str:
        ...

class TreeArgumentsWrapper(_AbstractArgumentsMixin):
    def __init__(self, arguments: Any) -> None:
        ...
    @property
    def context(self) -> Any:
        ...
    @property
    def argument_node(self) -> Any:
        ...
    @property
    def trailer(self) -> Any:
        ...
    def unpack(self, func: Any = None) -> None:
        ...
    def get_calling_nodes(self) -> List[Any]:
        ...
    def __repr__(self) -> str:
        ...

def _iterate_star_args(
    context: Any,
    array: Any,
    input_node: Any,
    funcdef: Optional[Any] = None
) -> Iterator[Any]:
    ...

def _star_star_dict(
    context: Any,
    array: Any,
    input_node: Any,
    funcdef: Optional[Any] = None
) -> Dict[Any, Any]:
    ...