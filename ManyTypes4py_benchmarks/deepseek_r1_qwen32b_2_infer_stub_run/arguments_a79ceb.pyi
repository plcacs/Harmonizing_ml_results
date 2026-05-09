import re
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple, Union
from parso.python.tree import Node as tree_Node
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, LazyTreeValue
from jedi.inference.names import ParamName, TreeNameDefinition
from jedi.inference.context import Context as InferenceContext
from jedi.inference.value import iterable
from jedi.inference.cache import inference_state_as_method_param_cache

class ParamIssue(Exception):
    pass

def try_iter_content(types: Any, depth: int = 0) -> None:
    ...

def repack_with_argument_clinic(clinic_string: str) -> Any:
    def decorator(func: Any) -> Any:
        def wrapper(value: Any, arguments: AbstractArguments) -> ValueSet:
            ...
        return wrapper
    return decorator

def iterate_argument_clinic(inference_state: Any, arguments: AbstractArguments, clinic_string: str) -> Any:
    ...

def _parse_argument_clinic(string: str) -> Tuple[str, bool, bool, int]:
    ...

class _AbstractArgumentsMixin:
    def unpack(self, funcdef: Any = None) -> Any:
        ...
    def get_calling_nodes(self) -> List[Any]:
        ...

class AbstractArguments(_AbstractArgumentsMixin):
    context: Any
    argument_node: Any
    trailer: Any

class TreeArguments(AbstractArguments):
    def __init__(self, inference_state: Any, context: InferenceContext, argument_node: Any, trailer: Any = None) -> None:
        ...
    
    @classmethod
    @inference_state_as_method_param_cache()
    def create_cached(cls: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    def unpack(self, funcdef: Any = None) -> Any:
        ...

    def _as_tree_tuple_objects(self) -> Any:
        ...

    def iter_calling_names_with_star(self) -> Any:
        ...

    def get_calling_nodes(self) -> List[ContextualizedNode]:
        ...

class ValuesArguments(AbstractArguments):
    def __init__(self, values_list: Any) -> None:
        ...

    def unpack(self, funcdef: Any = None) -> Any:
        ...

class TreeArgumentsWrapper(_AbstractArgumentsMixin):
    def __init__(self, arguments: AbstractArguments) -> None:
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

    def get_calling_nodes(self) -> List[ContextualizedNode]:
        ...

def _iterate_star_args(context: Any, array: Any, input_node: Any, funcdef: Any = None) -> Any:
    ...

def _star_star_dict(context: Any, array: Any, input_node: Any, funcdef: Any) -> Dict[str, LazyKnownValue]:
    ...