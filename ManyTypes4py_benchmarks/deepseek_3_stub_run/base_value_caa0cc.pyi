from typing import (
    Any, Optional, Union, Iterator, Iterable, Callable, TypeVar, Generic,
    Sequence, List, Set, FrozenSet, Dict, Tuple, overload, TYPE_CHECKING
)
from functools import reduce
from operator import add
from itertools import zip_longest
from parso.python.tree import Name
from jedi import debug
from jedi.inference.helpers import SimpleGetItemNotFound
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.cache import memoize_method

if TYPE_CHECKING:
    from jedi.inference.value import ValueSet
    from jedi.inference.arguments import ValuesArguments
    from jedi.inference.lazy_value import LazyValue, LazyKnownValues
    from jedi.inference.names import AbstractName, ValueName
    from jedi.inference.compiled import CompiledValueName
    from jedi.inference.gradual.annotation import merge_type_var_dicts
    from jedi.inference import finder, analysis
    from jedi.inference.gradual.conversion import convert_values

_T = TypeVar("_T")
_ValueType = TypeVar("_ValueType", bound="Value")
sentinel: object = ...

class HasNoContext(Exception):
    ...

class HelperValueMixin:
    def get_root_context(self) -> "Value": ...
    
    def execute(self, arguments: Any) -> "ValueSet": ...
    
    def execute_with_values(self, *value_list: "Value") -> "ValueSet": ...
    
    def execute_annotation(self) -> "ValueSet": ...
    
    def gather_annotation_classes(self) -> "ValueSet": ...
    
    def merge_types_of_iterate(
        self, 
        contextualized_node: Optional["ContextualizedNode"] = None, 
        is_async: bool = False
    ) -> "ValueSet": ...
    
    def _get_value_filters(
        self, 
        name_or_str: Union[str, Name]
    ) -> Iterator[Any]: ...
    
    def goto(
        self, 
        name_or_str: Union[str, Name], 
        name_context: Optional["Value"] = None, 
        analysis_errors: bool = True
    ) -> List["AbstractName"]: ...
    
    def py__getattribute__(
        self, 
        name_or_str: Union[str, Name], 
        name_context: Optional["Value"] = None, 
        position: Optional[Tuple[int, int]] = None, 
        analysis_errors: bool = True
    ) -> "ValueSet": ...
    
    def py__await__(self) -> "ValueSet": ...
    
    def py__name__(self) -> str: ...
    
    def iterate(
        self, 
        contextualized_node: Optional["ContextualizedNode"] = None, 
        is_async: bool = False
    ) -> Iterator["LazyValue"]: ...
    
    def is_sub_class_of(self, class_value: "Value") -> bool: ...
    
    def is_same_class(self, class2: "Value") -> bool: ...
    
    @memoize_method
    def as_context(self, *args: Any, **kwargs: Any) -> Any: ...

class Value(HelperValueMixin):
    tree_node: Optional[Any] = ...
    array_type: Optional[str] = ...
    api_type: str = ...
    
    def __init__(
        self, 
        inference_state: Any, 
        parent_context: Optional["Value"] = None
    ) -> None: ...
    
    def py__getitem__(
        self, 
        index_value_set: "ValueSet", 
        contextualized_node: "ContextualizedNode"
    ) -> "ValueSet": ...
    
    def py__simple_getitem__(self, index: Union[int, str, slice, bytes, float]) -> "ValueSet": ...
    
    def py__iter__(
        self, 
        contextualized_node: Optional["ContextualizedNode"] = None
    ) -> Iterator["LazyValue"]: ...
    
    def py__next__(
        self, 
        contextualized_node: Optional["ContextualizedNode"] = None
    ) -> Iterator["LazyValue"]: ...
    
    def get_signatures(self) -> List[Any]: ...
    
    def is_class(self) -> bool: ...
    
    def is_class_mixin(self) -> bool: ...
    
    def is_instance(self) -> bool: ...
    
    def is_function(self) -> bool: ...
    
    def is_module(self) -> bool: ...
    
    def is_namespace(self) -> bool: ...
    
    def is_compiled(self) -> bool: ...
    
    def is_bound_method(self) -> bool: ...
    
    def is_builtins_module(self) -> bool: ...
    
    def py__bool__(self) -> bool: ...
    
    def py__doc__(self) -> str: ...
    
    def get_safe_value(self, default: Any = sentinel) -> Any: ...
    
    def execute_operation(
        self, 
        other: "Value", 
        operator: str
    ) -> "ValueSet": ...
    
    def py__call__(self, arguments: Any) -> "ValueSet": ...
    
    def py__stop_iteration_returns(self) -> "ValueSet": ...
    
    def py__getattribute__alternatives(self, name_or_str: str) -> "ValueSet": ...
    
    def py__get__(
        self, 
        instance: Optional["Value"], 
        class_value: "Value"
    ) -> "ValueSet": ...
    
    def py__get__on_class(
        self, 
        calling_instance: "Value", 
        instance: Optional["Value"], 
        class_value: "Value"
    ) -> Any: ...
    
    def get_qualified_names(self) -> Optional[Tuple[str, ...]]: ...
    
    def is_stub(self) -> bool: ...
    
    def _as_context(self) -> Any: ...
    
    @property
    def name(self) -> Union["ValueName", "CompiledValueName"]: ...
    
    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]: ...
    
    def infer_type_vars(self, value_set: "ValueSet") -> Dict[str, "ValueSet"]: ...

def iterate_values(
    values: "ValueSet", 
    contextualized_node: Optional["ContextualizedNode"] = None, 
    is_async: bool = False
) -> "ValueSet": ...

class _ValueWrapperBase(HelperValueMixin):
    @property
    def name(self) -> Union["ValueName", "CompiledValueName"]: ...
    
    @classmethod
    @inference_state_as_method_param_cache()
    def create_cached(
        cls, 
        inference_state: Any, 
        *args: Any, 
        **kwargs: Any
    ) -> "_ValueWrapperBase": ...
    
    def __getattr__(self, name: str) -> Any: ...

class LazyValueWrapper(_ValueWrapperBase):
    @property
    @memoize_method
    def _wrapped_value(self) -> "Value": ...
    
    def __repr__(self) -> str: ...
    
    def _get_wrapped_value(self) -> "Value": ...

class ValueWrapper(_ValueWrapperBase):
    def __init__(self, wrapped_value: "Value") -> None: ...
    
    def __repr__(self) -> str: ...

class TreeValue(Value):
    def __init__(
        self, 
        inference_state: Any, 
        parent_context: Optional["Value"], 
        tree_node: Any
    ) -> None: ...
    
    def __repr__(self) -> str: ...

class ContextualizedNode:
    def __init__(self, context: "Value", node: Any) -> None: ...
    
    def get_root_context(self) -> "Value": ...
    
    def infer(self) -> "ValueSet": ...
    
    def __repr__(self) -> str: ...

def _getitem(
    value: "Value", 
    index_values: "ValueSet", 
    contextualized_node: "ContextualizedNode"
) -> "ValueSet": ...

class ValueSet:
    def __init__(self, iterable: Iterable["Value"]) -> None: ...
    
    @classmethod
    def _from_frozen_set(cls, frozenset_: FrozenSet["Value"]) -> "ValueSet": ...
    
    @classmethod
    def from_sets(cls, sets: Iterable[Union["ValueSet", Iterable["Value"]]]) -> "ValueSet": ...
    
    def __or__(self, other: "ValueSet") -> "ValueSet": ...
    
    def __and__(self, other: "ValueSet") -> "ValueSet": ...
    
    def __iter__(self) -> Iterator["Value"]: ...
    
    def __bool__(self) -> bool: ...
    
    def __len__(self) -> int: ...
    
    def __repr__(self) -> str: ...
    
    def filter(self, filter_func: Callable[["Value"], bool]) -> "ValueSet": ...
    
    def __getattr__(self, name: str) -> Callable[..., "ValueSet"]: ...
    
    def __eq__(self, other: object) -> bool: ...
    
    def __ne__(self, other: object) -> bool: ...
    
    def __hash__(self) -> int: ...
    
    def py__class__(self) -> "ValueSet": ...
    
    def iterate(
        self, 
        contextualized_node: Optional["ContextualizedNode"] = None, 
        is_async: bool = False
    ) -> Iterator["LazyValue"]: ...
    
    def execute(self, arguments: Any) -> "ValueSet": ...
    
    def execute_with_values(self, *args: Any, **kwargs: Any) -> "ValueSet": ...
    
    def goto(
        self, 
        name_or_str: Union[str, Name], 
        name_context: Optional["Value"] = None, 
        analysis_errors: bool = True
    ) -> List["AbstractName"]: ...
    
    def py__getattribute__(
        self, 
        name_or_str: Union[str, Name], 
        name_context: Optional["Value"] = None, 
        position: Optional[Tuple[int, int]] = None, 
        analysis_errors: bool = True
    ) -> "ValueSet": ...
    
    def get_item(
        self, 
        index_values: "ValueSet", 
        contextualized_node: "ContextualizedNode"
    ) -> "ValueSet": ...
    
    def try_merge(self, function_name: str) -> "ValueSet": ...
    
    def gather_annotation_classes(self) -> "ValueSet": ...
    
    def get_signatures(self) -> List[Any]: ...
    
    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]: ...
    
    def infer_type_vars(self, value_set: "ValueSet") -> Dict[str, "ValueSet"]: ...

NO_VALUES: ValueSet = ...

def iterator_to_value_set(
    func: Callable[..., Iterator["Value"]]
) -> Callable[..., "ValueSet"]: ...