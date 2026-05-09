"""
Stub file for base_value_caa0cc module
"""

from typing import (
    Any,
    Optional,
    Iterable,
    Union,
    List,
    FrozenSet,
    Callable,
    overload,
)
from jedi.inference.value import Value, LazyValueWrapper, ValueWrapper
from jedi.inference.value import TreeValue, ContextualizedNode
from jedi.inference.value import ValueSet, NO_VALUES
from parso.python.tree import Name

class HelperValueMixin:
    def get_root_context(self) -> Value:
        ...
    
    def execute(self, arguments: Any) -> ValueSet:
        ...
    
    def execute_with_values(self, *value_list: Any) -> ValueSet:
        ...
    
    def execute_annotation(self) -> ValueSet:
        ...
    
    def gather_annotation_classes(self) -> ValueSet:
        ...
    
    def merge_types_of_iterate(
        self,
        contextualized_node: Optional[ContextualizedNode] = None,
        is_async: bool = False,
    ) -> ValueSet:
        ...
    
    def _get_value_filters(self, name_or_str: Any) -> Iterable[Any]:
        ...
    
    def goto(
        self,
        name_or_str: Union[Name, str],
        name_context: Any = None,
        analysis_errors: bool = True,
    ) -> List[Any]:
        ...
    
    def py__getattribute__(
        self,
        name_or_str: Union[Name, str],
        name_context: Any = None,
        position: Optional[tuple[int, int]] = None,
        analysis_errors: bool = True,
    ) -> ValueSet:
        ...
    
    def py__await__(self) -> ValueSet:
        ...
    
    def py__name__(self) -> str:
        ...
    
    def iterate(
        self,
        contextualized_node: Optional[ContextualizedNode] = None,
        is_async: bool = False,
    ) -> Iterable[Any]:
        ...
    
    def is_sub_class_of(self, class_value: Value) -> bool:
        ...
    
    def is_same_class(self, class2: Any) -> bool:
        ...
    
    @memoize_method
    def as_context(self, *args: Any, **kwargs: Any) -> Any:
        ...

class Value(HelperValueMixin):
    tree_node: Optional[Name]
    array_type: Any
    api_type: str
    
    def __init__(
        self,
        inference_state: Any,
        parent_context: Optional[Value] = None,
    ) -> None:
        ...
    
    def py__getitem__(
        self,
        index_value_set: ValueSet,
        contextualized_node: ContextualizedNode,
    ) -> ValueSet:
        ...
    
    def py__simple_getitem__(self, index: Any) -> Any:
        ...
    
    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Iterable[Any]:
        ...
    
    def py__next__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Iterable[Any]:
        ...
    
    def get_signatures(self) -> List[Any]:
        ...
    
    def is_class(self) -> bool:
        ...
    
    def is_class_mixin(self) -> bool:
        ...
    
    def is_instance(self) -> bool:
        ...
    
    def is_function(self) -> bool:
        ...
    
    def is_module(self) -> bool:
        ...
    
    def is_namespace(self) -> bool:
        ...
    
    def is_compiled(self) -> bool:
        ...
    
    def is_bound_method(self) -> bool:
        ...
    
    def is_builtins_module(self) -> bool:
        ...
    
    def py__bool__(self) -> bool:
        ...
    
    def py__doc__(self) -> str:
        ...
    
    def get_safe_value(self, default: Any = ...) -> Any:
        ...
    
    def execute_operation(self, other: Any, operator: str) -> ValueSet:
        ...
    
    def py__call__(self, arguments: Any) -> ValueSet:
        ...
    
    def py__stop_iteration_returns(self) -> ValueSet:
        ...
    
    def py__getattribute__alternatives(self, name_or_str: Union[Name, str]) -> ValueSet:
        ...
    
    def py__get__(self, instance: Any, class_value: Value) -> ValueSet:
        ...
    
    def py__get__on_class(
        self,
        calling_instance: Any,
        instance: Any,
        class_value: Value,
    ) -> Any:
        ...
    
    def get_qualified_names(self) -> Optional[List[str]]:
        ...
    
    def is_stub(self) -> bool:
        ...
    
    @property
    def name(self) -> Any:
        ...
    
    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        ...
    
    def infer_type_vars(self, value_set: ValueSet) -> dict:
        ...

def iterate_values(
    values: Value,
    contextualized_node: Optional[ContextualizedNode] = None,
    is_async: bool = False,
) -> ValueSet:
    ...

class _ValueWrapperBase(HelperValueMixin):
    @safe_property
    def name(self) -> Any:
        ...
    
    @classmethod
    @inference_state_as_method_param_cache()
    def create_cached(cls, inference_state: Any, *args: Any, **kwargs: Any) -> Any:
        ...
    
    def __getattr__(self, name: str) -> Any:
        ...

class LazyValueWrapper(_ValueWrapperBase):
    @safe_property
    @memoize_method
    def _wrapped_value(self) -> Value:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def _get_wrapped_value(self) -> Value:
        ...

class ValueWrapper(_ValueWrapperBase):
    def __init__(self, wrapped_value: Value) -> None:
        ...
    
    def __repr__(self) -> str:
        ...

class TreeValue(Value):
    def __init__(
        self,
        inference_state: Any,
        parent_context: Optional[Value],
        tree_node: Name,
    ) -> None:
        ...
    
    def __repr__(self) -> str:
        ...

class ContextualizedNode:
    def __init__(self, context: Any, node: Any) -> None:
        ...
    
    def get_root_context(self) -> Any:
        ...
    
    def infer(self) -> ValueSet:
        ...
    
    def __repr__(self) -> str:
        ...

def _getitem(
    value: Value,
    index_values: Iterable[Any],
    contextualized_node: ContextualizedNode,
) -> ValueSet:
    ...

class ValueSet:
    def __init__(self, iterable: Iterable[Value]) -> None:
        ...
    
    @classmethod
    def _from_frozen_set(cls, frozenset_: FrozenSet[Value]) -> ValueSet:
        ...
    
    @classmethod
    def from_sets(cls, sets: Iterable[Iterable[Value]]) -> ValueSet:
        ...
    
    def __or__(self, other: ValueSet) -> ValueSet:
        ...
    
    def __and__(self, other: ValueSet) -> ValueSet:
        ...
    
    def __iter__(self) -> Iterable[Value]:
        ...
    
    def __bool__(self) -> bool:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def filter(self, filter_func: Callable[[Any], bool]) -> ValueSet:
        ...
    
    def __getattr__(self, name: str) -> Callable[..., ValueSet]:
        ...
    
    def __eq__(self, other: ValueSet) -> bool:
        ...
    
    def __ne__(self, other: ValueSet) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...
    
    def py__class__(self) -> ValueSet:
        ...
    
    def iterate(
        self,
        contextualized_node: Optional[ContextualizedNode] = None,
        is_async: bool = False,
    ) -> Iterable[Any]:
        ...
    
    def execute(self, arguments: Any) -> ValueSet:
        ...
    
    def execute_with_values(self, *args: Any, **kwargs: Any) -> ValueSet:
        ...
    
    def goto(self, *args: Any, **kwargs: Any) -> List[Any]:
        ...
    
    def py__getattribute__(self, *args: Any, **kwargs: Any) -> ValueSet:
        ...
    
    def get_item(self, *args: Any, **kwargs: Any) -> ValueSet:
        ...
    
    def try_merge(self, function_name: str) -> ValueSet:
        ...
    
    def gather_annotation_classes(self) -> ValueSet:
        ...
    
    def get_signatures(self) -> List[Any]:
        ...
    
    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        ...
    
    def infer_type_vars(self, value_set: ValueSet) -> dict:
        ...

NO_VALUES: ValueSet = ...

def iterator_to_value_set(func: Callable[..., Iterable[Any]]) -> Callable[..., ValueSet]:
    ...