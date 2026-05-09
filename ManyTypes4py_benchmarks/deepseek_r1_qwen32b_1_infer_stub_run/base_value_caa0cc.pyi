"""
Stub file for base_value_caa0cc module
"""

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from jedi.inference import (
    ValuesArguments,
    InferenceState,
    LazyKnownValues,
    ValueSet,
    Context,
)
from jedi.parser_utils import CleanScopeDocstring
from parso.python.tree import Name as ParsoName
import parso

T = TypeVar('T')
ValueT = TypeVar('ValueT', bound='Value')

class HelperValueMixin:
    inference_state: InferenceState
    parent_context: Optional['Value']

    def get_root_context(self) -> 'Value':
        ...

    def execute(self, arguments: ValuesArguments) -> ValueSet:
        ...

    def execute_with_values(self, *value_list: Value) -> ValueSet:
        ...

    def execute_annotation(self) -> ValueSet:
        ...

    def gather_annotation_classes(self) -> ValueSet:
        ...

    def merge_types_of_iterate(
        self,
        contextualized_node: Optional['ContextualizedNode'] = None,
        is_async: bool = False,
    ) -> ValueSet:
        ...

    def _get_value_filters(self, name_or_str: Union[str, ParsoName]) -> Iterable[Any]:
        ...

    def goto(
        self,
        name_or_str: Union[str, ParsoName],
        name_context: Optional['Value'] = None,
        analysis_errors: bool = True,
    ) -> List[Any]:
        ...

    def py__getattribute__(
        self,
        name_or_str: Union[str, ParsoName],
        name_context: Optional['Value'] = None,
        position: Optional[Tuple[int, int]] = None,
        analysis_errors: bool = True,
    ) -> ValueSet:
        ...

    def py__await__(self) -> ValueSet:
        ...

    def py__name__(self) -> str:
        ...

    def iterate(
        self,
        contextualized_node: Optional['ContextualizedNode'] = None,
        is_async: bool = False,
    ) -> Iterator['LazyKnownValues']:
        ...

    def is_sub_class_of(self, class_value: 'Value') -> bool:
        ...

    def is_same_class(self, class2: 'Value') -> bool:
        ...

    @memoize_method
    def as_context(self, *args: Any, **kwargs: Any) -> 'Value':
        ...

class Value(HelperValueMixin):
    tree_node: Optional[Any]
    array_type: Optional[Any]
    api_type: str

    def __init__(
        self,
        inference_state: InferenceState,
        parent_context: Optional['Value'] = None,
    ):
        ...

    def py__getitem__(
        self,
        index_value_set: ValueSet,
        contextualized_node: Optional['ContextualizedNode'],
    ) -> ValueSet:
        ...

    def py__simple_getitem__(self, index: Any) -> Any:
        ...

    def py__iter__(
        self,
        contextualized_node: Optional['ContextualizedNode'] = None,
    ) -> Iterator[Any]:
        ...

    def py__next__(
        self,
        contextualized_node: Optional['ContextualizedNode'] = None,
    ) -> ValueSet:
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

    def py__call__(self, arguments: ValuesArguments) -> ValueSet:
        ...

    def py__stop_iteration_returns(self) -> ValueSet:
        ...

    def py__getattribute__alternatives(self, name_or_str: str) -> ValueSet:
        ...

    def py__get__(
        self,
        instance: Optional['Value'],
        class_value: 'Value',
    ) -> ValueSet:
        ...

    def py__get__on_class(
        self,
        calling_instance: Optional['Value'],
        instance: Optional['Value'],
        class_value: 'Value',
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

    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, ValueSet]:
        ...

class TreeValue(Value):
    def __init__(
        self,
        inference_state: InferenceState,
        parent_context: Optional['Value'],
        tree_node: ParsoName,
    ):
        ...

class ContextualizedNode:
    context: Context
    node: parso.Node

    def __init__(self, context: Context, node: parso.Node):
        ...

    def get_root_context(self) -> 'Value':
        ...

    def infer(self) -> ValueSet:
        ...

class _ValueWrapperBase(HelperValueMixin):
    _wrapped_value: Any

    @property
    def name(self) -> Any:
        ...

class LazyValueWrapper(_ValueWrapperBase):
    def _get_wrapped_value(self) -> Any:
        ...

class ValueWrapper(_ValueWrapperBase):
    def __init__(self, wrapped_value: Any):
        ...

class ValueSet:
    _set: FrozenSet[Any]

    def __init__(self, iterable: Iterable[Any]):
        ...

    @classmethod
    def _from_frozen_set(cls, frozenset_: FrozenSet[Any]) -> 'ValueSet':
        ...

    @classmethod
    def from_sets(cls, sets: Iterable[Iterable[Any]]) -> 'ValueSet':
        ...

    def __or__(self, other: 'ValueSet') -> 'ValueSet':
        ...

    def __and__(self, other: 'ValueSet') -> 'ValueSet':
        ...

    def __iter__(self) -> Iterator[Any]:
        ...

    def __bool__(self) -> bool:
        ...

    def __len__(self) -> int:
        ...

    def filter(self, filter_func: Callable[[Any], bool]) -> 'ValueSet':
        ...

    def __getattr__(self, name: str) -> Callable[..., 'ValueSet']:
        ...

    def __eq__(self, other: 'ValueSet') -> bool:
        ...

    def __ne__(self, other: 'ValueSet') -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def py__class__(self) -> 'ValueSet':
        ...

    def iterate(
        self,
        contextualized_node: Optional['ContextualizedNode'] = None,
        is_async: bool = False,
    ) -> Iterator['LazyKnownValues']:
        ...

    def execute(self, arguments: ValuesArguments) -> 'ValueSet':
        ...

    def execute_with_values(self, *args: Any, **kwargs: Any) -> 'ValueSet':
        ...

    def goto(self, *args: Any, **kwargs: Any) -> List[Any]:
        ...

    def py__getattribute__(self, *args: Any, **kwargs: Any) -> 'ValueSet':
        ...

    def get_item(self, *args: Any, **kwargs: Any) -> 'ValueSet':
        ...

    def try_merge(self, function_name: str) -> 'ValueSet':
        ...

    def gather_annotation_classes(self) -> 'ValueSet':
        ...

    def get_signatures(self) -> List[Any]:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        ...

    def infer_type_vars(self, value_set: 'ValueSet') -> Dict[str, 'ValueSet']:
        ...

NO_VALUES: ValueSet = ...