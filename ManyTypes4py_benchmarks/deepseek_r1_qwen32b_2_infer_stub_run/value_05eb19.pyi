"""
Stub file for 'value_05eb19' module
"""

import re
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Any,
    Dict,
    Set,
    FrozenSet,
    Type,
    TypeVar,
    overload,
    TYPE_CHECKING,
)
from jedi.inference.base_value import Value, ValueSet
from jedi.inference.lazy_value import LazyKnownValue
from jedi.inference.signature import BuiltinSignature
from jedi.inference.context import CompiledContext, CompiledModuleContext
from jedi.inference.filters import AbstractFilter
from jedi.inference.names import (
    AbstractNameDefinition,
    ValueNameMixin,
    ParamNameInterface,
)
from jedi.inference.compiled.access import _sentinel
from jedi.inference.utils import to_list

if TYPE_CHECKING:
    from jedi.inference.value import CompiledInstance

T = TypeVar("T")

class CheckAttribute:
    def __init__(self, check_name: Optional[str] = None) -> None:
        ...

    def __call__(self, func: Callable) -> Any:
        ...

    def __get__(self, instance: Any, owner: Any) -> Any:
        ...

class CompiledValue(Value):
    def __init__(self, inference_state: Any, access_handle: Any, parent_context: Optional[Any] = None) -> None:
        ...

    def py__call__(self, arguments: Any) -> ValueSet:
        ...

    @CheckAttribute()
    def py__class__(self) -> ValueSet:
        ...

    @CheckAttribute()
    def py__mro__(self) -> Tuple[Any, ...]:
        ...

    @CheckAttribute()
    def py__bases__(self) -> Tuple[Any, ...]:
        ...

    def get_qualified_names(self) -> Tuple[str, ...]:
        ...

    def py__bool__(self) -> bool:
        ...

    def is_class(self) -> bool:
        ...

    def is_function(self) -> bool:
        ...

    def is_module(self) -> bool:
        ...

    def is_compiled(self) -> bool:
        ...

    def is_stub(self) -> bool:
        ...

    def is_instance(self) -> bool:
        ...

    def py__doc__(self) -> Optional[str]:
        ...

    @to_list
    def get_param_names(self) -> Generator[ParamNameInterface, None, None]:
        ...

    def get_signatures(self) -> List[BuiltinSignature]:
        ...

    def __repr__(self) -> str:
        ...

    @memoize_method
    def _parse_function_doc(self) -> Tuple[str, str]:
        ...

    @property
    def api_type(self) -> str:
        ...

    def get_filters(self, is_instance: bool = False, origin_scope: Any = None) -> Generator[CompiledValueFilter, None, None]:
        ...

    @memoize_method
    def _ensure_one_filter(self, is_instance: bool) -> CompiledValueFilter:
        ...

    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...

    def py__getitem__(self, index_value_set: Any, contextualized_node: Any) -> ValueSet:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterator[LazyKnownValue]:
        ...

    def py__name__(self) -> Optional[str]:
        ...

    @property
    def name(self) -> CompiledValueName:
        ...

    def _execute_function(self, params: Any) -> Generator[Any, None, None]:
        ...

    def get_safe_value(self, default: Any = _sentinel) -> Any:
        ...

    def execute_operation(self, other: Any, operator: Any) -> ValueSet:
        ...

    def execute_annotation(self) -> ValueSet:
        ...

    def negate(self) -> Any:
        ...

    def get_metaclasses(self) -> ValueSet:
        ...

    def _as_context(self) -> CompiledContext:
        ...

    @property
    def array_type(self) -> str:
        ...

    def get_key_values(self) -> List[Any]:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        ...

class CompiledModule(CompiledValue):
    file_io: None

    def _as_context(self) -> CompiledModuleContext:
        ...

    def py__path__(self) -> Optional[Path]:
        ...

    def is_package(self) -> bool:
        ...

    @property
    def string_names(self) -> Tuple[str, ...]:
        ...

    def py__file__(self) -> Optional[Path]:
        ...

class CompiledName(AbstractNameDefinition):
    def __init__(self, inference_state: Any, parent_value: Any, name: str) -> None:
        ...

    def py__doc__(self) -> Optional[str]:
        ...

    def _get_qualified_names(self) -> Optional[Tuple[str, ...]]:
        ...

    def get_defining_qualified_value(self) -> Optional[Any]:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def api_type(self) -> str:
        ...

    def infer(self) -> ValueSet:
        ...

    @memoize_method
    def infer_compiled_value(self) -> CompiledValue:
        ...

class SignatureParamName(ParamNameInterface, AbstractNameDefinition):
    def __init__(self, compiled_value: CompiledValue, signature_param: Any) -> None:
        ...

    @property
    def string_name(self) -> str:
        ...

    def to_string(self) -> str:
        ...

    def get_kind(self) -> int:
        ...

    def infer(self) -> ValueSet:
        ...

class UnresolvableParamName(ParamNameInterface, AbstractNameDefinition):
    def __init__(self, compiled_value: CompiledValue, name: str, default: str) -> None:
        ...

    def get_kind(self) -> int:
        ...

    def to_string(self) -> str:
        ...

    def infer(self) -> ValueSet:
        ...

class CompiledValueName(ValueNameMixin, AbstractNameDefinition):
    def __init__(self, value: CompiledValue, name: str) -> None:
        ...

class EmptyCompiledName(AbstractNameDefinition):
    def __init__(self, inference_state: Any, name: str) -> None:
        ...

    def infer(self) -> ValueSet:
        ...

class CompiledValueFilter(AbstractFilter):
    def __init__(self, inference_state: Any, compiled_value: CompiledValue, is_instance: bool = False) -> None:
        ...

    def get(self, name: str) -> List[Any]:
        ...

    def _get(self, name: str, allowed_getattr_callback: Callable, in_dir_callback: Callable, check_has_attribute: bool = False) -> List[Any]:
        ...

    @memoize_method
    def _get_cached_name(self, name: str, is_empty: bool = False) -> Union[EmptyCompiledName, CompiledName]:
        ...

    def values(self) -> List[Any]:
        ...

    def _create_name(self, name: str) -> CompiledName:
        ...

    def __repr__(self) -> str:
        ...

def _parse_function_doc(doc: str) -> Tuple[str, str]:
    ...

def create_from_name(inference_state: Any, compiled_value: CompiledValue, name: str) -> Optional[CompiledValue]:
    ...

def create_from_access_path(inference_state: Any, access_path: Any) -> CompiledValue:
    ...

@inference_state_function_cache()
def create_cached_compiled_value(inference_state: Any, access_handle: Any, parent_context: Optional[Any]) -> Union[CompiledValue, CompiledModule]:
    ...