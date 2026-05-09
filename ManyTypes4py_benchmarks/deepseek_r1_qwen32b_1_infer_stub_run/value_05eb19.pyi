"""
Stub file for 'value_05eb19' module.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
from pathlib import Path
from jedi.inference.base_value import Value, ValueSet
from jedi.inference.compiled.access import _sentinel
from jedi.inference.signature import BuiltinSignature
from jedi.inference.context import CompiledContext, CompiledModuleContext


class CheckAttribute:
    """
    Raises :exc:`AttributeError` if the attribute X is not available.
    """
    def __init__(self, check_name: Optional[str] = None) -> None:
        ...

    def __call__(self, func: Callable) -> 'CheckAttribute':
        ...

    def __get__(self, instance: Any, owner: Any) -> Callable:
        ...


class CompiledValue(Value):
    """
    Represents a compiled value.
    """
    def __init__(self, inference_state: Any, access_handle: Any, parent_context: Optional[Any] = None) -> None:
        ...

    @CheckAttribute()
    def py__class__(self) -> Value:
        ...

    @CheckAttribute()
    def py__mro__(self) -> Tuple[Value, ...]:
        ...

    @CheckAttribute()
    def py__bases__(self) -> Tuple[Value, ...]:
        ...

    def get_qualified_names(self) -> Iterable[str]:
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

    @property
    def api_type(self) -> str:
        ...

    def get_filters(self, is_instance: bool = False, origin_scope: Any = None) -> Iterable[Any]:
        ...

    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...

    def py__getitem__(self, index_value_set: Any, contextualized_node: Any) -> ValueSet:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterable[Any]:
        ...

    def py__name__(self) -> str:
        ...

    @property
    def name(self) -> Any:
        ...

    def get_safe_value(self, default: Any = _sentinel) -> Any:
        ...

    def execute_operation(self, other: Any, operator: str) -> ValueSet:
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
    def array_type(self) -> Any:
        ...

    def get_key_values(self) -> List[Any]:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        ...


class CompiledModule(CompiledValue):
    """
    Represents a compiled module.
    """
    file_io: Optional[Any]

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


class CompiledName:
    """
    Represents a compiled name definition.
    """
    def __init__(self, inference_state: Any, parent_value: Any, name: str) -> None:
        ...

    def py__doc__(self) -> Optional[str]:
        ...

    def _get_qualified_names(self) -> Optional[Tuple[str, ...]]:
        ...

    def get_defining_qualified_value(self) -> Optional[Any]:
        ...

    @property
    def api_type(self) -> str:
        ...

    def infer(self) -> ValueSet:
        ...

    @property
    def infer_compiled_value(self) -> CompiledValue:
        ...


class SignatureParamName:
    """
    Represents a signature parameter name.
    """
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


class UnresolvableParamName:
    """
    Represents an unresolvable parameter name.
    """
    def __init__(self, compiled_value: CompiledValue, name: str, default: str) -> None:
        ...

    def get_kind(self) -> int:
        ...

    def to_string(self) -> str:
        ...

    def infer(self) -> ValueSet:
        ...


class CompiledValueName:
    """
    Represents a compiled value name.
    """
    def __init__(self, value: CompiledValue, name: str) -> None:
        ...


class EmptyCompiledName:
    """
    Represents an empty compiled name.
    """
    def __init__(self, inference_state: Any, name: str) -> None:
        ...

    def infer(self) -> ValueSet:
        ...


class CompiledValueFilter:
    """
    Represents a filter for compiled values.
    """
    def __init__(self, inference_state: Any, compiled_value: CompiledValue, is_instance: bool = False) -> None:
        ...

    def get(self, name: str) -> List[Any]:
        ...

    def values(self) -> List[Any]:
        ...

    def _get(self, name: str, allowed_getattr_callback: Callable, in_dir_callback: Callable, check_has_attribute: bool = False) -> List[Any]:
        ...

    def _get_cached_name(self, name: str, is_empty: bool = False) -> Union[EmptyCompiledName, CompiledName]:
        ...

    def _create_name(self, name: str) -> CompiledName:
        ...


docstr_defaults: Dict[str, str] = ...


def _parse_function_doc(doc: str) -> Tuple[str, str]:
    ...


def create_from_name(inference_state: Any, compiled_value: CompiledValue, name: str) -> Optional[CompiledValue]:
    ...


def _normalize_create_args(func: Callable) -> Callable:
    ...


def create_from_access_path(inference_state: Any, access_path: Any) -> CompiledValue:
    ...


@overload
def create_cached_compiled_value(inference_state: Any, access_handle: Any, parent_context: Optional[Any]) -> CompiledValue:
    ...

@overload
def create_cached_compiled_value(inference_state: Any, access_handle: Any, parent_context: None) -> CompiledModule:
    ...

def create_cached_compiled_value(inference_state: Any, access_handle: Any, parent_context: Optional[Any] = None) -> Union[CompiledValue, CompiledModule]:
    ...