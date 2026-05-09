import asyncio
import collections
import enum
import inspect
import logging
import re
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    ForwardRef,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Any,
    Type,
    TypeVar,
    Dict,
    Optional,
    List,
    Tuple,
    Set,
    FrozenSet,
    Union,
    Any,
    Callable,
    TypeVar,
    overload,
    TYPE_CHECKING,
)

class FunctionKind(enum.Enum):
    MODULE = 0
    CLASS = 1
    INSTANCE = 2
    STATIC = 3
    PROPERTY = 4
    DJANGO_CACHED_PROPERTY = 5

class ExistingAnnotationStrategy(enum.Enum):
    REPLICATE = 0
    IGNORE = 1
    OMIT = 2

class ImportMap(DefaultDict[Any, Any]):
    def __init__(self) -> None:
        pass

    def merge(self, other: DefaultDict[Any, Any]) -> None:
        pass

class RenderAnnotation(GenericTypeRewriter[str]):
    def make_anonymous_typed_dict(self, required_fields: Dict[str, Any], optional_fields: Dict[str, Any]) -> Any:
        pass

    def make_builtin_typed_dict(self, name: str, annotations: Dict[str, Any], total: bool) -> Any:
        pass

    def generic_rewrite(self, typ: Any) -> str:
        pass

    def rewrite_container_type(self, container_type: Any) -> str:
        pass

    def rewrite_malformed_container(self, container: Any) -> str:
        pass

    def rewrite_type_variable(self, type_variable: Any) -> str:
        pass

    def make_builtin_tuple(self, elements: List[str]) -> str:
        pass

    def make_container_type(self, container_type: Any, elements: str) -> str:
        pass

    def rewrite_Union(self, union: Any) -> str:
        pass

    def rewrite(self, typ: Any) -> str:
        pass

class AttributeStub(Stub):
    def __init__(self, name: str, typ: Any) -> None:
        pass

    def render(self, prefix: str = '') -> str:
        pass

class FunctionStub(Stub):
    def __init__(self, name: str, signature: inspect.Signature, kind: FunctionKind, strip_modules: List[str] = [], is_async: bool = False) -> None:
        pass

    def render(self, prefix: str = '') -> str:
        pass

class ClassStub(Stub):
    def __init__(self, name: str, function_stubs: Optional[List[FunctionStub]] = None, attribute_stubs: Optional[List[AttributeStub]] = None) -> None:
        pass

    def render(self) -> str:
        pass

class ReplaceTypedDictsWithStubs(TypeRewriter):
    def __init__(self, class_name_hint: str) -> None:
        pass

    def _rewrite_container(self, cls: Any, container: Any) -> Any:
        pass

    def _add_typed_dict_class_stub(self, fields: Dict[str, Any], class_name: str, base_class_name: str = 'TypedDict', total: bool = True) -> None:
        pass

    def rewrite_anonymous_TypedDict(self, typed_dict: Any) -> ForwardRef:
        pass

    @staticmethod
    def rewrite_and_get_stubs(typ: Any, class_name_hint: str) -> Tuple[Any, List[Any]]:
        pass

class ModuleStub(Stub):
    def __init__(self, function_stubs: Optional[List[FunctionStub]] = None, class_stubs: Optional[List[ClassStub]] = None, imports_stub: Optional[ImportBlockStub] = None, typed_dict_class_stubs: Optional[List[Any]] = None) -> None:
        pass

    def render(self) -> str:
        pass

class FunctionDefinition:
    _KIND_WITH_SELF = {FunctionKind.CLASS, FunctionKind.INSTANCE, FunctionKind.PROPERTY, FunctionKind.DJANGO_CACHED_PROPERTY}

    def __init__(self, module: str, qualname: str, kind: FunctionKind, sig: inspect.Signature, is_async: bool = False, typed_dict_class_stubs: Optional[List[Any]] = None) -> None:
        pass

    @classmethod
    def from_callable(cls, func: Callable, kind: Optional[FunctionKind] = None) -> 'FunctionDefinition':
        pass

    @classmethod
    def from_callable_and_traced_types(cls, func: Callable, arg_types: Dict[str, Any], return_type: Optional[Any], yield_type: Optional[Any], existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE) -> 'FunctionDefinition':
        pass

    @property
    def has_self(self) -> bool:
        pass

    def __eq__(self, other: Any) -> bool:
        pass

    def __repr__(self) -> str:
        pass

def _get_import_for_qualname(qualname: str) -> str:
    pass

def get_imports_for_annotation(anno: Any) -> ImportMap:
    pass

def get_imports_for_signature(sig: inspect.Signature) -> ImportMap:
    pass

def update_signature_args(sig: inspect.Signature, arg_types: Dict[str, Any], has_self: bool, existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE) -> inspect.Signature:
    pass

def update_signature_return(sig: inspect.Signature, return_type: Optional[Any] = None, yield_type: Optional[Any] = None, existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE) -> inspect.Signature:
    pass

def shrink_traced_types(traces: Iterable[Any], max_typed_dict_size: int) -> Tuple[Dict[str, Any], Optional[Any], Optional[Any]]:
    pass

def get_typed_dict_class_name(parameter_name: str) -> str:
    pass

def render_annotation(anno: Any) -> str:
    pass

def render_parameter(param: inspect.Parameter) -> str:
    pass

def render_signature(sig: inspect.Signature, max_line_len: Optional[int] = None, prefix: str = '') -> str:
    pass

def get_updated_definition(func: Callable, traces: Iterable[Any], max_typed_dict_size: int, rewriter: Optional[TypeRewriter] = None, existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE) -> FunctionDefinition:
    pass

def build_module_stubs(entries: Iterable[FunctionDefinition]) -> Dict[str, ModuleStub]:
    pass

def build_module_stubs_from_traces(traces: Iterable[Any], max_typed_dict_size: int, existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE, rewriter: Optional[TypeRewriter] = None) -> Dict[str, ModuleStub]:
    pass

class StubIndexBuilder(CallTraceLogger):
    def __init__(self, module_re: str, max_typed_dict_size: int) -> None:
        pass

    def log(self, trace: Any) -> None:
        pass

    def get_stubs(self) -> Dict[str, ModuleStub]:
        pass