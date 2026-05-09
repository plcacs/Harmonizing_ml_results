import asyncio
import collections
import enum
import inspect
import logging
import re
from abc import ABCMeta, abstractmethod
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
    AnyStr,
    Type,
    TypeVar,
    overload,
    TYPE_CHECKING,
)
from monkeytype.typing import NoneType
from monkeytype.util import CallTrace

logger: logging.Logger

class FunctionKind(enum.Enum):
    MODULE: int
    CLASS: int
    INSTANCE: int
    STATIC: int
    PROPERTY: int
    DJANGO_CACHED_PROPERTY: int

    @classmethod
    def from_callable(cls, func: Callable) -> FunctionKind:
        ...

class ExistingAnnotationStrategy(enum.Enum):
    REPLICATE: int
    IGNORE: int
    OMIT: int

class ImportMap(DefaultDict[Any, Any]):
    def __init__(self) -> None:
        ...

    def merge(self, other: DefaultDict[Any, Any]) -> None:
        ...

def _get_import_for_qualname(qualname: str) -> str:
    ...

def get_imports_for_annotation(anno: Any) -> ImportMap:
    ...

def get_imports_for_signature(sig: inspect.Signature) -> ImportMap:
    ...

def update_signature_args(
    sig: inspect.Signature,
    arg_types: Dict[str, Any],
    has_self: bool,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> inspect.Signature:
    ...

def update_signature_return(
    sig: inspect.Signature,
    return_type: Optional[Any] = None,
    yield_type: Optional[Any] = None,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> inspect.Signature:
    ...

def shrink_traced_types(
    traces: Iterable[CallTrace],
    max_typed_dict_size: int,
) -> Tuple[Dict[str, Any], Optional[Any], Optional[Any]]:
    ...

def get_typed_dict_class_name(parameter_name: str) -> str:
    ...

class Stub(metaclass=ABCMeta):
    @abstractmethod
    def render(self) -> str:
        ...

class ImportBlockStub(Stub):
    def __init__(self, imports: Optional[ImportMap] = None) -> None:
        ...

    def render(self) -> str:
        ...

class AttributeStub(Stub):
    def __init__(self, name: str, typ: Any) -> None:
        ...

    def render(self, prefix: str = '') -> str:
        ...

class FunctionStub(Stub):
    def __init__(
        self,
        name: str,
        signature: inspect.Signature,
        kind: FunctionKind,
        strip_modules: Optional[List[str]] = None,
        is_async: bool = False,
    ) -> None:
        ...

    def render(self, prefix: str = '') -> str:
        ...

class ClassStub(Stub):
    def __init__(
        self,
        name: str,
        function_stubs: Optional[List[FunctionStub]] = None,
        attribute_stubs: Optional[List[AttributeStub]] = None,
    ) -> None:
        ...

    def render(self) -> str:
        ...

class ReplaceTypedDictsWithStubs:
    def __init__(self, class_name_hint: str) -> None:
        ...

    def rewrite_anonymous_TypedDict(self, typed_dict: Any) -> ForwardRef:
        ...

    @staticmethod
    def rewrite_and_get_stubs(
        typ: Any,
        class_name_hint: str,
    ) -> Tuple[Any, List[ClassStub]]:
        ...

class ModuleStub(Stub):
    def __init__(
        self,
        function_stubs: Optional[List[FunctionStub]] = None,
        class_stubs: Optional[List[ClassStub]] = None,
        imports_stub: Optional[ImportBlockStub] = None,
        typed_dict_class_stubs: Optional[List[ClassStub]] = None,
    ) -> None:
        ...

    def render(self) -> str:
        ...

class FunctionDefinition:
    _KIND_WITH_SELF: Set[FunctionKind]

    def __init__(
        self,
        module: str,
        qualname: str,
        kind: FunctionKind,
        sig: inspect.Signature,
        is_async: bool = False,
        typed_dict_class_stubs: Optional[List[ClassStub]] = None,
    ) -> None:
        ...

    @classmethod
    def from_callable(cls, func: Callable, kind: Optional[FunctionKind] = None) -> FunctionDefinition:
        ...

    @classmethod
    def from_callable_and_traced_types(
        cls,
        func: Callable,
        arg_types: Dict[str, Any],
        return_type: Optional[Any],
        yield_type: Optional[Any],
        existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
    ) -> FunctionDefinition:
        ...

    @property
    def has_self(self) -> bool:
        ...

def get_updated_definition(
    func: Callable,
    traces: Iterable[CallTrace],
    max_typed_dict_size: int,
    rewriter: Optional[Any] = None,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> FunctionDefinition:
    ...

def build_module_stubs(entries: Iterable[FunctionDefinition]) -> Dict[str, ModuleStub]:
    ...

def build_module_stubs_from_traces(
    traces: Iterable[CallTrace],
    max_typed_dict_size: int,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
    rewriter: Optional[Any] = None,
) -> Dict[str, ModuleStub]:
    ...

class StubIndexBuilder(CallTraceLogger):
    def __init__(self, module_re: str, max_typed_dict_size: int) -> None:
        ...

    def log(self, trace: CallTrace) -> None:
        ...

    def get_stubs(self) -> Dict[str, ModuleStub]:
        ...