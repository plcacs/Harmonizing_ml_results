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
    Pattern,
    Type,
    TypeVar,
    Iterator,
    Generator,
    Container,
    overload,
)

T = TypeVar("T")

class FunctionKind(enum.Enum):
    MODULE = 0
    CLASS = 1
    INSTANCE = 2
    STATIC = 3
    PROPERTY = 4
    DJANGO_CACHED_PROPERTY = 5

    @classmethod
    def from_callable(cls, func: Callable[..., Any]) -> "FunctionKind": ...

class ExistingAnnotationStrategy(enum.Enum):
    REPLICATE = 0
    IGNORE = 1
    OMIT = 2

class ImportMap(DefaultDict[str, Set[str]]):
    def __init__(self) -> None: ...
    def merge(self, other: "ImportMap") -> None: ...

def _get_import_for_qualname(qualname: str) -> str: ...

def get_imports_for_annotation(anno: Any) -> ImportMap: ...

def get_imports_for_signature(sig: inspect.Signature) -> ImportMap: ...

def update_signature_args(
    sig: inspect.Signature,
    arg_types: Dict[str, Any],
    has_self: bool,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> inspect.Signature: ...

def update_signature_return(
    sig: inspect.Signature,
    return_type: Optional[Any] = None,
    yield_type: Optional[Any] = None,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> inspect.Signature: ...

def shrink_traced_types(
    traces: Iterable["CallTrace"],
    max_typed_dict_size: int,
) -> Tuple[Dict[str, Any], Optional[Any], Optional[Any]]: ...

def get_typed_dict_class_name(parameter_name: str) -> str: ...

class Stub(metaclass=ABCMeta):
    def __eq__(self, other: Any) -> Union[bool, Any]: ...
    @abstractmethod
    def render(self) -> str: ...

class ImportBlockStub(Stub):
    def __init__(self, imports: Optional[ImportMap] = None) -> None: ...
    def render(self) -> str: ...
    def __repr__(self) -> str: ...

def _is_optional(anno: Any) -> bool: ...

def _get_optional_elem(anno: Any) -> Any: ...

class RenderAnnotation(GenericTypeRewriter[str]):
    def make_anonymous_typed_dict(
        self,
        required_fields: Dict[str, Any],
        optional_fields: Dict[str, Any],
    ) -> str: ...
    def make_builtin_typed_dict(
        self,
        name: str,
        annotations: Dict[str, Any],
        total: bool,
    ) -> str: ...
    def generic_rewrite(self, typ: Any) -> str: ...
    def rewrite_container_type(self, container_type: Any) -> str: ...
    def rewrite_malformed_container(self, container: Any) -> str: ...
    def rewrite_type_variable(self, type_variable: Any) -> str: ...
    def make_builtin_tuple(self, elements: List[str]) -> str: ...
    def make_container_type(self, container_type: str, elements: str) -> str: ...
    def rewrite_Union(self, union: Any) -> str: ...
    def rewrite(self, typ: Any) -> str: ...

def render_annotation(anno: Any) -> str: ...

def render_parameter(param: inspect.Parameter) -> str: ...

def render_signature(
    sig: inspect.Signature,
    max_line_len: Optional[int] = None,
    prefix: str = "",
) -> str: ...

class AttributeStub(Stub):
    def __init__(self, name: str, typ: Any) -> None: ...
    def render(self, prefix: str = "") -> str: ...
    def __repr__(self) -> str: ...

class FunctionStub(Stub):
    def __init__(
        self,
        name: str,
        signature: inspect.Signature,
        kind: FunctionKind,
        strip_modules: Optional[List[str]] = None,
        is_async: bool = False,
    ) -> None: ...
    def render(self, prefix: str = "") -> str: ...
    def __repr__(self) -> str: ...

class ClassStub(Stub):
    def __init__(
        self,
        name: str,
        function_stubs: Optional[Iterable[FunctionStub]] = None,
        attribute_stubs: Optional[Iterable[AttributeStub]] = None,
    ) -> None: ...
    def render(self) -> str: ...
    def __repr__(self) -> str: ...

class ReplaceTypedDictsWithStubs(TypeRewriter):
    def __init__(self, class_name_hint: str) -> None: ...
    def _rewrite_container(self, cls: Any, container: Any) -> Any: ...
    def _add_typed_dict_class_stub(
        self,
        fields: Dict[str, Any],
        class_name: str,
        base_class_name: str = "TypedDict",
        total: bool = True,
    ) -> None: ...
    def rewrite_anonymous_TypedDict(self, typed_dict: Any) -> ForwardRef: ...
    @staticmethod
    def rewrite_and_get_stubs(
        typ: Any,
        class_name_hint: str,
    ) -> Tuple[Any, List[ClassStub]]: ...

class ModuleStub(Stub):
    def __init__(
        self,
        function_stubs: Optional[Iterable[FunctionStub]] = None,
        class_stubs: Optional[Iterable[ClassStub]] = None,
        imports_stub: Optional[ImportBlockStub] = None,
        typed_dict_class_stubs: Optional[Iterable[ClassStub]] = None,
    ) -> None: ...
    def render(self) -> str: ...
    def __repr__(self) -> str: ...

class FunctionDefinition:
    _KIND_WITH_SELF: Set[FunctionKind] = ...

    def __init__(
        self,
        module: str,
        qualname: str,
        kind: FunctionKind,
        sig: inspect.Signature,
        is_async: bool = False,
        typed_dict_class_stubs: Optional[List[ClassStub]] = None,
    ) -> None: ...
    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
        kind: Optional[FunctionKind] = None,
    ) -> "FunctionDefinition": ...
    @classmethod
    def from_callable_and_traced_types(
        cls,
        func: Callable[..., Any],
        arg_types: Dict[str, Any],
        return_type: Optional[Any],
        yield_type: Optional[Any],
        existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
    ) -> "FunctionDefinition": ...
    @property
    def has_self(self) -> bool: ...
    def __eq__(self, other: Any) -> Union[bool, Any]: ...
    def __repr__(self) -> str: ...

def get_updated_definition(
    func: Callable[..., Any],
    traces: Iterable["CallTrace"],
    max_typed_dict_size: int,
    rewriter: Optional[TypeRewriter] = None,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> FunctionDefinition: ...

def build_module_stubs(
    entries: Iterable[FunctionDefinition],
) -> Dict[str, ModuleStub]: ...

def build_module_stubs_from_traces(
    traces: Iterable["CallTrace"],
    max_typed_dict_size: int,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
    rewriter: Optional[TypeRewriter] = None,
) -> Dict[str, ModuleStub]: ...

class StubIndexBuilder(CallTraceLogger):
    def __init__(self, module_re: Union[str, Pattern[str]], max_typed_dict_size: int) -> None: ...
    def log(self, trace: "CallTrace") -> None: ...
    def get_stubs(self) -> Dict[str, ModuleStub]: ...

logger: logging.Logger = ...