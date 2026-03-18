```python
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
)

class FunctionKind(enum.Enum):
    MODULE: int = 0
    CLASS: int = 1
    INSTANCE: int = 2
    STATIC: int = 3
    PROPERTY: int = 4
    DJANGO_CACHED_PROPERTY: int = 5

    @classmethod
    def from_callable(cls, func: Any) -> "FunctionKind": ...

class ExistingAnnotationStrategy(enum.Enum):
    REPLICATE: int = 0
    IGNORE: int = 1
    OMIT: int = 2

class ImportMap(DefaultDict[Any, Any]):
    def __init__(self) -> None: ...
    def merge(self, other: Any) -> None: ...

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
    return_type: Any = None,
    yield_type: Any = None,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> inspect.Signature: ...
def shrink_traced_types(
    traces: Iterable[Any],
    max_typed_dict_size: int,
) -> Tuple[Dict[str, Any], Any, Any]: ...
def get_typed_dict_class_name(parameter_name: str) -> str: ...

class Stub(metaclass=ABCMeta):
    def __eq__(self, other: Any) -> bool: ...
    @abstractmethod
    def render(self) -> str: ...

class ImportBlockStub(Stub):
    def __init__(self, imports: Optional[ImportMap] = None) -> None: ...
    def render(self) -> str: ...

def _is_optional(anno: Any) -> bool: ...
def _get_optional_elem(anno: Any) -> Any: ...

class RenderAnnotation(GenericTypeRewriter[str]):
    def make_anonymous_typed_dict(self, required_fields: Any, optional_fields: Any) -> str: ...
    def make_builtin_typed_dict(self, name: str, annotations: Any, total: bool) -> str: ...
    def generic_rewrite(self, typ: Any) -> str: ...
    def rewrite_container_type(self, container_type: Any) -> str: ...
    def rewrite_malformed_container(self, container: Any) -> str: ...
    def rewrite_type_variable(self, type_variable: Any) -> str: ...
    def make_builtin_tuple(self, elements: Any) -> str: ...
    def make_container_type(self, container_type: Any, elements: Any) -> str: ...
    def rewrite_Union(self, union: Any) -> str: ...
    def rewrite(self, typ: Any) -> str: ...

def render_annotation(anno: Any) -> str: ...
def render_parameter(param: inspect.Parameter) -> str: ...
def render_signature(sig: inspect.Signature, max_line_len: Optional[int] = None, prefix: str = "") -> str: ...

class AttributeStub(Stub):
    def __init__(self, name: str, typ: Any) -> None: ...
    def render(self, prefix: str = "") -> str: ...

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

class ClassStub(Stub):
    def __init__(
        self,
        name: str,
        function_stubs: Optional[Iterable[FunctionStub]] = None,
        attribute_stubs: Optional[Iterable[AttributeStub]] = None,
    ) -> None: ...
    def render(self) -> str: ...

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
    def rewrite_anonymous_TypedDict(self, typed_dict: Any) -> Any: ...
    @staticmethod
    def rewrite_and_get_stubs(typ: Any, class_name_hint: str) -> Tuple[Any, List[ClassStub]]: ...

class ModuleStub(Stub):
    def __init__(
        self,
        function_stubs: Optional[Iterable[FunctionStub]] = None,
        class_stubs: Optional[Iterable[ClassStub]] = None,
        imports_stub: Optional[ImportBlockStub] = None,
        typed_dict_class_stubs: Optional[Iterable[ClassStub]] = None,
    ) -> None: ...
    def render(self) -> str: ...

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
    def from_callable(cls, func: Any, kind: Optional[FunctionKind] = None) -> "FunctionDefinition": ...
    @classmethod
    def from_callable_and_traced_types(
        cls,
        func: Any,
        arg_types: Dict[str, Any],
        return_type: Any,
        yield_type: Any,
        existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
    ) -> "FunctionDefinition": ...
    @property
    def has_self(self) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...

def get_updated_definition(
    func: Any,
    traces: Iterable[Any],
    max_typed_dict_size: int,
    rewriter: Optional[TypeRewriter] = None,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> FunctionDefinition: ...
def build_module_stubs(entries: Iterable[FunctionDefinition]) -> Dict[str, ModuleStub]: ...
def build_module_stubs_from_traces(
    traces: Iterable[Any],
    max_typed_dict_size: int,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
    rewriter: Optional[TypeRewriter] = None,
) -> Dict[str, ModuleStub]: ...

class StubIndexBuilder(CallTraceLogger):
    def __init__(self, module_re: str, max_typed_dict_size: int) -> None: ...
    def log(self, trace: Any) -> None: ...
    def get_stubs(self) -> Dict[str, ModuleStub]: ...

logger: logging.Logger = ...
```