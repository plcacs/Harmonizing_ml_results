from abc import ABCMeta, abstractmethod
import inspect
import logging
from enum import Enum
from typing import Any, Callable, ClassVar, DefaultDict, Dict, ForwardRef, Iterable, List, Mapping, Optional, Pattern, Set, Tuple, Union

from monkeytype.tracing import CallTrace, CallTraceLogger
from monkeytype.typing import GenericTypeRewriter, TypeRewriter

logger: logging.Logger

class FunctionKind(Enum):
    @classmethod
    def from_callable(cls, func: Callable[..., Any]) -> "FunctionKind": ...

class ExistingAnnotationStrategy(Enum):
    REPLICATE: ClassVar["ExistingAnnotationStrategy"]
    IGNORE: ClassVar["ExistingAnnotationStrategy"]
    OMIT: ClassVar["ExistingAnnotationStrategy"]

class ImportMap(DefaultDict[str, Set[str]]):
    def __init__(self) -> None: ...
    def merge(self, other: Mapping[str, Iterable[str]]) -> None: ...

def _get_import_for_qualname(qualname: str) -> str: ...
def get_imports_for_annotation(anno: object) -> ImportMap: ...
def get_imports_for_signature(sig: inspect.Signature) -> ImportMap: ...
def update_signature_args(
    sig: inspect.Signature,
    arg_types: Mapping[str, object],
    has_self: bool,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> inspect.Signature: ...
def update_signature_return(
    sig: inspect.Signature,
    return_type: Optional[object] = ...,
    yield_type: Optional[object] = ...,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> inspect.Signature: ...
def shrink_traced_types(
    traces: Iterable[CallTrace], max_typed_dict_size: int
) -> Tuple[Dict[str, object], Optional[object], Optional[object]]: ...
def get_typed_dict_class_name(parameter_name: str) -> str: ...

class Stub(metaclass=ABCMeta):
    def __eq__(self, other: object) -> bool: ...
    @abstractmethod
    def render(self) -> str: ...

class ImportBlockStub(Stub):
    imports: ImportMap
    def __init__(self, imports: Optional[ImportMap] = None) -> None: ...
    def render(self) -> str: ...
    def __repr__(self) -> str: ...

def _is_optional(anno: object) -> bool: ...
def _get_optional_elem(anno: object) -> object: ...

class RenderAnnotation(GenericTypeRewriter[str]):
    def make_anonymous_typed_dict(self, required_fields: Mapping[str, object], optional_fields: Mapping[str, object]) -> str: ...
    def make_builtin_typed_dict(self, name: str, annotations: Mapping[str, object], total: bool) -> str: ...
    def generic_rewrite(self, typ: object) -> str: ...
    def rewrite_container_type(self, container_type: object) -> str: ...
    def rewrite_malformed_container(self, container: object) -> str: ...
    def rewrite_type_variable(self, type_variable: object) -> str: ...
    def make_builtin_tuple(self, elements: Iterable[str]) -> str: ...
    def make_container_type(self, container_type: object, elements: str) -> str: ...
    def rewrite_Union(self, union: object) -> str: ...
    def rewrite(self, typ: object) -> str: ...

def render_annotation(anno: object) -> str: ...
def render_parameter(param: inspect.Parameter) -> str: ...
def render_signature(sig: inspect.Signature, max_line_len: Optional[int] = None, prefix: str = "") -> str: ...

class AttributeStub(Stub):
    name: str
    typ: object
    def __init__(self, name: str, typ: object) -> None: ...
    def render(self, prefix: str = "") -> str: ...
    def __repr__(self) -> str: ...

class FunctionStub(Stub):
    name: str
    signature: inspect.Signature
    kind: FunctionKind
    strip_modules: List[str]
    is_async: bool
    def __init__(
        self,
        name: str,
        signature: inspect.Signature,
        kind: FunctionKind,
        strip_modules: Optional[Iterable[str]] = None,
        is_async: bool = False,
    ) -> None: ...
    def render(self, prefix: str = "") -> str: ...
    def __repr__(self) -> str: ...

class ClassStub(Stub):
    name: str
    function_stubs: Dict[str, FunctionStub]
    attribute_stubs: List[AttributeStub]
    def __init__(
        self,
        name: str,
        function_stubs: Optional[Iterable[FunctionStub]] = None,
        attribute_stubs: Optional[Iterable[AttributeStub]] = None,
    ) -> None: ...
    def render(self) -> str: ...
    def __repr__(self) -> str: ...

class ReplaceTypedDictsWithStubs(TypeRewriter):
    stubs: List[ClassStub]
    def __init__(self, class_name_hint: str) -> None: ...
    def _rewrite_container(self, cls: object, container: object) -> object: ...
    def _add_typed_dict_class_stub(
        self,
        fields: Mapping[str, object],
        class_name: str,
        base_class_name: str = "TypedDict",
        total: bool = True,
    ) -> None: ...
    def rewrite_anonymous_TypedDict(self, typed_dict: object) -> ForwardRef: ...
    @staticmethod
    def rewrite_and_get_stubs(typ: object, class_name_hint: str) -> Tuple[object, List[ClassStub]]: ...

class ModuleStub(Stub):
    function_stubs: Dict[str, FunctionStub]
    class_stubs: Dict[str, ClassStub]
    imports_stub: ImportBlockStub
    typed_dict_class_stubs: List[ClassStub]
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
    _KIND_WITH_SELF: ClassVar[Set[FunctionKind]]
    module: str
    qualname: str
    kind: FunctionKind
    signature: inspect.Signature
    is_async: bool
    typed_dict_class_stubs: List[ClassStub]
    def __init__(
        self,
        module: str,
        qualname: str,
        kind: FunctionKind,
        sig: inspect.Signature,
        is_async: bool = False,
        typed_dict_class_stubs: Optional[Iterable[ClassStub]] = None,
    ) -> None: ...
    @classmethod
    def from_callable(cls, func: Callable[..., Any], kind: Optional[FunctionKind] = None) -> "FunctionDefinition": ...
    @classmethod
    def from_callable_and_traced_types(
        cls,
        func: Callable[..., Any],
        arg_types: Mapping[str, object],
        return_type: Optional[object],
        yield_type: Optional[object],
        existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
    ) -> "FunctionDefinition": ...
    @property
    def has_self(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

def get_updated_definition(
    func: Callable[..., Any],
    traces: Iterable[CallTrace],
    max_typed_dict_size: int,
    rewriter: Optional[TypeRewriter] = None,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> FunctionDefinition: ...

def build_module_stubs(entries: Iterable[FunctionDefinition]) -> Dict[str, ModuleStub]: ...

def build_module_stubs_from_traces(
    traces: Iterable[CallTrace],
    max_typed_dict_size: int,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
    rewriter: Optional[TypeRewriter] = None,
) -> Dict[str, ModuleStub]: ...

class StubIndexBuilder(CallTraceLogger):
    re: Pattern[str]
    index: DefaultDict[Callable[..., Any], Set[CallTrace]]
    max_typed_dict_size: int
    def __init__(self, module_re: str, max_typed_dict_size: int) -> None: ...
    def log(self, trace: CallTrace) -> None: ...
    def get_stubs(self) -> Dict[str, ModuleStub]: ...