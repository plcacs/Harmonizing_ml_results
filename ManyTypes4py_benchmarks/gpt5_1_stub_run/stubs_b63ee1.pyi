from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple
import enum
import inspect
import logging
from monkeytype.tracing import CallTrace, CallTraceLogger
from monkeytype.typing import GenericTypeRewriter, TypeRewriter


logger: logging.Logger = ...


class FunctionKind(enum.Enum):
    MODULE: "FunctionKind"
    CLASS: "FunctionKind"
    INSTANCE: "FunctionKind"
    STATIC: "FunctionKind"
    PROPERTY: "FunctionKind"
    DJANGO_CACHED_PROPERTY: "FunctionKind"

    @classmethod
    def from_callable(cls, func: Callable[..., Any]) -> "FunctionKind": ...


class ExistingAnnotationStrategy(enum.Enum):
    REPLICATE: "ExistingAnnotationStrategy"
    IGNORE: "ExistingAnnotationStrategy"
    OMIT: "ExistingAnnotationStrategy"


class ImportMap(DefaultDict[str, Set[str]]):
    def __init__(self) -> None: ...
    def merge(self, other: Dict[str, Set[str]]) -> None: ...


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
    return_type: Optional[Any] = ...,
    yield_type: Optional[Any] = ...,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> inspect.Signature: ...
def shrink_traced_types(
    traces: Iterable[CallTrace],
    max_typed_dict_size: int,
) -> Tuple[Dict[str, Any], Optional[Any], Optional[Any]]: ...
def get_typed_dict_class_name(parameter_name: str) -> str: ...


class Stub:
    def __eq__(self, other: object) -> bool: ...
    def render(self) -> str: ...


class ImportBlockStub(Stub):
    imports: ImportMap

    def __init__(self, imports: Optional[ImportMap] = ...) -> None: ...
    def render(self) -> str: ...
    def __repr__(self) -> str: ...


class RenderAnnotation(GenericTypeRewriter[str]):
    def make_anonymous_typed_dict(self, required_fields: Any, optional_fields: Any) -> str: ...
    def make_builtin_typed_dict(self, name: Any, annotations: Any, total: Any) -> str: ...
    def generic_rewrite(self, typ: Any) -> str: ...
    def rewrite_container_type(self, container_type: Any) -> str: ...
    def rewrite_malformed_container(self, container: Any) -> str: ...
    def rewrite_type_variable(self, type_variable: Any) -> str: ...
    def make_builtin_tuple(self, elements: Iterable[str]) -> str: ...
    def make_container_type(self, container_type: Any, elements: str) -> str: ...
    def rewrite_Union(self, union: Any) -> str: ...
    def rewrite(self, typ: Any) -> str: ...


def render_annotation(anno: Any) -> str: ...
def render_parameter(param: inspect.Parameter) -> str: ...
def render_signature(sig: inspect.Signature, max_line_len: Optional[int] = ..., prefix: str = ...) -> str: ...


class AttributeStub(Stub):
    def __init__(self, name: str, typ: Any) -> None: ...
    def render(self, prefix: str = ...) -> str: ...
    def __repr__(self) -> str: ...


class FunctionStub(Stub):
    def __init__(
        self,
        name: str,
        signature: inspect.Signature,
        kind: FunctionKind,
        strip_modules: Optional[List[str]] = ...,
        is_async: bool = ...,
    ) -> None: ...
    def render(self, prefix: str = ...) -> str: ...
    def __repr__(self) -> str: ...


class ClassStub(Stub):
    def __init__(
        self,
        name: str,
        function_stubs: Optional[Iterable[FunctionStub]] = ...,
        attribute_stubs: Optional[Iterable[AttributeStub]] = ...,
    ) -> None: ...
    def render(self) -> str: ...
    def __repr__(self) -> str: ...


class ReplaceTypedDictsWithStubs(TypeRewriter):
    stubs: List[ClassStub]

    def __init__(self, class_name_hint: str) -> None: ...
    def _rewrite_container(self, cls: Any, container: Any) -> Any: ...
    def _add_typed_dict_class_stub(
        self,
        fields: Dict[str, Any],
        class_name: str,
        base_class_name: str = ...,
        total: bool = ...,
    ) -> None: ...
    def rewrite_anonymous_TypedDict(self, typed_dict: Any) -> Any: ...
    @staticmethod
    def rewrite_and_get_stubs(typ: Any, class_name_hint: str) -> Tuple[Any, List[ClassStub]]: ...


class ModuleStub(Stub):
    def __init__(
        self,
        function_stubs: Optional[Iterable[FunctionStub]] = ...,
        class_stubs: Optional[Iterable[ClassStub]] = ...,
        imports_stub: Optional[ImportBlockStub] = ...,
        typed_dict_class_stubs: Optional[Iterable[ClassStub]] = ...,
    ) -> None: ...
    def render(self) -> str: ...
    def __repr__(self) -> str: ...


class FunctionDefinition:
    _KIND_WITH_SELF: Set[FunctionKind]

    def __init__(
        self,
        module: str,
        qualname: str,
        kind: FunctionKind,
        sig: inspect.Signature,
        is_async: bool = ...,
        typed_dict_class_stubs: Optional[Iterable[ClassStub]] = ...,
    ) -> None: ...
    @classmethod
    def from_callable(cls, func: Callable[..., Any], kind: Optional[FunctionKind] = ...) -> "FunctionDefinition": ...
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
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...


def get_updated_definition(
    func: Callable[..., Any],
    traces: Iterable[CallTrace],
    max_typed_dict_size: int,
    rewriter: Optional[TypeRewriter] = ...,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
) -> FunctionDefinition: ...
def build_module_stubs(entries: Iterable[FunctionDefinition]) -> Dict[str, ModuleStub]: ...
def build_module_stubs_from_traces(
    traces: Iterable[CallTrace],
    max_typed_dict_size: int,
    existing_annotation_strategy: ExistingAnnotationStrategy = ExistingAnnotationStrategy.REPLICATE,
    rewriter: Optional[TypeRewriter] = ...,
) -> Dict[str, ModuleStub]: ...


class StubIndexBuilder(CallTraceLogger):
    def __init__(self, module_re: str, max_typed_dict_size: int) -> None: ...
    def log(self, trace: CallTrace) -> None: ...
    def get_stubs(self) -> Dict[str, ModuleStub]: ...