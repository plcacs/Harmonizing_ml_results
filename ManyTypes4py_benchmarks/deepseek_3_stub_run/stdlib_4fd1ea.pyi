import os
from inspect import Parameter
from typing import (
    Any,
    Callable,
    Dict,
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

from jedi import debug
from jedi.inference import analysis, compiled
from jedi.inference.arguments import (
    AbstractArguments,
    ParamIssue,
    TreeArgumentsWrapper,
)
from jedi.inference.base_value import (
    ContextualizedNode,
    NO_VALUES,
    ValueSet,
    ValueWrapper,
    LazyValueWrapper,
)
from jedi.inference.filters import (
    AttributeOverwrite,
    DictFilter,
    ParserTreeFilter,
    publish_method,
)
from jedi.inference.helpers import get_str_or_none
from jedi.inference.lazy_value import (
    LazyTreeValue,
    LazyKnownValue,
    LazyKnownValues,
)
from jedi.inference.names import BaseTreeParamName, ValueName
from jedi.inference.signature import AbstractSignature, SignatureWrapper
from jedi.inference.utils import safe_property
from jedi.inference.value import (
    ClassValue,
    ModuleValue,
    iterable,
    ClassMixin,
    FunctionMixin,
)
from jedi.inference.value.function import FunctionMixin
from jedi.inference.value.instance import (
    AnonymousMethodExecutionContext,
    MethodExecutionContext,
)
from jedi.inference.value.klass import ClassMixin
import parso

_T = TypeVar("_T")
_NAMEDTUPLE_CLASS_TEMPLATE: str = ...
_NAMEDTUPLE_FIELD_TEMPLATE: str = ...

def execute(callback: Callable[..., Any]) -> Callable[..., Any]: ...

def _follow_param(
    inference_state: Any, arguments: AbstractArguments, index: int
) -> ValueSet: ...

def argument_clinic(
    clinic_string: str,
    want_value: bool = False,
    want_context: bool = False,
    want_arguments: bool = False,
    want_inference_state: bool = False,
    want_callback: bool = False,
) -> Callable[..., Callable[..., ValueSet]]: ...

@argument_clinic("iterator[, default], /", want_inference_state=True)
def builtins_next(
    iterators: ValueSet, defaults: ValueSet, inference_state: Any
) -> ValueSet: ...

@argument_clinic("iterator[, default], /")
def builtins_iter(
    iterators_or_callables: ValueSet, defaults: ValueSet
) -> ValueSet: ...

@argument_clinic("object, name[, default], /")
def builtins_getattr(
    objects: ValueSet, names: ValueSet, defaults: Optional[ValueSet] = None
) -> ValueSet: ...

@argument_clinic("object[, bases, dict], /")
def builtins_type(
    objects: ValueSet, bases: ValueSet, dicts: ValueSet
) -> ValueSet: ...

class SuperInstance(LazyValueWrapper):
    def __init__(self, inference_state: Any, instance: Any) -> None: ...
    def _get_bases(self) -> List[Any]: ...
    def _get_wrapped_value(self) -> Any: ...
    def get_filters(
        self, origin_scope: Optional[Any] = None
    ) -> Generator[Any, None, None]: ...

@argument_clinic("[type[, value]], /", want_context=True)
def builtins_super(
    types: ValueSet, objects: ValueSet, context: Any
) -> ValueSet: ...

class ReversedObject(AttributeOverwrite):
    def __init__(self, reversed_obj: Any, iter_list: List[Any]) -> None: ...
    def py__iter__(
        self, contextualized_node: Optional[ContextualizedNode] = None
    ) -> List[Any]: ...
    @publish_method("__next__")
    def _next(self, arguments: AbstractArguments) -> ValueSet: ...

@argument_clinic("sequence, /", want_value=True, want_arguments=True)
def builtins_reversed(
    sequences: ValueSet, value: Any, arguments: AbstractArguments
) -> ValueSet: ...

@argument_clinic(
    "value, type, /", want_arguments=True, want_inference_state=True
)
def builtins_isinstance(
    objects: ValueSet,
    types: ValueSet,
    arguments: AbstractArguments,
    inference_state: Any,
) -> ValueSet: ...

class StaticMethodObject(ValueWrapper):
    def py__get__(
        self, instance: Optional[Any], class_value: Any
    ) -> ValueSet: ...

@argument_clinic("sequence, /")
def builtins_staticmethod(functions: ValueSet) -> ValueSet: ...

class ClassMethodObject(ValueWrapper):
    def __init__(self, class_method_obj: Any, function: Any) -> None: ...
    def py__get__(
        self, instance: Optional[Any], class_value: Any
    ) -> ValueSet: ...

class ClassMethodGet(ValueWrapper):
    def __init__(self, get_method: Any, klass: Any, function: Any) -> None: ...
    def get_signatures(self) -> List[Any]: ...
    def py__call__(self, arguments: AbstractArguments) -> ValueSet: ...

class ClassMethodArguments(TreeArgumentsWrapper):
    def __init__(self, klass: Any, arguments: AbstractArguments) -> None: ...
    def unpack(
        self, func: Optional[Any] = None
    ) -> Generator[Tuple[Optional[str], Any], None, None]: ...

@argument_clinic("sequence, /", want_value=True, want_arguments=True)
def builtins_classmethod(
    functions: ValueSet, value: Any, arguments: AbstractArguments
) -> ValueSet: ...

class PropertyObject(AttributeOverwrite, ValueWrapper):
    api_type: str = ...
    def __init__(self, property_obj: Any, function: Any) -> None: ...
    def py__get__(
        self, instance: Optional[Any], class_value: Any
    ) -> ValueSet: ...
    @publish_method("deleter")
    @publish_method("getter")
    @publish_method("setter")
    def _return_self(self, arguments: AbstractArguments) -> ValueSet: ...

@argument_clinic("func, /", want_callback=True)
def builtins_property(
    functions: ValueSet, callback: Callable[..., Any]
) -> ValueSet: ...

def collections_namedtuple(
    value: Any, arguments: AbstractArguments, callback: Callable[..., Any]
) -> ValueSet: ...

class PartialObject(ValueWrapper):
    def __init__(
        self,
        actual_value: Any,
        arguments: AbstractArguments,
        instance: Optional[Any] = None,
    ) -> None: ...
    def _get_functions(
        self, unpacked_arguments: Iterator[Tuple[Optional[str], Any]]
    ) -> Optional[ValueSet]: ...
    def get_signatures(self) -> List[Any]: ...
    def py__call__(self, arguments: AbstractArguments) -> ValueSet: ...
    def py__doc__(self) -> str: ...
    def py__get__(
        self, instance: Optional[Any], class_value: Any
    ) -> ValueSet: ...

class PartialMethodObject(PartialObject): ...

class PartialSignature(SignatureWrapper):
    def __init__(
        self,
        wrapped_signature: Any,
        skipped_arg_count: int,
        skipped_arg_set: Set[str],
    ) -> None: ...
    def get_param_names(self, resolve_stars: bool = False) -> List[Any]: ...

class MergedPartialArguments(AbstractArguments):
    def __init__(
        self,
        partial_arguments: AbstractArguments,
        call_arguments: AbstractArguments,
        instance: Optional[Any] = None,
    ) -> None: ...
    def unpack(
        self, funcdef: Optional[Any] = None
    ) -> Generator[Tuple[Optional[str], Any], None, None]: ...

def functools_partial(
    value: Any, arguments: AbstractArguments, callback: Callable[..., Any]
) -> ValueSet: ...

def functools_partialmethod(
    value: Any, arguments: AbstractArguments, callback: Callable[..., Any]
) -> ValueSet: ...

@argument_clinic("first, /")
def _return_first_param(firsts: ValueSet) -> ValueSet: ...

@argument_clinic("seq")
def _random_choice(sequences: ValueSet) -> ValueSet: ...

def _dataclass(
    value: Any, arguments: AbstractArguments, callback: Callable[..., Any]
) -> ValueSet: ...

class DataclassWrapper(ValueWrapper, ClassMixin):
    def get_signatures(self) -> List[Any]: ...

class DataclassSignature(AbstractSignature):
    def __init__(self, value: Any, param_names: List[Any]) -> None: ...
    def get_param_names(self, resolve_stars: bool = False) -> List[Any]: ...

class DataclassParamName(BaseTreeParamName):
    def __init__(
        self,
        parent_context: Any,
        tree_name: Any,
        annotation_node: Any,
        default_node: Optional[Any],
    ) -> None: ...
    def get_kind(self) -> Parameter: ...
    def infer(self) -> ValueSet: ...

class ItemGetterCallable(ValueWrapper):
    def __init__(self, instance: Any, args_value_set: ValueSet) -> None: ...
    @repack_with_argument_clinic("item, /")
    def py__call__(self, item_value_set: ValueSet) -> ValueSet: ...

@argument_clinic("func, /")
def _functools_wraps(funcs: ValueSet) -> ValueSet: ...

class WrapsCallable(ValueWrapper):
    @repack_with_argument_clinic("func, /")
    def py__call__(self, funcs: ValueSet) -> ValueSet: ...

class Wrapped(ValueWrapper, FunctionMixin):
    def __init__(self, func: Any, original_function: Any) -> None: ...
    @property
    def name(self) -> Any: ...
    def get_signature_functions(self) -> List[Any]: ...

@argument_clinic("*args, /", want_value=True, want_arguments=True)
def _operator_itemgetter(
    args_value_set: ValueSet, value: Any, arguments: AbstractArguments
) -> ValueSet: ...

def _create_string_input_function(
    func: Callable[[str], str]
) -> Callable[..., ValueSet]: ...

@argument_clinic("*args, /", want_callback=True)
def _os_path_join(args_set: ValueSet, callback: Callable[..., Any]) -> ValueSet: ...

_implemented: Dict[str, Dict[str, Callable[..., Any]]] = ...

def get_metaclass_filters(
    func: Callable[..., List[Any]]
) -> Callable[..., List[Any]]: ...

class EnumInstance(LazyValueWrapper):
    def __init__(self, cls: Any, name: Any) -> None: ...
    @safe_property
    def name(self) -> ValueName: ...
    def _get_wrapped_value(self) -> Any: ...
    def get_filters(
        self, origin_scope: Optional[Any] = None
    ) -> Generator[Any, None, None]: ...

def tree_name_to_values(
    func: Callable[..., ValueSet]
) -> Callable[..., ValueSet]: ...