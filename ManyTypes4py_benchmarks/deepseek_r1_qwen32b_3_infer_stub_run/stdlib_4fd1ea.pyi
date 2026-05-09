"""
Stub file for 'stdlib_4fd1ea' module.
"""

from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union, overload
)
from jedi.inference.base_value import ValueSet, LazyValueWrapper
from jedi.inference.value import ClassValue, FunctionMixin
from jedi.inference.signature import AbstractSignature
from jedi.inference.value.instance import AnonymousMethodExecutionContext, MethodExecutionContext
from jedi.inference.value.klass import ClassMixin
from jedi.inference.value.function import FunctionMixin
from jedi.inference.lazy_value import LazyTreeValue, LazyKnownValue, LazyKnownValues
from jedi.inference.names import ValueName, BaseTreeParamName
from jedi.inference.filters import AttributeOverwrite, ParserTreeFilter, DictFilter
from jedi.inference.utils import safe_property
from jedi.inference.helpers import get_str_or_none
from jedi.inference.arguments import iterate_argument_clinic, ParamIssue, repack_with_argument_clinic, AbstractArguments, TreeArgumentsWrapper
from jedi.inference import analysis, compiled

class SuperInstance(LazyValueWrapper):
    def _get_bases(self) -> ValueSet:
        ...
    def _get_wrapped_value(self) -> LazyValueWrapper:
        ...
    def get_filters(self, origin_scope=None) -> Iterable:
        ...

class ReversedObject(AttributeOverwrite):
    def __init__(self, reversed_obj: Any, iter_list: Iterable):
        ...
    def py__iter__(self, contextualized_node=None) -> Iterable:
        ...
    def _next(self, arguments: Any) -> ValueSet:
        ...

class StaticMethodObject(ValueWrapper):
    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...

class ClassMethodObject(ValueWrapper):
    def __init__(self, class_method_obj: Any, function: Any):
        ...
    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...

class ClassMethodGet(ValueWrapper):
    def __init__(self, get_method: Any, klass: Any, function: Any):
        ...
    def get_signatures(self) -> List[AbstractSignature]:
        ...
    def py__call__(self, arguments: Any) -> ValueSet:
        ...

class ClassMethodArguments(TreeArgumentsWrapper):
    def __init__(self, klass: Any, arguments: Any):
        ...
    def unpack(self, func=None) -> Iterable:
        ...

class PropertyObject(AttributeOverwrite, ValueWrapper):
    api_type: str = 'property'
    def __init__(self, property_obj: Any, function: Any):
        ...
    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...
    def _return_self(self, arguments: Any) -> ValueSet:
        ...

class PartialObject(ValueWrapper):
    def __init__(self, actual_value: Any, arguments: Any, instance: Any = None):
        ...
    def _get_functions(self, unpacked_arguments: Iterable) -> Optional[ValueSet]:
        ...
    def get_signatures(self) -> List[AbstractSignature]:
        ...
    def py__call__(self, arguments: Any) -> ValueSet:
        ...
    def py__doc__(self) -> str:
        ...
    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...

class PartialMethodObject(PartialObject):
    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...

class PartialSignature(SignatureWrapper):
    def __init__(self, wrapped_signature: AbstractSignature, skipped_arg_count: int, skipped_arg_set: Set[str]):
        ...
    def get_param_names(self, resolve_stars: bool = False) -> List[str]:
        ...

class MergedPartialArguments(AbstractArguments):
    def __init__(self, partial_arguments: Any, call_arguments: Any, instance: Any = None):
        ...
    def unpack(self, funcdef=None) -> Iterable:
        ...

class DataclassWrapper(ValueWrapper, ClassMixin):
    def get_signatures(self) -> List[AbstractSignature]:
        ...

class DataclassSignature(AbstractSignature):
    def __init__(self, value: Any, param_names: List[Any]):
        ...
    def get_param_names(self, resolve_stars: bool = False) -> List[Any]:
        ...

class DataclassParamName(BaseTreeParamName):
    def __init__(self, parent_context: Any, tree_name: Any, annotation_node: Any, default_node: Any):
        ...
    def get_kind(self) -> str:
        ...
    def infer(self) -> ValueSet:
        ...

class ItemGetterCallable(ValueWrapper):
    def __init__(self, instance: Any, args_value_set: ValueSet):
        ...
    def py__call__(self, item_value_set: ValueSet) -> ValueSet:
        ...

class WrapsCallable(ValueWrapper):
    def py__call__(self, funcs: ValueSet) -> ValueSet:
        ...

class Wrapped(ValueWrapper, FunctionMixin):
    def __init__(self, func: Any, original_function: Any):
        ...
    @property
    def name(self) -> Any:
        ...
    def get_signature_functions(self) -> List[Any]:
        ...

def execute(callback: Callable) -> Callable:
    ...

def _follow_param(inference_state: Any, arguments: Any, index: int) -> ValueSet:
    ...

def argument_clinic(clinic_string: str, want_value: bool = False, want_context: bool = False, want_arguments: bool = False, want_inference_state: bool = False, want_callback: bool = False) -> Callable:
    ...

@argument_clinic('iterator[, default], /', want_inference_state=True)
def builtins_next(iterators: Iterable, defaults: Iterable, inference_state: Any) -> ValueSet:
    ...

@argument_clinic('iterator[, default], /')
def builtins_iter(iterators_or_callables: Iterable, defaults: Iterable) -> ValueSet:
    ...

@argument_clinic('object, name[, default], /')
def builtins_getattr(objects: Iterable, names: Iterable, defaults: Optional[Any] = None) -> ValueSet:
    ...

@argument_clinic('object[, bases, dict], /')
def builtins_type(objects: Iterable, bases: Iterable, dicts: Iterable) -> ValueSet:
    ...

@argument_clinic('[type[, value]], /', want_context=True)
def builtins_super(types: Iterable, objects: Iterable, context: Any) -> ValueSet:
    ...

@argument_clinic('sequence, /', want_value=True, want_arguments=True)
def builtins_reversed(sequences: Iterable, value: Any, arguments: Any) -> ValueSet:
    ...

@argument_clinic('value, type, /', want_arguments=True, want_inference_state=True)
def builtins_isinstance(objects: Iterable, types: Iterable, arguments: Any, inference_state: Any) -> ValueSet:
    ...

@argument_clinic('sequence, /')
def builtins_staticmethod(functions: Iterable) -> ValueSet:
    ...

@argument_clinic('sequence, /', want_value=True, want_arguments=True)
def builtins_classmethod(functions: Iterable, value: Any, arguments: Any) -> ValueSet:
    ...

@argument_clinic('func, /', want_callback=True)
def builtins_property(functions: Iterable, callback: Callable) -> ValueSet:
    ...

def collections_namedtuple(value: Any, arguments: Any, callback: Callable) -> ValueSet:
    ...

def functools_partial(value: Any, arguments: Any, callback: Callable) -> ValueSet:
    ...

def functools_partialmethod(value: Any, arguments: Any, callback: Callable) -> ValueSet:
    ...

@argument_clinic('first, /')
def _return_first_param(firsts: Iterable) -> ValueSet:
    ...

@argument_clinic('seq')
def _random_choice(sequences: Iterable) -> ValueSet:
    ...

def _dataclass(value: Any, arguments: Any, callback: Callable) -> ValueSet:
    ...

def _functools_wraps(funcs: Iterable) -> ValueSet:
    ...

@argument_clinic('*args, /', want_callback=True)
def _operator_itemgetter(args_set: Iterable, callback: Callable) -> ValueSet:
    ...

def _create_string_input_function(func: Callable) -> Callable:
    ...

@argument_clinic('*args, /', want_callback=True)
def _os_path_join(args_set: Iterable, callback: Callable) -> ValueSet:
    ...

_implemented: Dict[str, Dict[str, Callable]] = {
    'builtins': {
        'getattr': builtins_getattr,
        'type': builtins_type,
        'super': builtins_super,
        'reversed': builtins_reversed,
        'isinstance': builtins_isinstance,
        'next': builtins_next,
        'iter': builtins_iter,
        'staticmethod': builtins_staticmethod,
        'classmethod': builtins_classmethod,
        'property': builtins_property,
    },
    'copy': {'copy': _return_first_param, 'deepcopy': _return_first_param},
    'json': {'load': Any, 'loads': Any},
    'collections': {'namedtuple': collections_namedtuple},
    'functools': {
        'partial': functools_partial,
        'partialmethod': functools_partialmethod,
        'wraps': _functools_wraps,
    },
    '_weakref': {'proxy': _return_first_param},
    'random': {'choice': _random_choice},
    'operator': {'itemgetter': _operator_itemgetter},
    'abc': {'abstractmethod': _return_first_param},
    'typing': {'_alias': Any, 'runtime_checkable': Any},
    'dataclasses': {'dataclass': _dataclass},
    'os.path': {
        'dirname': _create_string_input_function(os.path.dirname),
        'abspath': _create_string_input_function(os.path.abspath),
        'relpath': _create_string_input_function(os.path.relpath),
        'join': _os_path_join,
    },
}

def get_metaclass_filters(func: Callable) -> Callable:
    ...

class EnumInstance(LazyValueWrapper):
    def __init__(self, cls: Any, name: Any):
        ...
    @safe_property
    def name(self) -> ValueName:
        ...
    def _get_wrapped_value(self) -> Any:
        ...
    def get_filters(self, origin_scope=None) -> Iterable:
        ...

def tree_name_to_values(func: Callable) -> Callable:
    ...