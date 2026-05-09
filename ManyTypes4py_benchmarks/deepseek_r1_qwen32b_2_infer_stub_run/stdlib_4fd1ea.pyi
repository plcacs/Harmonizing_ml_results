"""
Stub file for 'stdlib_4fd1ea' module
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union, overload
from jedi.inference.base_value import ValueSet, LazyValueWrapper
from jedi.inference import InferenceState
from jedi.inference.signature import Signature
from jedi.inference.filters import Filter

def execute(callback: Callable) -> Callable:
    ...

def _follow_param(inference_state: InferenceState, arguments: Any, index: int) -> ValueSet:
    ...

def argument_clinic(clinic_string: str, want_value: bool = False, want_context: bool = False, want_arguments: bool = False, want_inference_state: bool = False, want_callback: bool = False) -> Callable:
    ...

@argument_clinic('iterator[, default], /', want_inference_state=True)
def builtins_next(iterators: ValueSet, defaults: ValueSet, inference_state: InferenceState) -> ValueSet:
    ...

@argument_clinic('iterator[, default], /')
def builtins_iter(iterators_or_callables: ValueSet, defaults: ValueSet) -> ValueSet:
    ...

@argument_clinic('object, name[, default], /')
def builtins_getattr(objects: ValueSet, names: ValueSet, defaults: Optional[ValueSet] = None) -> ValueSet:
    ...

@argument_clinic('object[, bases, dict], /')
def builtins_type(objects: ValueSet, bases: ValueSet, dicts: ValueSet) -> ValueSet:
    ...

class SuperInstance(LazyValueWrapper):
    def __init__(self, inference_state: InferenceState, instance: Any) -> None:
        ...
    
    def _get_bases(self) -> ValueSet:
        ...
    
    def _get_wrapped_value(self) -> Any:
        ...
    
    def get_filters(self, origin_scope: Any = None) -> Iterator[Filter]:
        ...

@argument_clinic('[type[, value]], /', want_context=True)
def builtins_super(types: ValueSet, objects: ValueSet, context: Any) -> ValueSet:
    ...

class ReversedObject(AttributeOverwrite):
    def __init__(self, reversed_obj: Any, iter_list: List[Any]) -> None:
        ...
    
    def py__iter__(self, contextualized_node: Any = None) -> ValueSet:
        ...
    
    def _next(self, arguments: Any) -> ValueSet:
        ...

@argument_clinic('sequence, /', want_value=True, want_arguments=True)
def builtins_reversed(sequences: ValueSet, value: Any, arguments: Any) -> ValueSet:
    ...

@argument_clinic('value, type, /', want_arguments=True, want_inference_state=True)
def builtins_isinstance(objects: ValueSet, types: ValueSet, arguments: Any, inference_state: InferenceState) -> ValueSet:
    ...

class StaticMethodObject(ValueWrapper):
    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...

@argument_clinic('sequence, /')
def builtins_staticmethod(functions: ValueSet) -> ValueSet:
    ...

class ClassMethodObject(ValueWrapper):
    def __init__(self, class_method_obj: Any, function: Any) -> None:
        ...
    
    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...

class ClassMethodGet(ValueWrapper):
    def __init__(self, get_method: Any, klass: Any, function: Any) -> None:
        ...
    
    def get_signatures(self) -> List[Signature]:
        ...
    
    def py__call__(self, arguments: Any) -> ValueSet:
        ...

class ClassMethodArguments(TreeArgumentsWrapper):
    def __init__(self, klass: Any, arguments: Any) -> None:
        ...
    
    def unpack(self, func: Any = None) -> Iterator[Tuple[Optional[Any], Any]]:
        ...

@argument_clinic('sequence, /', want_value=True, want_arguments=True)
def builtins_classmethod(functions: ValueSet, value: Any, arguments: Any) -> ValueSet:
    ...

class PropertyObject(AttributeOverwrite, ValueWrapper):
    api_type: str = ...
    
    def __init__(self, property_obj: Any, function: Any) -> None:
        ...
    
    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...
    
    def _return_self(self, arguments: Any) -> ValueSet:
        ...

@argument_clinic('func, /', want_callback=True)
def builtins_property(functions: ValueSet, callback: Callable) -> ValueSet:
    ...

def collections_namedtuple(value: Any, arguments: Any, callback: Callable) -> ValueSet:
    ...

class PartialObject(ValueWrapper):
    def __init__(self, actual_value: Any, arguments: Any, instance: Any = None) -> None:
        ...
    
    def _get_functions(self, unpacked_arguments: Iterator[Tuple[Optional[Any], Any]]) -> Optional[ValueSet]:
        ...
    
    def get_signatures(self) -> List[Signature]:
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
    def __init__(self, wrapped_signature: Signature, skipped_arg_count: int, skipped_arg_set: Set[str]) -> None:
        ...
    
    def get_param_names(self, resolve_stars: bool = False) -> List[Any]:
        ...

class MergedPartialArguments(AbstractArguments):
    def __init__(self, partial_arguments: Any, call_arguments: Any, instance: Any = None) -> None:
        ...
    
    def unpack(self, funcdef: Any = None) -> Iterator[Tuple[Optional[Any], Any]]:
        ...

def functools_partial(value: Any, arguments: Any, callback: Callable) -> ValueSet:
    ...

def functools_partialmethod(value: Any, arguments: Any, callback: Callable) -> ValueSet:
    ...

@argument_clinic('first, /')
def _return_first_param(firsts: ValueSet) -> ValueSet:
    ...

@argument_clinic('seq')
def _random_choice(sequences: ValueSet) -> ValueSet:
    ...

def _dataclass(value: Any, arguments: Any, callback: Callable) -> ValueSet:
    ...

class DataclassWrapper(ValueWrapper, ClassMixin):
    def get_signatures(self) -> List[Signature]:
        ...

class DataclassSignature(AbstractSignature):
    def __init__(self, value: Any, param_names: List[Any]) -> None:
        ...
    
    def get_param_names(self, resolve_stars: bool = False) -> List[Any]:
        ...

class DataclassParamName(BaseTreeParamName):
    def __init__(self, parent_context: Any, tree_name: Any, annotation_node: Any, default_node: Any) -> None:
        ...
    
    def get_kind(self) -> int:
        ...
    
    def infer(self) -> ValueSet:
        ...

class ItemGetterCallable(ValueWrapper):
    def __init__(self, instance: Any, args_value_set: ValueSet) -> None:
        ...
    
    def py__call__(self, item_value_set: ValueSet) -> ValueSet:
        ...

@argument_clinic('func, /')
def _functools_wraps(funcs: ValueSet) -> ValueSet:
    ...

class WrapsCallable(ValueWrapper):
    def py__call__(self, funcs: ValueSet) -> ValueSet:
        ...

class Wrapped(ValueWrapper, FunctionMixin):
    def __init__(self, func: Any, original_function: Any) -> None:
        ...
    
    @property
    def name(self) -> Any:
        ...
    
    def get_signature_functions(self) -> List[Any]:
        ...

@argument_clinic('*args, /', want_callback=True)
def _operator_itemgetter(args_set: ValueSet, callback: Callable) -> ValueSet:
    ...

def _create_string_input_function(func: Callable) -> Callable:
    ...

@argument_clinic('*args, /', want_callback=True)
def _os_path_join(args_set: ValueSet, callback: Callable) -> ValueSet:
    ...

_implemented: Dict[str, Dict[str, Callable]] = ...

def get_metaclass_filters(func: Callable) -> Callable:
    ...

class EnumInstance(LazyValueWrapper):
    def __init__(self, cls: Any, name: Any) -> None:
        ...
    
    @property
    def name(self) -> Any:
        ...
    
    def _get_wrapped_value(self) -> Any:
        ...
    
    def get_filters(self, origin_scope: Any = None) -> Iterator[Filter]:
        ...

def tree_name_to_values(func: Callable) -> Callable:
    ...