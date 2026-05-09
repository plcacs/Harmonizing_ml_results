"""
Stub file for 'typing_fe369a.py' module.
"""

from jedi.inference.base_value import Value, ValueSet
from jedi.inference.gradual.base import BaseTypingValue, BaseTypingClassWithGenerics, BaseTypingInstance
from jedi.inference.names import NameWrapper, FilterWrapper
from jedi.inference.value.klass import ClassMixin
from jedi.inference.compiled import builtin_from_name
from typing import Any, Generator, List, Optional, Dict, Union

class TypingModuleName(NameWrapper):
    def infer(self) -> ValueSet:
        ...
    
    def _remap(self) -> Generator[Value, None, None]:
        ...

class TypingModuleFilterWrapper(FilterWrapper):
    name_wrapper_class: type[TypingModuleName]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

class ProxyWithGenerics(BaseTypingClassWithGenerics):
    def execute_annotation(self) -> ValueSet:
        ...
    
    def gather_annotation_classes(self) -> ValueSet:
        ...
    
    def _create_instance_with_generics(self, generics_manager: Any) -> BaseTypingClassWithGenerics:
        ...

class ProxyTypingValue(BaseTypingValue):
    index_class: type[ProxyWithGenerics]
    def with_generics(self, generics_tuple: Any) -> BaseTypingValue:
        ...
    
    def py__getitem__(self, index_value_set: ValueSet, contextualized_node: Any) -> ValueSet:
        ...

class _TypingClassMixin(ClassMixin):
    def py__bases__(self) -> List[Value]:
        ...
    
    def get_metaclasses(self) -> List[ValueSet]:
        ...
    
    @property
    def name(self) -> ValueName:
        ...

class TypingClassWithGenerics(ProxyWithGenerics, _TypingClassMixin):
    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, Any]:
        ...
    
    def _create_instance_with_generics(self, generics_manager: Any) -> BaseTypingClassWithGenerics:
        ...

class ProxyTypingClassValue(ProxyTypingValue, _TypingClassMixin):
    index_class: type[TypingClassWithGenerics]
    ...

class TypeAlias(LazyValueWrapper):
    def __init__(self, parent_context: Any, origin_tree_name: Any, actual: str) -> None:
        ...
    
    def _get_wrapped_value(self) -> Value:
        ...
    
    def gather_annotation_classes(self) -> ValueSet:
        ...
    
    def get_signatures(self) -> List[Any]:
        ...

class Callable(BaseTypingInstance):
    def py__call__(self, arguments: Any) -> ValueSet:
        ...
    
    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...

class Tuple(BaseTypingInstance):
    def _is_homogenous(self) -> bool:
        ...
    
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    
    def py__iter__(self, contextualized_node: Optional[Any]) -> Generator[ValueSet, None, None]:
        ...
    
    def py__getitem__(self, index_value_set: ValueSet, contextualized_node: Any) -> ValueSet:
        ...
    
    def _get_wrapped_value(self) -> Value:
        ...
    
    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, Any]:
        ...

class Generic(BaseTypingInstance):
    ...

class Protocol(BaseTypingInstance):
    ...

class AnyClass(BaseTypingValue):
    def execute_annotation(self) -> ValueSet:
        ...

class OverloadFunction(BaseTypingValue):
    @repack_with_argument_clinic('func, /')
    def py__call__(self, func_value_set: ValueSet) -> ValueSet:
        ...

class NewTypeFunction(ValueWrapper):
    def py__call__(self, arguments: Any) -> ValueSet:
        ...

class NewType(Value):
    def __init__(self, inference_state: Any, parent_context: Any, tree_node: Any, type_value_set: ValueSet) -> None:
        ...
    
    def py__class__(self) -> ValueSet:
        ...
    
    def py__call__(self, arguments: Any) -> ValueSet:
        ...
    
    @property
    def name(self) -> ValueName:
        ...

class CastFunction(ValueWrapper):
    @repack_with_argument_clinic('type, object, /')
    def py__call__(self, type_value_set: ValueSet, object_value_set: ValueSet) -> ValueSet:
        ...

class TypedDictClass(BaseTypingValue):
    ...

class TypedDict(LazyValueWrapper):
    def __init__(self, definition_class: TypedDictClass) -> None:
        ...
    
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    
    def get_key_values(self) -> ValueSet:
        ...
    
    def _get_wrapped_value(self) -> Value:
        ...