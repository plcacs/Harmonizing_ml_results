"""
Stub file for 'typing_fe369a' module.
"""

from jedi.inference.base_value import ValueSet, NO_VALUES, Value, LazyValueWrapper, ValueWrapper
from jedi.inference.compiled import builtin_from_name, create_simple_object
from jedi.inference.gradual.base import BaseTypingValue, BaseTypingClassWithGenerics, BaseTypingInstance
from jedi.inference.gradual.type_var import TypeVarClass
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
from typing import Any, Union, Optional, List, Dict, Set, FrozenSet, Tuple, Protocol, Callable, ClassVar, NewType, Type, Any

class TypingModuleName(NameWrapper):
    def infer(self) -> ValueSet:
        ...
    def _remap(self) -> ValueSet:
        ...

class TypingModuleFilterWrapper(FilterWrapper):
    name_wrapper_class = TypingModuleName

class ProxyWithGenerics(BaseTypingClassWithGenerics):
    def execute_annotation(self) -> ValueSet:
        ...
    def gather_annotation_classes(self) -> ValueSet:
        ...
    def _create_instance_with_generics(self, generics_manager: LazyGenericManager) -> ProxyWithGenerics:
        ...
    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, Any]:
        ...

class ProxyTypingValue(BaseTypingValue):
    index_class = ProxyWithGenerics
    def with_generics(self, generics_tuple: tuple) -> ProxyWithGenerics:
        ...
    def py__getitem__(self, index_value_set: ValueSet, contextualized_node: Any) -> ValueSet:
        ...

class _TypingClassMixin(ClassMixin):
    def py__bases__(self) -> List[LazyKnownValues]:
        ...
    def get_metaclasses(self) -> List[Any]:
        ...
    @property
    def name(self) -> ValueName:
        ...

class TypingClassWithGenerics(ProxyWithGenerics, _TypingClassMixin):
    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, Any]:
        ...
    def _create_instance_with_generics(self, generics_manager: LazyGenericManager) -> TypingClassWithGenerics:
        ...

class ProxyTypingClassValue(ProxyTypingValue, _TypingClassMixin):
    index_class = TypingClassWithGenerics

class TypeAlias(LazyValueWrapper):
    def __init__(self, parent_context: Any, origin_tree_name: Any, actual: str) -> None:
        ...
    @property
    def name(self) -> ValueName:
        ...
    def py__name__(self) -> str:
        ...
    def _get_wrapped_value(self) -> Any:
        ...
    def gather_annotation_classes(self) -> ValueSet:
        ...

class Callable(BaseTypingInstance):
    def py__call__(self, arguments: Any) -> ValueSet:
        ...

class Tuple(BaseTypingInstance):
    def _is_homogenous(self) -> bool:
        ...
    def py__simple_getitem__(self, index: Union[int, Any]) -> ValueSet:
        ...
    def py__iter__(self, contextualized_node: Any) -> ValueSet:
        ...
    def py__getitem__(self, index_value_set: ValueSet, contextualized_node: Any) -> ValueSet:
        ...
    def _get_wrapped_value(self) -> Any:
        ...
    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, Any]:
        ...

class Generic(BaseTypingInstance):
    pass

class Protocol(BaseTypingInstance):
    pass

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
    pass

class TypedDict(LazyValueWrapper):
    def __init__(self, definition_class: TypedDictClass) -> None:
        ...
    @property
    def name(self) -> ValueName:
        ...
    def py__simple_getitem__(self, index: Union[str, Any]) -> ValueSet:
        ...
    def get_key_values(self) -> ValueSet:
        ...
    def _get_wrapped_value(self) -> Any:
        ...