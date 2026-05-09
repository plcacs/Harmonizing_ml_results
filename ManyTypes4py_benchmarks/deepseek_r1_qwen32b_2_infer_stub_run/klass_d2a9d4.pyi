"""
Stub file for klass_d2a9d4 module
"""

from typing import Any, Callable, Iterable, Iterator, List, Optional, Set, Tuple, Union, Any
from jedi.inference.base_value import ValueSet
from jedi.inference.context import NodeContext, Scope
from jedi.inference.value.function import FunctionAndClassBase
from jedi.inference.gradual.base import GenericClass
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value import TreeInstance
from jedi.inference.gradual.typing import TypedDict, TypedDictClass
from jedi.inference.value import LazyKnownValues, LazyTreeValue
from jedi.inference.names import TreeNameDefinition, ValueName
from jedi.inference.base_value import NO_VALUES
from jedi.inference.gradual.annotation import find_unknown_type_vars
from jedi.inference.signature import BoundSignature, Signature

class ClassName(TreeNameDefinition):
    def infer(self) -> Iterator[ValueSet]:
        ...
    
    @property
    def api_type(self) -> str:
        ...

class ClassFilter:
    def __init__(self, class_value: Any, node_context: Optional[NodeContext] = None, until_position: Optional[int] = None, origin_scope: Optional[Scope] = None, is_instance: bool = False) -> None:
        ...
    
    def _convert_names(self, names: Iterable[Any]) -> List[ClassName]:
        ...
    
    def _equals_origin_scope(self) -> bool:
        ...
    
    def _access_possible(self, name: Any) -> bool:
        ...
    
    def _filter(self, names: Iterable[Any]) -> List[ClassName]:
        ...

class ClassMixin:
    def is_class(self) -> bool:
        ...
    
    def is_class_mixin(self) -> bool:
        ...
    
    def py__call__(self, arguments: Any) -> ValueSet:
        ...
    
    def py__class__(self) -> Any:
        ...
    
    @property
    def name(self) -> ValueName:
        ...
    
    def py__name__(self) -> str:
        ...
    
    def py__mro__(self) -> Iterator[Any]:
        ...
    
    def get_filters(self, origin_scope: Optional[Scope] = None, is_instance: bool = False, include_metaclasses: bool = True, include_type_when_class: bool = True) -> Iterator[Any]:
        ...
    
    def get_signatures(self) -> List[Any]:
        ...
    
    def _as_context(self) -> ClassContext:
        ...
    
    def get_type_hint(self, add_class_info: bool = True) -> str:
        ...
    
    def is_typeddict(self) -> bool:
        ...
    
    def py__getitem__(self, index_value_set: Any, contextualized_node: Any) -> ValueSet:
        ...
    
    def with_generics(self, generics_tuple: Any) -> GenericClass:
        ...
    
    def define_generics(self, type_var_dict: Any) -> ValueSet:
        ...

class ClassValue(ClassMixin, FunctionAndClassBase, metaclass=CachedMetaClass):
    api_type: str = ...
    
    def list_type_vars(self) -> List[ValueSet]:
        ...
    
    def _get_bases_arguments(self) -> Optional[Any]:
        ...
    
    def py__bases__(self) -> List[LazyKnownValues]:
        ...
    
    def get_metaclass_filters(self, metaclasses: Any, is_instance: bool) -> Iterator[Any]:
        ...
    
    def get_metaclasses(self) -> ValueSet:
        ...
    
    def get_metaclass_signatures(self, metaclasses: Any) -> List[Any]:
        ...