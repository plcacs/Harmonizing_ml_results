```python
from typing import Any, Iterator, List, Optional, Tuple, Union
from jedi.inference.cache import inference_state_method_cache, inference_state_method_generator_cache
from jedi.inference.arguments import ValuesArguments
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.context import ClassContext
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.lazy_value import LazyTreeValue, LazyKnownValues
from jedi.inference.names import TreeNameDefinition, ValueName
from jedi.inference.value.function import FunctionAndClassBase
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
from jedi.plugins import plugin_manager
from parso.python.tree import NodeOrLeaf

class ClassName(TreeNameDefinition):
    _apply_decorators: bool
    _class_value: Any
    
    def __init__(
        self,
        class_value: Any,
        tree_name: Any,
        name_context: Any,
        apply_decorators: bool
    ) -> None: ...
    
    def infer(self) -> ValueSet: ...
    
    @property
    def api_type(self) -> str: ...

class ClassFilter(ParserTreeFilter):
    _class_value: Any
    _is_instance: bool
    
    def __init__(
        self,
        class_value: Any,
        node_context: Optional[Any] = None,
        until_position: Optional[Any] = None,
        origin_scope: Optional[Any] = None,
        is_instance: bool = False
    ) -> None: ...
    
    def _convert_names(self, names: List[Any]) -> List[ClassName]: ...
    
    def _equals_origin_scope(self) -> bool: ...
    
    def _access_possible(self, name: Any) -> bool: ...
    
    def _filter(self, names: List[Any]) -> List[Any]: ...

class ClassMixin:
    def is_class(self) -> bool: ...
    
    def is_class_mixin(self) -> bool: ...
    
    def py__call__(self, arguments: ValuesArguments) -> ValueSet: ...
    
    def py__class__(self) -> Any: ...
    
    @property
    def name(self) -> ValueName: ...
    
    def py__name__(self) -> str: ...
    
    @inference_state_method_generator_cache()
    def py__mro__(self) -> Iterator[Any]: ...
    
    def get_filters(
        self,
        origin_scope: Optional[Any] = None,
        is_instance: bool = False,
        include_metaclasses: bool = True,
        include_type_when_class: bool = True
    ) -> Iterator[Any]: ...
    
    def get_signatures(self) -> List[Any]: ...
    
    def _as_context(self) -> ClassContext: ...
    
    def get_type_hint(self, add_class_info: bool = True) -> str: ...
    
    @inference_state_method_cache(default=False)
    def is_typeddict(self) -> bool: ...
    
    def py__getitem__(
        self,
        index_value_set: ValueSet,
        contextualized_node: Any
    ) -> ValueSet: ...
    
    def with_generics(self, generics_tuple: Tuple[Any, ...]) -> Any: ...
    
    def define_generics(self, type_var_dict: Any) -> ValueSet: ...

class ClassValue(ClassMixin, FunctionAndClassBase):
    api_type: str = ...
    
    @inference_state_method_cache()
    def list_type_vars(self) -> List[Any]: ...
    
    def _get_bases_arguments(self) -> Optional[Any]: ...
    
    @inference_state_method_cache(default=())
    def py__bases__(self) -> List[Any]: ...
    
    @plugin_manager.decorate()
    def get_metaclass_filters(
        self,
        metaclasses: ValueSet,
        is_instance: bool
    ) -> Iterator[Any]: ...
    
    @inference_state_method_cache(default=NO_VALUES)
    def get_metaclasses(self) -> ValueSet: ...
    
    @plugin_manager.decorate()
    def get_metaclass_signatures(self, metaclasses: ValueSet) -> List[Any]: ...
```