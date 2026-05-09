from jedi.inference.base_value import ValueSet, NO_VALUES, Value, LazyValueWrapper, ValueWrapper
from jedi.inference.gradual.typing import AnyClass
from jedi.inference.gradual.generics import TupleGenericManager
from typing import Dict, List, Tuple, Optional, Union

class _BoundTypeVarName:
    def __init__(self, type_var: Any, value_set: ValueSet):
        ...

    def infer(self) -> ValueSet:
        ...

    def py__name__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

class _TypeVarFilter:
    def __init__(self, generics: List[ValueSet], type_vars: List[Any]):
        ...

    def get(self, name: str) -> List[AbstractNameDefinition]:
        ...

    def values(self) -> List:
        ...

class _AnnotatedClassContext:
    def get_filters(self, *args, **kwargs) -> List[ClassFilter]:
        ...

class DefineGenericBaseClass:
    def __init__(self, generics_manager: TupleGenericManager):
        ...

    def get_generics(self) -> Tuple[ValueSet, ...]:
        ...

    def define_generics(self, type_var_dict: Dict[str, ValueSet]) -> ValueSet:
        ...

    def is_same_class(self, other: DefineGenericBaseClass) -> bool:
        ...

    def get_signatures(self) -> List[Signature]:
        ...

    def __repr__(self) -> str:
        ...

class GenericClass(DefineGenericBaseClass):
    def __init__(self, class_value: Any, generics_manager: TupleGenericManager):
        ...

    def get_type_hint(self, add_class_info: bool = True) -> str:
        ...

    def get_type_var_filter(self) -> _TypeVarFilter:
        ...

    def py__call__(self, arguments: Any) -> ValueSet[_GenericInstanceWrapper]:
        ...

    def _as_context(self) -> _AnnotatedClassContext:
        ...

    def py__bases__(self) -> List[_LazyGenericBaseClass]:
        ...

    def is_sub_class_of(self, class_value: Any) -> bool:
        ...

    def with_generics(self, generics_tuple: Tuple) -> Any:
        ...

class _LazyGenericBaseClass:
    def __init__(self, class_value: Any, lazy_base_class: Any, generics_manager: TupleGenericManager):
        ...

    def infer(self) -> ValueSet:
        ...

    def _remap_type_vars(self, base: Any) -> ValueSet:
        ...

    def __repr__(self) -> str:
        ...

class _GenericInstanceWrapper(ValueWrapper):
    def py__stop_iteration_returns(self) -> ValueSet:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> str:
        ...

class _PseudoTreeNameClass(Value):
    def __init__(self, parent_context: Any, tree_name: Any):
        ...

    def get_filters(self, *args, **kwargs) -> List[ClassFilter]:
        ...

    def py__class__(self) -> Any:
        ...

    @property
    def name(self) -> ValueName:
        ...

    def get_qualified_names(self) -> Tuple[str, ...]:
        ...

    def __repr__(self) -> str:
        ...

class BaseTypingValue(LazyValueWrapper):
    def __init__(self, parent_context: Any, tree_name: Any):
        ...

    @property
    def name(self) -> ValueName:
        ...

    def _get_wrapped_value(self) -> _PseudoTreeNameClass:
        ...

    def get_signatures(self) -> List[Signature]:
        ...

    def __repr__(self) -> str:
        ...

class BaseTypingClassWithGenerics(DefineGenericBaseClass):
    def __init__(self, parent_context: Any, tree_name: Any, generics_manager: TupleGenericManager):
        ...

    def _get_wrapped_value(self) -> _PseudoTreeNameClass:
        ...

    def __repr__(self) -> str:
        ...

class BaseTypingInstance(LazyValueWrapper):
    def __init__(self, parent_context: Any, class_value: Any, tree_name: Any, generics_manager: TupleGenericManager):
        ...

    def py__class__(self) -> Any:
        ...

    def get_annotated_class_object(self) -> Any:
        ...

    def get_qualified_names(self) -> Tuple[str, ...]:
        ...

    @property
    def name(self) -> ValueName:
        ...

    def _get_wrapped_value(self) -> Any:
        ...

    def __repr__(self) -> str:
        ...