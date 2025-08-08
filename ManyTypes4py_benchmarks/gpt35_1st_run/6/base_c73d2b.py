from typing import List, Dict, Set, Tuple, Type

class _BoundTypeVarName(AbstractNameDefinition):
    def __init__(self, type_var: TypeVar, value_set: ValueSet) -> None:
    def infer(self) -> ValueSet:
    def py__name__(self) -> str:
    def __repr__(self) -> str:

class _TypeVarFilter:
    def __init__(self, generics: TupleGenericManager, type_vars: List[TypeVar]) -> None:
    def get(self, name: str) -> List[AbstractNameDefinition]:
    def values(self) -> List:

class _AnnotatedClassContext(ClassContext):
    def get_filters(self, *args, **kwargs) -> None:

class DefineGenericBaseClass(LazyValueWrapper):
    def __init__(self, generics_manager: TupleGenericManager) -> None:
    def _create_instance_with_generics(self, generics_manager: TupleGenericManager) -> None:
    def get_generics(self) -> TupleGenericManager:
    def define_generics(self, type_var_dict: dict) -> ValueSet:
    def is_same_class(self, other) -> bool:
    def get_signatures(self) -> List:
    def __repr__(self) -> str:

class GenericClass(DefineGenericBaseClass, ClassMixin):
    def __init__(self, class_value: Value, generics_manager: TupleGenericManager) -> None:
    def _get_wrapped_value(self) -> Value:
    def get_type_hint(self, add_class_info: bool = True) -> str:
    def get_type_var_filter(self) -> _TypeVarFilter:
    def py__call__(self, arguments: List) -> ValueSet:
    def _as_context(self) -> _AnnotatedClassContext:
    def py__bases__(self) -> List:
    def _create_instance_with_generics(self, generics_manager: TupleGenericManager) -> GenericClass:
    def is_sub_class_of(self, class_value: Value) -> bool:
    def with_generics(self, generics_tuple: Tuple) -> Value:
    def infer_type_vars(self, value_set: ValueSet) -> dict:

class _LazyGenericBaseClass:
    def __init__(self, class_value: Value, lazy_base_class: Value, generics_manager: TupleGenericManager) -> None:
    def infer(self) -> ValueSet:
    def _remap_type_vars(self, base: GenericClass) -> ValueSet:
    def __repr__(self) -> str:

class _GenericInstanceWrapper(ValueWrapper):
    def py__stop_iteration_returns(self) -> ValueSet:
    def get_type_hint(self, add_class_info: bool = True) -> str:

class _PseudoTreeNameClass(Value):
    def __init__(self, parent_context: ClassContext, tree_name: Value) -> None:
    def get_filters(self, *args, **kwargs) -> None:
    def py__class__(self) -> Value:
    @property
    def name(self) -> ValueName:
    def get_qualified_names(self) -> Tuple:
    def __repr__(self) -> str:

class BaseTypingValue(LazyValueWrapper):
    def __init__(self, parent_context: ClassContext, tree_name: Value) -> None:
    @property
    def name(self) -> ValueName:
    def _get_wrapped_value(self) -> _PseudoTreeNameClass:
    def get_signatures(self) -> List:
    def __repr__(self) -> str:

class BaseTypingClassWithGenerics(DefineGenericBaseClass):
    def __init__(self, parent_context: ClassContext, tree_name: Value, generics_manager: TupleGenericManager) -> None:
    def _get_wrapped_value(self) -> _PseudoTreeNameClass:
    def __repr__(self) -> str:

class BaseTypingInstance(LazyValueWrapper):
    def __init__(self, parent_context: ClassContext, class_value: Value, tree_name: Value, generics_manager: TupleGenericManager) -> None:
    def py__class__(self) -> Value:
    def get_annotated_class_object(self) -> Value:
    def get_qualified_names(self) -> Tuple:
    @property
    def name(self) -> ValueName:
    def _get_wrapped_value(self) -> Value:
    def __repr__(self) -> str:
