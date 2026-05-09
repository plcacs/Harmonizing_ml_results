from jedi.inference.base_value import (
    ValueSet,
    Value,
    LazyValueWrapper,
    ValueWrapper,
    NO_VALUES,
)
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager
from jedi.inference.gradual.typing import AnyClass
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    Iterator,
    Iterable,
    Type,
    Callable,
    Generic,
    TypeVar,
)

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
    def __init__(self, generics: Any, type_vars: Any):
        ...

    def get(self, name: str) -> List[Union[_BoundTypeVarName, str]]:
        ...

    def values(self) -> List[Any]:
        ...

class _AnnotatedClassContext(ClassContext):
    def get_filters(self, *args: Any, **kwargs: Any) -> Generator[Any, None, None]:
        ...

class DefineGenericBaseClass(LazyValueWrapper):
    def __init__(self, generics_manager: TupleGenericManager):
        ...

    def _create_instance_with_generics(self, generics_manager: TupleGenericManager) -> Any:
        ...

    @inference_state_method_cache()
    def get_generics(self) -> Tuple[ValueSet, ...]:
        ...

    def define_generics(self, type_var_dict: Dict[str, ValueSet]) -> ValueSet:
        ...

    def is_same_class(self, other: Any) -> bool:
        ...

    def get_signatures(self) -> List[Any]:
        ...

    def __repr__(self) -> str:
        ...

    def py__call__(self, arguments: Any) -> ValueSet:
        ...

    def _as_context(self) -> _AnnotatedClassContext:
        ...

    @to_list
    def py__bases__(self) -> Iterator[Any]:
        ...

    def with_generics(self, generics_tuple: Any) -> Any:
        ...

    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, ValueSet]:
        ...

class GenericClass(DefineGenericBaseClass, ClassMixin):
    def __init__(self, class_value: Any, generics_manager: TupleGenericManager):
        ...

    def _get_wrapped_value(self) -> Any:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> str:
        ...

    def get_type_var_filter(self) -> _TypeVarFilter:
        ...

    def py__call__(self, arguments: Any) -> ValueSet:
        ...

    def _as_context(self) -> _AnnotatedClassContext:
        ...

    def _create_instance_with_generics(self, generics_manager: TupleGenericManager) -> GenericClass:
        ...

    def is_sub_class_of(self, class_value: Any) -> bool:
        ...

    def with_generics(self, generics_tuple: Any) -> Any:
        ...

    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, ValueSet]:
        ...

class _LazyGenericBaseClass:
    def __init__(self, class_value: Any, lazy_base_class: Any, generics_manager: TupleGenericManager):
        ...

    @iterator_to_value_set
    def infer(self) -> Iterator[ValueSet]:
        ...

    def _remap_type_vars(self, base: Any) -> Iterator[ValueSet]:
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

    @property
    def tree_node(self) -> Any:
        ...

    def get_filters(self, *args: Any, **kwargs: Any) -> Generator[Any, None, None]:
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

    def get_signatures(self) -> List[Any]:
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