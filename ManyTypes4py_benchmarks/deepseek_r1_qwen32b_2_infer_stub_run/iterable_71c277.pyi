"""
Stub file for 'iterable_71c277' module.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Iterable,
    Iterator,
    Type,
    overload,
)
from jedi.inference.base_value import (
    ValueSet,
    Value,
    NO_VALUES,
    LazyValueWrapper,
    LazyKnownValue,
    LazyKnownValues,
)
from jedi.inference.compiled import CompiledValueName
from jedi.inference.context import CompForContext
from jedi.inference.gradual.generics import GenericClass
from jedi.inference.value.dynamic_arrays import FakeList, FakeTuple
from jedi.inference.value import LazyAttributeOverwrite
from jedi.inference.value.dynamic_arrays import FakeDict
from jedi.inference.value import Slice

class IterableMixin:
    def py__next__(self, contextualized_node: Optional[Any] = None) -> ValueSet:
        ...
    def py__stop_iteration_returns(self) -> ValueSet:
        ...
    @property
    def get_safe_value(self) -> Value:
        ...

class GeneratorBase(IterableMixin, LazyAttributeOverwrite):
    array_type: str
    def _get_wrapped_value(self) -> Any:
        ...
    def _get_cls(self) -> Any:
        ...
    def py__bool__(self) -> bool:
        ...
    @publish_method('__iter__')
    def _iter(self, arguments: Any) -> ValueSet:
        ...
    @publish_method('send')
    @publish_method('__next__')
    def _next(self, arguments: Any) -> ValueSet:
        ...
    @property
    def name(self) -> CompiledValueName:
        ...
    def get_annotated_class_object(self) -> GenericClass:
        ...

class Generator(GeneratorBase):
    def __init__(self, inference_state: Any, func_execution_context: Any) -> None:
        ...
    def py__iter__(self, contextualized_node: Optional[Any] = None) -> ValueSet:
        ...
    def py__stop_iteration_returns(self) -> ValueSet:
        ...
    def __repr__(self) -> str:
        ...

def comprehension_from_atom(
    inference_state: Any,
    value: Any,
    atom: Any
) -> Union[DictComprehension, SetComprehension, ListComprehension, GeneratorComprehension]:
    ...

class ComprehensionMixin:
    @inference_state_method_cache()
    def _get_comp_for_context(self, parent_context: Any, comp_for: Any) -> CompForContext:
        ...
    @inference_state_method_cache(default=[])
    @to_list
    def _iterate(self) -> List[ValueSet]:
        ...
    def py__iter__(self, contextualized_node: Optional[Any] = None) -> ValueSet:
        ...
    def __repr__(self) -> str:
        ...

class _DictMixin:
    def _get_generics(self) -> Tuple[Any, ...]:
        ...

class Sequence(LazyAttributeOverwrite, IterableMixin):
    api_type: str
    @property
    def name(self) -> CompiledValueName:
        ...
    def _get_generics(self) -> Tuple[Any, ...]:
        ...
    @inference_state_method_cache(default=())
    def _cached_generics(self) -> Tuple[Any, ...]:
        ...
    def _get_wrapped_value(self) -> Any:
        ...
    def py__bool__(self) -> Optional[bool]:
        ...
    @safe_property
    def parent(self) -> Any:
        ...
    def py__getitem__(self, index_value_set: ValueSet, contextualized_node: Any) -> ValueSet:
        ...

class _BaseComprehension(ComprehensionMixin):
    def __init__(self, inference_state: Any, defining_context: Any, sync_comp_for_node: Any, entry_node: Any) -> None:
        ...

class ListComprehension(_BaseComprehension, Sequence):
    array_type: str
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...

class SetComprehension(_BaseComprehension, Sequence):
    array_type: str

class GeneratorComprehension(_BaseComprehension, GeneratorBase):
    ...

class _DictKeyMixin:
    def get_mapping_item_values(self) -> Tuple[ValueSet, ValueSet]:
        ...
    def get_key_values(self) -> ValueSet:
        ...

class DictComprehension(ComprehensionMixin, Sequence, _DictKeyMixin):
    array_type: str
    def __init__(self, inference_state: Any, defining_context: Any, sync_comp_for_node: Any, key_node: Any, value_node: Any) -> None:
        ...
    def py__iter__(self, contextualized_node: Optional[Any] = None) -> ValueSet:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    def _dict_keys(self) -> ValueSet:
        ...
    def _dict_values(self) -> ValueSet:
        ...
    @publish_method('values')
    def _imitate_values(self, arguments: Any) -> ValueSet:
        ...
    @publish_method('items')
    def _imitate_items(self, arguments: Any) -> ValueSet:
        ...
    def exact_key_items(self) -> List[Tuple[Any, Any]]:
        ...

class SequenceLiteralValue(Sequence):
    _TUPLE_LIKE: Tuple[str, ...]
    mapping: Dict[str, str]
    def __init__(self, inference_state: Any, defining_context: Any, atom: Any) -> None:
        ...
    def _get_generics(self) -> Tuple[Any, ...]:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    def py__iter__(self, contextualized_node: Optional[Any] = None) -> ValueSet:
        ...
    def py__len__(self) -> int:
        ...
    def get_tree_entries(self) -> List[Any]:
        ...
    def __repr__(self) -> str:
        ...

class DictLiteralValue(_DictMixin, SequenceLiteralValue, _DictKeyMixin):
    array_type: str
    def __init__(self, inference_state: Any, defining_context: Any, atom: Any) -> None:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    def py__iter__(self, contextualized_node: Optional[Any] = None) -> ValueSet:
        ...
    @publish_method('values')
    def _imitate_values(self, arguments: Any) -> ValueSet:
        ...
    @publish_method('items')
    def _imitate_items(self, arguments: Any) -> ValueSet:
        ...
    def exact_key_items(self) -> List[Tuple[Any, Any]]:
        ...
    def _dict_values(self) -> ValueSet:
        ...
    def _dict_keys(self) -> ValueSet:
        ...

class _FakeSequence(Sequence):
    def __init__(self, inference_state: Any, lazy_value_list: List[LazyKnownValue]) -> None:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    def py__iter__(self, contextualized_node: Optional[Any] = None) -> ValueSet:
        ...
    def py__bool__(self) -> bool:
        ...
    def __repr__(self) -> str:
        ...

class FakeTuple(_FakeSequence):
    array_type: str

class FakeList(_FakeSequence):
    array_type: str

class FakeDict(_DictMixin, Sequence, _DictKeyMixin):
    array_type: str
    def __init__(self, inference_state: Any, dct: Dict[Any, Any]) -> None:
        ...
    def py__iter__(self, contextualized_node: Optional[Any] = None) -> ValueSet:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    @publish_method('values')
    def _values(self, arguments: Any) -> ValueSet:
        ...
    def _dict_values(self) -> ValueSet:
        ...
    def _dict_keys(self) -> ValueSet:
        ...
    def exact_key_items(self) -> List[Tuple[Any, Any]]:
        ...
    def __repr__(self) -> str:
        ...

class MergedArray(Sequence):
    def __init__(self, inference_state: Any, arrays: List[Sequence]) -> None:
        ...
    def py__iter__(self, contextualized_node: Optional[Any] = None) -> ValueSet:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...

def unpack_tuple_to_dict(context: Any, types: ValueSet, exprlist: Any) -> Dict[str, ValueSet]:
    ...

class Slice(LazyValueWrapper):
    def __init__(self, python_context: Any, start: Any, stop: Any, step: Any) -> None:
        ...
    def _get_wrapped_value(self) -> Any:
        ...
    def get_safe_value(self, default: Any = sentinel) -> slice:
        ...