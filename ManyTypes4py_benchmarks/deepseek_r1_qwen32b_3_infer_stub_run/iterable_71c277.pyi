"""
Stub file for 'iterable_71c277' module.
"""

from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    Type,
    overload,
    AnyStr,
    Iterator,
)
from jedi.inference.compiled.value import CompiledValue
from jedi.inference.value import LazyValueWrapper, LazyKnownValue, LazyKnownValues
from jedi.inference.base_value import ValueSet, Value, NO_VALUES
from jedi.inference.context import ContextualizedNode
from jedi.inference.filters import publish_method
from jedi.inference.utils import to_list
from jedi.parser_utils import get_sync_comp_fors

class IterableMixin:
    def py__next__(self, contextualized_node: Optional[ContextualizedNode] = None) -> ValueSet:
        ...
    def py__stop_iteration_returns(self) -> ValueSet:
        ...
    @staticmethod
    def get_safe_value(value: Any) -> Any:
        ...

class GeneratorBase(IterableMixin):
    array_type: str
    def _get_wrapped_value(self) -> CompiledValue:
        ...
    def _get_cls(self) -> CompiledValue:
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
    def name(self) -> CompiledValue:
        ...
    def get_annotated_class_object(self) -> Type[Value]:
        ...

class Generator(GeneratorBase):
    def __init__(self, inference_state: Any, func_execution_context: Any) -> None:
        ...
    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> ValueSet:
        ...
    def py__stop_iteration_returns(self) -> ValueSet:
        ...
    def __repr__(self) -> str:
        ...

def comprehension_from_atom(
    inference_state: Any,
    value: Any,
    atom: Any
) -> Union['DictComprehension', 'SetComprehension', 'GeneratorComprehension']:
    ...

class ComprehensionMixin:
    @to_list
    def _iterate(self) -> List[ValueSet]:
        ...
    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[LazyKnownValues, None, None]:
        ...

class _DictMixin:
    def _get_generics(self) -> Tuple[ValueSet, ...]:
        ...

class Sequence(IterableMixin):
    array_type: str
    def _get_generics(self) -> Tuple[ValueSet, ...]:
        ...
    def _get_wrapped_value(self) -> CompiledValue:
        ...
    def py__bool__(self) -> Optional[bool]:
        ...
    @property
    def parent(self) -> Any:
        ...
    def py__getitem__(self, index_value_set: ValueSet, contextualized_node: ContextualizedNode) -> ValueSet:
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
    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[LazyKnownValues, None, None]:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    def _dict_keys(self) -> ValueSet:
        ...
    def _dict_values(self) -> ValueSet:
        ...
    def _imitate_values(self, arguments: Any) -> ValueSet:
        ...
    def _imitate_items(self, arguments: Any) -> ValueSet:
        ...

class SequenceLiteralValue(Sequence):
    def __init__(self, inference_state: Any, defining_context: Any, atom: Any) -> None:
        ...
    def _get_generics(self) -> Tuple[ValueSet, ...]:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[LazyTreeValue, None, None]:
        ...
    def py__len__(self) -> int:
        ...
    def get_tree_entries(self) -> Union[List[Any], List[Tuple[Any, Any]]]:
        ...

class DictLiteralValue(SequenceLiteralValue, _DictMixin, _DictKeyMixin):
    array_type: str
    def __init__(self, inference_state: Any, defining_context: Any, atom: Any) -> None:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    def _dict_values(self) -> ValueSet:
        ...
    def _dict_keys(self) -> ValueSet:
        ...
    def exact_key_items(self) -> Generator[Tuple[str, LazyTreeValue], None, None]:
        ...

class _FakeSequence(Sequence):
    def __init__(self, inference_state: Any, lazy_value_list: List[LazyKnownValue]) -> None:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Iterator[LazyKnownValue]:
        ...
    def py__bool__(self) -> bool:
        ...

class FakeTuple(_FakeSequence):
    array_type: str

class FakeList(_FakeSequence):
    array_type: str

class FakeDict(Sequence, _DictMixin, _DictKeyMixin):
    def __init__(self, inference_state: Any, dct: Dict[Any, Any]) -> None:
        ...
    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[LazyKnownValue, None, None]:
        ...
    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...
    def _dict_values(self) -> ValueSet:
        ...
    def _dict_keys(self) -> ValueSet:
        ...
    def exact_key_items(self) -> Generator[Tuple[Any, Any], None, None]:
        ...

class MergedArray(Sequence):
    def __init__(self, inference_state: Any, arrays: List[Sequence]) -> None:
        ...
    def py__iter__(self, contextualized_node: Optional[ContextualizedNode] = None) -> Generator[ValueSet, None, None]:
        ...

def unpack_tuple_to_dict(context: Any, types: ValueSet, exprlist: Any) -> Dict[str, ValueSet]:
    ...

class Slice(LazyValueWrapper):
    def __init__(self, python_context: Any, start: Any, stop: Any, step: Any) -> None:
        ...
    def _get_wrapped_value(self) -> CompiledValue:
        ...
    def get_safe_value(self, default: Any = ...) -> slice:
        ...