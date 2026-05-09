from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)
from jedi.inference.base_value import ValueSet, Value, LazyValueWrapper
from jedi.inference.compiled import CompiledValueName
from jedi.inference.context import CompForContext
from jedi.inference.value.dynamic_arrays import check_array_additions
from jedi.parser_utils import get_sync_comp_fors


class IterableMixin:
    def py__next__(self, contextualized_node: Optional[Any] = None) -> Any:
        ...

    def py__stop_iteration_returns(self) -> ValueSet:
        ...

    get_safe_value: ClassVar[Callable[[Value, Any], Any]]


class GeneratorBase(LazyAttributeOverwrite, IterableMixin):
    array_type: ClassVar[str]

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

    def get_annotated_class_object(self) -> Any:
        ...


class Generator(GeneratorBase):
    def __init__(self, inference_state: Any, func_execution_context: Any) -> None:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Any:
        ...

    def py__stop_iteration_returns(self) -> ValueSet:
        ...

    def __repr__(self) -> str:
        ...


def comprehension_from_atom(
    inference_state: Any,
    value: Any,
    atom: Any
) -> Union[DictComprehension, SetComprehension, GeneratorComprehension]:
    ...


class ComprehensionMixin:
    @inference_state_method_cache()
    def _get_comp_for_context(self, parent_context: Any, comp_for: Any) -> CompForContext:
        ...

    def _nested(self, comp_fors: List[Any], parent_context: Optional[Any] = None) -> Iterable[Any]:
        ...

    @inference_state_method_cache(default=[])
    @to_list
    def _iterate(self) -> List[Any]:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterable[LazyKnownValues]:
        ...

    def __repr__(self) -> str:
        ...


class _DictMixin:
    def _get_generics(self) -> Tuple[Any, ...]:
        ...


class Sequence(LazyAttributeOverwrite, IterableMixin):
    api_type: ClassVar[str]

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

    def py__getitem__(self, index_value_set: Any, contextualized_node: Any) -> Any:
        ...


class _BaseComprehension(ComprehensionMixin):
    def __init__(self, inference_state: Any, defining_context: Any, sync_comp_for_node: Any, entry_node: Any) -> None:
        ...


class ListComprehension(_BaseComprehension, Sequence):
    array_type: ClassVar[str] = 'list'

    def py__simple_getitem__(self, index: Any) -> Any:
        ...


class SetComprehension(_BaseComprehension, Sequence):
    array_type: ClassVar[str] = 'set'


class GeneratorComprehension(_BaseComprehension, GeneratorBase):
    ...


class _DictKeyMixin:
    def get_mapping_item_values(self) -> Tuple[Any, Any]:
        ...

    def get_key_values(self) -> Any:
        ...


class DictComprehension(ComprehensionMixin, Sequence, _DictKeyMixin):
    array_type: ClassVar[str] = 'dict'

    def __init__(self, inference_state: Any, defining_context: Any, sync_comp_for_node: Any, key_node: Any, value_node: Any) -> None:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterable[LazyKnownValues]:
        ...

    def py__simple_getitem__(self, index: Any) -> Any:
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

    def exact_key_items(self) -> List[Any]:
        ...


class SequenceLiteralValue(Sequence):
    _TUPLE_LIKE: ClassVar[Tuple[str, ...]] = ('testlist_star_expr', 'testlist', 'subscriptlist')
    mapping: ClassVar[Dict[str, str]] = {'(': 'tuple', '[': 'list', '{': 'set'}

    def __init__(self, inference_state: Any, defining_context: Any, atom: Any) -> None:
        ...

    def _get_generics(self) -> Tuple[Any, ...]:
        ...

    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterable[LazyTreeValue]:
        ...

    def py__len__(self) -> int:
        ...

    def get_tree_entries(self) -> List[Any]:
        ...

    def __repr__(self) -> str:
        ...


class DictLiteralValue(_DictMixin, SequenceLiteralValue, _DictKeyMixin):
    array_type: ClassVar[str] = 'dict'

    def __init__(self, inference_state: Any, defining_context: Any, atom: Any) -> None:
        ...

    def py__simple_getitem__(self, index: Any) -> Any:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterable[LazyKnownValues]:
        ...

    @publish_method('values')
    def _imitate_values(self, arguments: Any) -> ValueSet:
        ...

    @publish_method('items')
    def _imitate_items(self, arguments: Any) -> ValueSet:
        ...

    def exact_key_items(self) -> Iterable[Tuple[Any, LazyTreeValue]]:
        ...

    def _dict_values(self) -> ValueSet:
        ...

    def _dict_keys(self) -> ValueSet:
        ...


class _FakeSequence(Sequence):
    def __init__(self, inference_state: Any, lazy_value_list: List[Any]) -> None:
        ...

    def py__simple_getitem__(self, index: Any) -> Any:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterable[Any]:
        ...

    def py__bool__(self) -> bool:
        ...

    def __repr__(self) -> str:
        ...


class FakeTuple(_FakeSequence):
    array_type: ClassVar[str] = 'tuple'


class FakeList(_FakeSequence):
    array_type: ClassVar[str] = 'tuple'


class FakeDict(_DictMixin, Sequence, _DictKeyMixin):
    array_type: ClassVar[str] = 'dict'

    def __init__(self, inference_state: Any, dct: Dict[Any, Any]) -> None:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterable[LazyKnownValue]:
        ...

    def py__simple_getitem__(self, index: Any) -> Any:
        ...

    @publish_method('values')
    def _values(self, arguments: Any) -> ValueSet:
        ...

    def _dict_values(self) -> ValueSet:
        ...

    def _dict_keys(self) -> ValueSet:
        ...

    def exact_key_items(self) -> Iterable[Tuple[Any, Any]]:
        ...

    def __repr__(self) -> str:
        ...


class MergedArray(Sequence):
    def __init__(self, inference_state: Any, arrays: List[Any]) -> None:
        ...

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterable[Any]:
        ...

    def py__simple_getitem__(self, index: Any) -> ValueSet:
        ...


def unpack_tuple_to_dict(context: Any, types: Any, exprlist: Any) -> Dict[Any, Any]:
    ...


class Slice(LazyValueWrapper):
    def __init__(self, python_context: Any, start: Any, stop: Any, step: Any) -> None:
        ...

    def _get_wrapped_value(self) -> Any:
        ...

    def get_safe_value(self, default: Any = sentinel) -> slice:
        ...