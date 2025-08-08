from typing import Any, Dict, List, Tuple, Union

class IterableMixin:

    def py__next__(self, contextualized_node=None) -> Any:
        ...

    def py__stop_iteration_returns(self) -> Any:
        ...

    def get_safe_value(self) -> Any:
        ...

class GeneratorBase(LazyAttributeOverwrite, IterableMixin):

    def _get_wrapped_value(self) -> Any:
        ...

    def _get_cls(self) -> Any:
        ...

    def py__bool__(self) -> bool:
        ...

    def _iter(self, arguments) -> Any:
        ...

    def _next(self, arguments) -> Any:
        ...

    def py__stop_iteration_returns(self) -> Any:
        ...

    @property
    def name(self) -> Any:
        ...

    def get_annotated_class_object(self) -> Any:
        ...

class Generator(GeneratorBase):

    def __init__(self, inference_state, func_execution_context) -> None:
        ...

    def py__iter__(self, contextualized_node=None) -> Any:
        ...

    def py__stop_iteration_returns(self) -> Any:
        ...

    def __repr__(self) -> str:
        ...

def comprehension_from_atom(inference_state, value, atom) -> Any:
    ...

class ComprehensionMixin:

    def _get_comp_for_context(self, parent_context, comp_for) -> Any:
        ...

    def _nested(self, comp_fors, parent_context=None) -> Any:
        ...

    def _iterate(self) -> Any:
        ...

    def py__iter__(self, contextualized_node=None) -> Any:
        ...

    def __repr__(self) -> str:
        ...

class _DictMixin:

    def _get_generics(self) -> Tuple:
        ...

class Sequence(LazyAttributeOverwrite, IterableMixin):

    def py__getitem__(self, index_value_set, contextualized_node) -> Any:
        ...

class _BaseComprehension(ComprehensionMixin):

    def __init__(self, inference_state, defining_context, sync_comp_for_node, entry_node) -> None:
        ...

class ListComprehension(_BaseComprehension, Sequence):

    def py__simple_getitem__(self, index) -> Any:
        ...

class SetComprehension(_BaseComprehension, Sequence):
    ...

class GeneratorComprehension(_BaseComprehension, GeneratorBase):
    ...

class _DictKeyMixin:

    def get_mapping_item_values(self) -> Tuple:
        ...

    def get_key_values(self) -> Any:
        ...

class DictComprehension(ComprehensionMixin, Sequence, _DictKeyMixin):

    def __init__(self, inference_state, defining_context, sync_comp_for_node, key_node, value_node) -> None:
        ...

    def py__iter__(self, contextualized_node=None) -> Any:
        ...

    def py__simple_getitem__(self, index) -> Any:
        ...

    def _dict_keys(self) -> Any:
        ...

    def _dict_values(self) -> Any:
        ...

    def exact_key_items(self) -> Any:
        ...

class SequenceLiteralValue(Sequence):

    def __init__(self, inference_state, defining_context, atom) -> None:
        ...

    def py__simple_getitem__(self, index) -> Any:
        ...

    def py__iter__(self, contextualized_node=None) -> Any:
        ...

    def py__len__(self) -> int:
        ...

    def get_tree_entries(self) -> Any:
        ...

    def __repr__(self) -> str:
        ...

class DictLiteralValue(_DictMixin, SequenceLiteralValue, _DictKeyMixin):

    def __init__(self, inference_state, defining_context, atom) -> None:
        ...

    def py__simple_getitem__(self, index) -> Any:
        ...

    def py__iter__(self, contextualized_node=None) -> Any:
        ...

    def _imitate_values(self, arguments) -> Any:
        ...

    def _imitate_items(self, arguments) -> Any:
        ...

    def exact_key_items(self) -> Any:
        ...

class _FakeSequence(Sequence):

    def __init__(self, inference_state, lazy_value_list) -> None:
        ...

    def py__simple_getitem__(self, index) -> Any:
        ...

    def py__iter__(self, contextualized_node=None) -> Any:
        ...

    def py__bool__(self) -> bool:
        ...

    def __repr__(self) -> str:
        ...

class FakeTuple(_FakeSequence):
    ...

class FakeList(_FakeSequence):
    ...

class FakeDict(_DictMixin, Sequence, _DictKeyMixin):

    def __init__(self, inference_state, dct) -> None:
        ...

    def py__iter__(self, contextualized_node=None) -> Any:
        ...

    def py__simple_getitem__(self, index) -> Any:
        ...

    def _values(self, arguments) -> Any:
        ...

    def _dict_values(self) -> Any:
        ...

    def _dict_keys(self) -> Any:
        ...

    def exact_key_items(self) -> Any:
        ...

    def __repr__(self) -> str:
        ...

class MergedArray(Sequence):

    def __init__(self, inference_state, arrays) -> None:
        ...

    def py__iter__(self, contextualized_node=None) -> Any:
        ...

    def py__simple_getitem__(self, index) -> Any:
        ...
