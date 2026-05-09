"""
Values are the "values" that Python would return. However Values are at the
same time also the "values" that a user is currently sitting in.

A ValueSet is typically used to specify the return of a function or any other
static analysis operation. In jedi there are always multiple returns and not
just one.
"""
from functools import reduce
from operator import add
from itertools import zip_longest
from parso.python.tree import Name
from jedi import debug
from jedi.parser_utils import clean_scope_docstring
from jedi.inference.helpers import SimpleGetItemNotFound
from jedi.inference.utils import safe_property
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.cache import memoize_method
sentinel = object()
Self = type('Self', (), {})

class HasNoContext(Exception):
    pass

class HelperValueMixin:

    def get_root_context(self) -> Self:
        pass

    def execute(self, arguments) -> 'ValueSet':
        pass

    def execute_with_values(self, *value_list: 'Value') -> 'ValueSet':
        pass

    def execute_annotation(self) -> 'ValueSet':
        pass

    def gather_annotation_classes(self) -> 'ValueSet':
        pass

    def merge_types_of_iterate(self, contextualized_node: 'ContextualizedNode' = None, is_async: bool = False) -> 'ValueSet':
        pass

    def _get_value_filters(self, name_or_str: Name | str) -> 'Generator':
        pass

    def goto(self, name_or_str: Name | str, name_context: Self = None, analysis_errors: bool = True) -> list:
        pass

    def py__getattribute__(self, name_or_str: Name | str, name_context: Self = None, position: tuple[int, int] = None, analysis_errors: bool = True) -> 'ValueSet':
        pass

    def py__await__(self) -> 'ValueSet':
        pass

    def py__name__(self) -> str:
        pass

    def iterate(self, contextualized_node: 'ContextualizedNode' = None, is_async: bool = False) -> 'Generator':
        pass

    def is_sub_class_of(self, class_value: 'Value') -> bool:
        pass

    def is_same_class(self, class2: 'Value') -> bool:
        pass

    @memoize_method
    def as_context(self, *args, **kwargs) -> Self:
        pass

class Value(HelperValueMixin):
    tree_node: Name | None
    array_type: None
    api_type: str

    def __init__(self, inference_state: object, parent_context: Self | None = None):
        pass

    def py__getitem__(self, index_value_set: 'ValueSet', contextualized_node: 'ContextualizedNode') -> 'ValueSet':
        pass

    def py__simple_getitem__(self, index: object) -> object:
        pass

    def py__iter__(self, contextualized_node: 'ContextualizedNode' = None) -> 'Generator':
        pass

    def py__next__(self, contextualized_node: 'ContextualizedNode' = None) -> 'Generator':
        pass

    def get_signatures(self) -> list:
        pass

    def is_class(self) -> bool:
        pass

    def is_class_mixin(self) -> bool:
        pass

    def is_instance(self) -> bool:
        pass

    def is_function(self) -> bool:
        pass

    def is_module(self) -> bool:
        pass

    def is_namespace(self) -> bool:
        pass

    def is_compiled(self) -> bool:
        pass

    def is_bound_method(self) -> bool:
        pass

    def is_builtins_module(self) -> bool:
        pass

    def py__bool__(self) -> bool:
        pass

    def py__doc__(self) -> str:
        pass

    def get_safe_value(self, default: object = sentinel) -> object:
        pass

    def execute_operation(self, other: object, operator: str) -> 'ValueSet':
        pass

    def py__call__(self, arguments: object) -> 'ValueSet':
        pass

    def py__stop_iteration_returns(self) -> 'ValueSet':
        pass

    def py__getattribute__alternatives(self, name_or_str: str) -> 'ValueSet':
        pass

    def py__get__(self, instance: object, class_value: 'Value') -> 'ValueSet':
        pass

    def py__get__on_class(self, calling_instance: object, instance: object, class_value: 'Value') -> object:
        pass

    def get_qualified_names(self) -> None:
        pass

    def is_stub(self) -> bool:
        pass

    def _as_context(self) -> Self:
        pass

    @property
    def name(self) -> object:
        pass

    def get_type_hint(self, add_class_info: bool = True) -> str | None:
        pass

    def infer_type_vars(self, value_set: 'ValueSet') -> dict[str, 'ValueSet']:
        pass

def iterate_values(values: 'ValueSet', contextualized_node: 'ContextualizedNode' = None, is_async: bool = False) -> 'ValueSet':
    pass

class _ValueWrapperBase(HelperValueMixin):

    @safe_property
    def name(self) -> object:
        pass

    @classmethod
    @inference_state_as_method_param_cache()
    def create_cached(cls, inference_state: object, *args, **kwargs) -> Self:
        pass

    def __getattr__(self, name: str) -> object:
        pass

class LazyValueWrapper(_ValueWrapperBase):

    @safe_property
    @memoize_method
    def _wrapped_value(self) -> object:
        pass

    def __repr__(self) -> str:
        pass

    def _get_wrapped_value(self) -> object:
        pass

class ValueWrapper(_ValueWrapperBase):

    def __init__(self, wrapped_value: object):
        pass

    def __repr__(self) -> str:
        pass

class TreeValue(Value):

    def __init__(self, inference_state: object, parent_context: Self | None, tree_node: Name):
        pass

    def __repr__(self) -> str:
        pass

class ContextualizedNode:

    def __init__(self, context: object, node: object):
        pass

    def get_root_context(self) -> Self:
        pass

    def infer(self) -> 'ValueSet':
        pass

    def __repr__(self) -> str:
        pass

def _getitem(value: 'Value', index_values: 'ValueSet', contextualized_node: 'ContextualizedNode') -> 'ValueSet':
    pass

class ValueSet:

    def __init__(self, iterable: Iterable['Value']):
        pass

    @classmethod
    def _from_frozen_set(cls, frozenset_: frozenset['Value']) -> Self:
        pass

    @classmethod
    def from_sets(cls, sets: Iterable[Iterable['Value']]) -> Self:
        pass

    def __or__(self, other: 'ValueSet') -> 'ValueSet':
        pass

    def __and__(self, other: 'ValueSet') -> 'ValueSet':
        pass

    def __iter__(self) -> Iterator['Value']:
        pass

    def __bool__(self) -> bool:
        pass

    def __len__(self) -> int:
        pass

    def __repr__(self) -> str:
        pass

    def filter(self, filter_func: Callable) -> 'ValueSet':
        pass

    def __getattr__(self, name: str) -> Callable:
        pass

    def __eq__(self, other: 'ValueSet') -> bool:
        pass

    def __ne__(self, other: 'ValueSet') -> bool:
        pass

    def __hash__(self) -> int:
        pass

    def py__class__(self) -> 'ValueSet':
        pass

    def iterate(self, contextualized_node: 'ContextualizedNode' = None, is_async: bool = False) -> 'Generator':
        pass

    def execute(self, arguments: object) -> 'ValueSet':
        pass

    def execute_with_values(self, *args, **kwargs) -> 'ValueSet':
        pass

    def goto(self, *args, **kwargs) -> list:
        pass

    def py__getattribute__(self, *args, **kwargs) -> 'ValueSet':
        pass

    def get_item(self, *args, **kwargs) -> 'ValueSet':
        pass

    def try_merge(self, function_name: str) -> 'ValueSet':
        pass

    def gather_annotation_classes(self) -> 'ValueSet':
        pass

    def get_signatures(self) -> list:
        pass

    def get_type_hint(self, add_class_info: bool = True) -> str | None:
        pass

    def infer_type_vars(self, value_set: 'ValueSet') -> dict[str, 'ValueSet']:
        pass
NO_VALUES = ValueSet([])