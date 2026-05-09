"""
Values are the "values" that Python would return. However Values are at the
same time also the "values" that a user is currently sitting in.

A ValueSet is typically used to specify the return of a function or any other
static analysis operation. In jedi there are always multiple returns and not
just one.
"""
from __future__ import annotations
from functools import reduce
from operator import add
from itertools import zip_longest
from typing import Any, Iterable, Iterator, List, Optional, Set, Tuple, Union, Dict, Callable, Type, TypeVar, Generator
from parso.python.tree import Name
from jedi import debug
from jedi.parser_utils import clean_scope_docstring
from jedi.inference.helpers import SimpleGetItemNotFound
from jedi.inference.utils import safe_property
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.cache import memoize_method

sentinel = object()

class HasNoContext(Exception):
    pass

class HelperValueMixin:

    def get_root_context(self) -> Any:
        value = self
        if value.parent_context is None:
            return value.as_context()
        while True:
            if value.parent_context is None:
                return value
            value = value.parent_context

    def execute(self, arguments: Any) -> ValueSet:
        return self.inference_state.execute(self, arguments=arguments)

    def execute_with_values(self, *value_list: Any) -> ValueSet:
        from jedi.inference.arguments import ValuesArguments
        arguments = ValuesArguments([ValueSet([value]) for value in value_list])
        return self.inference_state.execute(self, arguments)

    def execute_annotation(self) -> ValueSet:
        return self.execute_with_values()

    def gather_annotation_classes(self) -> ValueSet:
        return ValueSet([self])

    def merge_types_of_iterate(self, contextualized_node: Optional[Any] = None, is_async: bool = False) -> ValueSet:
        return ValueSet.from_sets((lazy_value.infer() for lazy_value in self.iterate(contextualized_node, is_async)))

    def _get_value_filters(self, name_or_str: Union[Name, str]) -> Generator[Any, None, None]:
        origin_scope = name_or_str if isinstance(name_or_str, Name) else None
        yield from self.get_filters(origin_scope=origin_scope)
        if self.is_stub():
            from jedi.inference.gradual.conversion import convert_values
            for c in convert_values(ValueSet({self})):
                yield from c.get_filters()

    def goto(self, name_or_str: Union[Name, str], name_context: Optional[Any] = None, analysis_errors: bool = True) -> List[Any]:
        from jedi.inference import finder
        filters = self._get_value_filters(name_or_str)
        names = finder.filter_name(filters, name_or_str)
        debug.dbg('context.goto %s in (%s): %s', name_or_str, self, names)
        return names

    def py__getattribute__(self, name_or_str: Union[Name, str], name_context: Optional[Any] = None, position: Optional[Tuple[int, int]] = None, analysis_errors: bool = True) -> ValueSet:
        """
        :param position: Position of the last statement -> tuple of line, column
        """
        if name_context is None:
            name_context = self
        names = self.goto(name_or_str, name_context, analysis_errors)
        values = ValueSet.from_sets((name.infer() for name in names))
        if not values:
            n = name_or_str.value if isinstance(name_or_str, Name) else name_or_str
            values = self.py__getattribute__alternatives(n)
        if not names and (not values) and analysis_errors:
            if isinstance(name_or_str, Name):
                from jedi.inference import analysis
                analysis.add_attribute_error(name_context, self, name_or_str)
        debug.dbg('context.names_to_types: %s -> %s', names, values)
        return values

    def py__await__(self) -> ValueSet:
        await_value_set = self.py__getattribute__('__await__')
        if not await_value_set:
            debug.warning('Tried to run __await__ on value %s', self)
        return await_value_set.execute_with_values()

    def py__name__(self) -> str:
        return self.name.string_name

    def iterate(self, contextualized_node: Optional[Any] = None, is_async: bool = False) -> Iterator[Any]:
        debug.dbg('iterate %s', self)
        if is_async:
            from jedi.inference.lazy_value import LazyKnownValues
            return iter([LazyKnownValues(self.py__getattribute__('__aiter__').execute_with_values().py__getattribute__('__anext__').execute_with_values().py__getattribute__('__await__').execute_with_values().py__stop_iteration_returns())])
        return self.py__iter__(contextualized_node)

    def is_sub_class_of(self, class_value: Any) -> bool:
        with debug.increase_indent_cm('subclass matching of %s <=> %s' % (self, class_value), color='BLUE'):
            for cls in self.py__mro__():
                if cls.is_same_class(class_value):
                    debug.dbg('matched subclass True', color='BLUE')
                    return True
            debug.dbg('matched subclass False', color='BLUE')
            return False

    def is_same_class(self, class2: Any) -> bool:
        if type(class2).is_same_class != HelperValueMixin.is_same_class:
            return class2.is_same_class(self)
        return self == class2

    @memoize_method
    def as_context(self, *args: Any, **kwargs: Any) -> Any:
        return self._as_context(*args, **kwargs)

class Value(HelperValueMixin):
    """
    To be implemented by subclasses.
    """
    tree_node: Any = None
    array_type: Any = None
    api_type: str = 'not_defined_please_report_bug'

    def __init__(self, inference_state: Any, parent_context: Optional[Any] = None) -> None:
        self.inference_state = inference_state
        self.parent_context = parent_context

    def py__getitem__(self, index_value_set: ValueSet, contextualized_node: Any) -> ValueSet:
        from jedi.inference import analysis
        analysis.add(contextualized_node.context, 'type-error-not-subscriptable', contextualized_node.node, message="TypeError: '%s' object is not subscriptable" % self)
        return NO_VALUES

    def py__simple_getitem__(self, index: Any) -> ValueSet:
        raise SimpleGetItemNotFound

    def py__iter__(self, contextualized_node: Optional[Any] = None) -> Iterator[Any]:
        if contextualized_node is not None:
            from jedi.inference import analysis
            analysis.add(contextualized_node.context, 'type-error-not-iterable', contextualized_node.node, message="TypeError: '%s' object is not iterable" % self)
        return iter([])

    def py__next__(self, contextualized_node: Optional[Any] = None) -> Iterator[Any]:
        return self.py__iter__(contextualized_node)

    def get_signatures(self) -> List[Any]:
        return []

    def is_class(self) -> bool:
        return False

    def is_class_mixin(self) -> bool:
        return False

    def is_instance(self) -> bool:
        return False

    def is_function(self) -> bool:
        return False

    def is_module(self) -> bool:
        return False

    def is_namespace(self) -> bool:
        return False

    def is_compiled(self) -> bool:
        return False

    def is_bound_method(self) -> bool:
        return False

    def is_builtins_module(self) -> bool:
        return False

    def py__bool__(self) -> bool:
        """
        Since Wrapper is a super class for classes, functions and modules,
        the return value will always be true.
        """
        return True

    def py__doc__(self) -> str:
        try:
            self.tree_node.get_doc_node
        except AttributeError:
            return ''
        else:
            return clean_scope_docstring(self.tree_node)

    def get_safe_value(self, default: Any = sentinel) -> Any:
        if default is sentinel:
            raise ValueError('There exists no safe value for value %s' % self)
        return default

    def execute_operation(self, other: Any, operator: Any) -> ValueSet:
        debug.warning('%s not possible between %s and %s', operator, self, other)
        return NO_VALUES

    def py__call__(self, arguments: Any) -> ValueSet:
        debug.warning('no execution possible %s', self)
        return NO_VALUES

    def py__stop_iteration_returns(self) -> ValueSet:
        debug.warning('Not possible to return the stop iterations of %s', self)
        return NO_VALUES

    def py__getattribute__alternatives(self, name_or_str: Union[Name, str]) -> ValueSet:
        """
        For now a way to add values in cases like __getattr__.
        """
        return NO_VALUES

    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        debug.warning('No __get__ defined on %s', self)
        return ValueSet([self])

    def py__get__on_class(self, calling_instance: Any, instance: Any, class_value: Any) -> Any:
        return NotImplemented

    def get_qualified_names(self) -> Optional[List[str]]:
        return None

    def is_stub(self) -> bool:
        return self.parent_context.is_stub()

    def _as_context(self) -> Any:
        raise HasNoContext

    @property
    def name(self) -> Any:
        raise NotImplementedError

    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        return None

    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, ValueSet]:
        return {}

def iterate_values(values: ValueSet, contextualized_node: Optional[Any] = None, is_async: bool = False) -> ValueSet:
    """
    Calls `iterate`, on all values but ignores the ordering and just returns
    all values that the iterate functions yield.
    """
    return ValueSet.from_sets((lazy_value.infer() for lazy_value in values.iterate(contextualized_node, is_async=is_async)))

class _ValueWrapperBase(HelperValueMixin):

    @safe_property
    def name(self) -> Any:
        from jedi.inference.names import ValueName
        wrapped_name = self._wrapped_value.name
        if wrapped_name.tree_name is not None:
            return ValueName(self, wrapped_name.tree_name)
        else:
            from jedi.inference.compiled import CompiledValueName
            return CompiledValueName(self, wrapped_name.string_name)

    @classmethod
    @inference_state_as_method_param_cache()
    def create_cached(cls, inference_state: Any, *args: Any, **kwargs: Any) -> _ValueWrapperBase:
        return cls(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        assert name != '_wrapped_value', 'Problem with _get_wrapped_value'
        return getattr(self._wrapped_value, name)

class LazyValueWrapper(_ValueWrapperBase):

    @safe_property
    @memoize_method
    def _wrapped_value(self) -> Any:
        with debug.increase_indent_cm('Resolve lazy value wrapper'):
            return self._get_wrapped_value()

    def __repr__(self) -> str:
        return '<%s>' % self.__class__.__name__

    def _get_wrapped_value(self) -> Any:
        raise NotImplementedError

class ValueWrapper(_ValueWrapperBase):

    def __init__(self, wrapped_value: Any) -> None:
        self._wrapped_value = wrapped_value

    def __repr__(self) -> str:
        return '%s(%s)' % (self.__class__.__name__, self._wrapped_value)

class TreeValue(Value):

    def __init__(self, inference_state: Any, parent_context: Any, tree_node: Any) -> None:
        super().__init__(inference_state, parent_context)
        self.tree_node = tree_node

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.tree_node)

class ContextualizedNode:

    def __init__(self, context: Any, node: Any) -> None:
        self.context = context
        self.node = node

    def get_root_context(self) -> Any:
        return self.context.get_root_context()

    def infer(self) -> ValueSet:
        return self.context.infer_node(self.node)

    def __repr__(self) -> str:
        return '<%s: %s in %s>' % (self.__class__.__name__, self.node, self.context)

def _getitem(value: Value, index_values: ValueSet, contextualized_node: ContextualizedNode) -> ValueSet:
    result = NO_VALUES
    unused_values: Set[Any] = set()
    for index_value in index_values:
        index = index_value.get_safe_value(default=None)
        if type(index) in (float, int, str, slice, bytes):
            try:
                result |= value.py__simple_getitem__(index)
                continue
            except SimpleGetItemNotFound:
                pass
        unused_values.add(index_value)
    if unused_values or not index_values:
        result |= value.py__getitem__(ValueSet(unused_values), contextualized_node)
    debug.dbg('py__getitem__ result: %s', result)
    return result

class ValueSet:

    def __init__(self, iterable: Iterable[Value]) -> None:
        self._set = frozenset(iterable)
        for value in iterable:
            assert not isinstance(value, ValueSet)

    @classmethod
    def _from_frozen_set(cls, frozenset_: frozenset) -> ValueSet:
        self = cls.__new__(cls)
        self._set = frozenset_
        return self

    @classmethod
    def from_sets(cls, sets: Iterable[Union[ValueSet, Iterable[Value]]]) -> ValueSet:
        """
        Used to work with an iterable of set.
        """
        aggregated = set()
        for set_ in sets:
            if isinstance(set_, ValueSet):
                aggregated |= set_._set
            else:
                aggregated |= frozenset(set_)
        return cls._from_frozen_set(frozenset(aggregated))

    def __or__(self, other: ValueSet) -> ValueSet:
        return self._from_frozen_set(self._set | other._set)

    def __and__(self, other: ValueSet) -> ValueSet:
        return self._from_frozen_set(self._set & other._set)

    def __iter__(self) -> Iterator[Value]:
        return iter(self._set)

    def __bool__(self) -> bool:
        return bool(self._set)

    def __len__(self) -> int:
        return len(self._set)

    def __repr__(self) -> str:
        return 'S{%s}' % ', '.join((str(s) for s in self._set))

    def filter(self, filter_func: Callable[[Value], bool]) -> ValueSet:
        return self.__class__(filter(filter_func, self._set))

    def __getattr__(self, name: str) -> Callable[..., ValueSet]:

        def mapper(*args: Any, **kwargs: Any) -> ValueSet:
            return self.from_sets((getattr(value, name)(*args, **kwargs) for value in self._set))
        return mapper

    def __eq__(self, other: object) -> bool:
        return self._set == getattr(other, '_set', None)

    def __ne__(self, other: object) -> bool:
        return not