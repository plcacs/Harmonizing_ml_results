from functools import reduce
from operator import add
from itertools import zip_longest
from typing import Any, Callable, Dict, FrozenSet, Generic, Iterable, Iterator, List, Optional, Set, Tuple, TypeVar, Union, cast

from parso.python.tree import Name
from jedi import debug
from jedi.parser_utils import clean_scope_docstring
from jedi.inference.helpers import SimpleGetItemNotFound
from jedi.inference.utils import safe_property
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.cache import memoize_method
sentinel = object()

T = TypeVar('T')
U = TypeVar('U')

class HasNoContext(Exception):
    pass

class HelperValueMixin:
    inference_state: Any
    parent_context: Optional['Value']
    name: Any
    
    def get_root_context(self) -> 'Value':
        value = self
        if value.parent_context is None:
            return value.as_context()
        while True:
            if value.parent_context is None:
                return value
            value = value.parent_context

    def execute(self, arguments: Any) -> 'ValueSet':
        return self.inference_state.execute(self, arguments=arguments)

    def execute_with_values(self, *value_list: Any) -> 'ValueSet':
        from jedi.inference.arguments import ValuesArguments
        arguments = ValuesArguments([ValueSet([value]) for value in value_list])
        return self.inference_state.execute(self, arguments)

    def execute_annotation(self) -> 'ValueSet':
        return self.execute_with_values()

    def gather_annotation_classes(self) -> 'ValueSet':
        return ValueSet([self])

    def merge_types_of_iterate(self, contextualized_node: Optional['ContextualizedNode'] = None, is_async: bool = False) -> 'ValueSet':
        return ValueSet.from_sets((lazy_value.infer() for lazy_value in self.iterate(contextualized_node, is_async)))

    def _get_value_filters(self, name_or_str: Union[str, Name]) -> Iterator[Any]:
        origin_scope = name_or_str if isinstance(name_or_str, Name) else None
        yield from self.get_filters(origin_scope=origin_scope)
        if self.is_stub():
            from jedi.inference.gradual.conversion import convert_values
            for c in convert_values(ValueSet({self})):
                yield from c.get_filters()

    def goto(self, name_or_str: Union[str, Name], name_context: Optional['Value'] = None, analysis_errors: bool = True) -> List[Any]:
        from jedi.inference import finder
        filters = self._get_value_filters(name_or_str)
        names = finder.filter_name(filters, name_or_str)
        debug.dbg('context.goto %s in (%s): %s', name_or_str, self, names)
        return names

    def py__getattribute__(self, name_or_str: Union[str, Name], name_context: Optional['Value'] = None, position: Optional[Tuple[int, int]] = None, analysis_errors: bool = True) -> 'ValueSet':
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

    def py__await__(self) -> 'ValueSet':
        await_value_set = self.py__getattribute__('__await__')
        if not await_value_set:
            debug.warning('Tried to run __await__ on value %s', self)
        return await_value_set.execute_with_values()

    def py__name__(self) -> str:
        return self.name.string_name

    def iterate(self, contextualized_node: Optional['ContextualizedNode'] = None, is_async: bool = False) -> Iterator[Any]:
        debug.dbg('iterate %s', self)
        if is_async:
            from jedi.inference.lazy_value import LazyKnownValues
            return iter([LazyKnownValues(self.py__getattribute__('__aiter__').execute_with_values().py__getattribute__('__anext__').execute_with_values().py__getattribute__('__await__').execute_with_values().py__stop_iteration_returns())])
        return self.py__iter__(contextualized_node)

    def is_sub_class_of(self, class_value: 'Value') -> bool:
        with debug.increase_indent_cm('subclass matching of %s <=> %s' % (self, class_value), color='BLUE'):
            for cls in self.py__mro__():
                if cls.is_same_class(class_value):
                    debug.dbg('matched subclass True', color='BLUE')
                    return True
            debug.dbg('matched subclass False', color='BLUE')
            return False

    def is_same_class(self, class2: 'Value') -> bool:
        if type(class2).is_same_class != HelperValueMixin.is_same_class:
            return class2.is_same_class(self)
        return self == class2

    @memoize_method
    def as_context(self, *args: Any, **kwargs: Any) -> 'Value':
        return self._as_context(*args, **kwargs)

class Value(HelperValueMixin):
    """
    To be implemented by subclasses.
    """
    tree_node: Optional[Any] = None
    array_type: Optional[Any] = None
    api_type: str = 'not_defined_please_report_bug'

    def __init__(self, inference_state: Any, parent_context: Optional['Value'] = None) -> None:
        self.inference_state = inference_state
        self.parent_context = parent_context

    def py__getitem__(self, index_value_set: 'ValueSet', contextualized_node: 'ContextualizedNode') -> 'ValueSet':
        from jedi.inference import analysis
        analysis.add(contextualized_node.context, 'type-error-not-subscriptable', contextualized_node.node, message="TypeError: '%s' object is not subscriptable" % self)
        return NO_VALUES

    def py__simple_getitem__(self, index: Union[int, str, slice, bytes]) -> 'ValueSet':
        raise SimpleGetItemNotFound

    def py__iter__(self, contextualized_node: Optional['ContextualizedNode'] = None) -> Iterator[Any]:
        if contextualized_node is not None:
            from jedi.inference import analysis
            analysis.add(contextualized_node.context, 'type-error-not-iterable', contextualized_node.node, message="TypeError: '%s' object is not iterable" % self)
        return iter([])

    def py__next__(self, contextualized_node: Optional['ContextualizedNode'] = None) -> Iterator[Any]:
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

    def execute_operation(self, other: 'Value', operator: str) -> 'ValueSet':
        debug.warning('%s not possible between %s and %s', operator, self, other)
        return NO_VALUES

    def py__call__(self, arguments: Any) -> 'ValueSet':
        debug.warning('no execution possible %s', self)
        return NO_VALUES

    def py__stop_iteration_returns(self) -> 'ValueSet':
        debug.warning('Not possible to return the stop iterations of %s', self)
        return NO_VALUES

    def py__getattribute__alternatives(self, name_or_str: str) -> 'ValueSet':
        """
        For now a way to add values in cases like __getattr__.
        """
        return NO_VALUES

    def py__get__(self, instance: 'Value', class_value: 'Value') -> 'ValueSet':
        debug.warning('No __get__ defined on %s', self)
        return ValueSet([self])

    def py__get__on_class(self, calling_instance: 'Value', instance: 'Value', class_value: 'Value') -> Any:
        return NotImplemented

    def get_qualified_names(self) -> Optional[Tuple[str, ...]]:
        return None

    def is_stub(self) -> bool:
        return self.parent_context.is_stub()

    def _as_context(self) -> 'Value':
        raise HasNoContext

    @property
    def name(self) -> Any:
        raise NotImplementedError

    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        return None

    def infer_type_vars(self, value_set: 'ValueSet') -> Dict[str, 'ValueSet']:
        """
        When the current instance represents a type annotation, this method
        tries to find information about undefined type vars and returns a dict
        from type var name to value set.

        This is for example important to understand what `iter([1])` returns.
        According to typeshed, `iter` returns an `Iterator[_T]`:

            def iter(iterable: Iterable[_T]) -> Iterator[_T]: ...

        This functions would generate `int` for `_T` in this case, because it
        unpacks the `Iterable`.

        Parameters
        ----------

        `self`: represents the annotation of the current parameter to infer the
            value for. In the above example, this would initially be the
            `Iterable[_T]` of the `iterable` parameter and then, when recursing,
            just the `_T` generic parameter.

        `value_set`: represents the actual argument passed to the parameter
            we're inferrined for, or (for recursive calls) their types. In the
            above example this would first be the representation of the list
            `[1]` and then, when recursing, just of `1`.
        """
        return {}

def iterate_values(values: 'ValueSet', contextualized_node: Optional['ContextualizedNode'] = None, is_async: bool = False) -> 'ValueSet':
    """
    Calls `iterate`, on all values but ignores the ordering and just returns
    all values that the iterate functions yield.
    """
    return ValueSet.from_sets((lazy_value.infer() for lazy_value in values.iterate(contextualized_node, is_async=is_async)))

class _ValueWrapperBase(HelperValueMixin):
    _wrapped_value: 'Value'

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
    def create_cached(cls, inference_state: Any, *args: Any, **kwargs: Any) -> '_ValueWrapperBase':
        return cls(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        assert name != '_wrapped_value', 'Problem with _get_wrapped_value'
        return getattr(self._wrapped_value, name)

class LazyValueWrapper(_ValueWrapperBase):

    @safe_property
    @memoize_method
    def _wrapped_value(self) -> 'Value':
        with debug.increase_indent_cm('Resolve lazy value wrapper'):
            return self._get_wrapped_value()

    def __repr__(self) -> str:
        return '<%s>' % self.__class__.__name__

    def _get_wrapped_value(self) -> 'Value':
        raise NotImplementedError

class ValueWrapper(_ValueWrapperBase):

    def __init__(self, wrapped_value: 'Value') -> None:
        self._wrapped_value = wrapped_value

    def __repr__(self) -> str:
        return '%s(%s)' % (self.__class__.__name__, self._wrapped_value)

class TreeValue(Value):

    def __init__(self, inference_state: Any, parent_context: Optional['Value'], tree_node: Any) -> None:
        super().__init__(inference_state, parent_context)
        self.tree_node = tree_node

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.tree_node)

class ContextualizedNode:

    def __init__(self, context: 'Value', node: Any) -> None:
        self.context = context
        self.node = node

    def get_root_context(self) -> 'Value':
        return self.context.get_root_context()

    def infer(self) -> 'ValueSet':
        return self.context.infer_node(self.node)

    def __repr__(self) -> str:
        return '<%s: %s in %s>' % (self.__class__.__name__, self.node, self.context)

def _getitem(value: 'Value', index_values: 'ValueSet', contextualized_node: 'ContextualizedNode') -> 'ValueSet':
    result = NO_VALUES
    unused_values: Set['Value'] = set()
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

class ValueSet(Generic[T]):

    def __init__(self, iterable: Iterable[T]) -> None:
        self._set: FrozenSet[T] = frozenset(iterable)
        for value in iterable:
            assert not isinstance(value, ValueSet)

    @classmethod
    def _from_frozen_set(cls, frozenset_: FrozenSet[T]) -> 'ValueSet[T]':
        self = cls.__new__(cls)
        self._set = frozenset_
        return self

    @classmethod
    def from_sets(cls, sets: Iterable[Union['ValueSet[T]', Iterable[T]]]) -> 'ValueSet[T]':
        """
        Used to work with an iterable of set.
        """
        aggregated: Set[T] = set()
        for set_ in sets:
            if isinstance(set_, ValueSet):
                aggregated |= set_._set
            else:
                aggregated |= frozenset(set_)
        return cls._from_frozen_set(frozenset(aggregated))

    def __or__(self, other: 'ValueSet[T]') -> 'ValueSet[T]':
        return self._from_frozen_set(self._set | other._set)

    def __and__(self, other: 'ValueSet[T]') -> 'ValueSet[T]':
        return self._from_frozen_set(self._set & other._set)

    def __iter__(self) -> Iterator[T]:
        return iter(self._set)

    def __bool__(self) -> bool:
        return bool(self._set)

    def __len__(self) -> int:
        return len(self._set)

    def __repr__(self) -> str:
        return 'S{%s}' % ', '.join((str(s) for s in self._set))

    def filter(self, filter_func: Callable[[T], bool]) -> 'ValueSet[T]':
        return self.__class__(filter(filter_func, self._set))

    def __getattr__(self, name: str) -> Callable[..., 'ValueSet']:
        def mapper(*args: Any, **kwargs: Any) -> 'ValueSet':
            return self.from_sets((getattr(value, name)(*args, **kwargs) for value in self._set))
        return mapper

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValueSet):
            return NotImplemented
        return self._set == other._set

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._set)

    def py__class__(self) -> 'ValueSet':
        return ValueSet((c.py__class__() for c in self._set))

    def iterate(self, contextualized_node: Optional['ContextualizedNode'] = None, is_async: bool = False) -> Iterator[Any]:
        from jedi.inference.lazy_value import get_merged_lazy_value
        type_iters = [c.iterate(contextualized_node, is_async=is_async) for c in self._set]
        for lazy_values in zip_longest(*type_iters):
            yield get_merged_lazy_value([l for l in lazy_values if l is not None])

    def execute(self, arguments: Any) -> 'ValueSet':
        return ValueSet.from_sets((c.inference_state.execute(c, arguments) for c in self._set))

    def execute_with_values(self, *args: Any, **kwargs: Any) -> 'ValueSet':
        return ValueSet.from_sets((c.execute_with_values(*args, **kwargs) for c in self._set))

    def goto(self, *args: Any, **kwargs: Any) -> List[Any]:
        return reduce(add, [c.goto(*args, **kwargs) for c in self._set], [])

    def py__getattribute__(self, *args: Any, **kwargs: Any) -> 'ValueSet':
        return ValueSet.from_sets((c.py__getattribute__(*args, **kwargs) for c in self._set))

    def get_item(self, *args: Any, **kwargs: Any) -> 'ValueSet':
        return ValueSet.from_sets((_getitem(c, *args, **kwargs) for c in self._set))

    def try_merge(self, function_name: str) -> 'ValueSet':
        value_set = self.__class__([])
        for c in self._set:
            try:
                method = getattr(c, function_name)
            except AttributeError:
                pass
            else:
                value_set |= method()
        return value_set

    def gather_annotation_classes(self) -> 'ValueSet':
        return ValueSet.from_sets([c.gather_annotation_classes() for c in self._set])

    def get_signatures(self) -> List[Any]:
        return [sig for c in self._set for sig in c.get_signatures()]

    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        t = [v.get_type_hint(add_class_info=add_class_info) for v in self._set]
        type_hints = sorted(filter(None, t))
        if len(type_hints) == 1:
            return type_hints[0]
        optional = 'None' in type_hints
        if optional:
            type_hints.remove('None')
        if len(type_hints) == 0:
            return None
        elif len(type_hints) == 1:
            s = type_hints[0]
        else:
            s = 'Union[%s]' % ', '.join(type_hints)
        if optional:
            s = 'Optional[%s]' % s
        return s

    def infer_type_vars(self, value_set: 'ValueSet') -> Dict[str, 'ValueSet']:
        from jedi.inference.gradual.annotation import merge_type_var_dicts
        type_var_dict: Dict[str, 'ValueSet'] = {}
        for value in self._set:
            merge_type_var_dicts(type_var_dict, value.infer_type_vars(value_set))
        return type_var_dict
NO_VALUES = ValueSet([])

def iterator_to_value_set(func: Callable[..., Iterator[T]]) -> Callable[..., ValueSet[T]]:
    def wrapper(*args: Any, **kwargs: Any) -> ValueSet[T]:
        return ValueSet(func(*args, **kwargs))
    return wrapper
