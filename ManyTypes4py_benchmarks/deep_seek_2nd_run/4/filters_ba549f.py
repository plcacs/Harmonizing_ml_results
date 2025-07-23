"""
Filters are objects that you can use to filter names in different scopes. They
are needed for name resolution.
"""
from abc import abstractmethod
from typing import (
    List, 
    MutableMapping, 
    Type, 
    Dict, 
    Any, 
    Tuple, 
    Iterator, 
    Optional, 
    Union, 
    Callable, 
    TypeVar, 
    Generic, 
    Set,
    Iterable,
    cast
)
import weakref
from parso.tree import search_ancestor
from parso.python.tree import Name, UsedNamesMapping
from jedi.inference import flow_analysis
from jedi.inference.base_value import ValueSet, ValueWrapper, LazyValueWrapper
from jedi.parser_utils import get_cached_parent_scope, get_parso_cache_node
from jedi.inference.utils import to_list
from jedi.inference.names import TreeNameDefinition, ParamName, AnonymousParamName, AbstractNameDefinition, NameWrapper
from jedi.inference.context import AbstractContext
from jedi.inference.arguments import AbstractArguments
from jedi.inference.value import Value
from parso.python.tree import PythonNode
from parso.tree import NodeOrLeaf

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)

_definition_name_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

class AbstractFilter:
    _until_position: Optional[Tuple[int, int]] = None

    def _filter(self, names: List[T]) -> List[T]:
        if self._until_position is not None:
            return [n for n in names if n.start_pos < self._until_position]
        return names

    @abstractmethod
    def get(self, name: str) -> List[AbstractNameDefinition]:
        raise NotImplementedError

    @abstractmethod
    def values(self) -> List[AbstractNameDefinition]:
        raise NotImplementedError

class FilterWrapper:
    name_wrapper_class: Type[NameWrapper]

    def __init__(self, wrapped_filter: AbstractFilter) -> None:
        self._wrapped_filter = wrapped_filter

    def wrap_names(self, names: List[AbstractNameDefinition]) -> List[NameWrapper]:
        return [self.name_wrapper_class(name) for name in names]

    def get(self, name: str) -> List[NameWrapper]:
        return self.wrap_names(self._wrapped_filter.get(name))

    def values(self) -> List[NameWrapper]:
        return self.wrap_names(self._wrapped_filter.values())

def _get_definition_names(
    parso_cache_node: Optional[PythonNode],
    used_names: UsedNamesMapping,
    name_key: str
) -> Tuple[Name, ...]:
    if parso_cache_node is None:
        names = used_names.get(name_key, ())
        return tuple((name for name in names if name.is_definition(include_setitem=True)))
    try:
        for_module = _definition_name_cache[parso_cache_node]
    except KeyError:
        for_module = _definition_name_cache[parso_cache_node] = {}
    try:
        return for_module[name_key]
    except KeyError:
        names = used_names.get(name_key, ())
        result = for_module[name_key] = tuple((name for name in names if name.is_definition(include_setitem=True)))
        return result

class _AbstractUsedNamesFilter(AbstractFilter):
    name_class: Type[TreeNameDefinition] = TreeNameDefinition

    def __init__(
        self,
        parent_context: AbstractContext,
        node_context: Optional[AbstractContext] = None
    ) -> None:
        if node_context is None:
            node_context = parent_context
        self._node_context = node_context
        self._parser_scope = node_context.tree_node
        module_context = node_context.get_root_context()
        path = module_context.py__file__()
        if path is None:
            self._parso_cache_node = None
        else:
            self._parso_cache_node = get_parso_cache_node(
                module_context.inference_state.latest_grammar if module_context.is_stub() 
                else module_context.inference_state.grammar, 
                path
            )
        self._used_names = module_context.tree_node.get_used_names()
        self.parent_context = parent_context

    def get(self, name: str) -> List[TreeNameDefinition]:
        return self._convert_names(self._filter(_get_definition_names(self._parso_cache_node, self._used_names, name)))

    def _convert_names(self, names: Iterable[Name]) -> List[TreeNameDefinition]:
        return [self.name_class(self.parent_context, name) for name in names]

    def values(self) -> List[TreeNameDefinition]:
        return self._convert_names((name for name_key in self._used_names for name in self._filter(_get_definition_names(self._parso_cache_node, self._used_names, name_key))))

    def __repr__(self) -> str:
        return '<%s: %s>' % (self.__class__.__name__, self.parent_context)

class ParserTreeFilter(_AbstractUsedNamesFilter):
    def __init__(
        self,
        parent_context: AbstractContext,
        node_context: Optional[AbstractContext] = None,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional[Any] = None
    ) -> None:
        super().__init__(parent_context, node_context)
        self._origin_scope = origin_scope
        self._until_position = until_position

    def _filter(self, names: List[Name]) -> List[Name]:
        names = super()._filter(names)
        names = [n for n in names if self._is_name_reachable(n)]
        return list(self._check_flows(names))

    def _is_name_reachable(self, name: Name) -> bool:
        parent = name.parent
        if parent.type == 'trailer':
            return False
        base_node = parent if parent.type in ('classdef', 'funcdef') else name
        return get_cached_parent_scope(self._parso_cache_node, base_node) == self._parser_scope

    def _check_flows(self, names: List[Name]) -> Iterator[Name]:
        for name in sorted(names, key=lambda name: name.start_pos, reverse=True):
            check = flow_analysis.reachability_check(
                context=self._node_context,
                value_scope=self._parser_scope,
                node=name,
                origin_scope=self._origin_scope
            )
            if check is not flow_analysis.UNREACHABLE:
                yield name
            if check is flow_analysis.REACHABLE:
                break

class _FunctionExecutionFilter(ParserTreeFilter):
    def __init__(
        self,
        parent_context: AbstractContext,
        function_value: Value,
        until_position: Optional[Tuple[int, int]],
        origin_scope: Optional[Any]
    ) -> None:
        super().__init__(parent_context, until_position=until_position, origin_scope=origin_scope)
        self._function_value = function_value

    def _convert_param(self, param: NodeOrLeaf, name: Name) -> Union[ParamName, AnonymousParamName]:
        raise NotImplementedError

    @to_list
    def _convert_names(self, names: List[Name]) -> Iterator[Union[TreeNameDefinition, ParamName, AnonymousParamName]]:
        for name in names:
            param = search_ancestor(name, 'param')
            if param:
                yield self._convert_param(param, name)
            else:
                yield TreeNameDefinition(self.parent_context, name)

class FunctionExecutionFilter(_FunctionExecutionFilter):
    def __init__(
        self,
        *args: Any,
        arguments: AbstractArguments,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._arguments = arguments

    def _convert_param(self, param: NodeOrLeaf, name: Name) -> ParamName:
        return ParamName(self._function_value, name, self._arguments)

class AnonymousFunctionExecutionFilter(_FunctionExecutionFilter):
    def _convert_param(self, param: NodeOrLeaf, name: Name) -> AnonymousParamName:
        return AnonymousParamName(self._function_value, name)

class GlobalNameFilter(_AbstractUsedNamesFilter):
    def get(self, name: str) -> List[TreeNameDefinition]:
        try:
            names = self._used_names[name]
        except KeyError:
            return []
        return self._convert_names(self._filter(names))

    @to_list
    def _filter(self, names: List[Name]) -> Iterator[Name]:
        for name in names:
            if name.parent.type == 'global_stmt':
                yield name

    def values(self) -> List[TreeNameDefinition]:
        return self._convert_names((name for name_list in self._used_names.values() for name in self._filter(name_list)))

class DictFilter(AbstractFilter):
    def __init__(self, dct: Dict[str, Any]) -> None:
        self._dct = dct

    def get(self, name: str) -> List[Any]:
        try:
            value = self._convert(name, self._dct[name])
        except KeyError:
            return []
        else:
            return list(self._filter([value]))

    def values(self) -> List[Any]:
        def yielder() -> Iterator[Any]:
            for item in self._dct.items():
                try:
                    yield self._convert(*item)
                except KeyError:
                    pass
        return self._filter(yielder())

    def _convert(self, name: str, value: Any) -> Any:
        return value

    def __repr__(self) -> str:
        keys = ', '.join(self._dct.keys())
        return '<%s: for {%s}>' % (self.__class__.__name__, keys)

class MergedFilter:
    def __init__(self, *filters: AbstractFilter) -> None:
        self._filters = filters

    def get(self, name: str) -> List[AbstractNameDefinition]:
        return [n for filter in self._filters for n in filter.get(name)]

    def values(self) -> List[AbstractNameDefinition]:
        return [n for filter in self._filters for n in filter.values()]

    def __repr__(self) -> str:
        return '%s(%s)' % (self.__class__.__name__, ', '.join((str(f) for f in self._filters)))

class _BuiltinMappedMethod(ValueWrapper):
    """``Generator.__next__`` ``dict.values`` methods and so on."""
    api_type = 'function'

    def __init__(self, value: Value, method: Callable, builtin_func: Value) -> None:
        super().__init__(builtin_func)
        self._value = value
        self._method = method

    def py__call__(self, arguments: AbstractArguments) -> ValueSet:
        return self._method(self._value, arguments)

class SpecialMethodFilter(DictFilter):
    """
    A filter for methods that are defined in this module on the corresponding
    classes like Generator (for __next__, etc).
    """

    class SpecialMethodName(AbstractNameDefinition):
        api_type = 'function'

        def __init__(
            self,
            parent_context: AbstractContext,
            string_name: str,
            callable_: Callable,
            builtin_value: Value
        ) -> None:
            self.parent_context = parent_context
            self.string_name = string_name
            self._callable = callable_
            self._builtin_value = builtin_value

        def infer(self) -> ValueSet:
            for filter in self._builtin_value.get_filters():
                for name in filter.get(self.string_name):
                    builtin_func = next(iter(name.infer()))
                    break
                else:
                    continue
                break
            return ValueSet([_BuiltinMappedMethod(self.parent_context, self._callable, builtin_func)])

    def __init__(self, value: Value, dct: Dict[str, Callable], builtin_value: Value) -> None:
        super().__init__(dct)
        self.value = value
        self._builtin_value = builtin_value

    def _convert(self, name: str, value: Callable) -> 'SpecialMethodFilter.SpecialMethodName':
        return self.SpecialMethodName(self.value, name, value, self._builtin_value)

class _OverwriteMeta(type):
    def __init__(cls, name: str, bases: Tuple[type, ...], dct: Dict[str, Any]) -> None:
        super().__init__(name, bases, dct)
        base_dct: Dict[str, Any] = {}
        for base_cls in reversed(cls.__bases__):
            try:
                base_dct.update(base_cls.overwritten_methods)
            except AttributeError:
                pass
        for func in cls.__dict__.values():
            try:
                base_dct.update(func.registered_overwritten_methods)
            except AttributeError:
                pass
        cls.overwritten_methods = base_dct

class _AttributeOverwriteMixin:
    def get_filters(self, *args: Any, **kwargs: Any) -> Iterator[AbstractFilter]:
        yield SpecialMethodFilter(self, self.overwritten_methods, self._wrapped_value)
        yield from self._wrapped_value.get_filters(*args, **kwargs)

class LazyAttributeOverwrite(_AttributeOverwriteMixin, LazyValueWrapper, metaclass=_OverwriteMeta):
    def __init__(self, inference_state: Any) -> None:
        self.inference_state = inference_state

class AttributeOverwrite(_AttributeOverwriteMixin, ValueWrapper, metaclass=_OverwriteMeta):
    pass

def publish_method(method_name: str) -> Callable[[Callable], Callable]:
    def decorator(func: Callable) -> Callable:
        dct = func.__dict__.setdefault('registered_overwritten_methods', {})
        dct[method_name] = func
        return func
    return decorator
