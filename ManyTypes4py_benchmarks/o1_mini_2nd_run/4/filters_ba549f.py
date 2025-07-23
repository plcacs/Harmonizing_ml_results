"""
Filters are objects that you can use to filter names in different scopes. They
are needed for name resolution.
"""
from abc import ABC, abstractmethod
from typing import (
    List,
    MutableMapping,
    Type,
    Optional,
    Tuple,
    Dict,
    Any,
    Callable,
    Iterable,
    Generator,
)
import weakref
from parso.tree import search_ancestor
from parso.python.tree import Name, UsedNamesMapping
from jedi.inference import flow_analysis
from jedi.inference.base_value import ValueSet, ValueWrapper, LazyValueWrapper
from jedi.parser_utils import get_cached_parent_scope, get_parso_cache_node
from jedi.inference.utils import to_list
from jedi.inference.names import (
    TreeNameDefinition,
    ParamName,
    AnonymousParamName,
    AbstractNameDefinition,
    NameWrapper,
)
from jedi.inference import inference
from jedi.inference import context

_definition_name_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class AbstractFilter(ABC):
    _until_position: Optional[Tuple[int, int]] = None

    def _filter(self, names: Iterable[Name]) -> List[Name]:
        if self._until_position is not None:
            return [n for n in names if n.start_pos < self._until_position]
        return list(names)

    @abstractmethod
    def get(self, name: str) -> List['AbstractNameDefinition']:
        raise NotImplementedError

    @abstractmethod
    def values(self) -> List['AbstractNameDefinition']:
        raise NotImplementedError


class FilterWrapper:
    name_wrapper_class: Type['NameWrapper']

    def __init__(self, wrapped_filter: AbstractFilter) -> None:
        self._wrapped_filter: AbstractFilter = wrapped_filter

    def wrap_names(self, names: Iterable['AbstractNameDefinition']) -> List['NameWrapper']:
        return [self.name_wrapper_class(name) for name in names]

    def get(self, name: str) -> List['NameWrapper']:
        return self.wrap_names(self._wrapped_filter.get(name))

    def values(self) -> List['NameWrapper']:
        return self.wrap_names(self._wrapped_filter.values())


def _get_definition_names(
    parso_cache_node: Optional[Any],
    used_names: UsedNamesMapping,
    name_key: str
) -> Tuple[Name, ...]:
    if parso_cache_node is None:
        names = used_names.get(name_key, ())
        return tuple(name for name in names if name.is_definition(include_setitem=True))
    try:
        for_module = _definition_name_cache[parso_cache_node]
    except KeyError:
        for_module = _definition_name_cache[parso_cache_node] = {}
    try:
        return for_module[name_key]
    except KeyError:
        names = used_names.get(name_key, ())
        result = tuple(name for name in names if name.is_definition(include_setitem=True))
        for_module[name_key] = result
        return result


class _AbstractUsedNamesFilter(AbstractFilter):
    name_class: Type['TreeNameDefinition'] = TreeNameDefinition

    parent_context: 'context.BaseContext'
    _node_context: 'context.BaseContext'
    _parser_scope: Any
    _parso_cache_node: Optional[Any]
    _used_names: UsedNamesMapping
    parent_context: 'context.BaseContext'

    def __init__(self, parent_context: 'context.BaseContext', node_context: Optional['context.BaseContext'] = None) -> None:
        if node_context is None:
            node_context = parent_context
        self._node_context: 'context.BaseContext' = node_context
        self._parser_scope = node_context.tree_node
        module_context = node_context.get_root_context()
        path = module_context.py__file__()
        if path is None:
            self._parso_cache_node = None
        else:
            grammar = module_context.inference_state.latest_grammar if module_context.is_stub() else module_context.inference_state.grammar
            self._parso_cache_node = get_parso_cache_node(grammar, path)
        self._used_names: UsedNamesMapping = module_context.tree_node.get_used_names()
        self.parent_context = parent_context

    def get(self, name: str) -> List['TreeNameDefinition']:
        return self._convert_names(self._filter(_get_definition_names(self._parso_cache_node, self._used_names, name)))

    def _convert_names(self, names: Iterable[Name]) -> List['TreeNameDefinition']:
        return [self.name_class(self.parent_context, name) for name in names]

    def values(self) -> List['TreeNameDefinition']:
        return self._convert_names(
            (
                name
                for name_key in self._used_names
                for name in self._filter(_get_definition_names(self._parso_cache_node, self._used_names, name_key))
            )
        )

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self.parent_context}>'


class ParserTreeFilter(_AbstractUsedNamesFilter):
    _origin_scope: Any

    def __init__(
        self,
        parent_context: 'context.BaseContext',
        node_context: Optional['context.BaseContext'] = None,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional[Any] = None
    ) -> None:
        """
        node_context is an option to specify a second value for use cases
        like the class mro where the parent class of a new name would be the
        value, but for some type inference it's important to have a local
        value of the other classes.
        """
        super().__init__(parent_context, node_context)
        self._origin_scope = origin_scope
        self._until_position = until_position

    def _filter(self, names: Iterable[Name]) -> List[Name]:
        names = super()._filter(names)
        names = [n for n in names if self._is_name_reachable(n)]
        return list(self._check_flows(names))

    def _is_name_reachable(self, name: Name) -> bool:
        parent = name.parent
        if parent.type == 'trailer':
            return False
        base_node = parent if parent.type in ('classdef', 'funcdef') else name
        cached_scope = get_cached_parent_scope(self._parso_cache_node, base_node)
        return cached_scope == self._parser_scope

    def _check_flows(self, names: List[Name]) -> Generator[Name, None, None]:
        sorted_names = sorted(names, key=lambda name: name.start_pos, reverse=True)
        for name in sorted_names:
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
    _function_value: Any

    def __init__(
        self,
        parent_context: 'context.BaseContext',
        function_value: Any,
        until_position: Optional[Tuple[int, int]],
        origin_scope: Optional[Any]
    ) -> None:
        super().__init__(parent_context, until_position=until_position, origin_scope=origin_scope)
        self._function_value = function_value

    def _convert_param(self, param: Any, name: Name) -> 'AbstractNameDefinition':
        raise NotImplementedError

    @to_list
    def _convert_names(self, names: Iterable[Name]) -> List['AbstractNameDefinition']:
        for name in names:
            param = search_ancestor(name, 'param')
            if param:
                yield self._convert_param(param, name)
            else:
                yield TreeNameDefinition(self.parent_context, name)


class FunctionExecutionFilter(_FunctionExecutionFilter):
    _arguments: Any

    def __init__(
        self,
        parent_context: 'context.BaseContext',
        function_value: Any,
        until_position: Optional[Tuple[int, int]],
        origin_scope: Optional[Any],
        arguments: Any
    ) -> None:
        super().__init__(parent_context, function_value, until_position, origin_scope)
        self._arguments = arguments

    def _convert_param(self, param: Any, name: Name) -> 'ParamName':
        return ParamName(self._function_value, name, self._arguments)


class AnonymousFunctionExecutionFilter(_FunctionExecutionFilter):
    def _convert_param(self, param: Any, name: Name) -> 'AnonymousParamName':
        return AnonymousParamName(self._function_value, name)


class GlobalNameFilter(_AbstractUsedNamesFilter):
    def get(self, name: str) -> List['TreeNameDefinition']:
        try:
            names = self._used_names[name]
        except KeyError:
            return []
        return self._convert_names(self._filter(names))

    @to_list
    def _filter(self, names: Iterable[Name]) -> Generator[Name, None, None]:
        for name in names:
            if name.parent.type == 'global_stmt':
                yield name

    def values(self) -> List['TreeNameDefinition']:
        return self._convert_names(
            (
                name
                for name_list in self._used_names.values()
                for name in self._filter(name_list)
            )
        )


class DictFilter(AbstractFilter):
    _dct: Dict[str, Any]

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
        def yielder() -> Iterable[Any]:
            for key, value in self._dct.items():
                try:
                    yield self._convert(key, value)
                except KeyError:
                    pass

        return self._filter(yielder())

    def _convert(self, name: str, value: Any) -> Any:
        return value

    def __repr__(self) -> str:
        keys = ', '.join(self._dct.keys())
        return f'<{self.__class__.__name__}: for {{{keys}}}>'


class MergedFilter:
    _filters: Tuple[AbstractFilter, ...]

    def __init__(self, *filters: AbstractFilter) -> None:
        self._filters = filters

    def get(self, name: str) -> List[Any]:
        return [n for filter_ in self._filters for n in filter_.get(name)]

    def values(self) -> List[Any]:
        return [n for filter_ in self._filters for n in filter_.values()]

    def __repr__(self) -> str:
        filters_str = ', '.join(str(f) for f in self._filters)
        return f'{self.__class__.__name__}({filters_str})'


class _BuiltinMappedMethod(ValueWrapper):
    """``Generator.__next__`` ``dict.values`` methods and so on."""
    api_type: str = 'function'

    _value: Any
    _method: Callable[..., Any]
    _builtin_func: Any

    def __init__(self, value: Any, method: Callable[..., Any], builtin_func: Any) -> None:
        super().__init__(builtin_func)
        self._value = value
        self._method = method
        self._builtin_func = builtin_func

    def py__call__(self, arguments: Any) -> ValueSet:
        return self._method(self._value, arguments)


class SpecialMethodFilter(DictFilter):
    """
    A filter for methods that are defined in this module on the corresponding
    classes like Generator (for __next__, etc).
    """

    class SpecialMethodName(AbstractNameDefinition):
        api_type: str = 'function'

        parent_context: 'context.BaseContext'
        string_name: str
        _callable: Callable[..., Any]
        _builtin_value: Any

        def __init__(
            self,
            parent_context: 'context.BaseContext',
            string_name: str,
            callable_: Callable[..., Any],
            builtin_value: Any
        ) -> None:
            self.parent_context = parent_context
            self.string_name = string_name
            self._callable = callable_
            self._builtin_value = builtin_value

        def infer(self) -> ValueSet:
            for filter_ in self._builtin_value.get_filters():
                for name in filter_.get(self.string_name):
                    builtin_func = next(iter(name.infer()), None)
                    if builtin_func is not None:
                        break
                else:
                    continue
                break
            else:
                return ValueSet()
            return ValueSet([_BuiltinMappedMethod(self.parent_context, self._callable, builtin_func)])

    value: Any
    _builtin_value: Any

    def __init__(self, value: Any, dct: Dict[str, Any], builtin_value: Any) -> None:
        super().__init__(dct)
        self.value = value
        self._builtin_value = builtin_value
        '\n        This value is what will be used to introspect the name, whereas the\n        other value will be used to execute the function.\n\n        We distinguish, because we have to.\n        '

    def _convert(self, name: str, value: Any) -> 'SpecialMethodName':
        return self.SpecialMethodName(self.value, name, value, self._builtin_value)


class _OverwriteMeta(type):
    overwritten_methods: Dict[str, Callable[..., Any]] = {}

    def __init__(cls, name: str, bases: Tuple[Type, ...], dct: Dict[str, Any]) -> None:
        super().__init__(name, bases, dct)
        base_dct: Dict[str, Callable[..., Any]] = {}
        for base_cls in reversed(cls.__bases__):
            if hasattr(base_cls, 'overwritten_methods'):
                base_dct.update(base_cls.overwritten_methods)
        for func in cls.__dict__.values():
            if hasattr(func, 'registered_overwritten_methods'):
                base_dct.update(func.registered_overwritten_methods)
        cls.overwritten_methods = base_dct


class _AttributeOverwriteMixin:
    def get_filters(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Generator[AbstractFilter, None, None]:
        yield SpecialMethodFilter(self, self.overwritten_methods, self._wrapped_value)
        yield from self._wrapped_value.get_filters(*args, **kwargs)


class LazyAttributeOverwrite(_AttributeOverwriteMixin, LazyValueWrapper, metaclass=_OverwriteMeta):
    inference_state: 'inference.InferenceState'

    def __init__(self, inference_state: 'inference.InferenceState') -> None:
        self.inference_state = inference_state


class AttributeOverwrite(_AttributeOverwriteMixin, ValueWrapper, metaclass=_OverwriteMeta):
    pass


def publish_method(method_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        dct = func.__dict__.setdefault('registered_overwritten_methods', {})
        dct[method_name] = func
        return func
    return decorator
