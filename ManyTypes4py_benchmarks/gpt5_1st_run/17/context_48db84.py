from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi.inference.filters import ParserTreeFilter, MergedFilter, GlobalNameFilter
from jedi.inference.names import AnonymousParamName, TreeNameDefinition
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.parser_utils import get_parent_scope
from jedi import debug
from jedi import parser_utils


class AbstractContext:
    def __init__(self, inference_state: Any):
        self.inference_state: Any = inference_state
        self.predefined_names: Dict[Any, Dict[str, ValueSet]] = {}

    @abstractmethod
    def get_filters(
        self,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional[Name] = None,
    ) -> Iterator[Any]:
        raise NotImplementedError

    def goto(
        self,
        name_or_str: Union[Name, str],
        position: Optional[Tuple[int, int]],
    ) -> list[Any]:
        from jedi.inference import finder

        filters = _get_global_filters_for_name(
            self,
            name_or_str if isinstance(name_or_str, Name) else None,
            position,
        )
        names = finder.filter_name(filters, name_or_str)
        debug.dbg('context.goto %s in (%s): %s', name_or_str, self, names)
        return names

    def py__getattribute__(
        self,
        name_or_str: Union[Name, str],
        name_context: Optional[AbstractContext] = None,
        position: Optional[Tuple[int, int]] = None,
        analysis_errors: bool = True,
    ) -> ValueSet:
        """
        :param position: Position of the last statement -> tuple of line, column
        """
        if name_context is None:
            name_context = self
        names = self.goto(name_or_str, position)
        string_name = name_or_str.value if isinstance(name_or_str, Name) else name_or_str
        found_predefined_types: Optional[ValueSet] = None
        if self.predefined_names and isinstance(name_or_str, Name):
            node: Any = name_or_str
            while node is not None and (not parser_utils.is_scope(node)):
                node = node.parent
                if node.type in ('if_stmt', 'for_stmt', 'comp_for', 'sync_comp_for'):
                    try:
                        name_dict = self.predefined_names[node]
                        types = name_dict[string_name]
                    except KeyError:
                        continue
                    else:
                        found_predefined_types = types
                        break
        if found_predefined_types is not None and names:
            from jedi.inference import flow_analysis
            check = flow_analysis.reachability_check(context=self, value_scope=self.tree_node, node=name_or_str)
            if check is flow_analysis.UNREACHABLE:
                values = NO_VALUES
            else:
                values = found_predefined_types
        else:
            values = ValueSet.from_sets((name.infer() for name in names))
        if not names and (not values) and analysis_errors:
            if isinstance(name_or_str, Name):
                from jedi.inference import analysis
                message = "NameError: name '%s' is not defined." % string_name
                analysis.add(name_context, 'name-error', name_or_str, message)
        debug.dbg('context.names_to_types: %s -> %s', names, values)
        if values:
            return values
        return self._check_for_additional_knowledge(name_or_str, name_context, position)

    def _check_for_additional_knowledge(
        self,
        name_or_str: Union[Name, str],
        name_context: Optional[AbstractContext],
        position: Optional[Tuple[int, int]],
    ) -> ValueSet:
        name_context = name_context or self
        if isinstance(name_or_str, Name) and (not name_context.is_instance()):
            flow_scope: Any = name_or_str
            base_nodes = [name_context.tree_node]
            if any((b.type in ('comp_for', 'sync_comp_for') for b in base_nodes)):
                return NO_VALUES
            from jedi.inference.finder import check_flow_information
            while True:
                flow_scope = get_parent_scope(flow_scope, include_flows=True)
                n = check_flow_information(name_context, flow_scope, name_or_str, position)
                if n is not None:
                    return n
                if flow_scope in base_nodes:
                    break
        return NO_VALUES

    def get_root_context(self) -> AbstractContext:
        parent_context = self.parent_context
        if parent_context is None:
            return self
        return parent_context.get_root_context()

    def is_module(self) -> bool:
        return False

    def is_builtins_module(self) -> bool:
        return False

    def is_class(self) -> bool:
        return False

    def is_stub(self) -> bool:
        return False

    def is_instance(self) -> bool:
        return False

    def is_compiled(self) -> bool:
        return False

    def is_bound_method(self) -> bool:
        return False

    @abstractmethod
    def py__name__(self) -> str:
        raise NotImplementedError

    def get_value(self) -> Any:
        raise NotImplementedError

    @property
    def name(self) -> Any:
        return None

    def get_qualified_names(self) -> Tuple[str, ...]:
        return ()

    def py__doc__(self) -> str:
        return ''

    @contextmanager
    def predefine_names(self, flow_scope: Any, dct: Dict[str, ValueSet]) -> Iterator[None]:
        predefined = self.predefined_names
        predefined[flow_scope] = dct
        try:
            yield
        finally:
            del predefined[flow_scope]


class ValueContext(AbstractContext):
    """
    Should be defined, otherwise the API returns empty types.
    """

    def __init__(self, value: Any):
        super().__init__(value.inference_state)
        self._value: Any = value

    @property
    def tree_node(self) -> Any:
        return self._value.tree_node

    @property
    def parent_context(self) -> Optional[AbstractContext]:
        return self._value.parent_context

    def is_module(self) -> bool:
        return self._value.is_module()

    def is_builtins_module(self) -> bool:
        return self._value == self.inference_state.builtins_module

    def is_class(self) -> bool:
        return self._value.is_class()

    def is_stub(self) -> bool:
        return self._value.is_stub()

    def is_instance(self) -> bool:
        return self._value.is_instance()

    def is_compiled(self) -> bool:
        return self._value.is_compiled()

    def is_bound_method(self) -> bool:
        return self._value.is_bound_method()

    def py__name__(self) -> str:
        return self._value.py__name__()

    @property
    def name(self) -> Any:
        return self._value.name

    def get_qualified_names(self) -> Tuple[str, ...]:
        return self._value.get_qualified_names()

    def py__doc__(self) -> str:
        return self._value.py__doc__()

    def get_value(self) -> Any:
        return self._value

    def __repr__(self) -> str:
        return '%s(%s)' % (self.__class__.__name__, self._value)


class TreeContextMixin:
    def infer_node(self, node: Any) -> ValueSet:
        from jedi.inference.syntax_tree import infer_node
        return infer_node(self, node)

    def create_value(self, node: Any) -> Any:
        from jedi.inference import value
        if node == self.tree_node:
            assert self.is_module()
            return self.get_value()
        parent_context = self.create_context(node)
        if node.type in ('funcdef', 'lambdef'):
            func = value.FunctionValue.from_context(parent_context, node)
            if parent_context.is_class():
                class_value = parent_context.parent_context.create_value(parent_context.tree_node)
                instance = value.AnonymousInstance(self.inference_state, parent_context.parent_context, class_value)
                func = value.BoundMethod(instance=instance, class_context=class_value.as_context(), function=func)
            return func
        elif node.type == 'classdef':
            return value.ClassValue(self.inference_state, parent_context, node)
        else:
            raise NotImplementedError("Probably shouldn't happen: %s" % node)

    def create_context(self, node: Any) -> AbstractContext:
        def from_scope_node(scope_node: Any, is_nested: bool = True) -> AbstractContext:
            if scope_node == self.tree_node:
                return self
            if scope_node.type in ('funcdef', 'lambdef', 'classdef'):
                return self.create_value(scope_node).as_context()
            elif scope_node.type in ('comp_for', 'sync_comp_for'):
                parent_context = from_scope_node(parent_scope(scope_node.parent))
                if node.start_pos >= scope_node.children[-1].start_pos:
                    return parent_context
                return CompForContext(parent_context, scope_node)
            raise Exception("There's a scope that was not managed: %s" % scope_node)

        def parent_scope(node: Any) -> Any:
            while True:
                node = node.parent
                if parser_utils.is_scope(node):
                    return node
                elif node.type in ('argument', 'testlist_comp'):
                    if node.children[1].type in ('comp_for', 'sync_comp_for'):
                        return node.children[1]
                elif node.type == 'dictorsetmaker':
                    for n in node.children[1:4]:
                        if n.type in ('comp_for', 'sync_comp_for'):
                            return n

        scope_node = parent_scope(node)
        if scope_node.type in ('funcdef', 'classdef'):
            colon = scope_node.children[scope_node.children.index(':')]
            if node.start_pos < colon.start_pos:
                parent = node.parent
                if not (parent.type == 'param' and parent.name == node):
                    scope_node = parent_scope(scope_node)
        return from_scope_node(scope_node, is_nested=True)

    def create_name(self, tree_name: Name) -> Union[AnonymousParamName, TreeNameDefinition]:
        definition = tree_name.get_definition()
        if definition and definition.type == 'param' and (definition.name == tree_name):
            funcdef = search_ancestor(definition, 'funcdef', 'lambdef')
            func = self.create_value(funcdef)
            return AnonymousParamName(func, tree_name)
        else:
            context = self.create_context(tree_name)
            return TreeNameDefinition(context, tree_name)


class FunctionContext(TreeContextMixin, ValueContext):
    def get_filters(
        self,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional[Name] = None,
    ) -> Iterator[ParserTreeFilter]:
        yield ParserTreeFilter(self.inference_state, parent_context=self, until_position=until_position, origin_scope=origin_scope)


class ModuleContext(TreeContextMixin, ValueContext):
    def py__file__(self) -> Optional[str]:
        return self._value.py__file__()

    def get_filters(
        self,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional[Name] = None,
    ) -> Iterator[Any]:
        filters = self._value.get_filters(origin_scope)
        next(filters, None)
        yield MergedFilter(ParserTreeFilter(parent_context=self, until_position=until_position, origin_scope=origin_scope), self.get_global_filter())
        yield from filters

    def get_global_filter(self) -> GlobalNameFilter:
        return GlobalNameFilter(self)

    @property
    def string_names(self) -> Sequence[str]:
        return self._value.string_names

    @property
    def code_lines(self) -> Sequence[str]:
        return self._value.code_lines

    def get_value(self) -> Any:
        """
        This is the only function that converts a context back to a value.
        This is necessary for stub -> python conversion and vice versa. However
        this method shouldn't be moved to AbstractContext.
        """
        return self._value


class NamespaceContext(TreeContextMixin, ValueContext):
    def get_filters(
        self,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional[Name] = None,
    ) -> Iterator[Any]:
        return self._value.get_filters()

    def get_value(self) -> Any:
        return self._value

    @property
    def string_names(self) -> Sequence[str]:
        return self._value.string_names

    def py__file__(self) -> Optional[str]:
        return self._value.py__file__()


class ClassContext(TreeContextMixin, ValueContext):
    def get_filters(
        self,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional[Name] = None,
    ) -> Iterator[ParserTreeFilter]:
        yield self.get_global_filter(until_position, origin_scope)

    def get_global_filter(
        self,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional[Name] = None,
    ) -> ParserTreeFilter:
        return ParserTreeFilter(parent_context=self, until_position=until_position, origin_scope=origin_scope)


class CompForContext(TreeContextMixin, AbstractContext):
    def __init__(self, parent_context: AbstractContext, comp_for: Any):
        super().__init__(parent_context.inference_state)
        self.tree_node: Any = comp_for
        self.parent_context: AbstractContext = parent_context

    def get_filters(
        self,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional[Name] = None,
    ) -> Iterator[ParserTreeFilter]:
        yield ParserTreeFilter(self)

    def get_value(self) -> None:
        return None

    def py__name__(self) -> str:
        return '<comprehension context>'

    def __repr__(self) -> str:
        return '%s(%s)' % (self.__class__.__name__, self.tree_node)


class CompiledContext(ValueContext):
    def get_filters(
        self,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional[Name] = None,
    ) -> Iterator[Any]:
        return self._value.get_filters()


class CompiledModuleContext(CompiledContext):
    code_lines: Optional[Sequence[str]] = None

    def get_value(self) -> Any:
        return self._value

    @property
    def string_names(self) -> Sequence[str]:
        return self._value.string_names

    def py__file__(self) -> Optional[str]:
        return self._value.py__file__()


def _get_global_filters_for_name(
    context: AbstractContext,
    name_or_none: Optional[Name],
    position: Optional[Tuple[int, int]],
) -> Iterator[Any]:
    if name_or_none is not None:
        ancestor: Any = search_ancestor(name_or_none, 'funcdef', 'classdef', 'lambdef')
        lambdef: Optional[Any] = None
        if ancestor == 'lambdef':
            lambdef = ancestor
            ancestor = search_ancestor(name_or_none, 'funcdef', 'classdef')
        if ancestor is not None:
            colon = ancestor.children[-2]
            if position is not None and position < colon.start_pos:
                if lambdef is None or position < lambdef.children[-2].start_pos:
                    position = ancestor.start_pos
    return get_global_filters(context, position, name_or_none)


def get_global_filters(
    context: AbstractContext,
    until_position: Optional[Tuple[int, int]],
    origin_scope: Optional[Name],
) -> Iterator[Any]:
    """
    Returns all filters in order of priority for name resolution.

    For global name lookups. The filters will handle name resolution
    themselves, but here we gather possible filters downwards.

    >>> from jedi import Script
    >>> script = Script('''
    ... x = ['a', 'b', 'c']
    ... def func():
    ...     y = None
    ... ''')
    >>> module_node = script._module_node
    >>> scope = next(module_node.iter_funcdefs())
    >>> scope
    <Function: func@3-5>
    >>> context = script._get_module_context().create_context(scope)
    >>> filters = list(get_global_filters(context, (4, 0), None))

    First we get the names from the function scope.

    >>> print(filters[0])  # doctest: +ELLIPSIS
    MergedFilter(<ParserTreeFilter: ...>, <GlobalNameFilter: ...>)
    >>> sorted(str(n) for n in filters[0].values())  # doctest: +NORMALIZE_WHITESPACE
    ['<TreeNameDefinition: string_name=func start_pos=(3, 4)>',
     '<TreeNameDefinition: string_name=x start_pos=(2, 0)>']
    >>> filters[0]._filters[0]._until_position
    (4, 0)
    >>> filters[0]._filters[1]._until_position

    Then it yields the names from one level "lower". In this example, this is
    the module scope (including globals).
    As a side note, you can see, that the position in the filter is None on the
    globals filter, because there the whole module is searched.

    >>> list(filters[1].values())  # package modules -> Also empty.
    []
    >>> sorted(name.string_name for name in filters[2].values())  # Module attributes
    ['__doc__', '__name__', '__package__']

    Finally, it yields the builtin filter, if `include_builtin` is
    true (default).

    >>> list(filters[3].values())  # doctest: +ELLIPSIS
    [...]
    """
    base_context = context
    from jedi.inference.value.function import BaseFunctionExecutionContext
    while context is not None:
        yield from context.get_filters(until_position=until_position, origin_scope=origin_scope)
        if isinstance(context, (BaseFunctionExecutionContext, ModuleContext)):
            until_position = None
        context = context.parent_context
    b = next(base_context.inference_state.builtins_module.get_filters(), None)
    assert b is not None
    yield b