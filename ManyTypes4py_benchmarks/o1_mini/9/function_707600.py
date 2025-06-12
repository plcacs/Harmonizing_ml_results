from parso.python import tree
from jedi import debug
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import docstrings
from jedi.inference import flow_analysis
from jedi.inference.signature import TreeSignature
from jedi.inference.filters import ParserTreeFilter, FunctionExecutionFilter, AnonymousFunctionExecutionFilter
from jedi.inference.names import ValueName, AbstractNameDefinition, AnonymousParamName, ParamName, NameWrapper
from jedi.inference.base_value import (
    ContextualizedNode,
    NO_VALUES,
    ValueSet,
    TreeValue,
    ValueWrapper,
)
from jedi.inference.lazy_value import LazyKnownValues, LazyKnownValue, LazyTreeValue
from jedi.inference.context import ValueContext, TreeContextMixin
from jedi.inference.value import iterable
from jedi import parser_utils
from jedi.inference.parser_cache import get_yield_exprs
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.gradual.generics import TupleGenericManager
from typing import Optional, List, Tuple, Union, Iterator, Callable, Type, Any, Iterable


class LambdaName(AbstractNameDefinition):
    string_name: str = '<lambda>'
    api_type: str = 'function'

    _lambda_value: 'FunctionValue'
    parent_context: ValueContext

    def __init__(self, lambda_value: 'FunctionValue') -> None:
        self._lambda_value = lambda_value
        self.parent_context = lambda_value.parent_context

    @property
    def start_pos(self) -> Tuple[int, int]:
        return self._lambda_value.tree_node.start_pos

    def infer(self) -> ValueSet:
        return ValueSet([self._lambda_value])


class FunctionAndClassBase(TreeValue):

    def get_qualified_names(self) -> Optional[Tuple[str, ...]]:
        if self.parent_context.is_class():
            n = self.parent_context.get_qualified_names()
            if n is None:
                return None
            return n + (self.py__name__(),)
        elif self.parent_context.is_module():
            return (self.py__name__(),)
        else:
            return None


class FunctionMixin:
    api_type: str = 'function'

    def get_filters(
        self, origin_scope: Optional['TreeContextMixin'] = None
    ) -> Iterator['FunctionExecutionFilter']:
        cls = self.py__class__()
        for instance in cls.execute_with_values():
            yield from instance.get_filters(origin_scope=origin_scope)

    def py__get__(
        self, instance: Optional['ValueSet'], class_value: 'ValueSet'
    ) -> ValueSet:
        from jedi.inference.value.instance import BoundMethod
        if instance is None:
            return ValueSet([self])
        return ValueSet([BoundMethod(instance, class_value.as_context(), self)])

    def get_param_names(self) -> List[AnonymousParamName]:
        return [
            AnonymousParamName(self, param.name)
            for param in self.tree_node.get_params()
        ]

    @property
    def name(self) -> Union[LambdaName, ValueName]:
        if self.tree_node.type == 'lambdef':
            return LambdaName(self)
        return ValueName(self, self.tree_node.name)

    def is_function(self) -> bool:
        return True

    def py__name__(self) -> str:
        return self.name.string_name

    def get_type_hint(self, add_class_info: bool = True) -> str:
        return_annotation = self.tree_node.annotation
        if return_annotation is None:

            def param_name_to_str(n: AnonymousParamName) -> str:
                s = n.string_name
                annotation = n.infer().get_type_hint()
                if annotation is not None:
                    s += ': ' + annotation
                if n.default_node is not None:
                    s += '=' + n.default_node.get_code(include_prefix=False)
                return s

            function_execution = self.as_context()
            result = function_execution.infer()
            return_hint = result.get_type_hint()
            body = (
                self.py__name__()
                + '(%s)' % ', '.join(
                    [param_name_to_str(n) for n in function_execution.get_param_names()]
                )
            )
            if return_hint is None:
                return body
        else:
            return_hint = return_annotation.get_code(include_prefix=False)
            body = self.py__name__() + self.tree_node.children[2].get_code(
                include_prefix=False
            )
        return body + ' -> ' + return_hint

    def py__call__(self, arguments: 'Arguments') -> ValueSet:
        function_execution = self.as_context(arguments)
        return function_execution.infer()

    def _as_context(
        self, arguments: Optional['Arguments'] = None
    ) -> 'FunctionExecutionContext':
        if arguments is None:
            return AnonymousFunctionExecution(self)
        return FunctionExecutionContext(self, arguments)

    def get_signatures(self) -> List[TreeSignature]:
        return [TreeSignature(f) for f in self.get_signature_functions()]


class FunctionValue(FunctionMixin, FunctionAndClassBase, metaclass=CachedMetaClass):

    inference_state: 'InferenceState'
    parent_context: ValueContext
    tree_node: tree.Function

    @classmethod
    def from_context(
        cls,
        context: 'ValueContext',
        tree_node: tree.Function,
    ) -> 'FunctionValue':
        def create(t_node: tree.Function) -> 'FunctionValue':
            if context.is_class():
                return MethodValue(
                    context.inference_state,
                    context,
                    parent_context=parent_context,
                    tree_node=t_node,
                )
            else:
                return cls(
                    context.inference_state, parent_context=parent_context, tree_node=t_node
                )

        overloaded_funcs = list(_find_overload_functions(context, tree_node))
        parent_context = context
        while parent_context.is_class() or parent_context.is_instance():
            parent_context = parent_context.parent_context
        function = create(tree_node)
        if overloaded_funcs:
            return OverloadedFunctionValue(
                function, list(reversed([create(f) for f in overloaded_funcs]))
            )
        return function

    def py__class__(self) -> 'ValueSet':
        c, = values_from_qualified_names(self.inference_state, 'types', 'FunctionType')
        return c

    def get_default_param_context(self) -> 'ValueContext':
        return self.parent_context

    def get_signature_functions(self) -> List['FunctionValue']:
        return [self]


class FunctionNameInClass(NameWrapper):

    _class_context: 'ValueContext'
    _name: Union[LambdaName, ValueName]

    def __init__(self, class_context: 'ValueContext', name: Union[LambdaName, ValueName]) -> None:
        super().__init__(name)
        self._class_context = class_context

    def get_defining_qualified_value(self) -> 'ValueSet':
        return self._class_context.get_value()


class MethodValue(FunctionValue):

    class_context: 'ValueContext'

    def __init__(
        self,
        inference_state: 'InferenceState',
        class_context: 'ValueContext',
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(inference_state, *args, **kwargs)
        self.class_context = class_context

    def get_default_param_context(self) -> 'ValueContext':
        return self.class_context

    def get_qualified_names(self) -> Optional[Tuple[str, ...]]:
        names = self.class_context.get_qualified_names()
        if names is None:
            return None
        return names + (self.py__name__(),)

    @property
    def name(self) -> FunctionNameInClass:
        return FunctionNameInClass(self.class_context, super().name)


class BaseFunctionExecutionContext(ValueContext, TreeContextMixin):

    tree_node: tree.Function

    def infer_annotations(self) -> ValueSet:
        raise NotImplementedError

    @inference_state_method_cache(default=NO_VALUES)
    @recursion.execution_recursion_decorator()
    def get_return_values(
        self, check_yields: bool = False
    ) -> 'ValueSet':
        funcdef = self.tree_node
        if funcdef.type == 'lambdef':
            return self.infer_node(funcdef.children[-1])
        if check_yields:
            value_set = NO_VALUES
            returns = get_yield_exprs(self.inference_state, funcdef)
        else:
            value_set = self.infer_annotations()
            if value_set:
                return value_set
            value_set |= docstrings.infer_return_types(self._value)
            returns = funcdef.iter_return_stmts()
        for r in returns:
            if check_yields:
                value_set |= ValueSet.from_sets(
                    (lazy_value.infer() for lazy_value in self._get_yield_lazy_value(r))
                )
            else:
                check = flow_analysis.reachability_check(self, funcdef, r)
                if check is flow_analysis.UNREACHABLE:
                    debug.dbg('Return unreachable: %s', r)
                else:
                    try:
                        children = r.children
                    except AttributeError:
                        ctx = compiled.builtin_from_name(self.inference_state, 'None')
                        value_set |= ValueSet([ctx])
                    else:
                        value_set |= self.infer_node(children[1])
                if check is flow_analysis.REACHABLE:
                    debug.dbg('Return reachable: %s', r)
                    break
        return value_set

    def _get_yield_lazy_value(self, yield_expr: tree.YieldExpr) -> Iterator['LazyValue']:
        if yield_expr.type == 'keyword':
            ctx = compiled.builtin_from_name(self.inference_state, 'None')
            yield LazyKnownValue(ctx)
            return
        node = yield_expr.children[1]
        if node.type == 'yield_arg':
            cn = ContextualizedNode(self, node.children[1])
            yield from cn.infer().iterate(cn)
        else:
            yield LazyTreeValue(self, node)

    @recursion.execution_recursion_decorator(default=iter([]))
    def get_yield_lazy_values(self, is_async: bool = False) -> Iterator['LazyValue']:
        for_parents = [
            (y, tree.search_ancestor(y, 'for_stmt', 'funcdef', 'while_stmt', 'if_stmt'))
            for y in get_yield_exprs(self.inference_state, self.tree_node)
        ]
        yields_order: List[Tuple[Optional[tree.ForStmt], List[tree.YieldExpr]]] = []
        last_for_stmt: Optional[tree.ForStmt] = None
        for yield_, for_stmt in for_parents:
            if for_stmt is None:
                continue
            parent = for_stmt.parent
            if parent.type == 'suite':
                parent = parent.parent
            if (
                for_stmt.type == 'for_stmt'
                and parent == self.tree_node
                and parser_utils.for_stmt_defines_one_name(for_stmt)
            ):
                if for_stmt == last_for_stmt:
                    yields_order[-1][1].append(yield_)
                else:
                    yields_order.append((for_stmt, [yield_]))
            elif for_stmt == self.tree_node:
                yields_order.append((None, [yield_]))
            else:
                types = self.get_return_values(check_yields=True)
                if types:
                    yield LazyKnownValues(types, min=0, max=float('inf'))
                return
            last_for_stmt = for_stmt
        for for_stmt, yields in yields_order:
            if for_stmt is None:
                for yield_ in yields:
                    yield from self._get_yield_lazy_value(yield_)
            else:
                input_node = for_stmt.get_testlist()
                cn = ContextualizedNode(self, input_node)
                ordered = cn.infer().iterate(cn)
                ordered = list(ordered)
                for lazy_value in ordered:
                    dct = {str(for_stmt.children[1].value): lazy_value.infer()}
                    with self.predefine_names(for_stmt, dct):
                        for yield_in_same_for_stmt in yields:
                            yield from self._get_yield_lazy_value(yield_in_same_for_stmt)

    def merge_yield_values(self, is_async: bool = False) -> 'ValueSet':
        return ValueSet.from_sets(
            (lazy_value.infer() for lazy_value in self.get_yield_lazy_values())
        )

    def is_generator(self) -> bool:
        return bool(get_yield_exprs(self.inference_state, self.tree_node))

    def infer(self) -> 'ValueSet':
        """
        Created to be used by inheritance.
        """
        inference_state = self.inference_state
        is_coroutine = self.tree_node.parent.type in ('async_stmt', 'async_funcdef')
        from jedi.inference.gradual.base import GenericClass
        if is_coroutine:
            if self.is_generator():
                async_generator_classes = inference_state.typing_module.py__getattribute__('AsyncGenerator')
                yield_values = self.merge_yield_values(is_async=True)
                generics = (yield_values.py__class__(), NO_VALUES)
                return ValueSet(
                    (
                        GenericClass(c, TupleGenericManager(generics))
                        for c in async_generator_classes
                    )
                ).execute_annotation()
            else:
                async_classes = inference_state.typing_module.py__getattribute__('Coroutine')
                return_values = self.get_return_values()
                generics = (return_values.py__class__(), NO_VALUES, NO_VALUES)
                return ValueSet(
                    (
                        GenericClass(c, TupleGenericManager(generics))
                        for c in async_classes
                    )
                ).execute_annotation()
        elif self.is_generator() and (not self.infer_annotations()):
            return ValueSet([iterable.Generator(inference_state, self)])
        else:
            return self.get_return_values()


class FunctionExecutionContext(BaseFunctionExecutionContext):

    _arguments: 'Arguments'
    _value: 'FunctionValue'

    def __init__(self, function_value: 'FunctionValue', arguments: 'Arguments') -> None:
        super().__init__(function_value)
        self._arguments = arguments

    def get_filters(
        self,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional['TreeContextMixin'] = None,
    ) -> Iterator['FunctionExecutionFilter']:
        yield FunctionExecutionFilter(
            self, self._value, until_position=until_position, origin_scope=origin_scope, arguments=self._arguments
        )

    def infer_annotations(self) -> 'ValueSet':
        from jedi.inference.gradual.annotation import infer_return_types
        return infer_return_types(self._value, self._arguments)

    def get_param_names(self) -> List[ParamName]:
        return [
            ParamName(self._value, param.name, self._arguments)
            for param in self._value.tree_node.get_params()
        ]


class AnonymousFunctionExecution(BaseFunctionExecutionContext):

    _value: 'FunctionValue'

    def infer_annotations(self) -> 'ValueSet':
        return NO_VALUES

    def get_filters(
        self,
        until_position: Optional[Tuple[int, int]] = None,
        origin_scope: Optional['TreeContextMixin'] = None,
    ) -> Iterator['AnonymousFunctionExecutionFilter']:
        yield AnonymousFunctionExecutionFilter(
            self, self._value, until_position=until_position, origin_scope=origin_scope
        )

    def get_param_names(self) -> List[AnonymousParamName]:
        return self._value.get_param_names()


class OverloadedFunctionValue(FunctionMixin, ValueWrapper):

    _wrapped_value: 'FunctionValue'
    _overloaded_functions: List['FunctionValue']

    def __init__(
        self, function: 'FunctionValue', overloaded_functions: List['FunctionValue']
    ) -> None:
        super().__init__(function)
        self._overloaded_functions = overloaded_functions

    def py__call__(self, arguments: 'Arguments') -> 'ValueSet':
        debug.dbg('Execute overloaded function %s', self._wrapped_value, color='BLUE')
        function_executions: List['FunctionExecutionContext'] = []
        for signature in self.get_signatures():
            function_execution = signature.value.as_context(arguments)
            function_executions.append(function_execution)
            if signature.matches_signature(arguments):
                return function_execution.infer()
        if self.inference_state.is_analysis:
            return NO_VALUES
        return ValueSet.from_sets((fe.infer() for fe in function_executions))

    def get_signature_functions(self) -> List['FunctionValue']:
        return self._overloaded_functions

    def get_type_hint(self, add_class_info: bool = True) -> str:
        return 'Union[%s]' % ', '.join(
            (f.get_type_hint() for f in self._overloaded_functions)
        )


def _find_overload_functions(
    context: 'ValueContext', tree_node: tree.Function
) -> Iterator[tree.Function]:
    def _is_overload_decorated(funcdef: tree.Function) -> bool:
        if funcdef.parent.type == 'decorated':
            decorators = funcdef.parent.children[0]
            if decorators.type == 'decorator':
                decorators_list = [decorators]
            else:
                decorators_list = decorators.children
            for decorator in decorators_list:
                dotted_name = decorator.children[1]
                if dotted_name.type == 'name' and dotted_name.value == 'overload':
                    return True
        return False

    if tree_node.type == 'lambdef':
        return
    if _is_overload_decorated(tree_node):
        yield tree_node
    while True:
        filter_ = ParserTreeFilter(context, until_position=tree_node.start_pos)
        names = filter_.get(tree_node.name.value)
        assert isinstance(names, list)
        if not names:
            break
        found = False
        for name in names:
            funcdef = name.tree_name.parent
            if funcdef.type == 'funcdef' and _is_overload_decorated(funcdef):
                tree_node = funcdef
                found = True
                yield funcdef
        if not found:
            break
