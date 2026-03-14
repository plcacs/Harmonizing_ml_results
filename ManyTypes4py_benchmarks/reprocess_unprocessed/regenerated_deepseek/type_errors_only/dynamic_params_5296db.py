"""
One of the really important features of |jedi| is to have an option to
understand code like this::

    def foo(bar):
        bar. # completion here
    foo(1)

There's no doubt wheter bar is an ``int`` or not, but if there's also a call
like ``foo('str')``, what would happen? Well, we'll just show both. Because
that's what a human would expect.

It works as follows:

- |Jedi| sees a param
- search for function calls named ``foo``
- execute these calls and check the input.
"""
from typing import Optional, Iterator, Generator, List, Tuple, Any, TYPE_CHECKING
from jedi import settings
from jedi import debug
from jedi.parser_utils import get_parent_scope
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.arguments import TreeArguments
from jedi.inference.param import get_executed_param_names
from jedi.inference.helpers import is_stdlib_path
from jedi.inference.utils import to_list
from jedi.inference.value import instance
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.references import get_module_contexts_containing_name
from jedi.inference import recursion

if TYPE_CHECKING:
    from jedi.inference.value import AbstractInstanceContext, TreeInstance
    from jedi.inference.base_value import Value
    from jedi.inference.context import ModuleContext, Context
    from jedi.inference.arguments import AbstractArguments
    from jedi.inference.value.function import BaseFunctionExecutionContext
    from parso.tree import Node, BaseNode
    from parso.python.tree import Name, Lambda, Funcdef, ClassDef

MAX_PARAM_SEARCHES = 20


def _avoid_recursions(func: Any) -> Any:
    def wrapper(function_value: 'Value', param_index: int) -> ValueSet:
        inf = function_value.inference_state
        with recursion.execution_allowed(inf, function_value.tree_node) as allowed:
            if allowed:
                inf.dynamic_params_depth += 1
                try:
                    return func(function_value, param_index)
                finally:
                    inf.dynamic_params_depth -= 1
            return NO_VALUES
    return wrapper


@debug.increase_indent
@_avoid_recursions
def dynamic_param_lookup(function_value: 'Value', param_index: int) -> ValueSet:
    """
    A dynamic search for param values. If you try to complete a type:

    >>> def func(foo):
    ...     foo
    >>> func(1)
    >>> func("")

    It is not known what the type ``foo`` without analysing the whole code. You
    have to look for all calls to ``func`` to find out what ``foo`` possibly
    is.
    """
    funcdef: 'Funcdef' = function_value.tree_node
    if not settings.dynamic_params:
        return NO_VALUES
    path: Optional[str] = function_value.get_root_context().py__file__()
    if path is not None and is_stdlib_path(path):
        return NO_VALUES
    if funcdef.type == 'lambdef':
        string_name: Optional[str] = _get_lambda_name(funcdef)
        if string_name is None:
            return NO_VALUES
    else:
        string_name = funcdef.name.value
    debug.dbg('Dynamic param search in %s.', string_name, color='MAGENTA')
    module_context: 'ModuleContext' = function_value.get_root_context()
    arguments_list: List['AbstractArguments'] = _search_function_arguments(
        module_context, funcdef, string_name
    )
    values = ValueSet.from_sets(
        (
            get_executed_param_names(function_value, arguments)[param_index].infer()
            for arguments in arguments_list
        )
    )
    debug.dbg('Dynamic param result finished', color='MAGENTA')
    return values


@inference_state_method_cache(default=None)
@to_list
def _search_function_arguments(
    module_context: 'ModuleContext',
    funcdef: 'Funcdef',
    string_name: str
) -> List['AbstractArguments']:
    """
    Returns a list of param names.
    """
    compare_node: 'BaseNode' = funcdef
    if string_name == '__init__':
        cls: 'ClassDef' = get_parent_scope(funcdef)
        if cls.type == 'classdef':
            string_name = cls.name.value
            compare_node = cls
    found_arguments: bool = False
    i: int = 0
    inference_state = module_context.inference_state
    if settings.dynamic_params_for_other_modules:
        module_contexts: List['ModuleContext'] = get_module_contexts_containing_name(
            inference_state, [module_context], string_name, limit_reduction=5
        )
    else:
        module_contexts = [module_context]
    for for_mod_context in module_contexts:
        for name, trailer in _get_potential_nodes(for_mod_context, string_name):
            i += 1
            if i * inference_state.dynamic_params_depth > MAX_PARAM_SEARCHES:
                return []
            random_context: 'Context' = for_mod_context.create_context(name)
            for arguments in _check_name_for_execution(
                inference_state, random_context, compare_node, name, trailer
            ):
                found_arguments = True
                yield arguments
        if found_arguments:
            return []
    return []


def _get_lambda_name(node: 'Lambda') -> Optional[str]:
    stmt: 'Node' = node.parent
    if stmt.type == 'expr_stmt':
        first_operator = next(stmt.yield_operators(), None)
        if first_operator == '=':
            first: 'Node' = stmt.children[0]
            if first.type == 'name':
                return first.value
    return None


def _get_potential_nodes(
    module_value: 'ModuleContext',
    func_string_name: str
) -> Generator[Tuple['Name', 'Node'], None, None]:
    try:
        names: List['Name'] = module_value.tree_node.get_used_names()[func_string_name]
    except KeyError:
        return
    for name in names:
        bracket: 'Node' = name.get_next_leaf()
        trailer: 'Node' = bracket.parent
        if trailer.type == 'trailer' and bracket == '(':
            yield (name, trailer)


def _check_name_for_execution(
    inference_state: Any,
    context: 'Context',
    compare_node: 'BaseNode',
    name: 'Name',
    trailer: 'Node'
) -> Generator['AbstractArguments', None, None]:
    from jedi.inference.value.function import BaseFunctionExecutionContext

    def create_args(value: 'Value') -> 'AbstractArguments':
        arglist: Optional['Node'] = trailer.children[1]
        if arglist == ')':
            arglist = None
        args: TreeArguments = TreeArguments(inference_state, context, arglist, trailer)
        from jedi.inference.value.instance import InstanceArguments
        if value.tree_node.type == 'classdef':
            created_instance: 'TreeInstance' = instance.TreeInstance(
                inference_state, value.parent_context, value, args
            )
            return InstanceArguments(created_instance, args)
        else:
            if value.is_bound_method():
                args = InstanceArguments(value.instance, args)
            return args

    for value in inference_state.infer(context, name):
        value_node: 'BaseNode' = value.tree_node
        if compare_node == value_node:
            yield create_args(value)
        elif isinstance(value.parent_context, BaseFunctionExecutionContext) and compare_node.type == 'funcdef':
            param_names = value.parent_context.get_param_names()
            if len(param_names) != 1:
                continue
            values: ValueSet = param_names[0].infer()
            if [v.tree_node for v in values] == [compare_node]:
                module_context: 'ModuleContext' = context.get_root_context()
                execution_context: 'BaseFunctionExecutionContext' = value.as_context(create_args(value))
                potential_nodes = _get_potential_nodes(module_context, param_names[0].string_name)
                for name, trailer in potential_nodes:
                    if value_node.start_pos < name.start_pos < value_node.end_pos:
                        random_context: 'Context' = execution_context.create_context(name)
                        yield from _check_name_for_execution(
                            inference_state, random_context, compare_node, name, trailer
                        )