from jedi import settings
from jedi import debug
from jedi.parser_utils import get_parent_scope
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.arguments import TreeArguments
from jedi.inference.param import get_executed_param_names
from jedi.inference.helpers import is_stdlib_path
from jedi.inference.utils import to_list
from jedi.inference.value import instance, Value, ValueSet, NO_VALUES
from jedi.inference.base_context import Context
from jedi.inference.base_value import ValueSet
from jedi.inference.references import get_module_contexts_containing_name
from jedi.inference import recursion
from jedi.parser import TreeNode
from jedi.api.classes import FunctionExecutionContext
from typing import Callable, Iterator, Optional, Tuple, List, Any

MAX_PARAM_SEARCHES: int = 20

def _avoid_recursions(func: Callable[['Value', int], ValueSet]) -> Callable[['Value', int], ValueSet]:
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
    funcdef: TreeNode = function_value.tree_node
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
        string_name: str = funcdef.name.value
    debug.dbg('Dynamic param search in %s.', string_name, color='MAGENTA')
    module_context: Context = function_value.get_root_context()
    arguments_list: List[TreeArguments] = _search_function_arguments(module_context, funcdef, string_name)
    values: ValueSet = ValueSet.from_sets(
        (get_executed_param_names(function_value, arguments).param_values[param_index].infer() for arguments in arguments_list)
    )
    debug.dbg('Dynamic param result finished', color='MAGENTA')
    return values

@inference_state_method_cache(default=None)
@to_list
def _search_function_arguments(module_context: Context, funcdef: TreeNode, string_name: str) -> Iterator[TreeArguments]:
    """
    Returns a list of param names.
    """
    compare_node: TreeNode = funcdef
    if string_name == '__init__':
        cls: TreeNode = get_parent_scope(funcdef)
        if cls.type == 'classdef':
            string_name = cls.name.value
            compare_node = cls
    found_arguments: bool = False
    i: int = 0
    inference_state = module_context.inference_state
    if settings.dynamic_params_for_other_modules:
        module_contexts: List[Context] = get_module_contexts_containing_name(
            inference_state, [module_context], string_name, limit_reduction=5
        )
    else:
        module_contexts = [module_context]
    for for_mod_context in module_contexts:
        for name, trailer in _get_potential_nodes(for_mod_context, string_name):
            i += 1
            if i * inference_state.dynamic_params_depth > MAX_PARAM_SEARCHES:
                return
            random_context: Context = for_mod_context.create_context(name)
            for arguments in _check_name_for_execution(inference_state, random_context, compare_node, name, trailer):
                found_arguments = True
                yield arguments
        if found_arguments:
            return

def _get_lambda_name(node: TreeNode) -> Optional[str]:
    stmt: TreeNode = node.parent
    if stmt.type == 'expr_stmt':
        first_operator: Optional[str] = next(stmt.yield_operators(), None)
        if first_operator == '=':
            first = stmt.children[0]
            if first.type == 'name':
                return first.value
    return None

def _get_potential_nodes(module_value: Context, func_string_name: str) -> Iterator[Tuple[Any, Any]]:
    try:
        names = module_value.tree_node.get_used_names()[func_string_name]
    except KeyError:
        return
    for name in names:
        bracket = name.get_next_leaf()
        trailer = bracket.parent if bracket else None
        if trailer and trailer.type == 'trailer' and bracket.value == '(':
            yield (name, trailer)

def _check_name_for_execution(
    inference_state: Any, 
    context: Context, 
    compare_node: TreeNode, 
    name: Any, 
    trailer: Any
) -> Iterator[TreeArguments]:
    from jedi.inference.value.function import BaseFunctionExecutionContext

    def create_args(value: 'Value') -> TreeArguments:
        arglist = trailer.children[1]
        if arglist.value == ')':
            arglist = None
        args = TreeArguments(inference_state, context, arglist, trailer)
        from jedi.inference.value.instance import InstanceArguments
        if value.tree_node.type == 'classdef':
            created_instance = instance.TreeInstance(inference_state, value.parent_context, value, args)
            return InstanceArguments(created_instance, args)
        else:
            if value.is_bound_method():
                args = InstanceArguments(value.instance, args)
            return args

    for value in inference_state.infer(context, name):
        value_node: TreeNode = value.tree_node
        if compare_node == value_node:
            yield create_args(value)
        elif isinstance(value.parent_context, BaseFunctionExecutionContext) and compare_node.type == 'funcdef':
            param_names = value.parent_context.get_param_names()
            if len(param_names) != 1:
                continue
            values = param_names[0].infer()
            if [v.tree_node for v in values] == [compare_node]:
                module_context: Context = context.get_root_context()
                execution_context: FunctionExecutionContext = value.as_context(create_args(value))
                potential_nodes = _get_potential_nodes(module_context, param_names[0].string_name)
                for name, trailer in potential_nodes:
                    if compare_node.start_pos < name.start_pos < compare_node.end_pos:
                        random_context: Context = execution_context.create_context(name)
                        yield from _check_name_for_execution(inference_state, random_context, compare_node, name, trailer)
