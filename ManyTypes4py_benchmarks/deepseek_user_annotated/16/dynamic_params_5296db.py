from typing import Optional, Iterator, List, Tuple, Generator, Set, Dict, Any
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
from jedi.inference.value.function import BaseFunctionExecutionContext
from jedi.inference.base_value import Value
from jedi.inference.context import ModuleContext, Context
from jedi.inference import InferenceState
from jedi.api.interpreter import ModuleValue
from jedi.parser_utils import get_cached_parent_scope
from parso.python.tree import FuncDef, Lambda, Name, Node, ExprStmt, ClassDef


MAX_PARAM_SEARCHES = 20


def _avoid_recursions(func: Any) -> Any:
    def wrapper(function_value: Value, param_index: int) -> ValueSet:
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
def dynamic_param_lookup(function_value: Value, param_index: int) -> ValueSet:
    funcdef = function_value.tree_node

    if not settings.dynamic_params:
        return NO_VALUES

    path = function_value.get_root_context().py__file__()
    if path is not None and is_stdlib_path(path):
        return NO_VALUES

    string_name: Optional[str]
    if funcdef.type == 'lambdef':
        string_name = _get_lambda_name(funcdef)
        if string_name is None:
            return NO_VALUES
    else:
        string_name = funcdef.name.value
    debug.dbg('Dynamic param search in %s.', string_name, color='MAGENTA')

    module_context = function_value.get_root_context()
    arguments_list = _search_function_arguments(module_context, funcdef, string_name)
    values = ValueSet.from_sets(
        get_executed_param_names(
            function_value, arguments
        )[param_index].infer()
        for arguments in arguments_list
    )
    debug.dbg('Dynamic param result finished', color='MAGENTA')
    return values


@inference_state_method_cache(default=None)
@to_list
def _search_function_arguments(
    module_context: ModuleContext,
    funcdef: Node,
    string_name: str
) -> List[TreeArguments]:
    compare_node = funcdef
    if string_name == '__init__':
        cls = get_parent_scope(funcdef)
        if cls.type == 'classdef':
            string_name = cls.name.value
            compare_node = cls

    found_arguments = False
    i = 0
    inference_state = module_context.inference_state

    if settings.dynamic_params_for_other_modules:
        module_contexts = get_module_contexts_containing_name(
            inference_state, [module_context], string_name,
            limit_reduction=5,
        )
    else:
        module_contexts = [module_context]

    for for_mod_context in module_contexts:
        for name, trailer in _get_potential_nodes(for_mod_context, string_name):
            i += 1

            if i * inference_state.dynamic_params_depth > MAX_PARAM_SEARCHES:
                return []

            random_context = for_mod_context.create_context(name)
            for arguments in _check_name_for_execution(
                    inference_state, random_context, compare_node, name, trailer):
                found_arguments = True
                yield arguments

        if found_arguments:
            return []


def _get_lambda_name(node: Lambda) -> Optional[str]:
    stmt = node.parent
    if stmt.type == 'expr_stmt':
        first_operator = next(stmt.yield_operators(), None)
        if first_operator == '=':
            first = stmt.children[0]
            if first.type == 'name':
                return first.value
    return None


def _get_potential_nodes(
    module_value: ModuleValue,
    func_string_name: str
) -> Generator[Tuple[Name, Node], None, None]:
    try:
        names = module_value.tree_node.get_used_names()[func_string_name]
    except KeyError:
        return

    for name in names:
        bracket = name.get_next_leaf()
        trailer = bracket.parent
        if trailer.type == 'trailer' and bracket == '(':
            yield name, trailer


def _check_name_for_execution(
    inference_state: InferenceState,
    context: Context,
    compare_node: Node,
    name: Name,
    trailer: Node
) -> Generator[TreeArguments, None, None]:
    def create_args(value: Value) -> TreeArguments:
        arglist = trailer.children[1]
        if arglist == ')':
            arglist = None
        args = TreeArguments(inference_state, context, arglist, trailer)
        if value.tree_node.type == 'classdef':
            created_instance = instance.TreeInstance(
                inference_state,
                value.parent_context,
                value,
                args
            )
            return InstanceArguments(created_instance, args)
        else:
            if value.is_bound_method():
                args = InstanceArguments(value.instance, args)
            return args

    for value in inference_state.infer(context, name):
        value_node = value.tree_node
        if compare_node == value_node:
            yield create_args(value)
        elif isinstance(value.parent_context, BaseFunctionExecutionContext) \
                and compare_node.type == 'funcdef':
            param_names = value.parent_context.get_param_names()
            if len(param_names) != 1:
                continue
            values = param_names[0].infer()
            if [v.tree_node for v in values] == [compare_node]:
                module_context = context.get_root_context()
                execution_context = value.as_context(create_args(value))
                potential_nodes = _get_potential_nodes(module_context, param_names[0].string_name)
                for name, trailer in potential_nodes:
                    if value_node.start_pos < name.start_pos < value_node.end_pos:
                        random_context = execution_context.create_context(name)
                        yield from _check_name_for_execution(
                            inference_state,
                            random_context,
                            compare_node,
                            name,
                            trailer
                        )
