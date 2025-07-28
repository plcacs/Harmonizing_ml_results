#!/usr/bin/env python3
from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
from typing import Any, List, Optional, Tuple, Dict, Iterator, Generator

_DEFINITION_SCOPES: Tuple[str, ...] = ('suite', 'file_input')
_VARIABLE_EXCTRACTABLE: List[str] = EXPRESSION_PARTS + 'atom testlist_star_expr testlist test lambdef lambdef_nocond keyword name number string fstring'.split()

def extract_variable(inference_state: Any, path: str, module_node: Any, name: str, pos: Tuple[int, int], until_pos: Optional[Tuple[int, int]]) -> Refactoring:
    nodes: List[Any] = _find_nodes(module_node, pos, until_pos)
    debug.dbg('Extracting nodes: %s', nodes)
    is_expression, message = _is_expression_with_error(nodes)
    if not is_expression:
        raise RefactoringError(message)
    generated_code: str = name + ' = ' + _expression_nodes_to_string(nodes)
    file_to_node_changes: Dict[str, Dict[Any, str]] = {path: _replace(nodes, name, generated_code, pos)}
    return Refactoring(inference_state, file_to_node_changes)

def _is_expression_with_error(nodes: List[Any]) -> Tuple[bool, str]:
    """
    Returns a tuple (is_expression, error_string).
    """
    if any((node.type == 'name' and node.is_definition() for node in nodes)):
        return (False, 'Cannot extract a name that defines something')
    if nodes[0].type not in _VARIABLE_EXCTRACTABLE:
        return (False, 'Cannot extract a "%s"' % nodes[0].type)
    return (True, '')

def _find_nodes(module_node: Any, pos: Tuple[int, int], until_pos: Optional[Tuple[int, int]]) -> List[Any]:
    """
    Looks up a module and tries to find the appropriate amount of nodes that
    are in there.
    """
    start_node: Any = module_node.get_leaf_for_position(pos, include_prefixes=True)
    if until_pos is None:
        if start_node.type == 'operator':
            next_leaf = start_node.get_next_leaf()
            if next_leaf is not None and next_leaf.start_pos == pos:
                start_node = next_leaf
        if _is_not_extractable_syntax(start_node):
            start_node = start_node.parent
        if start_node.parent.type == 'trailer':
            start_node = start_node.parent.parent
        while start_node.parent.type in EXPRESSION_PARTS:
            start_node = start_node.parent
        nodes: List[Any] = [start_node]
    else:
        if start_node.end_pos == pos:
            next_leaf = start_node.get_next_leaf()
            if next_leaf is not None:
                start_node = next_leaf
        if _is_not_extractable_syntax(start_node):
            start_node = start_node.parent
        end_leaf: Any = module_node.get_leaf_for_position(until_pos, include_prefixes=True)
        if end_leaf.start_pos > until_pos:
            end_leaf = end_leaf.get_previous_leaf()
            if end_leaf is None:
                raise RefactoringError('Cannot extract anything from that')
        parent_node: Any = start_node
        while parent_node.end_pos < end_leaf.end_pos:
            parent_node = parent_node.parent
        nodes = _remove_unwanted_expression_nodes(parent_node, pos, until_pos)
    if len(nodes) == 1 and start_node.type in ('return_stmt', 'yield_expr'):
        return [nodes[0].children[1]]
    return nodes

def _replace(nodes: List[Any], expression_replacement: str, extracted: str, pos: Tuple[int, int],
             insert_before_leaf: Optional[Any] = None, remaining_prefix: Optional[str] = None) -> Dict[Any, str]:
    definition: Any = _get_parent_definition(nodes[0])
    if insert_before_leaf is None:
        insert_before_leaf = definition.get_first_leaf()
    first_node_leaf: Any = nodes[0].get_first_leaf()
    lines: List[str] = split_lines(insert_before_leaf.prefix, keepends=True)
    if first_node_leaf is insert_before_leaf:
        if remaining_prefix is not None:
            lines[:-1] = [remaining_prefix]
    lines[-1:-1] = [indent_block(extracted, lines[-1]) + '\n']
    extracted_prefix: str = ''.join(lines)
    replacement_dct: Dict[Any, str] = {}
    if first_node_leaf is insert_before_leaf:
        replacement_dct[nodes[0]] = extracted_prefix + expression_replacement
    else:
        if remaining_prefix is None:
            p: str = first_node_leaf.prefix
        else:
            p = remaining_prefix + _get_indentation(nodes[0])
        replacement_dct[nodes[0]] = p + expression_replacement
        replacement_dct[insert_before_leaf] = extracted_prefix + insert_before_leaf.value
    for node in nodes[1:]:
        replacement_dct[node] = ''
    return replacement_dct

def _expression_nodes_to_string(nodes: List[Any]) -> str:
    return ''.join((n.get_code(include_prefix=i != 0) for i, n in enumerate(nodes)))

def _suite_nodes_to_string(nodes: List[Any], pos: Tuple[int, int]) -> Tuple[str, str]:
    n: Any = nodes[0]
    prefix, part_of_code = _split_prefix_at(n.get_first_leaf(), pos[0] - 1)
    code: str = part_of_code + n.get_code(include_prefix=False) + ''.join((n.get_code() for n in nodes[1:]))
    return (prefix, code)

def _split_prefix_at(leaf: Any, until_line: int) -> Tuple[str, str]:
    """
    Returns a tuple of the leaf's prefix, split at the until_line
    position.
    """
    second_line_count: int = leaf.start_pos[0] - until_line
    lines: List[str] = split_lines(leaf.prefix, keepends=True)
    return (''.join(lines[:-second_line_count]), ''.join(lines[-second_line_count:]))

def _get_indentation(node: Any) -> str:
    return split_lines(node.get_first_leaf().prefix)[-1]

def _get_parent_definition(node: Any) -> Any:
    """
    Returns the statement where a node is defined.
    """
    while node is not None:
        if node.parent.type in _DEFINITION_SCOPES:
            return node
        node = node.parent
    raise NotImplementedError('We should never even get here')

def _remove_unwanted_expression_nodes(parent_node: Any, pos: Tuple[int, int], until_pos: Tuple[int, int]) -> List[Any]:
    """
    This function makes it so for `1 * 2 + 3` you can extract `2 + 3`, even
    though it is not part of the expression.
    """
    typ: str = parent_node.type
    is_suite_part: bool = typ in ('suite', 'file_input')
    if typ in EXPRESSION_PARTS or is_suite_part:
        nodes: List[Any] = parent_node.children
        start_index: int = 0
        for i, n in enumerate(nodes):
            if n.end_pos > pos:
                start_index = i
                if n.type == 'operator':
                    start_index -= 1
                break
        end_index: int = len(nodes) - 1
        for i, n in reversed(list(enumerate(nodes))):
            if n.start_pos < until_pos:
                end_index = i
                if n.type == 'operator':
                    end_index += 1
                for n2 in nodes[i:]:
                    if _is_not_extractable_syntax(n2):
                        end_index += 1
                    else:
                        break
                break
        nodes = nodes[start_index:end_index + 1]
        if not is_suite_part:
            nodes[0:1] = _remove_unwanted_expression_nodes(nodes[0], pos, until_pos)
            nodes[-1:] = _remove_unwanted_expression_nodes(nodes[-1], pos, until_pos)
        return nodes
    return [parent_node]

def _is_not_extractable_syntax(node: Any) -> bool:
    return node.type == 'operator' or (node.type == 'keyword' and node.value not in ('None', 'True', 'False'))

def extract_function(inference_state: Any, path: str, module_context: Any, name: str, pos: Tuple[int, int], until_pos: Tuple[int, int]) -> Refactoring:
    nodes: List[Any] = _find_nodes(module_context.tree_node, pos, until_pos)
    assert len(nodes)
    is_expression, _ = _is_expression_with_error(nodes)
    context: Any = module_context.create_context(nodes[0])
    is_bound_method: bool = context.is_bound_method()
    params, return_variables = list(_find_inputs_and_outputs(module_context, context, nodes))
    if context.is_module():
        insert_before_leaf: Optional[Any] = None
    else:
        node: Any = _get_code_insertion_node(context.tree_node, is_bound_method)
        insert_before_leaf = node.get_first_leaf()
    if is_expression:
        code_block: str = 'return ' + _expression_nodes_to_string(nodes) + '\n'
        remaining_prefix: Optional[str] = None
        has_ending_return_stmt: bool = False
    else:
        has_ending_return_stmt = _is_node_ending_return_stmt(nodes[-1])
        if not has_ending_return_stmt:
            # If return_variables is empty, fallback to empty list.
            return_variables = list(_find_needed_output_variables(context, nodes[0].parent, nodes[-1].end_pos, return_variables)) or ([return_variables[-1]] if return_variables else [])
        remaining_prefix, code_block = _suite_nodes_to_string(nodes, pos)
        after_leaf: Any = nodes[-1].get_next_leaf()
        first: str
        second: str
        first, second = _split_prefix_at(after_leaf, until_pos[0])
        code_block += first
        code_block = dedent(code_block)
        if not has_ending_return_stmt:
            output_var_str: str = ', '.join(return_variables)
            code_block += 'return ' + output_var_str + '\n'
    _check_for_non_extractables(nodes[:-1] if has_ending_return_stmt else nodes)
    decorator: str = ''
    self_param: Optional[str] = None
    if is_bound_method:
        if not function_is_staticmethod(context.tree_node):
            function_param_names = context.get_value().get_param_names()
            if len(function_param_names):
                self_param = function_param_names[0].string_name
                params = [p for p in params if p != self_param]
        if function_is_classmethod(context.tree_node):
            decorator = '@classmethod\n'
    else:
        code_block += '\n'
    function_code: str = '%sdef %s(%s):\n%s' % (
        decorator, 
        name, 
        ', '.join(params if self_param is None else [self_param] + params),
        indent_block(code_block)
    )
    function_call: str = '%s(%s)' % ((('' if self_param is None else self_param + '.') + name), ', '.join(params))
    if is_expression:
        replacement: str = function_call
    elif has_ending_return_stmt:
        replacement = 'return ' + function_call + '\n'
    else:
        output_var_str = ', '.join(return_variables)  # Recalculate output_var_str if needed.
        replacement = output_var_str + ' = ' + function_call + '\n'
    replacement_dct: Dict[Any, str] = _replace(nodes, replacement, function_code, pos, insert_before_leaf, remaining_prefix)
    if not is_expression:
        after_leaf = nodes[-1].get_next_leaf()
        replacement_dct[after_leaf] = second + after_leaf.value
    file_to_node_changes: Dict[str, Dict[Any, str]] = {path: replacement_dct}
    return Refactoring(inference_state, file_to_node_changes)

def _check_for_non_extractables(nodes: List[Any]) -> None:
    for n in nodes:
        try:
            children = n.children
        except AttributeError:
            if n.value == 'return':
                raise RefactoringError('Can only extract return statements if they are at the end.')
            if n.value == 'yield':
                raise RefactoringError('Cannot extract yield statements.')
        else:
            _check_for_non_extractables(children)

def _is_name_input(module_context: Any, names: List[Any], first: Tuple[int, int], last: Tuple[int, int]) -> bool:
    for name in names:
        if name.api_type == 'param' or not name.parent_context.is_module():
            if name.get_root_context() is not module_context:
                return True
            if name.start_pos is None or not (first <= name.start_pos < last):
                return True
    return False

def _find_inputs_and_outputs(module_context: Any, context: Any, nodes: List[Any]) -> Tuple[List[str], List[str]]:
    first: Tuple[int, int] = nodes[0].start_pos
    last: Tuple[int, int] = nodes[-1].end_pos
    inputs: List[str] = []
    outputs: List[str] = []
    for name in _find_non_global_names(nodes):
        if name.is_definition():
            if name not in outputs:
                outputs.append(name.value)
        elif name.value not in inputs:
            name_definitions = context.goto(name, name.start_pos)
            if not name_definitions or _is_name_input(module_context, name_definitions, first, last):
                inputs.append(name.value)
    return (inputs, outputs)

def _find_non_global_names(nodes: List[Any]) -> Iterator[Any]:
    for node in nodes:
        try:
            children = node.children
        except AttributeError:
            if node.type == 'name':
                yield node
        else:
            if node.type == 'trailer' and node.children[0] == '.':
                continue
            yield from _find_non_global_names(children)

def _get_code_insertion_node(node: Any, is_bound_method: bool) -> Any:
    if not is_bound_method or function_is_staticmethod(node):
        while node.parent.type != 'file_input':
            node = node.parent
    while node.parent.type in ('async_funcdef', 'decorated', 'async_stmt'):
        node = node.parent
    return node

def _find_needed_output_variables(context: Any, search_node: Any, at_least_pos: Tuple[int, int], return_variables: List[str]) -> Generator[str, None, None]:
    """
    Searches everything after at_least_pos in a node and checks if any of the
    return_variables are used in there and returns those.
    """
    for node in search_node.children:
        if node.start_pos < at_least_pos:
            continue
        vars_set = set(return_variables)
        for name in _find_non_global_names([node]):
            if not name.is_definition() and name.value in vars_set:
                vars_set.remove(name.value)
                yield name.value

def _is_node_ending_return_stmt(node: Any) -> bool:
    t: str = node.type
    if t == 'simple_stmt':
        return _is_node_ending_return_stmt(node.children[0])
    return t == 'return_stmt'
