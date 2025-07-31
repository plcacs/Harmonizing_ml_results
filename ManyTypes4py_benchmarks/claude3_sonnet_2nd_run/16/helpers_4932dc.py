"""
Helpers for the API
"""
import re
from collections import namedtuple
from textwrap import dedent
from itertools import chain
from functools import wraps
from inspect import Parameter
from typing import List, Dict, Tuple, Set, Optional, Iterator, Any, Callable, Union, Iterable, TypeVar, cast
from parso.python.parser import Parser
from parso.python import tree
from jedi.inference.base_value import NO_VALUES
from jedi.inference.syntax_tree import infer_atom
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.compiled import get_string_value_set
from jedi.cache import signature_time_cache, memoize_method
from jedi.parser_utils import get_parent_scope
CompletionParts = namedtuple('CompletionParts', ['path', 'has_dot', 'name'])

def _start_match(string: str, like_name: str) -> bool:
    return string.startswith(like_name)

def _fuzzy_match(string: str, like_name: str) -> bool:
    if len(like_name) <= 1:
        return like_name in string
    pos = string.find(like_name[0])
    if pos >= 0:
        return _fuzzy_match(string[pos + 1:], like_name[1:])
    return False

def match(string: str, like_name: str, fuzzy: bool = False) -> bool:
    if fuzzy:
        return _fuzzy_match(string, like_name)
    else:
        return _start_match(string, like_name)

def sorted_definitions(defs: List[Any]) -> List[Any]:
    return sorted(defs, key=lambda x: (str(x.module_path or ''), x.line or 0, x.column or 0, x.name))

def get_on_completion_name(module_node: Any, lines: List[str], position: Tuple[int, int]) -> str:
    leaf = module_node.get_leaf_for_position(position)
    if leaf is None or leaf.type in ('string', 'error_leaf'):
        line = lines[position[0] - 1]
        return re.search('(?!\\d)\\w+$|$', line[:position[1]]).group(0)
    elif leaf.type not in ('name', 'keyword'):
        return ''
    return leaf.value[:position[1] - leaf.start_pos[1]]

def _get_code(code_lines: List[str], start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> str:
    lines = code_lines[start_pos[0] - 1:end_pos[0]]
    lines[-1] = lines[-1][:end_pos[1]]
    lines[0] = lines[0][start_pos[1]:]
    return ''.join(lines)

class OnErrorLeaf(Exception):

    @property
    def error_leaf(self) -> Any:
        return self.args[0]

def _get_code_for_stack(code_lines: List[str], leaf: Any, position: Tuple[int, int]) -> str:
    if leaf.start_pos >= position:
        leaf = leaf.get_previous_leaf()
        if leaf is None:
            return ''
    is_after_newline = leaf.type == 'newline'
    while leaf.type == 'newline':
        leaf = leaf.get_previous_leaf()
        if leaf is None:
            return ''
    if leaf.type == 'error_leaf' or leaf.type == 'string':
        if leaf.start_pos[0] < position[0]:
            return ''
        raise OnErrorLeaf(leaf)
    else:
        user_stmt = leaf
        while True:
            if user_stmt.parent.type in ('file_input', 'suite', 'simple_stmt'):
                break
            user_stmt = user_stmt.parent
        if is_after_newline:
            if user_stmt.start_pos[1] > position[1]:
                return ''
        return _get_code(code_lines, user_stmt.get_start_pos_of_prefix(), position)

def get_stack_at_position(grammar: Any, code_lines: List[str], leaf: Any, pos: Tuple[int, int]) -> List[Any]:
    """
    Returns the possible node names (e.g. import_from, xor_test or yield_stmt).
    """

    class EndMarkerReached(Exception):
        pass

    def tokenize_without_endmarker(code: str) -> Iterator[Any]:
        tokens = grammar._tokenize(code)
        for token in tokens:
            if token.string == safeword:
                raise EndMarkerReached()
            elif token.prefix.endswith(safeword):
                raise EndMarkerReached()
            elif token.string.endswith(safeword):
                yield token
                raise EndMarkerReached()
            else:
                yield token
    code = dedent(_get_code_for_stack(code_lines, leaf, pos))
    safeword = 'ZZZ_USER_WANTS_TO_COMPLETE_HERE_WITH_JEDI'
    code = code + ' ' + safeword
    p = Parser(grammar._pgen_grammar, error_recovery=True)
    try:
        p.parse(tokens=tokenize_without_endmarker(code))
    except EndMarkerReached:
        return p.stack
    raise SystemError("This really shouldn't happen. There's a bug in Jedi:\n%s" % list(tokenize_without_endmarker(code)))

def infer(inference_state: Any, context: Any, leaf: Any) -> Any:
    if leaf.type == 'name':
        return inference_state.infer(context, leaf)
    parent = leaf.parent
    definitions = NO_VALUES
    if parent.type == 'atom':
        definitions = context.infer_node(leaf.parent)
    elif parent.type == 'trailer':
        definitions = infer_call_of_leaf(context, leaf)
    elif isinstance(leaf, tree.Literal):
        return infer_atom(context, leaf)
    elif leaf.type in ('fstring_string', 'fstring_start', 'fstring_end'):
        return get_string_value_set(inference_state)
    return definitions

def filter_follow_imports(names: Iterable[Any], follow_builtin_imports: bool = False) -> Iterator[Any]:
    for name in names:
        if name.is_import():
            new_names = list(filter_follow_imports(name.goto(), follow_builtin_imports=follow_builtin_imports))
            found_builtin = False
            if follow_builtin_imports:
                for new_name in new_names:
                    if new_name.start_pos is None:
                        found_builtin = True
            if found_builtin:
                yield name
            else:
                yield from new_names
        else:
            yield name

class CallDetails:

    def __init__(self, bracket_leaf: Any, children: List[Any], position: Tuple[int, int]) -> None:
        self.bracket_leaf = bracket_leaf
        self._children = children
        self._position = position

    @property
    def index(self) -> Optional[int]:
        return _get_index_and_key(self._children, self._position)[0]

    @property
    def keyword_name_str(self) -> Optional[str]:
        return _get_index_and_key(self._children, self._position)[1]

    @memoize_method
    def _list_arguments(self) -> List[Tuple[int, Optional[str], bool]]:
        return list(_iter_arguments(self._children, self._position))

    def calculate_index(self, param_names: List[Any]) -> Optional[int]:
        positional_count = 0
        used_names: Set[str] = set()
        star_count = -1
        args = self._list_arguments()
        if not args:
            if param_names:
                return 0
            else:
                return None
        is_kwarg = False
        for i, (star_count, key_start, had_equal) in enumerate(args):
            is_kwarg |= had_equal | (star_count == 2)
            if star_count:
                pass
            elif i + 1 != len(args):
                if had_equal:
                    used_names.add(key_start)
                else:
                    positional_count += 1
        for i, param_name in enumerate(param_names):
            kind = param_name.get_kind()
            if not is_kwarg:
                if kind == Parameter.VAR_POSITIONAL:
                    return i
                if kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.POSITIONAL_ONLY):
                    if i == positional_count:
                        return i
            if key_start is not None and (not star_count == 1) or star_count == 2:
                if param_name.string_name not in used_names and (kind == Parameter.KEYWORD_ONLY or (kind == Parameter.POSITIONAL_OR_KEYWORD and positional_count <= i)):
                    if star_count:
                        return i
                    if had_equal:
                        if param_name.string_name == key_start:
                            return i
                    elif param_name.string_name.startswith(key_start):
                        return i
                if kind == Parameter.VAR_KEYWORD:
                    return i
        return None

    def iter_used_keyword_arguments(self) -> Iterator[str]:
        for star_count, key_start, had_equal in list(self._list_arguments()):
            if had_equal and key_start:
                yield key_start

    def count_positional_arguments(self) -> int:
        count = 0
        for star_count, key_start, had_equal in self._list_arguments()[:-1]:
            if star_count:
                break
            count += 1
        return count

def _iter_arguments(nodes: List[Any], position: Tuple[int, int]) -> Iterator[Tuple[int, Optional[str], bool]]:

    def remove_after_pos(name: Any) -> Optional[str]:
        if name.type != 'name':
            return None
        return name.value[:position[1] - name.start_pos[1]]
    nodes_before = [c for c in nodes if c.start_pos < position]
    if nodes_before[-1].type == 'arglist':
        yield from _iter_arguments(nodes_before[-1].children, position)
        return
    previous_node_yielded = False
    stars_seen = 0
    for i, node in enumerate(nodes_before):
        if node.type == 'argument':
            previous_node_yielded = True
            first = node.children[0]
            second = node.children[1]
            if second == '=':
                if second.start_pos < position:
                    yield (0, first.value, True)
                else:
                    yield (0, remove_after_pos(first), False)
            elif first in ('*', '**'):
                yield (len(first.value), remove_after_pos(second), False)
            else:
                first_leaf = node.get_first_leaf()
                if first_leaf.type == 'name' and first_leaf.start_pos >= position:
                    yield (0, remove_after_pos(first_leaf), False)
                else:
                    yield (0, None, False)
            stars_seen = 0
        elif node.type == 'testlist_star_expr':
            for n in node.children[::2]:
                if n.type == 'star_expr':
                    stars_seen = 1
                    n = n.children[1]
                yield (stars_seen, remove_after_pos(n), False)
                stars_seen = 0
            previous_node_yielded = bool(len(node.children) % 2)
        elif isinstance(node, tree.PythonLeaf) and node.value == ',':
            if not previous_node_yielded:
                yield (stars_seen, '', False)
                stars_seen = 0
            previous_node_yielded = False
        elif isinstance(node, tree.PythonLeaf) and node.value in ('*', '**'):
            stars_seen = len(node.value)
        elif node == '=' and nodes_before[-1]:
            previous_node_yielded = True
            before = nodes_before[i - 1]
            if before.type == 'name':
                yield (0, before.value, True)
            else:
                yield (0, None, False)
            stars_seen = 0
    if not previous_node_yielded:
        if nodes_before[-1].type == 'name':
            yield (stars_seen, remove_after_pos(nodes_before[-1]), False)
        else:
            yield (stars_seen, '', False)

def _get_index_and_key(nodes: List[Any], position: Tuple[int, int]) -> Tuple[int, Optional[str]]:
    """
    Returns the amount of commas and the keyword argument string.
    """
    nodes_before = [c for c in nodes if c.start_pos < position]
    if nodes_before[-1].type == 'arglist':
        return _get_index_and_key(nodes_before[-1].children, position)
    key_str = None
    last = nodes_before[-1]
    if last.type == 'argument' and last.children[1] == '=' and (last.children[1].end_pos <= position):
        key_str = last.children[0].value
    elif last == '=':
        key_str = nodes_before[-2].value
    return (nodes_before.count(','), key_str)

def _get_signature_details_from_error_node(node: Any, additional_children: List[Any], position: Tuple[int, int]) -> Optional[CallDetails]:
    for index, element in reversed(list(enumerate(node.children))):
        if element == '(' and element.end_pos <= position and (index > 0):
            children = node.children[index:]
            name = element.get_previous_leaf()
            if name is None:
                continue
            if name.type == 'name' or name.parent.type in ('trailer', 'atom'):
                return CallDetails(element, children + additional_children, position)
    return None

def get_signature_details(module: Any, position: Tuple[int, int]) -> Optional[CallDetails]:
    leaf = module.get_leaf_for_position(position, include_prefixes=True)
    if leaf.start_pos >= position:
        leaf = leaf.get_previous_leaf()
        if leaf is None:
            return None
    node = leaf.parent
    while node is not None:
        if node.type in ('funcdef', 'classdef', 'decorated', 'async_stmt'):
            return None
        additional_children: List[Any] = []
        for n in reversed(node.children):
            if n.start_pos < position:
                if n.type == 'error_node':
                    result = _get_signature_details_from_error_node(n, additional_children, position)
                    if result is not None:
                        return result
                    additional_children[0:0] = n.children
                    continue
                additional_children.insert(0, n)
        if node.type == 'trailer' and node.children[0] == '(' or (node.type == 'decorator' and node.children[2] == '('):
            if not (leaf is node.children[-1] and position >= leaf.end_pos):
                leaf = node.get_previous_leaf()
                if leaf is None:
                    return None
                return CallDetails(node.children[0] if node.type == 'trailer' else node.children[2], node.children, position)
        node = node.parent
    return None

@signature_time_cache('call_signatures_validity')
def cache_signatures(inference_state: Any, context: Any, bracket_leaf: Any, code_lines: List[str], user_pos: Tuple[int, int]) -> Iterator[Any]:
    """This function calculates the cache key."""
    line_index = user_pos[0] - 1
    before_cursor = code_lines[line_index][:user_pos[1]]
    other_lines = code_lines[bracket_leaf.start_pos[0]:line_index]
    whole = ''.join(other_lines + [before_cursor])
    before_bracket = re.match('.*\\(', whole, re.DOTALL)
    module_path = context.get_root_context().py__file__()
    if module_path is None:
        yield None
    else:
        yield (module_path, before_bracket, bracket_leaf.start_pos)
    yield infer(inference_state, context, bracket_leaf.get_previous_leaf())

def validate_line_column(func: Callable) -> Callable:

    @wraps(func)
    def wrapper(self: Any, line: Optional[int] = None, column: Optional[int] = None, *args: Any, **kwargs: Any) -> Any:
        line = max(len(self._code_lines), 1) if line is None else line
        if not 0 < line <= len(self._code_lines):
            raise ValueError('`line` parameter is not in a valid range.')
        line_string = self._code_lines[line - 1]
        line_len = len(line_string)
        if line_string.endswith('\r\n'):
            line_len -= 2
        elif line_string.endswith('\n'):
            line_len -= 1
        column = line_len if column is None else column
        if not 0 <= column <= line_len:
            raise ValueError('`column` parameter (%d) is not in a valid range (0-%d) for line %d (%r).' % (column, line_len, line, line_string))
        return func(self, line, column, *args, **kwargs)
    return wrapper

def get_module_names(module: Any, all_scopes: bool, definitions: bool = True, references: bool = False) -> List[Any]:
    """
    Returns a dictionary with name parts as keys and their call paths as
    values.
    """

    def def_ref_filter(name: Any) -> bool:
        is_def = name.is_definition()
        return definitions and is_def or (references and (not is_def))
    names = list(chain.from_iterable(module.get_used_names().values()))
    if not all_scopes:

        def is_module_scope_name(name: Any) -> bool:
            parent_scope = get_parent_scope(name)
            if parent_scope and parent_scope.type == 'async_stmt':
                parent_scope = parent_scope.parent
            return parent_scope in (module, None)
        names = [n for n in names if is_module_scope_name(n)]
    return list(filter(def_ref_filter, names))

def split_search_string(name: str) -> Tuple[str, List[str]]:
    type, _, dotted_names = name.rpartition(' ')
    if type == 'def':
        type = 'function'
    return (type, dotted_names.split('.'))
