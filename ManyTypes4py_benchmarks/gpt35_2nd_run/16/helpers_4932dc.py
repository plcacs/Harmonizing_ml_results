import re
from collections import namedtuple
from textwrap import dedent
from itertools import chain
from functools import wraps
from inspect import Parameter
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

def sorted_definitions(defs) -> list:
    return sorted(defs, key=lambda x: (str(x.module_path or ''), x.line or 0, x.column or 0, x.name))

def get_on_completion_name(module_node, lines, position) -> str:
    leaf = module_node.get_leaf_for_position(position)
    if leaf is None or leaf.type in ('string', 'error_leaf'):
        line = lines[position[0] - 1]
        return re.search('(?!\\d)\\w+$|$', line[:position[1]]).group(0)
    elif leaf.type not in ('name', 'keyword'):
        return ''
    return leaf.value[:position[1] - leaf.start_pos[1]

def _get_code(code_lines, start_pos, end_pos) -> str:
    lines = code_lines[start_pos[0] - 1:end_pos[0]]
    lines[-1] = lines[-1][:end_pos[1]]
    lines[0] = lines[0][start_pos[1]:]
    return ''.join(lines)

class OnErrorLeaf(Exception):

    @property
    def error_leaf(self) -> str:
        return self.args[0]

def _get_code_for_stack(code_lines, leaf, position) -> str:
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

def get_stack_at_position(grammar, code_lines, leaf, pos) -> list:
    ...

def infer(inference_state, context, leaf) -> list:
    ...

def filter_follow_imports(names, follow_builtin_imports=False) -> list:
    ...

class CallDetails:

    def __init__(self, bracket_leaf, children, position) -> None:
        ...

    @property
    def index(self) -> int:
        ...

    @property
    def keyword_name_str(self) -> str:
        ...

    @memoize_method
    def _list_arguments(self) -> list:
        ...

    def calculate_index(self, param_names) -> int:
        ...

    def iter_used_keyword_arguments(self) -> list:
        ...

    def count_positional_arguments(self) -> int:
        ...

def _iter_arguments(nodes, position) -> list:
    ...

def _get_index_and_key(nodes, position) -> tuple:
    ...

def _get_signature_details_from_error_node(node, additional_children, position) -> CallDetails:
    ...

def get_signature_details(module, position) -> CallDetails:
    ...

@signature_time_cache('call_signatures_validity')
def cache_signatures(inference_state, context, bracket_leaf, code_lines, user_pos) -> list:
    ...

def validate_line_column(func) -> callable:

    @wraps(func)
    def wrapper(self, line=None, column=None, *args, **kwargs) -> any:
        ...

    return wrapper

def get_module_names(module, all_scopes, definitions=True, references=False) -> list:
    ...

def split_search_string(name) -> tuple:
    ...
