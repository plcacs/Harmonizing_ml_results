import os
from typing import Generator, List, Optional, Tuple, Union

from jedi.api import classes
from jedi.api.strings import StringName, get_quote_ending
from jedi.api.helpers import match
from jedi.inference.helpers import get_str_or_none
from jedi.inference.base_value import Value
from jedi.inference.context import ModuleContext
from jedi.inference.value import TreeValue
from jedi.parser_utils import get_last_leaf
from jedi.inference.gradual.typing import TypeVar
from jedi.inference.names import AbstractNameDefinition
from jedi.inference.signature import AbstractSignature
from jedi.parser_utils import get_last_leaf
from jedi.inference.gradual.typing import TypeVar
from jedi.inference.names import AbstractNameDefinition
from jedi.inference.signature import AbstractSignature

_T = TypeVar('_T')

class PathName(StringName):
    api_type: str = 'path'

def complete_file_name(
    inference_state: 'InferenceState',
    module_context: ModuleContext,
    start_leaf: 'Leaf',
    quote: str,
    string: str,
    like_name: str,
    signatures_callback: 'Callable',
    code_lines: List[str],
    position: Tuple[int, int],
    fuzzy: bool
) -> Generator[classes.Completion, None, None]:
    like_name_length: int = len(os.path.basename(string))

    addition: Optional[str] = _get_string_additions(module_context, start_leaf)
    if string.startswith('~'):
        string: str = os.path.expanduser(string)
    if addition is None:
        return
    string: str = addition + string

    must_start_with: str = os.path.basename(string)
    string: str = os.path.dirname(string)

    sigs: List[AbstractSignature] = signatures_callback(*position)
    is_in_os_path_join: bool = sigs and all(s.full_name == 'os.path.join' for s in sigs)
    if is_in_os_path_join:
        to_be_added: Optional[str] = _add_os_path_join(module_context, start_leaf, sigs[0].bracket_start)
        if to_be_added is None:
            is_in_os_path_join: bool = False
        else:
            string: str = to_be_added + string
    base_path: str = os.path.join(inference_state.project.path, string)
    try:
        listed: List[os.DirEntry] = sorted(os.scandir(base_path), key=lambda e: e.name)
    except (FileNotFoundError, OSError):
        return
    quote_ending: str = get_quote_ending(quote, code_lines, position)
    for entry in listed:
        name: str = entry.name
        if match(name, must_start_with, fuzzy=fuzzy):
            if is_in_os_path_join or not entry.is_dir():
                name += quote_ending
            else:
                name += os.path.sep

            yield classes.Completion(
                inference_state,
                PathName(inference_state, name[len(must_start_with) - like_name_length:]),
                stack=None,
                like_name_length=like_name_length,
                is_fuzzy=fuzzy,
            )

def _get_string_additions(module_context: ModuleContext, start_leaf: 'Leaf') -> Optional[str]:
    def iterate_nodes() -> Generator['Leaf', None, None]:
        node: 'Leaf' = addition.parent
        was_addition: bool = True
        for child_node in reversed(node.children[:node.children.index(addition)]):
            if was_addition:
                was_addition: bool = False
                yield child_node
                continue

            if child_node != '+':
                break
            was_addition: bool = True

    addition: 'Leaf' = start_leaf.get_previous_leaf()
    if addition != '+':
        return ''
    context: ModuleContext = module_context.create_context(start_leaf)
    return _add_strings(context, reversed(list(iterate_nodes())))

def _add_strings(context: ModuleContext, nodes: List['Leaf'], add_slash: bool = False) -> Optional[str]:
    string: str = ''
    first: bool = True
    for child_node in nodes:
        values: List[Value] = context.infer_node(child_node)
        if len(values) != 1:
            return None
        c: Value = values[0]
        s: Optional[str] = get_str_or_none(c)
        if s is None:
            return None
        if not first and add_slash:
            string += os.path.sep
        string += s
        first: bool = False
    return string

def _add_os_path_join(module_context: ModuleContext, start_leaf: 'Leaf', bracket_start: Tuple[int, int]) -> Optional[str]:
    def check(maybe_bracket: 'Leaf', nodes: List['Leaf']) -> Optional[str]:
        if maybe_bracket.start_pos != bracket_start:
            return None

        if not nodes:
            return ''
        context: ModuleContext = module_context.create_context(nodes[0])
        return _add_strings(context, nodes, add_slash=True) or ''

    if start_leaf.type == 'error_leaf':
        value_node: 'Leaf' = start_leaf.parent
        index: int = value_node.children.index(start_leaf)
        if index > 0:
            error_node: 'Leaf' = value_node.children[index - 1]
            if error_node.type == 'error_node' and len(error_node.children) >= 2:
                index: int = -2
                if error_node.children[-1].type == 'arglist':
                    arglist_nodes: List['Leaf'] = error_node.children[-1].children
                    index -= 1
                else:
                    arglist_nodes: List['Leaf'] = []

                return check(error_node.children[index + 1], arglist_nodes[::2])
        return None

    searched_node_child: 'Leaf' = start_leaf
    while searched_node_child.parent is not None \
            and searched_node_child.parent.type not in ('arglist', 'trailer', 'error_node'):
        searched_node_child: 'Leaf' = searched_node_child.parent

    if searched_node_child.get_first_leaf() is not start_leaf:
        return None
    searched_node: 'Leaf' = searched_node_child.parent
    if searched_node is None:
        return None

    index: int = searched_node.children.index(searched_node_child)
    arglist_nodes: List['Leaf'] = searched_node.children[:index]
    if searched_node.type == 'arglist':
        trailer: 'Leaf' = searched_node.parent
        if trailer.type == 'error_node':
            trailer_index: int = trailer.children.index(searched_node)
            assert trailer_index >= 2
            assert trailer.children[trailer_index - 1] == '('
            return check(trailer.children[trailer_index - 1], arglist_nodes[::2])
        elif trailer.type == 'trailer':
            return check(trailer.children[0], arglist_nodes[::2])
    elif searched_node.type == 'trailer':
        return check(searched_node.children[0], [])
    elif searched_node.type == 'error_node':
        return check(arglist_nodes[-1], [])
