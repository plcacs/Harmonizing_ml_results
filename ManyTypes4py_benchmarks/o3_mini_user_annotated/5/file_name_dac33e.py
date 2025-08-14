from typing import Any, Callable, Iterator, List, Optional, Tuple
import os

from jedi.api import classes
from jedi.api.strings import StringName, get_quote_ending
from jedi.api.helpers import match
from jedi.inference.helpers import get_str_or_none


class PathName(StringName):
    api_type = 'path'


def complete_file_name(
    inference_state: Any,
    module_context: Any,
    start_leaf: Any,
    quote: str,
    string: str,
    like_name: str,
    signatures_callback: Callable[[int, int], List[Any]],
    code_lines: List[str],
    position: Tuple[int, int],
    fuzzy: bool
) -> Iterator[classes.Completion]:
    # First we want to find out what can actually be changed as a name.
    like_name_length: int = len(os.path.basename(string))

    addition: Optional[str] = _get_string_additions(module_context, start_leaf)
    if string.startswith('~'):
        string = os.path.expanduser(string)
    if addition is None:
        return
    string = addition + string

    # Here we use basename again, because if strings are added like
    # `'foo' + 'bar`, it should complete to `foobar/`.
    must_start_with: str = os.path.basename(string)
    string = os.path.dirname(string)

    sigs: List[Any] = signatures_callback(*position)
    is_in_os_path_join: bool = bool(sigs and all(s.full_name == 'os.path.join' for s in sigs))
    if is_in_os_path_join:
        to_be_added: Optional[str] = _add_os_path_join(module_context, start_leaf, sigs[0].bracket_start)
        if to_be_added is None:
            is_in_os_path_join = False
        else:
            string = to_be_added + string
    base_path: str = os.path.join(inference_state.project.path, string)
    try:
        listed: List[os.DirEntry[Any]] = sorted(os.scandir(base_path), key=lambda e: e.name)
        # OSError: [Errno 36] File name too long: '...'
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


def _get_string_additions(module_context: Any, start_leaf: Any) -> Optional[str]:
    def iterate_nodes() -> Iterator[Any]:
        node = addition.parent
        was_addition: bool = True
        for child_node in reversed(node.children[:node.children.index(addition)]):
            if was_addition:
                was_addition = False
                yield child_node
                continue

            if child_node != '+':
                break
            was_addition = True

    addition: Any = start_leaf.get_previous_leaf()
    if addition != '+':
        return ''
    context: Any = module_context.create_context(start_leaf)
    return _add_strings(context, list(reversed(list(iterate_nodes()))))


def _add_strings(context: Any, nodes: List[Any], add_slash: bool = False) -> Optional[str]:
    string: str = ''
    first: bool = True
    for child_node in nodes:
        values: List[Any] = context.infer_node(child_node)
        if len(values) != 1:
            return None
        c: Any = values[0]
        s: Optional[str] = get_str_or_none(c)
        if s is None:
            return None
        if not first and add_slash:
            string += os.path.sep
        string += s
        first = False
    return string


def _add_os_path_join(module_context: Any, start_leaf: Any, bracket_start: Tuple[int, int]) -> Optional[str]:
    def check(maybe_bracket: Any, nodes: List[Any]) -> Optional[str]:
        if maybe_bracket.start_pos != bracket_start:
            return None

        if not nodes:
            return ''
        context: Any = module_context.create_context(nodes[0])
        return _add_strings(context, nodes, add_slash=True) or ''

    if start_leaf.type == 'error_leaf':
        # Unfinished string literal, like `join('`
        value_node: Any = start_leaf.parent
        index: int = value_node.children.index(start_leaf)
        if index > 0:
            error_node: Any = value_node.children[index - 1]
            if error_node.type == 'error_node' and len(error_node.children) >= 2:
                index = -2
                if error_node.children[-1].type == 'arglist':
                    arglist_nodes: List[Any] = error_node.children[-1].children
                    index -= 1
                else:
                    arglist_nodes = []

                return check(error_node.children[index + 1], arglist_nodes[::2])
        return None

    # Maybe an arglist or some weird error case. Therefore checked below.
    searched_node_child: Any = start_leaf
    while (searched_node_child.parent is not None and
           searched_node_child.parent.type not in ('arglist', 'trailer', 'error_node')):
        searched_node_child = searched_node_child.parent

    if searched_node_child.get_first_leaf() is not start_leaf:
        return None
    searched_node: Any = searched_node_child.parent
    if searched_node is None:
        return None

    index = searched_node.children.index(searched_node_child)
    arglist_nodes: List[Any] = searched_node.children[:index]
    if searched_node.type == 'arglist':
        trailer: Any = searched_node.parent
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
        # Stuff like `join(""`
        return check(arglist_nodes[-1], [])
    return None