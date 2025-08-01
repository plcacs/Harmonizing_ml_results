import re
from typing import Any, List, Optional, Iterator, Tuple
from jedi.inference.names import AbstractArbitraryName
from jedi.inference.helpers import infer_call_of_leaf
from jedi.api.classes import Completion
from jedi.parser_utils import cut_value_at_position

_sentinel = object()


class StringName(AbstractArbitraryName):
    api_type: str = 'string'
    is_value_name: bool = False


def complete_dict(module_context: Any, code_lines: List[str], leaf: Any, position: Tuple[int, int],
                  string: Optional[str], fuzzy: bool) -> List[Completion]:
    bracket_leaf: Any = leaf
    if bracket_leaf != '[':
        bracket_leaf = leaf.get_previous_leaf()
    cut_end_quote: str = ''
    if string:
        cut_end_quote = get_quote_ending(string, code_lines, position, invert_result=True)
    if bracket_leaf == '[':
        if string is None and leaf is not bracket_leaf:
            string = cut_value_at_position(leaf, position)
        context: Any = module_context.create_context(bracket_leaf)
        before_bracket_leaf: Any = bracket_leaf.get_previous_leaf()
        if before_bracket_leaf.type in ('atom', 'trailer', 'name'):
            values = infer_call_of_leaf(context, before_bracket_leaf)
            return list(_completions_for_dicts(module_context.inference_state,
                                               values,
                                               '' if string is None else string,
                                               cut_end_quote,
                                               fuzzy=fuzzy))
    return []


def _completions_for_dicts(inference_state: Any, dicts: Any, literal_string: str, cut_end_quote: str,
                           fuzzy: bool) -> Iterator[Completion]:
    for dict_key in sorted(_get_python_keys(dicts), key=lambda x: repr(x)):
        dict_key_str: str = _create_repr_string(literal_string, dict_key)
        if dict_key_str.startswith(literal_string):
            name = StringName(inference_state, dict_key_str[:-len(cut_end_quote) or None])
            yield Completion(inference_state, name, stack=None, like_name_length=len(literal_string), is_fuzzy=fuzzy)


def _create_repr_string(literal_string: str, dict_key: Any) -> str:
    if not isinstance(dict_key, (str, bytes)) or not literal_string:
        return repr(dict_key)
    r: str = repr(dict_key)
    prefix, quote = _get_string_prefix_and_quote(literal_string)
    if quote is None:
        return r
    if quote == r[0]:
        return prefix + r
    return prefix + quote + r[1:-1] + quote


def _get_python_keys(dicts: Any) -> Iterator[Any]:
    for dct in dicts:
        if dct.array_type == 'dict':
            for key in dct.get_key_values():
                dict_key = key.get_safe_value(default=_sentinel)
                if dict_key is not _sentinel:
                    yield dict_key


def _get_string_prefix_and_quote(string: str) -> Tuple[Optional[str], Optional[str]]:
    match = re.match(r'(\w*)("""|\'\'\'|"|\')', string)
    if match is None:
        return (None, None)
    return (match.group(1), match.group(2))


def _matches_quote_at_position(code_lines: List[str], quote: str, position: Tuple[int, int]) -> bool:
    line_index, col_index = position[0] - 1, position[1]
    string_at_pos = code_lines[line_index][col_index:col_index + len(quote)]
    return string_at_pos == quote


def get_quote_ending(string: str, code_lines: List[str], position: Tuple[int, int], invert_result: bool = False) -> str:
    _, quote = _get_string_prefix_and_quote(string)
    if quote is None:
        return ''
    if _matches_quote_at_position(code_lines, quote, position) != invert_result:
        return ''
    return quote