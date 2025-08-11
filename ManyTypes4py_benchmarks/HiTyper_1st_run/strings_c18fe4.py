"""
This module is here for string completions. This means mostly stuff where
strings are returned, like `foo = dict(bar=3); foo["ba` would complete to
`"bar"]`.

It however does the same for numbers. The difference between string completions
and other completions is mostly that this module doesn't return defined
names in a module, but pretty much an arbitrary string.
"""
import re
from jedi.inference.names import AbstractArbitraryName
from jedi.inference.helpers import infer_call_of_leaf
from jedi.api.classes import Completion
from jedi.parser_utils import cut_value_at_position
_sentinel = object()

class StringName(AbstractArbitraryName):
    api_type = 'string'
    is_value_name = False

def complete_dict(module_context: Union[int, str, dict], code_lines: Union[str, bool, None], leaf: int, position: Union[str, int, typing.Iterable[str]], string: Union[str, int], fuzzy: Union[str, None, list]) -> list:
    bracket_leaf = leaf
    if bracket_leaf != '[':
        bracket_leaf = leaf.get_previous_leaf()
    cut_end_quote = ''
    if string:
        cut_end_quote = get_quote_ending(string, code_lines, position, invert_result=True)
    if bracket_leaf == '[':
        if string is None and leaf is not bracket_leaf:
            string = cut_value_at_position(leaf, position)
        context = module_context.create_context(bracket_leaf)
        before_bracket_leaf = bracket_leaf.get_previous_leaf()
        if before_bracket_leaf.type in ('atom', 'trailer', 'name'):
            values = infer_call_of_leaf(context, before_bracket_leaf)
            return list(_completions_for_dicts(module_context.inference_state, values, '' if string is None else string, cut_end_quote, fuzzy=fuzzy))
    return []

def _completions_for_dicts(inference_state: Union[str, None, typing.Iterable[str]], dicts: Union[tuple[str], None, list[str], dict[str, str]], literal_string: Union[str, list[tuple[str]]], cut_end_quote: str, fuzzy: Union[str, list[str], bool]) -> typing.Generator[Completion]:
    for dict_key in sorted(_get_python_keys(dicts), key=lambda x: repr(x)):
        dict_key_str = _create_repr_string(literal_string, dict_key)
        if dict_key_str.startswith(literal_string):
            name = StringName(inference_state, dict_key_str[:-len(cut_end_quote) or None])
            yield Completion(inference_state, name, stack=None, like_name_length=len(literal_string), is_fuzzy=fuzzy)

def _create_repr_string(literal_string: Union[str, None, bytes], dict_key: Union[str, bool, None]) -> str:
    if not isinstance(dict_key, (str, bytes)) or not literal_string:
        return repr(dict_key)
    r = repr(dict_key)
    prefix, quote = _get_string_prefix_and_quote(literal_string)
    if quote is None:
        return r
    if quote == r[0]:
        return prefix + r
    return prefix + quote + r[1:-1] + quote

def _get_python_keys(dicts: Union[list[dict], dict[str, set[str]]]) -> typing.Generator[str]:
    for dct in dicts:
        if dct.array_type == 'dict':
            for key in dct.get_key_values():
                dict_key = key.get_safe_value(default=_sentinel)
                if dict_key is not _sentinel:
                    yield dict_key

def _get_string_prefix_and_quote(string: str) -> Union[tuple[None], tuple]:
    match = re.match('(\\w*)("""|\\\'{3}|"|\\\')', string)
    if match is None:
        return (None, None)
    return (match.group(1), match.group(2))

def _matches_quote_at_position(code_lines: Union[str, qutebrowser.api.apitypes.Tab], quote: Union[str, tuple[str]], position: Union[str, qutebrowser.api.apitypes.Tab]) -> bool:
    string = code_lines[position[0] - 1][position[1]:position[1] + len(quote)]
    return string == quote

def get_quote_ending(string: str, code_lines: str, position: str, invert_result: bool=False) -> Union[typing.Text, str, None]:
    _, quote = _get_string_prefix_and_quote(string)
    if quote is None:
        return ''
    if _matches_quote_at_position(code_lines, quote, position) != invert_result:
        return ''
    return quote