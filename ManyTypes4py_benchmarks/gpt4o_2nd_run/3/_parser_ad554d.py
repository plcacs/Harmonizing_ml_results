import string
import warnings
from types import MappingProxyType
from typing import IO, Any, Callable, Dict, FrozenSet, Iterable, NamedTuple, Optional, Tuple, Union
from ._re import RE_DATETIME, RE_LOCALTIME, RE_NUMBER, match_to_datetime, match_to_localtime, match_to_number

ASCII_CTRL: FrozenSet[str] = frozenset((chr(i) for i in range(32))) | frozenset(chr(127))
ILLEGAL_BASIC_STR_CHARS: FrozenSet[str] = ASCII_CTRL - frozenset('\t')
ILLEGAL_MULTILINE_BASIC_STR_CHARS: FrozenSet[str] = ASCII_CTRL - frozenset('\t\n\r')
ILLEGAL_LITERAL_STR_CHARS: FrozenSet[str] = ILLEGAL_BASIC_STR_CHARS
ILLEGAL_MULTILINE_LITERAL_STR_CHARS: FrozenSet[str] = ASCII_CTRL - frozenset('\t\n')
ILLEGAL_COMMENT_CHARS: FrozenSet[str] = ILLEGAL_BASIC_STR_CHARS
TOML_WS: FrozenSet[str] = frozenset(' \t')
TOML_WS_AND_NEWLINE: FrozenSet[str] = TOML_WS | frozenset('\n')
BARE_KEY_CHARS: FrozenSet[str] = frozenset(string.ascii_letters + string.digits + '-_')
KEY_INITIAL_CHARS: FrozenSet[str] = BARE_KEY_CHARS | frozenset('"\'')
HEXDIGIT_CHARS: FrozenSet[str] = frozenset(string.hexdigits)
BASIC_STR_ESCAPE_REPLACEMENTS: MappingProxyType = MappingProxyType({'\\b': '\x08', '\\t': '\t', '\\n': '\n', '\\f': '\x0c', '\\r': '\r', '\\"': '"', '\\\\': '\\'})
ParseFloat = Callable[[str], Any]
Key = Tuple[str, ...]
Pos = int

class TOMLDecodeError(ValueError):
    """An error raised if a document is not valid TOML."""

def load(fp: IO[bytes], *, parse_float: ParseFloat = float) -> Dict[str, Any]:
    """Parse TOML from a file object."""
    s = fp.read()
    if isinstance(s, bytes):
        s = s.decode()
    else:
        warnings.warn('Text file object support is deprecated in favor of binary file objects. Use `open("foo.toml", "rb")` to open the file in binary mode.', DeprecationWarning)
    return loads(s, parse_float=parse_float)

def loads(s: str, *, parse_float: ParseFloat = float) -> Dict[str, Any]:
    """Parse TOML from a string."""
    src = s.replace('\r\n', '\n')
    pos = 0
    out = Output(NestedDict(), Flags())
    header: Key = ()
    while True:
        pos = skip_chars(src, pos, TOML_WS)
        try:
            char = src[pos]
        except IndexError:
            break
        if char == '\n':
            pos += 1
            continue
        if char in KEY_INITIAL_CHARS:
            pos = key_value_rule(src, pos, out, header, parse_float)
            pos = skip_chars(src, pos, TOML_WS)
        elif char == '[':
            try:
                second_char = src[pos + 1]
            except IndexError:
                second_char = None
            if second_char == '[':
                pos, header = create_list_rule(src, pos, out)
            else:
                pos, header = create_dict_rule(src, pos, out)
            pos = skip_chars(src, pos, TOML_WS)
        elif char != '#':
            raise suffixed_err(src, pos, 'Invalid statement')
        pos = skip_comment(src, pos)
        try:
            char = src[pos]
        except IndexError:
            break
        if char != '\n':
            raise suffixed_err(src, pos, 'Expected newline or end of document after a statement')
        pos += 1
    return out.data.dict

class Flags:
    """Flags that map to parsed keys/namespaces."""
    FROZEN = 0
    EXPLICIT_NEST = 1

    def __init__(self) -> None:
        self._flags: Dict[str, Any] = {}

    def unset_all(self, key: Key) -> None:
        cont = self._flags
        for k in key[:-1]:
            if k not in cont:
                return
            cont = cont[k]['nested']
        cont.pop(key[-1], None)

    def set_for_relative_key(self, head_key: Key, rel_key: Key, flag: int) -> None:
        cont = self._flags
        for k in head_key:
            if k not in cont:
                cont[k] = {'flags': set(), 'recursive_flags': set(), 'nested': {}}
            cont = cont[k]['nested']
        for k in rel_key:
            if k in cont:
                cont[k]['flags'].add(flag)
            else:
                cont[k] = {'flags': {flag}, 'recursive_flags': set(), 'nested': {}}
            cont = cont[k]['nested']

    def set(self, key: Key, flag: int, *, recursive: bool) -> None:
        cont = self._flags
        key_parent, key_stem = (key[:-1], key[-1])
        for k in key_parent:
            if k not in cont:
                cont[k] = {'flags': set(), 'recursive_flags': set(), 'nested': {}}
            cont = cont[k]['nested']
        if key_stem not in cont:
            cont[key_stem] = {'flags': set(), 'recursive_flags': set(), 'nested': {}}
        cont[key_stem]['recursive_flags' if recursive else 'flags'].add(flag)

    def is_(self, key: Key, flag: int) -> bool:
        if not key:
            return False
        cont = self._flags
        for k in key[:-1]:
            if k not in cont:
                return False
            inner_cont = cont[k]
            if flag in inner_cont['recursive_flags']:
                return True
            cont = inner_cont['nested']
        key_stem = key[-1]
        if key_stem in cont:
            cont = cont[key_stem]
            return flag in cont['flags'] or flag in cont['recursive_flags']
        return False

class NestedDict:

    def __init__(self) -> None:
        self.dict: Dict[str, Any] = {}

    def get_or_create_nest(self, key: Key, *, access_lists: bool = True) -> Dict[str, Any]:
        cont = self.dict
        for k in key:
            if k not in cont:
                cont[k] = {}
            cont = cont[k]
            if access_lists and isinstance(cont, list):
                cont = cont[-1]
            if not isinstance(cont, dict):
                raise KeyError('There is no nest behind this key')
        return cont

    def append_nest_to_list(self, key: Key) -> None:
        cont = self.get_or_create_nest(key[:-1])
        last_key = key[-1]
        if last_key in cont:
            list_ = cont[last_key]
            if not isinstance(list_, list):
                raise KeyError('An object other than list found behind this key')
            list_.append({})
        else:
            cont[last_key] = [{}]

class Output(NamedTuple):
    data: NestedDict
    flags: Flags

def skip_chars(src: str, pos: Pos, chars: FrozenSet[str]) -> Pos:
    try:
        while src[pos] in chars:
            pos += 1
    except IndexError:
        pass
    return pos

def skip_until(src: str, pos: Pos, expect: str, *, error_on: FrozenSet[str], error_on_eof: bool) -> Pos:
    try:
        new_pos = src.index(expect, pos)
    except ValueError:
        new_pos = len(src)
        if error_on_eof:
            raise suffixed_err(src, new_pos, f'Expected "{expect!r}"')
    if not error_on.isdisjoint(src[pos:new_pos]):
        while src[pos] not in error_on:
            pos += 1
        raise suffixed_err(src, pos, f'Found invalid character "{src[pos]!r}"')
    return new_pos

def skip_comment(src: str, pos: Pos) -> Pos:
    try:
        char = src[pos]
    except IndexError:
        char = None
    if char == '#':
        return skip_until(src, pos + 1, '\n', error_on=ILLEGAL_COMMENT_CHARS, error_on_eof=False)
    return pos

def skip_comments_and_array_ws(src: str, pos: Pos) -> Pos:
    while True:
        pos_before_skip = pos
        pos = skip_chars(src, pos, TOML_WS_AND_NEWLINE)
        pos = skip_comment(src, pos)
        if pos == pos_before_skip:
            return pos

def create_dict_rule(src: str, pos: Pos, out: Output) -> Tuple[Pos, Key]:
    pos += 1
    pos = skip_chars(src, pos, TOML_WS)
    pos, key = parse_key(src, pos)
    if out.flags.is_(key, Flags.EXPLICIT_NEST) or out.flags.is_(key, Flags.FROZEN):
        raise suffixed_err(src, pos, f'Can not declare {key} twice')
    out.flags.set(key, Flags.EXPLICIT_NEST, recursive=False)
    try:
        out.data.get_or_create_nest(key)
    except KeyError:
        raise suffixed_err(src, pos, 'Can not overwrite a value')
    if not src.startswith(']', pos):
        raise suffixed_err(src, pos, 'Expected "]" at the end of a table declaration')
    return (pos + 1, key)

def create_list_rule(src: str, pos: Pos, out: Output) -> Tuple[Pos, Key]:
    pos += 2
    pos = skip_chars(src, pos, TOML_WS)
    pos, key = parse_key(src, pos)
    if out.flags.is_(key, Flags.FROZEN):
        raise suffixed_err(src, pos, f'Can not mutate immutable namespace {key}')
    out.flags.unset_all(key)
    out.flags.set(key, Flags.EXPLICIT_NEST, recursive=False)
    try:
        out.data.append_nest_to_list(key)
    except KeyError:
        raise suffixed_err(src, pos, 'Can not overwrite a value')
    if not src.startswith(']]', pos):
        raise suffixed_err(src, pos, 'Expected "]]" at the end of an array declaration')
    return (pos + 2, key)

def key_value_rule(src: str, pos: Pos, out: Output, header: Key, parse_float: ParseFloat) -> Pos:
    pos, key, value = parse_key_value_pair(src, pos, parse_float)
    key_parent, key_stem = (key[:-1], key[-1])
    abs_key_parent = header + key_parent
    if out.flags.is_(abs_key_parent, Flags.FROZEN):
        raise suffixed_err(src, pos, f'Can not mutate immutable namespace {abs_key_parent}')
    out.flags.set_for_relative_key(header, key, Flags.EXPLICIT_NEST)
    try:
        nest = out.data.get_or_create_nest(abs_key_parent)
    except KeyError:
        raise suffixed_err(src, pos, 'Can not overwrite a value')
    if key_stem in nest:
        raise suffixed_err(src, pos, 'Can not overwrite a value')
    if isinstance(value, (dict, list)):
        out.flags.set(header + key, Flags.FROZEN, recursive=True)
    nest[key_stem] = value
    return pos

def parse_key_value_pair(src: str, pos: Pos, parse_float: ParseFloat) -> Tuple[Pos, Key, Any]:
    pos, key = parse_key(src, pos)
    try:
        char = src[pos]
    except IndexError:
        char = None
    if char != '=':
        raise suffixed_err(src, pos, 'Expected "=" after a key in a key/value pair')
    pos += 1
    pos = skip_chars(src, pos, TOML_WS)
    pos, value = parse_value(src, pos, parse_float)
    return (pos, key, value)

def parse_key(src: str, pos: Pos) -> Tuple[Pos, Key]:
    pos, key_part = parse_key_part(src, pos)
    key = (key_part,)
    pos = skip_chars(src, pos, TOML_WS)
    while True:
        try:
            char = src[pos]
        except IndexError:
            char = None
        if char != '.':
            return (pos, key)
        pos += 1
        pos = skip_chars(src, pos, TOML_WS)
        pos, key_part = parse_key_part(src, pos)
        key += (key_part,)
        pos = skip_chars(src, pos, TOML_WS)

def parse_key_part(src: str, pos: Pos) -> Tuple[Pos, str]:
    try:
        char = src[pos]
    except IndexError:
        char = None
    if char in BARE_KEY_CHARS:
        start_pos = pos
        pos = skip_chars(src, pos, BARE_KEY_CHARS)
        return (pos, src[start_pos:pos])
    if char == "'":
        return parse_literal_str(src, pos)
    if char == '"':
        return parse_one_line_basic_str(src, pos)
    raise suffixed_err(src, pos, 'Invalid initial character for a key part')

def parse_one_line_basic_str(src: str, pos: Pos) -> Tuple[Pos, str]:
    pos += 1
    return parse_basic_str(src, pos, multiline=False)

def parse_array(src: str, pos: Pos, parse_float: ParseFloat) -> Tuple[Pos, list]:
    pos += 1
    array = []
    pos = skip_comments_and_array_ws(src, pos)
    if src.startswith(']', pos):
        return (pos + 1, array)
    while True:
        pos, val = parse_value(src, pos, parse_float)
        array.append(val)
        pos = skip_comments_and_array_ws(src, pos)
        c = src[pos:pos + 1]
        if c == ']':
            return (pos + 1, array)
        if c != ',':
            raise suffixed_err(src, pos, 'Unclosed array')
        pos += 1
        pos = skip_comments_and_array_ws(src, pos)
        if src.startswith(']', pos):
            return (pos + 1, array)

def parse_inline_table(src: str, pos: Pos, parse_float: ParseFloat) -> Tuple[Pos, Dict[str, Any]]:
    pos += 1
    nested_dict = NestedDict()
    flags = Flags()
    pos = skip_chars(src, pos, TOML_WS)
    if src.startswith('}', pos):
        return (pos + 1, nested_dict.dict)
    while True:
        pos, key, value = parse_key_value_pair(src, pos, parse_float)
        key_parent, key_stem = (key[:-1], key[-1])
        if flags.is_(key, Flags.FROZEN):
            raise suffixed_err(src, pos, f'Can not mutate immutable namespace {key}')
        try:
            nest = nested_dict.get_or_create_nest(key_parent, access_lists=False)
        except KeyError:
            raise suffixed_err(src, pos, 'Can not overwrite a value')
        if key_stem in nest:
            raise suffixed_err(src, pos, f'Duplicate inline table key "{key_stem}"')
        nest[key_stem] = value
        pos = skip_chars(src, pos, TOML_WS)
        c = src[pos:pos + 1]
        if c == '}':
            return (pos + 1, nested_dict.dict)
        if c != ',':
            raise suffixed_err(src, pos, 'Unclosed inline table')
        if isinstance(value, (dict, list)):
            flags.set(key, Flags.FROZEN, recursive=True)
        pos += 1
        pos = skip_chars(src, pos, TOML_WS)

def parse_basic_str_escape(src: str, pos: Pos, *, multiline: bool = False) -> Tuple[Pos, str]:
    escape_id = src[pos:pos + 2]
    pos += 2
    if multiline and escape_id in {'\\ ', '\\\t', '\\\n'}:
        if escape_id != '\\\n':
            pos = skip_chars(src, pos, TOML_WS)
            try:
                char = src[pos]
            except IndexError:
                return (pos, '')
            if char != '\n':
                raise suffixed_err(src, pos, 'Unescaped "\\" in a string')
            pos += 1
        pos = skip_chars(src, pos, TOML_WS_AND_NEWLINE)
        return (pos, '')
    if escape_id == '\\u':
        return parse_hex_char(src, pos, 4)
    if escape_id == '\\U':
        return parse_hex_char(src, pos, 8)
    try:
        return (pos, BASIC_STR_ESCAPE_REPLACEMENTS[escape_id])
    except KeyError:
        if len(escape_id) != 2:
            raise suffixed_err(src, pos, 'Unterminated string')
        raise suffixed_err(src, pos, 'Unescaped "\\" in a string')

def parse_basic_str_escape_multiline(src: str, pos: Pos) -> Tuple[Pos, str]:
    return parse_basic_str_escape(src, pos, multiline=True)

def parse_hex_char(src: str, pos: Pos, hex_len: int) -> Tuple[Pos, str]:
    hex_str = src[pos:pos + hex_len]
    if len(hex_str) != hex_len or not HEXDIGIT_CHARS.issuperset(hex_str):
        raise suffixed_err(src, pos, 'Invalid hex value')
    pos += hex_len
    hex_int = int(hex_str, 16)
    if not is_unicode_scalar_value(hex_int):
        raise suffixed_err(src, pos, 'Escaped character is not a Unicode scalar value')
    return (pos, chr(hex_int))

def parse_literal_str(src: str, pos: Pos) -> Tuple[Pos, str]:
    pos += 1
    start_pos = pos
    pos = skip_until(src, pos, "'", error_on=ILLEGAL_LITERAL_STR_CHARS, error_on_eof=True)
    return (pos + 1, src[start_pos:pos])

def parse_multiline_str(src: str, pos: Pos, *, literal: bool) -> Tuple[Pos, str]:
    pos += 3
    if src.startswith('\n', pos):
        pos += 1
    if literal:
        delim = "'"
        end_pos = skip_until(src, pos, "'''", error_on=ILLEGAL_MULTILINE_LITERAL_STR_CHARS, error_on_eof=True)
        result = src[pos:end_pos]
        pos = end_pos + 3
    else:
        delim = '"'
        pos, result = parse_basic_str(src, pos, multiline=True)
    if not src.startswith(delim, pos):
        return (pos, result)
    pos += 1
    if not src.startswith(delim, pos):
        return (pos, result + delim)
    pos += 1
    return (pos, result + delim * 2)

def parse_basic_str(src: str, pos: Pos, *, multiline: bool) -> Tuple[Pos, str]:
    if multiline:
        error_on = ILLEGAL_MULTILINE_BASIC_STR_CHARS
        parse_escapes = parse_basic_str_escape_multiline
    else:
        error_on = ILLEGAL_BASIC_STR_CHARS
        parse_escapes = parse_basic_str_escape
    result = ''
    start_pos = pos
    while True:
        try:
            char = src[pos]
        except IndexError:
            raise suffixed_err(src, pos, 'Unterminated string')
        if char == '"':
            if not multiline:
                return (pos + 1, result + src[start_pos:pos])
            if src.startswith('"""', pos):
                return (pos + 3, result + src[start_pos:pos])
            pos += 1
            continue
        if char == '\\':
            result += src[start_pos:pos]
            pos, parsed_escape = parse_escapes(src, pos)
            result += parsed_escape
            start_pos = pos
            continue
        if char in error_on:
            raise suffixed_err(src, pos, f'Illegal character "{char!r}"')
        pos += 1

def parse_value(src: str, pos: Pos, parse_float: ParseFloat) -> Tuple[Pos, Any]:
    try:
        char = src[pos]
    except IndexError:
        char = None
    if char == '"':
        if src.startswith('"""', pos):
            return parse_multiline_str(src, pos, literal=False)
        return parse_one_line_basic_str(src, pos)
    if char == "'":
        if src.startswith("'''", pos):
            return parse_multiline_str(src, pos, literal=True)
        return parse_literal_str(src, pos)
    if char == 't':
        if src.startswith('true', pos):
            return (pos + 4, True)
    if char == 'f':
        if src.startswith('false', pos):
            return (pos + 5, False)
    datetime_match = RE_DATETIME.match(src, pos)
    if datetime_match:
        try:
            datetime_obj = match_to_datetime(datetime_match)
        except ValueError:
            raise suffixed_err(src, pos, 'Invalid date or datetime')
        return (datetime_match.end(), datetime_obj)
    localtime_match = RE_LOCALTIME.match(src, pos)
    if localtime_match:
        return (localtime_match.end(), match_to_localtime(localtime_match))
    number_match = RE_NUMBER.match(src, pos)
    if number_match:
        return (number_match.end(), match_to_number(number_match, parse_float))
    if char == '[':
        return parse_array(src, pos, parse_float)
    if char == '{':
        return parse_inline_table(src, pos, parse_float)
    first_three = src[pos:pos + 3]
    if first_three in {'inf', 'nan'}:
        return (pos + 3, parse_float(first_three))
    first_four = src[pos:pos + 4]
    if first_four in {'-inf', '+inf', '-nan', '+nan'}:
        return (pos + 4, parse_float(first_four))
    raise suffixed_err(src, pos, 'Invalid value')

def suffixed_err(src: str, pos: Pos, msg: str) -> TOMLDecodeError:
    """Return a `TOMLDecodeError` where error message is suffixed with
    coordinates in source."""

    def coord_repr(src: str, pos: Pos) -> str:
        if pos >= len(src):
            return 'end of document'
        line = src.count('\n', 0, pos) + 1
        if line == 1:
            column = pos + 1
        else:
            column = pos - src.rindex('\n', 0, pos)
        return f'line {line}, column {column}'
    return TOMLDecodeError(f'{msg} (at {coord_repr(src, pos)})')

def is_unicode_scalar_value(codepoint: int) -> bool:
    return 0 <= codepoint <= 55295 or 57344 <= codepoint <= 1114111
