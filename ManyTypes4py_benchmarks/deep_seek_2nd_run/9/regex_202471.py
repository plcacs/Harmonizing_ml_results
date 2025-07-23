import operator
import re
from hypothesis.errors import InvalidArgument
from hypothesis.internal import charmap
from hypothesis.strategies._internal.lazy import unwrap_strategies
from hypothesis.strategies._internal.strings import OneCharStringStrategy
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
try:
    import re._constants as sre
    import re._parser as sre_parse
    ATOMIC_GROUP = sre.ATOMIC_GROUP
    POSSESSIVE_REPEAT = sre.POSSESSIVE_REPEAT
except ImportError:
    import sre_constants as sre
    import sre_parse
    ATOMIC_GROUP = object()
    POSSESSIVE_REPEAT = object()
from hypothesis import reject, strategies as st
from hypothesis.internal.charmap import as_general_categories, categories
from hypothesis.internal.compat import add_note, int_to_byte
from hypothesis.strategies import SearchStrategy, DrawFn

UNICODE_CATEGORIES = set(categories())
SPACE_CHARS = set(' \t\n\r\x0c\x0b')
UNICODE_SPACE_CHARS = SPACE_CHARS | set('\x1c\x1d\x1e\x1f\x85')
UNICODE_DIGIT_CATEGORIES = {'Nd'}
UNICODE_SPACE_CATEGORIES = set(as_general_categories(['Z']))
UNICODE_LETTER_CATEGORIES = set(as_general_categories(['L']))
UNICODE_WORD_CATEGORIES = set(as_general_categories(['L', 'N']))
BYTES_ALL = {int_to_byte(i) for i in range(256)}
BYTES_DIGIT = {b for b in BYTES_ALL if re.match(b'\\d', b)}
BYTES_SPACE = {b for b in BYTES_ALL if re.match(b'\\s', b)}
BYTES_WORD = {b for b in BYTES_ALL if re.match(b'\\w', b)}
BYTES_LOOKUP = {sre.CATEGORY_DIGIT: BYTES_DIGIT, sre.CATEGORY_SPACE: BYTES_SPACE, sre.CATEGORY_WORD: BYTES_WORD, sre.CATEGORY_NOT_DIGIT: BYTES_ALL - BYTES_DIGIT, sre.CATEGORY_NOT_SPACE: BYTES_ALL - BYTES_SPACE, sre.CATEGORY_NOT_WORD: BYTES_ALL - BYTES_WORD}
GROUP_CACHE_STRATEGY = st.shared(st.builds(dict), key='hypothesis.regex.group_cache')

class IncompatibleWithAlphabet(InvalidArgument):
    pass

@st.composite
def update_group(draw: DrawFn, group_name: str, strategy: SearchStrategy[str]) -> str:
    cache = draw(GROUP_CACHE_STRATEGY)
    result = draw(strategy)
    cache[group_name] = result
    return result

@st.composite
def reuse_group(draw: DrawFn, group_name: str) -> str:
    cache = draw(GROUP_CACHE_STRATEGY)
    try:
        return cache[group_name]
    except KeyError:
        reject()

@st.composite
def group_conditional(draw: DrawFn, group_name: str, if_yes: SearchStrategy[str], if_no: SearchStrategy[str]) -> str:
    cache = draw(GROUP_CACHE_STRATEGY)
    if group_name in cache:
        return draw(if_yes)
    else:
        return draw(if_no)

@st.composite
def clear_cache_after_draw(draw: DrawFn, base_strategy: SearchStrategy[str]) -> str:
    cache = draw(GROUP_CACHE_STRATEGY)
    result = draw(base_strategy)
    cache.clear()
    return result

def chars_not_in_alphabet(alphabet: Optional[SearchStrategy[str]], string: str) -> Tuple[str, ...]:
    if alphabet is None:
        return ()
    intset = unwrap_strategies(alphabet).intervals
    return tuple((c for c in string if c not in intset))

class Context:
    __slots__ = ['flags']
    
    def __init__(self, flags: int) -> None:
        self.flags = flags

class CharactersBuilder:
    """Helper object that allows to configure `characters` strategy with
    various unicode categories and characters. Also allows negation of
    configured set.

    :param negate: If True, configure :func:`hypothesis.strategies.characters`
        to match anything other than configured character set
    :param flags: Regex flags. They affect how and which characters are matched
    """

    def __init__(self, *, negate: bool = False, flags: int = 0, alphabet: Optional[SearchStrategy[str]] = None) -> None:
        self._categories: Set[str] = set()
        self._whitelist_chars: Set[str] = set()
        self._blacklist_chars: Set[str] = set()
        self._negate = negate
        self._ignorecase = flags & re.IGNORECASE
        self.code_to_char = chr
        self._alphabet = unwrap_strategies(alphabet) if alphabet is not None else None
        if flags & re.ASCII and self._alphabet is not None:
            self._alphabet = OneCharStringStrategy(self._alphabet.intervals & charmap.query(max_codepoint=127))

    @property
    def strategy(self) -> SearchStrategy[str]:
        """Returns resulting strategy that generates configured char set."""
        white_chars = self._whitelist_chars - self._blacklist_chars
        multi_chars = {c for c in white_chars if len(c) > 1}
        intervals = charmap.query(categories=self._categories, exclude_characters=self._blacklist_chars, include_characters=white_chars - multi_chars)
        if self._negate:
            intervals = charmap.query() - intervals
            multi_chars.clear()
        return OneCharStringStrategy(intervals & (self._alphabet.intervals if self._alphabet is not None else charmap.query())) | (st.sampled_from(sorted(multi_chars)) if multi_chars else st.nothing())

    def add_category(self, category: int) -> None:
        """Update unicode state to match sre_parse object ``category``."""
        if category == sre.CATEGORY_DIGIT:
            self._categories |= UNICODE_DIGIT_CATEGORIES
        elif category == sre.CATEGORY_NOT_DIGIT:
            self._categories |= UNICODE_CATEGORIES - UNICODE_DIGIT_CATEGORIES
        elif category == sre.CATEGORY_SPACE:
            self._categories |= UNICODE_SPACE_CATEGORIES
            self._whitelist_chars |= UNICODE_SPACE_CHARS
        elif category == sre.CATEGORY_NOT_SPACE:
            self._categories |= UNICODE_CATEGORIES - UNICODE_SPACE_CATEGORIES
            self._blacklist_chars |= UNICODE_SPACE_CHARS
        elif category == sre.CATEGORY_WORD:
            self._categories |= UNICODE_WORD_CATEGORIES
            self._whitelist_chars.add('_')
        elif category == sre.CATEGORY_NOT_WORD:
            self._categories |= UNICODE_CATEGORIES - UNICODE_WORD_CATEGORIES
            self._blacklist_chars.add('_')
        else:
            raise NotImplementedError(f'Unknown character category: {category}')

    def add_char(self, c: str) -> None:
        """Add given char to the whitelist."""
        self._whitelist_chars.add(c)
        if self._ignorecase and re.match(re.escape(c), c.swapcase(), flags=re.IGNORECASE) is not None:
            self._whitelist_chars.add(c.swapcase())

class BytesBuilder(CharactersBuilder):

    def __init__(self, *, negate: bool = False, flags: int = 0) -> None:
        self._whitelist_chars: Set[bytes] = set()
        self._blacklist_chars: Set[bytes] = set()
        self._negate = negate
        self._alphabet = None
        self._ignorecase = flags & re.IGNORECASE
        self.code_to_char = int_to_byte

    @property
    def strategy(self) -> SearchStrategy[bytes]:
        """Returns resulting strategy that generates configured char set."""
        allowed = self._whitelist_chars
        if self._negate:
            allowed = BYTES_ALL - allowed
        return st.sampled_from(sorted(allowed))

    def add_category(self, category: int) -> None:
        """Update characters state to match sre_parse object ``category``."""
        self._whitelist_chars |= BYTES_LOOKUP[category]

@st.composite
def maybe_pad(draw: DrawFn, regex: re.Pattern, strategy: SearchStrategy[str], left_pad_strategy: SearchStrategy[str], right_pad_strategy: SearchStrategy[str]) -> str:
    """Attempt to insert padding around the result of a regex draw while
    preserving the match."""
    result = draw(strategy)
    left_pad = draw(left_pad_strategy)
    if left_pad and regex.search(left_pad + result):
        result = left_pad + result
    right_pad = draw(right_pad_strategy)
    if right_pad and regex.search(result + right_pad):
        result += right_pad
    return result

def base_regex_strategy(regex: re.Pattern, parsed: Optional[Any] = None, alphabet: Optional[SearchStrategy[str]] = None) -> SearchStrategy[str]:
    if parsed is None:
        parsed = sre_parse.parse(regex.pattern, flags=regex.flags)
    try:
        s = _strategy(parsed, context=Context(flags=regex.flags), is_unicode=isinstance(regex.pattern, str), alphabet=alphabet)
    except Exception as err:
        add_note(err, f'alphabet={alphabet!r} regex={regex!r}')
        raise
    return clear_cache_after_draw(s)

def regex_strategy(regex: Union[str, re.Pattern], fullmatch: bool, *, alphabet: Optional[SearchStrategy[str]], _temp_jsonschema_hack_no_end_newline: bool = False) -> SearchStrategy[str]:
    if not hasattr(regex, 'pattern'):
        regex = re.compile(regex)
    is_unicode = isinstance(regex.pattern, str)
    parsed = sre_parse.parse(regex.pattern, flags=regex.flags)
    if fullmatch:
        if not parsed:
            return st.just('' if is_unicode else b'')
        return base_regex_strategy(regex, parsed, alphabet).filter(regex.fullmatch)
    if not parsed:
        if is_unicode:
            return st.text(alphabet=alphabet)
        else:
            return st.binary()
    if is_unicode:
        base_padding_strategy = st.text(alphabet=alphabet)
        empty = st.just('')
        newline = st.just('\n')
    else:
        base_padding_strategy = st.binary()
        empty = st.just(b'')
        newline = st.just(b'\n')
    right_pad = base_padding_strategy
    left_pad = base_padding_strategy
    if parsed[-1][0] == sre.AT:
        if parsed[-1][1] == sre.AT_END_STRING:
            right_pad = empty
        elif parsed[-1][1] == sre.AT_END:
            if regex.flags & re.MULTILINE:
                right_pad = st.one_of(empty, st.builds(operator.add, newline, right_pad))
            else:
                right_pad = st.one_of(empty, newline)
            if _temp_jsonschema_hack_no_end_newline:
                right_pad = empty
    if parsed[0][0] == sre.AT:
        if parsed[0][1] == sre.AT_BEGINNING_STRING:
            left_pad = empty
        elif parsed[0][1] == sre.AT_BEGINNING:
            if regex.flags & re.MULTILINE:
                left_pad = st.one_of(empty, st.builds(operator.add, left_pad, newline))
            else:
                left_pad = empty
    base = base_regex_strategy(regex, parsed, alphabet).filter(regex.search)
    return maybe_pad(regex, base, left_pad, right_pad)

def _strategy(codes: Any, context: Context, is_unicode: bool, *, alphabet: Optional[SearchStrategy[str]]) -> SearchStrategy[str]:
    """Convert SRE regex parse tree to strategy that generates strings matching
    that regex represented by that parse tree.

    `codes` is either a list of SRE regex elements representations or a
    particular element representation. Each element is a tuple of element code
    (as string) and parameters. E.g. regex 'ab[0-9]+' compiles to following
    elements:

        [
            (LITERAL, 97),
            (LITERAL, 98),
            (MAX_REPEAT, (1, 4294967295, [
                (IN, [
                    (RANGE, (48, 57))
                ])
            ]))
        ]

    The function recursively traverses regex element tree and converts each
    element to strategy that generates strings that match that element.

    Context stores
    1. List of groups (for backreferences)
    2. Active regex flags (e.g. IGNORECASE, DOTALL, UNICODE, they affect
       behavior of various inner strategies)
    """

    def recurse(codes: Any) -> SearchStrategy[str]:
        return _strategy(codes, context, is_unicode, alphabet=alphabet)
    if is_unicode:
        empty = ''
        to_char = chr
    else:
        empty = b''
        to_char = int_to_byte
        binary_char = st.binary(min_size=1, max_size=1)
    if not isinstance(codes, tuple):
        strategies: List[SearchStrategy[str]] = []
        i = 0
        while i < len(codes):
            if codes[i][0] == sre.LITERAL and (not context.flags & re.IGNORECASE):
                j = i + 1
                while j < len(codes) and codes[j][0] == sre.LITERAL:
                    j += 1
                if i + 1 < j:
                    chars = empty.join((to_char(charcode) for _, charcode in codes[i:j]))
                    if (invalid := chars_not_in_alphabet(alphabet, chars)):
                        raise IncompatibleWithAlphabet(f'Literal {chars!r} contains characters {invalid!r} which are not in the specified alphabet')
                    strategies.append(st.just(chars))
                    i = j
                    continue
            strategies.append(recurse(codes[i]))
            i += 1
        if not strategies:
            return st.just(empty)
        if len(strategies) == 1:
            return strategies[0]
        return st.tuples(*strategies).map(empty.join)
    else:
        code, value = codes
        if code == sre.LITERAL:
            c = to_char(value)
            if chars_not_in_alphabet(alphabet, c):
                raise IncompatibleWithAlphabet(f'Literal {c!r} is not in the specified alphabet')
            if context.flags & re.IGNORECASE and c != c.swapcase() and (re.match(re.escape(c), c.swapcase(), re.IGNORECASE) is not None) and (not chars_not_in_alphabet(alphabet, c.swapcase())):
                return st.sampled_from([c, c.swapcase()])
            return st.just(c)
        elif code == sre.NOT_LITERAL:
            c = to_char(value)
            blacklist = {c}
            if context.flags & re.IGNORECASE and re.match(re.escape(c), c.swapcase(), re.IGNORECASE) is not None:
                stack = [c.swapcase()]
                while stack:
                    for char in stack.pop():
                        blacklist.add(char)
                        stack.extend(set(char.swapcase()) - blacklist)
            if is_unicode:
                return OneCharStringStrategy(unwrap_strategies(alphabet).intervals & charmap.query(exclude_characters=blacklist) if alphabet is not None else OneCharStringStrategy(charmap.query(exclude_characters=blacklist))
            else:
                return binary_char.filter(lambda c: c not in blacklist)
        elif code == sre.IN:
            negate = value[0][0] == sre.NEGATE
            if is_unicode:
                builder = CharactersBuilder(flags=context.flags, negate=negate, alphabet=alphabet)
            else:
                builder = BytesBuilder(flags=context.flags, negate=negate)
            for charset_code, charset_value in value:
                if charset_code == sre.NEGATE:
                    pass
                elif charset_code == sre.LITERAL:
                    c = builder.code_to_char(charset_value)
                    if chars_not_in_alphabet(builder._alphabet, c):
                        raise IncompatibleWithAlphabet(f'Literal {c!r} is not in the specified alphabet')
                    builder.add_char(c)
                elif charset_code == sre.RANGE:
                    low, high = charset_value
                    chars = empty.join(map(builder.code_to_char, range(low, high + 1)))
                    if len(chars) == len((invalid := set(chars_not_in_alphabet(alphabet, chars)))):
                        raise IncompatibleWithAlphabet(f"Charset '[{chr(low)}-{chr(high)}]' contains characters {invalid!r} which are not in the specified alphabet")
                    for c in chars:
                        if isinstance(c, int):
                            c = int_to_byte(c)
                        if c not in invalid:
                            builder.add_char(c)
                elif charset_code == sre.CATEGORY:
                    builder.add_category(charset_value)
                else:
                    raise NotImplementedError(f'Unknown charset code: {charset_code}')
            return builder.strategy
        elif code == sre.ANY:
            if is_unicode:
                assert alphabet is not None
                if context.flags & re.DOTALL