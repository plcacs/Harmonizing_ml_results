import operator
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

from hypothesis.errors import InvalidArgument
from hypothesis.internal import charmap
from hypothesis.internal.charmap import as_general_categories, categories
from hypothesis.internal.compat import add_note, int_to_byte
from hypothesis.strategies._internal.lazy import unwrap_strategies
from hypothesis.strategies._internal.strings import OneCharStringStrategy
from hypothesis import reject, strategies as st

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

UNICODE_CATEGORIES: Set[str] = set(categories())
SPACE_CHARS: Set[str] = set(' \t\n\r\x0c\x0b')
UNICODE_SPACE_CHARS: Set[str] = SPACE_CHARS | set('\x1c\x1d\x1e\x1f\x85')
UNICODE_DIGIT_CATEGORIES: Set[str] = {'Nd'}
UNICODE_SPACE_CATEGORIES: Set[str] = set(as_general_categories(['Z']))
UNICODE_LETTER_CATEGORIES: Set[str] = set(as_general_categories(['L']))
UNICODE_WORD_CATEGORIES: Set[str] = set(as_general_categories(['L', 'N']))
BYTES_ALL: Set[bytes] = {int_to_byte(i) for i in range(256)}
BYTES_DIGIT: Set[bytes] = {b for b in BYTES_ALL if re.match(b'\\d', b)}
BYTES_SPACE: Set[bytes] = {b for b in BYTES_ALL if re.match(b'\\s', b)}
BYTES_WORD: Set[bytes] = {b for b in BYTES_ALL if re.match(b'\\w', b)}
BYTES_LOOKUP: Dict[Any, Set[bytes]] = {
    sre.CATEGORY_DIGIT: BYTES_DIGIT,
    sre.CATEGORY_SPACE: BYTES_SPACE,
    sre.CATEGORY_WORD: BYTES_WORD,
    sre.CATEGORY_NOT_DIGIT: BYTES_ALL - BYTES_DIGIT,
    sre.CATEGORY_NOT_SPACE: BYTES_ALL - BYTES_SPACE,
    sre.CATEGORY_NOT_WORD: BYTES_ALL - BYTES_WORD,
}
GROUP_CACHE_STRATEGY = st.shared(st.builds(dict), key='hypothesis.regex.group_cache')


class IncompatibleWithAlphabet(InvalidArgument):
    pass


@st.composite
def update_group(
    draw: Callable[[st.SearchStrategy[Any]], Any],
    group_name: str,
    strategy: st.SearchStrategy[Any]
) -> Any:
    cache: Dict[str, Any] = draw(GROUP_CACHE_STRATEGY)
    result: Any = draw(strategy)
    cache[group_name] = result
    return result


@st.composite
def reuse_group(
    draw: Callable[[st.SearchStrategy[Any]], Any],
    group_name: str
) -> Any:
    cache: Dict[str, Any] = draw(GROUP_CACHE_STRATEGY)
    try:
        return cache[group_name]
    except KeyError:
        reject()


@st.composite
def group_conditional(
    draw: Callable[[st.SearchStrategy[Any]], Any],
    group_name: str,
    if_yes: st.SearchStrategy[Any],
    if_no: st.SearchStrategy[Any]
) -> Any:
    cache: Dict[str, Any] = draw(GROUP_CACHE_STRATEGY)
    if group_name in cache:
        return draw(if_yes)
    else:
        return draw(if_no)


@st.composite
def clear_cache_after_draw(
    draw: Callable[[st.SearchStrategy[Any]], Any],
    base_strategy: st.SearchStrategy[Any]
) -> Any:
    cache: Dict[str, Any] = draw(GROUP_CACHE_STRATEGY)
    result: Any = draw(base_strategy)
    cache.clear()
    return result


def chars_not_in_alphabet(alphabet: Optional[unwrap_strategies], string: str) -> Tuple[str, ...]:
    if alphabet is None:
        return ()
    intset = unwrap_strategies(alphabet).intervals
    return tuple(c for c in string if c not in intset)


class Context:
    __slots__ = ['flags']

    flags: int

    def __init__(self, flags: int) -> None:
        self.flags = flags


class CharactersBuilder:
    """Helper object that allows to configure `characters` strategy with
    various unicode categories and characters. Also allows negation of
    configured set.

    :param negate: If True, configure :func:`hypothesis.strategies.characters`
        to match anything other than configured character set
    :param flags: Regex flags. They affect how and which characters are matched
    :param alphabet: The alphabet to restrict characters to
    """

    def __init__(
        self,
        *,
        negate: bool = False,
        flags: int = 0,
        alphabet: st.SearchStrategy[Any]
    ) -> None:
        self._categories: Set[str] = set()
        self._whitelist_chars: Set[str] = set()
        self._blacklist_chars: Set[str] = set()
        self._negate: bool = negate
        self._ignorecase: bool = bool(flags & re.IGNORECASE)
        self.code_to_char: Callable[[int], str] = chr
        self._alphabet: unwrap_strategies = unwrap_strategies(alphabet)
        if flags & re.ASCII:
            self._alphabet = OneCharStringStrategy(self._alphabet.intervals & charmap.query(max_codepoint=127))

    @property
    def strategy(self) -> st.SearchStrategy[str]:
        """Returns resulting strategy that generates configured char set."""
        white_chars: Set[str] = self._whitelist_chars - self._blacklist_chars
        multi_chars: Set[str] = {c for c in white_chars if len(c) > 1}
        intervals = charmap.query(
            categories=self._categories,
            exclude_characters=self._blacklist_chars,
            include_characters=self._whitelist_chars - multi_chars
        )
        if self._negate:
            intervals = charmap.query() - intervals
            multi_chars.clear()
        base_strategy = OneCharStringStrategy(intervals & self._alphabet.intervals)
        if multi_chars:
            base_strategy |= st.sampled_from(sorted(multi_chars))
        return base_strategy

    def add_category(self, category: Any) -> None:
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
        if self._ignorecase and re.match(re.escape(c), c.swapcase(), flags=re.IGNORECASE):
            self._whitelist_chars.add(c.swapcase())


class BytesBuilder(CharactersBuilder):
    def __init__(
        self,
        *,
        negate: bool = False,
        flags: int = 0
    ) -> None:
        self._whitelist_chars: Set[bytes] = set()
        self._blacklist_chars: Set[bytes] = set()
        self._negate: bool = negate
        self._alphabet: Optional[Any] = None
        self._ignorecase: bool = bool(flags & re.IGNORECASE)
        self.code_to_char: Callable[[int], bytes] = int_to_byte

    @property
    def strategy(self) -> st.SearchStrategy[bytes]:
        """Returns resulting strategy that generates configured char set."""
        allowed: Set[bytes] = self._whitelist_chars
        if self._negate:
            allowed = BYTES_ALL - allowed
        return st.sampled_from(sorted(allowed))

    def add_category(self, category: Any) -> None:
        """Update characters state to match sre_parse object ``category``."""
        self._whitelist_chars |= BYTES_LOOKUP[category]


@st.composite
def maybe_pad(
    draw: Callable[[st.SearchStrategy[Any]], Any],
    regex: re.Pattern[Any],
    strategy: st.SearchStrategy[str],
    left_pad_strategy: st.SearchStrategy[str],
    right_pad_strategy: st.SearchStrategy[str]
) -> str:
    """Attempt to insert padding around the result of a regex draw while
    preserving the match."""
    result: str = draw(strategy)
    left_pad: Optional[str] = draw(left_pad_strategy)
    if left_pad and regex.search(left_pad + result):
        result = left_pad + result
    right_pad: Optional[str] = draw(right_pad_strategy)
    if right_pad and regex.search(result + right_pad):
        result += right_pad
    return result


def base_regex_strategy(
    regex: re.Pattern[Any],
    parsed: Optional[sre_parse.SubPattern] = None,
    alphabet: Optional[st.SearchStrategy[Any]] = None
) -> st.SearchStrategy[str]:
    if parsed is None:
        parsed = sre_parse.parse(regex.pattern, flags=regex.flags)
    try:
        s: st.SearchStrategy[str] = _strategy(
            parsed,
            context=Context(flags=regex.flags),
            is_unicode=isinstance(regex.pattern, str),
            alphabet=alphabet
        )
    except Exception as err:
        add_note(err, f'alphabet={alphabet!r} regex={regex!r}')
        raise
    return clear_cache_after_draw(s)


def regex_strategy(
    regex: Union[str, re.Pattern[Any]],
    fullmatch: bool,
    *,
    alphabet: Optional[st.SearchStrategy[Any]],
    _temp_jsonschema_hack_no_end_newline: bool = False
) -> st.SearchStrategy[Union[str, bytes]]:
    if not hasattr(regex, 'pattern'):
        regex = re.compile(regex)
    is_unicode: bool = isinstance(regex.pattern, str)
    parsed: sre_parse.SubPattern = sre_parse.parse(regex.pattern, flags=regex.flags)
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
        base_padding_strategy: st.SearchStrategy[str] = st.text(alphabet=alphabet)
        empty: st.SearchStrategy[str] = st.just('')
        newline: st.SearchStrategy[str] = st.just('\n')
    else:
        base_padding_strategy = st.binary()
        empty = st.just(b'')
        newline = st.just(b'\n')
    right_pad: st.SearchStrategy[Any] = base_padding_strategy
    left_pad: st.SearchStrategy[Any] = base_padding_strategy
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
    base: st.SearchStrategy[Any] = base_regex_strategy(regex, parsed, alphabet).filter(regex.search)
    return maybe_pad(regex, base, left_pad, right_pad)


def _strategy(
    codes: Union[sre_parse.SubPattern, List[Tuple[Any, Any]]],
    context: Context,
    is_unicode: bool,
    *,
    alphabet: Optional[st.SearchStrategy[Any]]
) -> st.SearchStrategy[Union[str, bytes]]:
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
    if is_unicode:
        empty: str = ''
        to_char: Callable[[int], str] = chr
    else:
        empty = b''
        to_char = int_to_byte
        binary_char: st.SearchStrategy[bytes] = st.binary(min_size=1, max_size=1)

    def recurse(c: Union[Tuple[Any, Any], List[Tuple[Any, Any]]]) -> st.SearchStrategy[Any]:
        return _strategy(c, context, is_unicode, alphabet=alphabet)

    if not isinstance(codes, tuple):
        strategies: List[st.SearchStrategy[Any]] = []
        i: int = 0
        while i < len(codes):
            if codes[i][0] == sre.LITERAL and not (context.flags & re.IGNORECASE):
                j: int = i + 1
                while j < len(codes) and codes[j][0] == sre.LITERAL:
                    j += 1
                if i + 1 < j:
                    chars: str = empty.join(to_char(charcode) for _, charcode in codes[i:j])
                    invalid = chars_not_in_alphabet(alphabet, chars)
                    if invalid:
                        raise IncompatibleWithAlphabet(
                            f'Literal {chars!r} contains characters {invalid!r} which are not in the specified alphabet'
                        )
                    strategies.append(st.just(chars))
                    i = j
                    continue
            strategies.append(recurse(codes[i]))
            i += 1
        if not strategies:
            return st.just(empty)
        if len(strategies) == 1:
            return strategies[0]
        return st.tuples(*strategies).map(lambda parts: empty.join(parts))  # type: ignore

    else:
        code, value = codes
        if code == sre.LITERAL:
            c: Union[str, bytes] = to_char(value)
            if chars_not_in_alphabet(alphabet, c):
                raise IncompatibleWithAlphabet(
                    f'Literal {c!r} is not in the specified alphabet'
                )
            if context.flags & re.IGNORECASE and c != getattr(c, 'swapcase', lambda: c)() and \
               re.match(re.escape(c), getattr(c, 'swapcase', lambda: c)(), flags=re.IGNORECASE) and \
               not chars_not_in_alphabet(alphabet, getattr(c, 'swapcase', lambda: c)()):
                return st.sampled_from([c, getattr(c, 'swapcase', lambda: c)()])  # type: ignore
            return st.just(c)
        elif code == sre.NOT_LITERAL:
            c: Union[str, bytes] = to_char(value)
            blacklist: Set[Union[str, bytes]] = {c}
            if context.flags & re.IGNORECASE and re.match(re.escape(c), getattr(c, 'swapcase', lambda: c)(), flags=re.IGNORECASE):
                stack: List[Union[str, bytes]] = [getattr(c, 'swapcase', lambda: c)()]
                while stack:
                    for char in stack.pop():
                        blacklist.add(char)
                        swapped = getattr(char, 'swapcase', lambda: char)()
                        if swapped not in blacklist:
                            stack.append(swapped)
            if is_unicode:
                return OneCharStringStrategy(
                    unwrap_strategies(alphabet).intervals & charmap.query(exclude_characters=blacklist)
                )
            else:
                return binary_char.filter(lambda c: c not in blacklist)
        elif code == sre.IN:
            negate: bool = value[0][0] == sre.NEGATE
            if is_unicode:
                builder = CharactersBuilder(flags=context.flags, negate=negate, alphabet=alphabet)
            else:
                builder = BytesBuilder(flags=context.flags, negate=negate)
            for charset_code, charset_value in value:
                if charset_code == sre.NEGATE:
                    continue
                elif charset_code == sre.LITERAL:
                    c = builder.code_to_char(charset_value)
                    if chars_not_in_alphabet(builder._alphabet, c):
                        raise IncompatibleWithAlphabet(
                            f'Literal {c!r} is not in the specified alphabet'
                        )
                    builder.add_char(c)
                elif charset_code == sre.RANGE:
                    low, high = charset_value
                    chars = empty.join(map(builder.code_to_char, range(low, high + 1)))
                    invalid = set(chars_not_in_alphabet(alphabet, chars))
                    if len(chars) == len(invalid):
                        raise IncompatibleWithAlphabet(
                            f"Charset '[{chr(low)}-{chr(high)}]' contains characters {invalid!r} which are not in the specified alphabet"
                        )
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
                if context.flags & re.DOTALL:
                    return alphabet
                return OneCharStringStrategy(unwrap_strategies(alphabet).intervals & charmap.query(exclude_characters='\n'))
            else:
                if context.flags & re.DOTALL:
                    return binary_char
                return binary_char.filter(lambda c: c != b'\n')
        elif code == sre.AT:
            return st.just(empty)
        elif code == sre.SUBPATTERN:
            old_flags: int = context.flags
            context.flags = (context.flags | value[1]) & ~value[2]
            strat: st.SearchStrategy[Any] = _strategy(value[-1], context, is_unicode, alphabet=alphabet)
            context.flags = old_flags
            if value[0]:
                strat = update_group(value[0], strat)
            return strat
        elif code == sre.GROUPREF:
            return reuse_group(value)
        elif code == sre.ASSERT:
            return recurse(value[1])
        elif code == sre.ASSERT_NOT:
            return st.just(empty)
        elif code == sre.BRANCH:
            branches: List[st.SearchStrategy[Any]] = []
            errors: List[str] = []
            for branch in value[1]:
                try:
                    branches.append(recurse(branch))
                except IncompatibleWithAlphabet as e:
                    errors.append(str(e))
            if errors and not branches:
                raise IncompatibleWithAlphabet('\n'.join(errors))
            return st.one_of(branches)
        elif code in {sre.MIN_REPEAT, sre.MAX_REPEAT, POSSESSIVE_REPEAT}:
            at_least: int
            at_most: Optional[int]
            subregex: Union[sre_parse.SubPattern, List[Tuple[Any, Any]]]
            at_least, at_most, subregex = value
            if at_most == sre.MAXREPEAT:
                at_most = None
            if at_least == 0 and at_most == 1:
                return st.just(empty) | recurse(subregex)
            return st.lists(recurse(subregex), min_size=at_least, max_size=at_most).map(lambda parts: empty.join(parts))  # type: ignore
        elif code == sre.GROUPREF_EXISTS:
            conditional_strat: st.SearchStrategy[Any] = group_conditional(
                value[0],
                recurse(value[1]),
                recurse(value[2]) if value[2] else st.just(empty)
            )
            return conditional_strat
        elif code == ATOMIC_GROUP:
            return _strategy(value, context, is_unicode, alphabet=alphabet)
        else:
            raise NotImplementedError(f'Unknown code point: {code!r}.  Please open an issue.')
