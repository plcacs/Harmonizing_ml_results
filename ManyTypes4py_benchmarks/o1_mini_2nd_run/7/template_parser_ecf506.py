from collections.abc import Callable
from typing import Optional, List
from typing_extensions import override

class FormattedError(Exception):
    pass

class TemplateParserError(Exception):

    def __init__(self, message: str) -> None:
        self.message: str = message

    @override
    def __str__(self) -> str:
        return self.message

class TokenizationError(Exception):

    def __init__(self, message: str, line_content: Optional[str] = None) -> None:
        self.message: str = message
        self.line_content: Optional[str] = line_content

class TokenizerState:

    def __init__(self) -> None:
        self.i: int = 0
        self.line: int = 1
        self.col: int = 1

class Token:

    def __init__(self, kind: str, s: str, tag: str, line: int, col: int, line_span: int) -> None:
        self.kind: str = kind
        self.s: str = s
        self.tag: str = tag
        self.line: int = line
        self.col: int = col
        self.line_span: int = line_span
        self.start_token: Optional['Token'] = None
        self.end_token: Optional['Token'] = None
        self.new_s: str = ''
        self.indent: Optional[str] = None
        self.orig_indent: Optional[str] = None
        self.child_indent: Optional[str] = None
        self.indent_is_final: bool = False
        self.parent_token: Optional['Token'] = None

def tokenize(text: str, template_format: Optional[str] = None) -> List[Token]:
    in_code_block: bool = False

    def advance(n: int) -> None:
        for _ in range(n):
            state.i += 1
            if state.i >= 0 and text[state.i - 1] == '\n':
                state.line += 1
                state.col = 1
            else:
                state.col += 1

    def looking_at(s: str) -> bool:
        return text[state.i:state.i + len(s)] == s

    def looking_at_htmlcomment() -> bool:
        return looking_at('<!--')

    def looking_at_handlebars_comment() -> bool:
        return looking_at('{{!')

    def looking_at_djangocomment() -> bool:
        return template_format == 'django' and looking_at('{#')

    def looking_at_handlebars_partial() -> bool:
        return template_format == 'handlebars' and looking_at('{{>')

    def looking_at_handlebars_partial_block() -> bool:
        return template_format == 'handlebars' and looking_at('{{#>')

    def looking_at_html_start() -> bool:
        return looking_at('<') and (not looking_at('</'))

    def looking_at_html_end() -> bool:
        return looking_at('</')

    def looking_at_handlebars_start() -> bool:
        return looking_at('{{#') or looking_at('{{^') or looking_at('{{~#')

    def looking_at_handlebars_else() -> bool:
        return template_format == 'handlebars' and looking_at('{{else')

    def looking_at_template_var() -> bool:
        return looking_at('{')

    def looking_at_handlebars_end() -> bool:
        return template_format == 'handlebars' and (looking_at('{{/') or looking_at('{{~/'))

    def looking_at_django_start() -> bool:
        return template_format == 'django' and looking_at('{% ')

    def looking_at_django_else() -> bool:
        return template_format == 'django' and (looking_at('{% else') or looking_at('{% elif') or looking_at('{%- else') or looking_at('{%- elif'))

    def looking_at_django_end() -> bool:
        return template_format == 'django' and looking_at('{% end')

    def looking_at_jinja2_end_whitespace_stripped() -> bool:
        return template_format == 'django' and looking_at('{%- end')

    def looking_at_jinja2_start_whitespace_stripped_type2() -> bool:
        return template_format == 'django' and looking_at('{%-') and (not looking_at('{%- end'))

    def looking_at_whitespace() -> bool:
        return looking_at('\n') or looking_at(' ')

    state = TokenizerState()
    tokens: List[Token] = []
    while state.i < len(text):
        try:
            if in_code_block:
                in_code_block = False
                s = get_code(text, state.i)
                if s == '':
                    continue
                tag = ''
                kind = 'code'
            elif looking_at_htmlcomment():
                s = get_html_comment(text, state.i)
                tag = s[4:-3]
                kind = 'html_comment'
            elif looking_at_handlebars_comment():
                s = get_handlebars_comment(text, state.i)
                tag = s[3:-2]
                kind = 'handlebars_comment'
            elif looking_at_djangocomment():
                s = get_django_comment(text, state.i)
                tag = s[2:-2]
                kind = 'django_comment'
            elif looking_at_handlebars_partial():
                s = get_handlebars_partial(text, state.i)
                tag = s[9:-2]
                kind = 'handlebars_partial'
            elif looking_at_handlebars_partial_block():
                s = get_handlebars_partial(text, state.i)
                tag = s[5:-2].split(None, 1)[0]
                kind = 'handlebars_partial_block'
            elif looking_at_html_start():
                s = get_html_tag(text, state.i)
                if s.endswith('/>'):
                    end_offset = -2
                else:
                    end_offset = -1
                tag_parts = s[1:end_offset].split()
                if not tag_parts:
                    raise TemplateParserError('Tag name missing')
                tag = tag_parts[0]
                if tag == '!DOCTYPE':
                    kind = 'html_doctype'
                elif s.endswith('/>'):
                    kind = 'html_singleton'
                else:
                    kind = 'html_start'
                if tag in ('code', 'pre', 'script'):
                    in_code_block = True
            elif looking_at_html_end():
                s = get_html_tag(text, state.i)
                tag = s[2:-1]
                kind = 'html_end'
            elif looking_at_handlebars_else():
                s = get_handlebars_tag(text, state.i)
                tag = 'else'
                kind = 'handlebars_else'
            elif looking_at_handlebars_start():
                s = get_handlebars_tag(text, state.i)
                tag = s[3:-2].split()[0].strip('#').removeprefix('*')
                kind = 'handlebars_start'
            elif looking_at_handlebars_end():
                s = get_handlebars_tag(text, state.i)
                tag = s[3:-2].strip('/#~')
                kind = 'handlebars_end'
            elif looking_at_django_else():
                s = get_django_tag(text, state.i)
                tag = 'else'
                kind = 'django_else'
            elif looking_at_django_end():
                s = get_django_tag(text, state.i)
                tag = s[6:-3]
                kind = 'django_end'
            elif looking_at_django_start():
                s = get_django_tag(text, state.i)
                tag = s[3:-2].split()[0]
                kind = 'django_start'
                if s[-3] == '-':
                    kind = 'jinja2_whitespace_stripped_start'
            elif looking_at_jinja2_end_whitespace_stripped():
                s = get_django_tag(text, state.i)
                tag = s[7:-3]
                kind = 'jinja2_whitespace_stripped_end'
            elif looking_at_jinja2_start_whitespace_stripped_type2():
                s = get_django_tag(text, state.i, stripped=True)
                tag = s[3:-3].split()[0]
                kind = 'jinja2_whitespace_stripped_type2_start'
            elif looking_at_template_var():
                s = get_template_var(text, state.i)
                tag = 'var'
                kind = 'template_var'
            elif looking_at('\n'):
                s = '\n'
                tag = 'newline'
                kind = 'newline'
            elif looking_at(' '):
                s = get_spaces(text, state.i)
                tag = ''
                if not tokens or tokens[-1].kind == 'newline':
                    kind = 'indent'
                else:
                    kind = 'whitespace'
            elif text[state.i] in '{<':
                snippet = text[state.i:][:15]
                raise AssertionError(f'tool cannot parse {snippet}')
            else:
                s = get_text(text, state.i)
                if s == '':
                    continue
                tag = ''
                kind = 'text'
        except TokenizationError as e:
            raise FormattedError(f'{e.message} at line {state.line} col {state.col}:"{e.line_content}"') from e
        line_span = len(s.strip('\n').split('\n'))
        token = Token(kind=kind, s=s, tag=tag.strip(), line=state.line, col=state.col, line_span=line_span)
        tokens.append(token)
        advance(len(s))
    return tokens

HTML_VOID_TAGS = {'area', 'base', 'br', 'col', 'command', 'embed', 'hr', 'img', 'input', 'keygen', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
HTML_INLINE_TAGS = {'a', 'b', 'br', 'button', 'cite', 'code', 'em', 'i', 'img', 'input', 'kbd', 'label', 'object', 'script', 'select', 'small', 'span', 'strong', 'textarea'}

def tag_flavor(token: Token) -> Optional[str]:
    kind = token.kind
    tag = token.tag
    if kind in ('code', 'django_comment', 'handlebars_comment', 'handlebars_partial', 'html_comment', 'html_doctype', 'html_singleton', 'indent', 'newline', 'template_var', 'text', 'whitespace'):
        return None
    if kind in ('handlebars_start', 'handlebars_partial_block', 'html_start'):
        return 'start'
    elif kind in ('django_else', 'django_end', 'handlebars_else', 'handlebars_end', 'html_end', 'jinja2_whitespace_stripped_end'):
        return 'end'
    elif kind in {'django_start', 'django_else', 'jinja2_whitespace_stripped_start', 'jinja2_whitespace_stripped_type2_start'}:
        if is_django_block_tag(tag):
            return 'start'
        else:
            return None
    else:
        raise AssertionError(f'tools programmer neglected to handle {kind} tokens')

def validate(fn: Optional[str] = None, text: Optional[str] = None, template_format: Optional[str] = None) -> List[Token]:
    assert fn or text
    if fn is None:
        fn = '<in memory file>'
    if text is None:
        with open(fn) as f:
            text = f.read()
    lines = text.split('\n')
    try:
        tokens = tokenize(text, template_format=template_format)
    except FormattedError as e:
        raise TemplateParserError(f'\n            fn: {fn}\n            {e}') from e
    prevent_whitespace_violations(fn, tokens)

    class State:

        def __init__(self, func: Callable[[Optional[Token]], None]) -> None:
            self.depth: int = 0
            self.foreign: bool = False
            self.matcher: Callable[[Optional[Token]], None] = func

    def no_start_tag(token: Optional[Token]) -> None:
        assert token
        raise TemplateParserError(f'\n            No start tag\n            fn: {fn}\n            end tag:\n                {token.tag}\n                line {token.line}, col {token.col}\n            ')

    state = State(no_start_tag)

    def start_tag_matcher(start_token: Token) -> None:
        state.depth += 1
        start_tag = start_token.tag.strip('~')
        start_line = start_token.line
        start_col = start_token.col
        old_matcher = state.matcher
        old_foreign = state.foreign
        if start_tag in ['math', 'svg']:
            state.foreign = True

        def f(end_token: Optional[Token]) -> None:
            if end_token is None:
                raise TemplateParserError(f"\n\n    Problem with {fn}\n    Missing end tag for the token at row {start_line} {start_col}!\n\n{start_token.s}\n\n    It's possible you have a typo in a token that you think is\n    matching this tag.\n                    ")
            is_else_tag = end_token.tag == 'else'
            end_tag = end_token.tag.strip('~')
            end_line = end_token.line
            end_col = end_token.col

            def report_problem() -> Optional[str]:
                if start_tag == 'code' and end_line == start_line + 1:
                    return 'Code tag is split across two lines.'
                if is_else_tag:
                    if start_tag not in ('if', 'else', 'unless'):
                        return f'Unexpected else/elif tag encountered after {start_tag} tag.'
                elif start_tag != end_tag:
                    return f'Mismatched tags: ({start_tag} != {end_tag})'
                return None
            problem = report_problem()
            if problem:
                raise TemplateParserError(f'\n                    fn: {fn}\n                   {problem}\n                    start:\n                        {start_token.s}\n                        line {start_line}, col {start_col}\n                    end tag:\n                        {end_tag}\n                        line {end_line}, col {end_col}\n                    ')
            if not is_else_tag:
                state.matcher = old_matcher
                state.foreign = old_foreign
                state.depth -= 1
            end_token.start_token = start_token
            start_token.end_token = end_token
        state.matcher = f

    for token in tokens:
        kind = token.kind
        tag = token.tag
        if not state.foreign:
            if kind == 'html_start' and tag in HTML_VOID_TAGS:
                raise TemplateParserError(f'Tag must be self-closing: {tag} at {fn} line {token.line}, col {token.col}')
            elif kind == 'html_singleton' and tag not in HTML_VOID_TAGS:
                raise TemplateParserError(f'Tag must not be self-closing: {tag} at {fn} line {token.line}, col {token.col}')
        flavor = tag_flavor(token)
        if flavor == 'start':
            start_tag_matcher(token)
        elif flavor == 'end':
            state.matcher(token)
    if state.depth != 0:
        state.matcher(None)
    ensure_matching_indentation(fn, tokens, lines)
    return tokens

def ensure_matching_indentation(fn: str, tokens: List[Token], lines: List[str]) -> None:

    def has_bad_indentation() -> bool:
        is_inline_tag = start_tag in HTML_INLINE_TAGS and start_token.kind == 'html_start'
        if end_line > start_line + 1:
            if is_inline_tag:
                end_row_text = lines[end_line - 1]
                if end_row_text.lstrip().startswith(end_token.s) and end_col != start_col:
                    return True
            elif end_col != start_col:
                return True
        return False

    for token in tokens:
        if token.start_token is None:
            continue
        end_token = token
        start_token = token.start_token
        start_line = start_token.line
        start_col = start_token.col
        start_tag = start_token.tag
        end_tag = end_token.tag.strip('~')
        end_line = end_token.line
        end_col = end_token.col
        if has_bad_indentation():
            raise TemplateParserError(f'\n                fn: {fn}\n                Indentation for start/end tags does not match.\n                start tag: {start_token.s}\n\n                start:\n                    line {start_line}, col {start_col}\n                end:\n                    {end_tag}\n                    line {end_line}, col {end_col}\n                ')

def prevent_extra_newlines(fn: str, tokens: List[Token]) -> None:
    count: int = 0
    for token in tokens:
        if token.kind != 'newline':
            count = 0
            continue
        count += 1
        if count >= 4:
            raise TemplateParserError(f'Please avoid so many blank lines near row {token.line} in {fn}.')

def prevent_whitespace_violations(fn: str, tokens: List[Token]) -> None:
    if tokens[0].kind in ('indent', 'whitespace'):
        raise TemplateParserError(f' Please remove the whitespace at the beginning of {fn}.')
    prevent_extra_newlines(fn, tokens)
    for i in range(1, len(tokens) - 1):
        token = tokens[i]
        next_token = tokens[i + 1]
        if token.kind == 'indent':
            if next_token.kind in ('indent', 'whitespace'):
                raise AssertionError('programming error parsing indents')
            if next_token.kind == 'newline':
                raise TemplateParserError(f'Please just make row {token.line} in {fn} a truly blank line (no spaces).')
            if len(token.s) % 4 != 0:
                raise TemplateParserError(f"\n                        Please use 4-space indents for template files. Most of our\n                        codebase (including Python and JavaScript) uses 4-space indents,\n                        so it's worth investing in configuring your editor to use\n                        4-space indents for files like\n                        {fn}\n\n                        The line at row {token.line} is indented with {len(token.s)} spaces.\n                    ")
        if token.kind == 'whitespace':
            if len(token.s) > 1:
                raise TemplateParserError(f'\n                        We did not expect this much whitespace at row {token.line} column {token.col} in {fn}.\n                    ')
            if next_token.kind == 'newline':
                raise TemplateParserError(f'\n                        Unexpected trailing whitespace at row {token.line} column {token.col} in {fn}.\n                    ')

def is_django_block_tag(tag: str) -> bool:
    return tag in ['autoescape', 'block', 'comment', 'for', 'if', 'ifequal', 'macro', 'verbatim', 'blocktrans', 'trans', 'raw', 'with']

def get_handlebars_tag(text: str, i: int) -> str:
    end: int = i + 2
    while end < len(text) - 1 and text[end] != '}':
        end += 1
    if text[end] != '}' or text[end + 1] != '}':
        raise TokenizationError('Tag missing "}}"', text[i:end + 2])
    s: str = text[i:end + 2]
    return s

def get_spaces(text: str, i: int) -> str:
    s: str = ''
    while i < len(text) and text[i] == ' ':
        s += text[i]
        i += 1
    return s

def get_code(text: str, i: int) -> str:
    s: str = ''
    while i < len(text) and text[i] not in '<':
        s += text[i]
        i += 1
    return s

def get_text(text: str, i: int) -> str:
    s: str = ''
    while i < len(text) and text[i] not in '{<':
        s += text[i]
        i += 1
    return s.strip()

def get_django_tag(text: str, i: int, stripped: bool = False) -> str:
    end: int = i + 2
    if stripped:
        end += 1
    while end < len(text) - 1 and text[end] != '%':
        end += 1
    if text[end] != '%' or text[end + 1] != '}':
        raise TokenizationError('Tag missing "%}"', text[i:end + 2])
    s: str = text[i:end + 2]
    return s

def get_html_tag(text: str, i: int) -> str:
    quote_count: int = 0
    end: int = i + 1
    unclosed_end: int = 0
    while end < len(text) and (text[end] != '>' or (quote_count % 2 != 0 and text[end] != '<')):
        if text[end] == '"':
            quote_count += 1
        if not unclosed_end and text[end] == '<':
            unclosed_end = end
        end += 1
    if quote_count % 2 != 0:
        if unclosed_end:
            raise TokenizationError('Unbalanced quotes', text[i:unclosed_end])
        else:
            raise TokenizationError('Unbalanced quotes', text[i:end + 1])
    if end == len(text) or text[end] != '>':
        raise TokenizationError('Tag missing ">"', text[i:end + 1])
    s: str = text[i:end + 1]
    return s

def get_html_comment(text: str, i: int) -> str:
    end: int = i + 7
    unclosed_end: int = 0
    while end <= len(text):
        if text[end - 3:end] == '-->':
            s: str = text[i:end]
            return s
        if not unclosed_end and text[end] == '<':
            unclosed_end = end
        end += 1
    raise TokenizationError('Unclosed comment', text[i:unclosed_end])

def get_handlebars_comment(text: str, i: int) -> str:
    end: int = i + 5
    unclosed_end: int = 0
    while end <= len(text):
        if text[end - 2:end] == '}}':
            s: str = text[i:end]
            return s
        if not unclosed_end and text[end] == '<':
            unclosed_end = end
        end += 1
    raise TokenizationError('Unclosed comment', text[i:unclosed_end])

def get_template_var(text: str, i: int) -> str:
    end: int = i + 3
    unclosed_end: int = 0
    while end <= len(text):
        if text[end - 1] == '}':
            if end < len(text) and text[end] == '}':
                end += 1
            s: str = text[i:end]
            return s
        if not unclosed_end and text[end] == '<':
            unclosed_end = end
        end += 1
    raise TokenizationError('Unclosed var', text[i:unclosed_end])

def get_django_comment(text: str, i: int) -> str:
    end: int = i + 4
    unclosed_end: int = 0
    while end <= len(text):
        if text[end - 2:end] == '#}':
            s: str = text[i:end]
            return s
        if not unclosed_end and text[end] == '<':
            unclosed_end = end
        end += 1
    raise TokenizationError('Unclosed comment', text[i:unclosed_end])

def get_handlebars_partial(text: str, i: int) -> str:
    """Works for both partials and partial blocks."""
    end: int = i + 10
    unclosed_end: int = 0
    while end <= len(text):
        if text[end - 2:end] == '}}':
            s: str = text[i:end]
            return s
        if not unclosed_end and text[end] == '<':
            unclosed_end = end
        end += 1
    raise TokenizationError('Unclosed partial', text[i:unclosed_end])
