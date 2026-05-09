from collections.abc import Callable
from typing_extensions import override

class FormattedError(Exception):
    pass

class TemplateParserError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

    @override
    def __str__(self) -> str:
        return self.message

class TokenizationError(Exception):
    def __init__(self, message: str, line_content: str = '') -> None:
        self.message = message
        self.line_content = line_content

class TokenizerState:
    def __init__(self) -> None:
        self.i: int = 0
        self.line: int = 1
        self.col: int = 1

class Token:
    def __init__(self, 
                 kind: str, 
                 s: str, 
                 tag: str, 
                 line: int, 
                 col: int, 
                 line_span: int) -> None:
        self.kind: str = kind
        self.s: str = s
        self.tag: str = tag
        self.line: int = line
        self.col: int = col
        self.line_span: int = line_span
        self.start_token: Token | None = None
        self.end_token: Token | None = None
        self.new_s: str = ''
        self.indent: str | None = None
        self.orig_indent: str | None = None
        self.child_indent: str | None = None
        self.indent_is_final: bool = False
        self.parent_token: Token | None = None

def tokenize(text: str, template_format: str | None = None) -> list[Token]:
    ...

def tag_flavor(token: Token) -> str | None:
    ...

def validate(fn: str, text: str | None = None, template_format: str | None = None) -> list[Token]:
    ...

def ensure_matching_indentation(fn: str, tokens: list[Token], lines: list[str]) -> None:
    ...

def prevent_extra_newlines(fn: str, tokens: list[Token]) -> None:
    ...

def prevent_whitespace_violations(fn: str, tokens: list[Token]) -> None:
    ...

def is_django_block_tag(tag: str) -> bool:
    ...

def get_handlebars_tag(text: str, i: int) -> str:
    ...

def get_spaces(text: str, i: int) -> str:
    ...

def get_code(text: str, i: int) -> str:
    ...

def get_text(text: str, i: int) -> str:
    ...

def get_django_tag(text: str, i: int, stripped: bool = False) -> str:
    ...

def get_html_tag(text: str, i: int) -> str:
    ...

def get_html_comment(text: str, i: int) -> str:
    ...

def get_handlebars_comment(text: str, i: int) -> str:
    ...

def get_template_var(text: str, i: int) -> str:
    ...

def get_django_comment(text: str, i: int) -> str:
    ...

def get_handlebars_partial(text: str, i: int) -> str:
    ...
