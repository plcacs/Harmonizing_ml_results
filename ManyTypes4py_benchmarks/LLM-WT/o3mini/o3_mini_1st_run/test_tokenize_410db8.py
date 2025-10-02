import io
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Tuple, Callable
import black
from blib2to3.pgen2 import token, tokenize


@dataclass
class Token:
    type: str
    string: str
    start: Tuple[int, int]
    end: Tuple[int, int]


def get_tokens(text: str) -> List[Token]:
    """Return the tokens produced by the tokenizer."""
    readline = io.StringIO(text).readline
    tokens: List[Token] = []

    def tokeneater(
        tok_type: int,
        tok_string: str,
        start: Tuple[int, int],
        end: Tuple[int, int],
        line: str,
    ) -> None:
        tokens.append(Token(token.tok_name[tok_type], tok_string, start, end))

    tokenize.tokenize(readline, tokeneater)
    return tokens


def assert_tokenizes(text: str, expected_tokens: List[Token]) -> None:
    """Assert that the tokenizer produces the expected tokens."""
    actual_tokens: List[Token] = get_tokens(text)
    assert actual_tokens == expected_tokens


def test_simple() -> None:
    assert_tokenizes(
        "1",
        [
            Token("NUMBER", "1", (1, 0), (1, 1)),
            Token("ENDMARKER", "", (2, 0), (2, 0)),
        ],
    )
    assert_tokenizes(
        "'a'",
        [
            Token("STRING", "'a'", (1, 0), (1, 3)),
            Token("ENDMARKER", "", (2, 0), (2, 0)),
        ],
    )
    assert_tokenizes(
        "a",
        [
            Token("NAME", "a", (1, 0), (1, 1)),
            Token("ENDMARKER", "", (2, 0), (2, 0)),
        ],
    )


def test_fstring() -> None:
    assert_tokenizes(
        'f"x"',
        [
            Token("FSTRING_START", 'f"', (1, 0), (1, 2)),
            Token("FSTRING_MIDDLE", "x", (1, 2), (1, 3)),
            Token("FSTRING_END", '"', (1, 3), (1, 4)),
            Token("ENDMARKER", "", (2, 0), (2, 0)),
        ],
    )
    assert_tokenizes(
        'f"{x}"',
        [
            Token("FSTRING_START", 'f"', (1, 0), (1, 2)),
            Token("FSTRING_MIDDLE", "", (1, 2), (1, 2)),
            Token("LBRACE", "{", (1, 2), (1, 3)),
            Token("NAME", "x", (1, 3), (1, 4)),
            Token("RBRACE", "}", (1, 4), (1, 5)),
            Token("FSTRING_MIDDLE", "", (1, 5), (1, 5)),
            Token("FSTRING_END", '"', (1, 5), (1, 6)),
            Token("ENDMARKER", "", (2, 0), (2, 0)),
        ],
    )
    assert_tokenizes(
        'f"{x:y}"\n',
        [
            Token("FSTRING_START", 'f"', (1, 0), (1, 2)),
            Token("FSTRING_MIDDLE", "", (1, 2), (1, 2)),
            Token("LBRACE", "{", (1, 2), (1, 3)),
            Token("NAME", "x", (1, 3), (1, 4)),
            Token("OP", ":", (1, 4), (1, 5)),
            Token("FSTRING_MIDDLE", "y", (1, 5), (1, 6)),
            Token("RBRACE", "}", (1, 6), (1, 7)),
            Token("FSTRING_MIDDLE", "", (1, 7), (1, 7)),
            Token("FSTRING_END", '"', (1, 7), (1, 8)),
            Token("NEWLINE", "\n", (1, 8), (1, 9)),
            Token("ENDMARKER", "", (2, 0), (2, 0)),
        ],
    )
    assert_tokenizes(
        'f"x\\\n{a}"\n',
        [
            Token("FSTRING_START", 'f"', (1, 0), (1, 2)),
            Token("FSTRING_MIDDLE", "x\\\n", (1, 2), (2, 0)),
            Token("LBRACE", "{", (2, 0), (2, 1)),
            Token("NAME", "a", (2, 1), (2, 2)),
            Token("RBRACE", "}", (2, 2), (2, 3)),
            Token("FSTRING_MIDDLE", "", (2, 3), (2, 3)),
            Token("FSTRING_END", '"', (2, 3), (2, 4)),
            Token("NEWLINE", "\n", (2, 4), (2, 5)),
            Token("ENDMARKER", "", (3, 0), (3, 0)),
        ],
    )


if __name__ == "__main__":
    code: str = sys.stdin.read()
    tokens: List[Token] = get_tokens(code)
    text: str = f"assert_tokenizes({code!r}, {tokens!r})"
    text = black.format_str(text, mode=black.Mode())
    print(textwrap.indent(text, "    "))