"""Safely evaluate Python string literals without using eval()."""
import re
from typing import Dict, Match, Tuple, Pattern

simple_escapes: Dict[str, str] = {
    'a': '\x07',
    'b': '\x08',
    'f': '\x0c',
    'n': '\n',
    'r': '\r',
    't': '\t',
    'v': '\x0b',
    "'": "'",
    '"': '"',
    '\\': '\\'
}

def escape(m: Match[str]) -> str:
    groups: Tuple[str, str] = m.group(0, 1)
    all: str = groups[0]
    tail: str = groups[1]
    assert all.startswith('\\')
    esc: str | None = simple_escapes.get(tail)
    if esc is not None:
        return esc
    if tail.startswith('x'):
        hexes: str = tail[1:]
        if len(hexes) < 2:
            raise ValueError(f"invalid hex string escape ('\\{tail}')")
        try:
            i: int = int(hexes, 16)
        except ValueError:
            raise ValueError(f"invalid hex string escape ('\\{tail}')") from None
    else:
        try:
            i = int(tail, 8)
        except ValueError:
            raise ValueError(f"invalid octal string escape ('\\{tail}')") from None
    return chr(i)

def evalString(s: str) -> str:
    assert s.startswith("'") or s.startswith('"'), repr(s[:1])
    q: str = s[0]
    if s[:3] == (q * 3):
        q = q * 3
    assert s.endswith(q), repr(s[-len(q):])
    assert len(s) >= (2 * len(q))
    s = s[len(q):-len(q)]
    return re.sub(r'\\([\'"]|\\|[abfnrtv]|x.{0,2}|[0-7]{1,3})', escape, s)

def test() -> None:
    for i in range(256):
        c: str = chr(i)
        s: str = repr(c)
        e: str = evalString(s)
        if e != c:
            print(i, c, s, e)

if __name__ == '__main__':
    test()
