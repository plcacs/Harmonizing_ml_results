import re
from codecs import BOM_UTF8
from typing import Tuple, Iterator, Optional, Any
from parso.python.tokenize import group

unicode_bom: str = BOM_UTF8.decode('utf-8')


class PrefixPart:
    parent: Any
    type: str
    value: str
    spacing: str
    start_pos: Tuple[int, int]

    def __init__(self, leaf: Any, typ: str, value: str, spacing: str = '', start_pos: Tuple[int, int] = (0, 0)) -> None:
        assert start_pos is not None
        self.parent = leaf
        self.type = typ
        self.value = value
        self.spacing = spacing
        self.start_pos = start_pos

    @property
    def end_pos(self) -> Tuple[int, int]:
        if self.value.endswith('\n') or self.value.endswith('\r'):
            return (self.start_pos[0] + 1, 0)
        if self.value == unicode_bom:
            return self.start_pos
        return (self.start_pos[0], self.start_pos[1] + len(self.value))

    def create_spacing_part(self) -> "PrefixPart":
        column: int = self.start_pos[1] - len(self.spacing)
        return PrefixPart(self.parent, 'spacing', self.spacing, start_pos=(self.start_pos[0], column))

    def __repr__(self) -> str:
        return '%s(%s, %s, %s)' % (self.__class__.__name__, self.type, repr(self.value), self.start_pos)

    def search_ancestor(self, *node_types: str) -> Optional[Any]:
        node: Optional[Any] = self.parent
        while node is not None:
            if node.type in node_types:
                return node
            node = node.parent
        return None


_comment: str = r'#[^\n\r\f]*'
_backslash: str = r'\\\r?\n|\\\r'
_newline: str = r'\r?\n|\r'
_form_feed: str = r'\f'
_only_spacing: str = r'$'
_spacing: str = r'[ \t]*'
_bom: str = unicode_bom
_regex: str = group(_comment, _backslash, _newline, _form_feed, _only_spacing, _bom, capture=True)
_regex = re.compile(group(_spacing, capture=True) + _regex)
_types: dict = {'#': 'comment', '\\': 'backslash', '\x0c': 'formfeed', '\n': 'newline', '\r': 'newline', unicode_bom: 'bom'}


def split_prefix(leaf: Any, start_pos: Tuple[int, int]) -> Iterator[PrefixPart]:
    line: int
    column: int
    line, column = start_pos
    start: int = 0
    value: str = ''
    spacing: str = ''
    bom: bool = False
    while start != len(leaf.prefix):
        match = _regex.match(leaf.prefix, start)
        if not match:
            break
        spacing = match.group(1)
        value = match.group(2)
        if not value:
            break
        type_ = _types[value[0]]
        yield PrefixPart(
            leaf, 
            type_, 
            value, 
            spacing, 
            start_pos=(line, column + start - int(bom) + len(spacing))
        )
        if type_ == 'bom':
            bom = True
        start = match.end(0)
        if value.endswith('\n') or value.endswith('\r'):
            line += 1
            column = -start
    if value:
        spacing = ''
    yield PrefixPart(leaf, 'spacing', spacing, start_pos=(line, column + start))