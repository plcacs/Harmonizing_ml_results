from collections.abc import Iterable, Iterator
from typing import Any, Optional, TypeVar, Union, Dict, List, Tuple, Sequence, Set, FrozenSet, cast
from blib2to3.pgen2.grammar import Grammar
import sys
from io import StringIO

__author__ = 'Guido van Rossum <guido@python.org>'

HUGE = 2147483647
_type_reprs: Dict[int, str] = {}

def type_repr(type_num: int) -> Union[int, str]:
    global _type_reprs
    if not _type_reprs:
        from . import pygram
        if not hasattr(pygram, 'python_symbols'):
            pygram.initialize(cache_dir=None)
        for name in dir(pygram.python_symbols):
            val = getattr(pygram.python_symbols, name)
            if type(val) == int:
                _type_reprs[val] = name
    return _type_reprs.setdefault(type_num, type_num)

_P = TypeVar('_P', bound='Base')
NL = Union['Node', 'Leaf']
Context = Tuple[str, Tuple[int, int]]
RawNode = Tuple[int, Optional[str], Optional[Context], Optional[List[NL]]]

class Base:
    parent: Optional['Base'] = None
    was_changed: bool = False
    was_checked: bool = False
    prev_sibling_map: Optional[Dict[int, Optional['Base']]] = None
    next_sibling_map: Optional[Dict[int, Optional['Base']]] = None
    children: List[NL]

    def __new__(cls, *args: Any, **kwds: Any) -> Any:
        assert cls is not Base, 'Cannot instantiate Base'
        return object.__new__(cls)

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self._eq(other)

    @property
    def prefix(self) -> str:
        raise NotImplementedError

    def _eq(self, other: 'Base') -> bool:
        raise NotImplementedError

    def __deepcopy__(self, memo: Any) -> 'Base':
        return self.clone()

    def clone(self) -> 'Base':
        raise NotImplementedError

    def post_order(self) -> Iterator['Base']:
        raise NotImplementedError

    def pre_order(self) -> Iterator['Base']:
        raise NotImplementedError

    def replace(self, new: Union['Base', List['Base']]) -> None:
        assert self.parent is not None, str(self)
        assert new is not None
        if not isinstance(new, list):
            new = [new]
        l_children: List[NL] = []
        found = False
        for ch in self.parent.children:
            if ch is self:
                assert not found, (self.parent.children, self, new)
                if new is not None:
                    l_children.extend(new)
                found = True
            else:
                l_children.append(ch)
        assert found, (self.children, self, new)
        self.parent.children = l_children
        self.parent.changed()
        self.parent.invalidate_sibling_maps()
        for x in new:
            x.parent = self.parent
        self.parent = None

    def get_lineno(self) -> Optional[int]:
        node = self
        while not isinstance(node, Leaf):
            if not node.children:
                return None
            node = node.children[0]
        return node.lineno

    def changed(self) -> None:
        if self.was_changed:
            return
        if self.parent:
            self.parent.changed()
        self.was_changed = True

    def remove(self) -> Optional[int]:
        if self.parent:
            for i, node in enumerate(self.parent.children):
                if node is self:
                    del self.parent.children[i]
                    self.parent.changed()
                    self.parent.invalidate_sibling_maps()
                    self.parent = None
                    return i
        return None

    @property
    def next_sibling(self) -> Optional['Base']:
        if self.parent is None:
            return None
        if self.parent.next_sibling_map is None:
            self.parent.update_sibling_maps()
        assert self.parent.next_sibling_map is not None
        return self.parent.next_sibling_map[id(self)]

    @property
    def prev_sibling(self) -> Optional['Base']:
        if self.parent is None:
            return None
        if self.parent.prev_sibling_map is None:
            self.parent.update_sibling_maps()
        assert self.parent.prev_sibling_map is not None
        return self.parent.prev_sibling_map[id(self)]

    def leaves(self) -> Iterator['Leaf']:
        for child in self.children:
            yield from child.leaves()

    def depth(self) -> int:
        if self.parent is None:
            return 0
        return 1 + self.parent.depth()

    def get_suffix(self) -> str:
        next_sib = self.next_sibling
        if next_sib is None:
            return ''
        prefix = next_sib.prefix
        return prefix

class Node(Base):
    def __init__(
        self,
        type: int,
        children: List[NL],
        context: Optional[Context] = None,
        prefix: Optional[str] = None,
        fixers_applied: Optional[List[Any]] = None
    ) -> None:
        assert type >= 256, type
        self.type = type
        self.children = list(children)
        for ch in self.children:
            assert ch.parent is None, repr(ch)
            ch.parent = self
        self.invalidate_sibling_maps()
        if prefix is not None:
            self.prefix = prefix
        if fixers_applied:
            self.fixers_applied = fixers_applied[:]
        else:
            self.fixers_applied = None

    def __repr__(self) -> str:
        assert self.type is not None
        return '{}({}, {!r})'.format(self.__class__.__name__, type_repr(self.type), self.children)

    def __str__(self) -> str:
        return ''.join(map(str, self.children))

    def _eq(self, other: 'Node') -> bool:
        return (self.type, self.children) == (other.type, other.children)

    def clone(self) -> 'Node':
        assert self.type is not None
        return Node(self.type, [ch.clone() for ch in self.children], fixers_applied=self.fixers_applied)

    def post_order(self) -> Iterator['Base']:
        for child in self.children:
            yield from child.post_order()
        yield self

    def pre_order(self) -> Iterator['Base']:
        yield self
        for child in self.children:
            yield from child.pre_order()

    @property
    def prefix(self) -> str:
        if not self.children:
            return ''
        return self.children[0].prefix

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        if self.children:
            self.children[0].prefix = prefix

    def set_child(self, i: int, child: NL) -> None:
        child.parent = self
        self.children[i].parent = None
        self.children[i] = child
        self.changed()
        self.invalidate_sibling_maps()

    def insert_child(self, i: int, child: NL) -> None:
        child.parent = self
        self.children.insert(i, child)
        self.changed()
        self.invalidate_sibling_maps()

    def append_child(self, child: NL) -> None:
        child.parent = self
        self.children.append(child)
        self.changed()
        self.invalidate_sibling_maps()

    def invalidate_sibling_maps(self) -> None:
        self.prev_sibling_map = None
        self.next_sibling_map = None

    def update_sibling_maps(self) -> None:
        _prev: Dict[int, Optional[Base]] = {}
        _next: Dict[int, Optional[Base]] = {}
        self.prev_sibling_map = _prev
        self.next_sibling_map = _next
        previous: Optional[Base] = None
        for current in self.children:
            _prev[id(current)] = previous
            _next[id(previous)] = current
            previous = current
        _next[id(current)] = None

class Leaf(Base):
    opening_bracket: Optional[Any] = None
    _prefix: str = ''
    lineno: int = 0
    column: int = 0
    fmt_pass_converted_first_leaf: Optional[Any] = None

    def __init__(
        self,
        type: int,
        value: str,
        context: Optional[Context] = None,
        prefix: Optional[str] = None,
        fixers_applied: List[Any] = [],
        opening_bracket: Optional[Any] = None,
        fmt_pass_converted_first_leaf: Optional[Any] = None
    ) -> None:
        assert 0 <= type < 256, type
        if context is not None:
            self._prefix, (self.lineno, self.column) = context
        self.type = type
        self.value = value
        if prefix is not None:
            self._prefix = prefix
        self.fixers_applied = fixers_applied[:]
        self.children: List[NL] = []
        self.opening_bracket = opening_bracket
        self.fmt_pass_converted_first_leaf = fmt_pass_converted_first_leaf

    def __repr__(self) -> str:
        from .pgen2.token import tok_name
        assert self.type is not None
        return '{}({}, {!r})'.format(self.__class__.__name__, tok_name.get(self.type, self.type), self.value)

    def __str__(self) -> str:
        return self._prefix + str(self.value)

    def _eq(self, other: 'Leaf') -> bool:
        return (self.type, self.value) == (other.type, other.value)

    def clone(self) -> 'Leaf':
        assert self.type is not None
        return Leaf(self.type, self.value, (self.prefix, (self.lineno, self.column)), fixers_applied=self.fixers_applied)

    def leaves(self) -> Iterator['Leaf']:
        yield self

    def post_order(self) -> Iterator['Leaf']:
        yield self

    def pre_order(self) -> Iterator['Leaf']:
        yield self

    @property
    def prefix(self) -> str:
        return self._prefix

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        self.changed()
        self._prefix = prefix

def convert(gr: Grammar, raw_node: RawNode) -> NL:
    type, value, context, children = raw_node
    if children or type in gr.number2symbol:
        assert children is not None
        if len(children) == 1:
            return children[0]
        return Node(type, children, context=context)
    else:
        return Leaf(type, value or '', context=context)

_Results = Dict[str, NL]

class BasePattern:
    type: Optional[int]
    content: Optional[Any]
    name: Optional[str]

    def __new__(cls, *args: Any, **kwds: Any) -> Any:
        assert cls is not BasePattern, 'Cannot instantiate BasePattern'
        return object.__new__(cls)

    def __repr__(self) -> str:
        assert self.type is not None
        args = [type_repr(self.type), self.content, self.name]
        while args and args[-1] is None:
            del args[-1]
        return '{}({})'.format(self.__class__.__name__, ', '.join(map(repr, args)))

    def _submatch(self, node: NL, results: Optional[_Results] = None) -> bool:
        raise NotImplementedError

    def optimize(self) -> 'BasePattern':
        return self

    def match(self, node: NL, results: Optional[_Results] = None) -> bool:
        if self.type is not None and node.type != self.type:
            return False
        if self.content is not None:
            r = None
            if results is not None:
                r = {}
            if not self._submatch(node, r):
                return False
            if r:
                assert results is not None
                results.update(r)
        if results is not None and self.name:
            results[self.name] = node
        return True

    def match_seq(self, nodes: List[NL], results: Optional[_Results] = None) -> bool:
        if len(nodes) != 1:
            return False
        return self.match(nodes[0], results)

    def generate_matches(self, nodes: List[NL]) -> Iterator[Tuple[int, _Results]]:
        r: _Results = {}
        if nodes and self.match(nodes[0], r):
            yield (1, r)

class LeafPattern(BasePattern):
    def __init__(
        self,
        type: Optional[int] = None,
        content: Optional[str] = None,
        name: Optional[str] = None
    ) -> None:
        if type is not None:
            assert 0 <= type < 256, type
        if content is not None:
            assert isinstance(content, str), repr(content)
        self.type = type
        self.content = content
        self.name = name

    def match(self, node: NL, results: Optional[_Results] = None) -> bool:
        if not isinstance(node, Leaf):
            return False
        return BasePattern.match(self, node, results)

    def _submatch(self, node: Leaf, results: Optional[_Results] = None) -> bool:
        return self.content == node.value

class NodePattern(BasePattern):
    wildcards: bool = False

    def __init__(
        self,
        type: Optional[int] = None,
        content: Optional[List[BasePattern]] = None,
        name: Optional[str] = None
    ) -> None:
        if type is not None:
            assert type >= 256, type
        newcontent: Optional[List[BasePattern]] = None
        if content is not None:
            assert not isinstance(content, str), repr(content)
            newcontent = list(content)
            for i, item in enumerate(newcontent):
                assert isinstance(item, BasePattern), (i, item)
                if isinstance(item, WildcardPattern):
                    self.wildcards = True
        self.type = type
        self.content = newcontent
        self.name = name

    def _submatch(self, node: NL, results: Optional[_Results] = None) -> bool:
        if self.wildcards:
            for c, r in generate_matches(cast(List[BasePattern], self.content), node.children):
                if c == len(node.children):
                    if results is not None:
                        results.update(r)
                    return True
            return False
        if len(cast(List[BasePattern], self.content)) != len(node.children):
            return False
        for subpattern, child in zip(cast(List[BasePattern], self.content), node.children):
            if not subpattern.match(child, results):
                return False
        return True

class WildcardPattern(BasePattern):
    def __init__(
        self,
        content: Optional[List[List[BasePattern]]] = None,
        min: int = 0,
        max: int = HUGE,
        name: Optional[str] = None
    ) -> None:
        assert 0 <= min <= max <= HUGE, (min, max)
        wrapped_content: Optional[Tuple[Tuple[BasePattern, ...], ...]] = None
        if content is not None:
            f = lambda s: tuple(s)
            wrapped_content = tuple(map(f, content))
            assert len(wrapped_content), repr(wrapped_content)
            for alt in wrapped_content:
                assert len(alt), repr(alt)
        self.content = wrapped_content
        self.min = min
        self.max = max
        self.name = name

    def optimize(self) -> BasePattern:
        subpattern = None
        if self.content is not None and len(self.content) == 1 and (len(self.content[0]) == 1):
            subpattern = self.content[0][0]
        if self.min == 1 and self.max == 1:
            if self.content is None:
                return NodePattern(name=self.name)
            if subpattern is not None and self.name == subpattern.name:
                return subpattern.optimize()
        if self.min <= 1 and isinstance(subpattern, WildcardPattern) and (subpattern.min <= 1) and (self.name == subpattern.name):
            return WildcardPattern(subpattern.content, self.min * subpattern.min, self.max * subpattern.max, subpattern.name)
        return self

    def match(self, node: NL, results: Optional[_Results] = None) -> bool:
        return self.match_seq([node], results)

    def match_seq(self, nodes: List[NL], results: Optional[_Results] = None) -> bool:
        for c, r in self.generate_matches(nodes):
            if c == len(nodes):
                if results is not None:
                    results.update(r)
                    if self.name:
                        results[self.name] = list(nodes)
                return True
        return False

    def generate_matches(self, nodes: List[NL]) -> Iterator[Tuple[int, _Results]]:
        if self.content is None:
            for count in range(self.min, 1 + min(len(nodes), self.max)):
                r: _Results = {}
                if self.name:
                    r[self.name] = nodes[:count]
                yield (count, r)
        elif self.name == 'bare_name':
            yield self._bare_name_matches(nodes)
        else:
            if hasattr(sys, 'getrefcount'):
                save_stderr = sys.stderr
                sys.stderr = StringIO()
            try:
                for count, r in self._recursive_matches(nodes, 0):
                    if self.name:
                        r[self.name] = nodes[:count]
                    yield (count, r)
            except RuntimeError:
                for count, r in self._iterative_matches(nodes):
                    if self.name:
                        r[self.name] = nodes[:count]
                    yield (count, r)
            finally:
                if hasattr(sys, 'getrefcount'):
                    sys.stderr = save_stderr

    def _iterative_matches(self, nodes: List[NL]) -> Iterator[Tuple[int, _Results]]:
        nodelen = len(nodes)
        if 0 >= self.min:
            yield (0, {})
        results: List[Tuple[int, _Results]] = []
        for alt in