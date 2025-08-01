from typing import Any, Dict, Generator, List, Optional, Tuple, Union

__author__ = 'Guido van Rossum <guido@python.org>'
import sys
import warnings
from io import StringIO

HUGE: int = 2147483647
_type_reprs: Dict[int, str] = {}

def type_repr(type_num: int) -> Union[int, str]:
    global _type_reprs
    if not _type_reprs:
        from .pygram import python_symbols
        for name, val in python_symbols.__dict__.items():
            if type(val) == int:
                _type_reprs[val] = name
    return _type_reprs.setdefault(type_num, type_num)

class Base(object):
    type: Optional[int] = None
    parent: Optional['Base'] = None
    children: Tuple['Base', ...] = ()
    was_changed: bool = False
    was_checked: bool = False

    def __new__(cls, *args: Any, **kwds: Any) -> 'Base':
        assert cls is not Base, 'Cannot instantiate Base'
        return object.__new__(cls)

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self._eq(other)

    __hash__ = None

    def __ne__(self, other: Any) -> bool:
        if self.__class__ is not other.__class__:
            return NotImplemented
        return not self._eq(other)

    def _eq(self, other: 'Base') -> bool:
        raise NotImplementedError

    def clone(self) -> 'Base':
        raise NotImplementedError

    def post_order(self) -> Generator['Base', None, None]:
        raise NotImplementedError

    def pre_order(self) -> Generator['Base', None, None]:
        raise NotImplementedError

    def set_prefix(self, prefix: str) -> None:
        warnings.warn('set_prefix() is deprecated; use the prefix property', DeprecationWarning, stacklevel=2)
        self.prefix = prefix

    def get_prefix(self) -> str:
        warnings.warn('get_prefix() is deprecated; use the prefix property', DeprecationWarning, stacklevel=2)
        return self.prefix

    def replace(self, new: Union['Base', List['Base']]) -> None:
        assert self.parent is not None, str(self)
        assert new is not None
        if not isinstance(new, list):
            new = [new]
        l_children: List['Base'] = []
        found: bool = False
        for ch in self.parent.children:
            if ch is self:
                assert not found, (self.parent.children, self, new)
                if new is not None:
                    l_children.extend(new)
                found = True
            else:
                l_children.append(ch)
        assert found, (self.children, self, new)
        self.parent.changed()
        self.parent.children = tuple(l_children)
        for x in new:
            x.parent = self.parent
        self.parent = None

    def get_lineno(self) -> Optional[int]:
        node: 'Base' = self
        while not isinstance(node, Leaf):
            if not node.children:
                return None
            node = node.children[0]
        return node.lineno

    def changed(self) -> None:
        if self.parent:
            self.parent.changed()
        self.was_changed = True

    def remove(self) -> Optional[int]:
        if self.parent:
            for i, node in enumerate(self.parent.children):
                if node is self:
                    self.parent.changed()
                    del self.parent.children[i]
                    self.parent = None
                    return i
        return None

    @property
    def next_sibling(self) -> Optional['Base']:
        if self.parent is None:
            return None
        for i, child in enumerate(self.parent.children):
            if child is self:
                try:
                    return self.parent.children[i + 1]
                except IndexError:
                    return None
        return None

    @property
    def prev_sibling(self) -> Optional['Base']:
        if self.parent is None:
            return None
        for i, child in enumerate(self.parent.children):
            if child is self:
                if i == 0:
                    return None
                return self.parent.children[i - 1]
        return None

    def leaves(self) -> Generator['Leaf', None, None]:
        for child in self.children:
            for x in child.leaves():
                yield x

    def depth(self) -> int:
        if self.parent is None:
            return 0
        return 1 + self.parent.depth()

    def get_suffix(self) -> str:
        next_sib = self.next_sibling
        if next_sib is None:
            return ''
        return next_sib.prefix

    if sys.version_info < (3, 0):
        def __str__(self) -> bytes:
            return str(self).encode('ascii')

class Node(Base):
    def __init__(self, type: int, children: List[Base], context: Optional[Any] = None, prefix: Optional[str] = None, fixers_applied: Optional[List[Any]] = None) -> None:
        assert type >= 256, type
        self.type = type
        self.children = list(children)
        for ch in self.children:
            assert ch.parent is None, repr(ch)
            ch.parent = self
        if prefix is not None:
            self.prefix = prefix
        if fixers_applied:
            self.fixers_applied = fixers_applied[:]
        else:
            self.fixers_applied = None

    def __repr__(self) -> str:
        return '%s(%s, %r)' % (self.__class__.__name__, type_repr(self.type), self.children)

    def __unicode__(self) -> str:
        return ''.join(map(str, self.children))

    if sys.version_info > (3, 0):
        __str__ = __unicode__

    def _eq(self, other: Base) -> bool:
        return (self.type, self.children) == (other.type, other.children)

    def clone(self) -> 'Node':
        return Node(self.type, [ch.clone() for ch in self.children], fixers_applied=self.fixers_applied)

    def post_order(self) -> Generator[Base, None, None]:
        for child in self.children:
            for node in child.post_order():
                yield node
        yield self

    def pre_order(self) -> Generator[Base, None, None]:
        yield self
        for child in self.children:
            for node in child.pre_order():
                yield node

    def _prefix_getter(self) -> str:
        if not self.children:
            return ''
        return self.children[0].prefix

    def _prefix_setter(self, prefix: str) -> None:
        if self.children:
            self.children[0].prefix = prefix

    prefix = property(_prefix_getter, _prefix_setter)

    def set_child(self, i: int, child: Base) -> None:
        child.parent = self
        self.children[i].parent = None
        self.children[i] = child
        self.changed()

    def insert_child(self, i: int, child: Base) -> None:
        child.parent = self
        self.children.insert(i, child)
        self.changed()

    def append_child(self, child: Base) -> None:
        child.parent = self
        self.children.append(child)
        self.changed()

class Leaf(Base):
    _prefix: str = ''
    lineno: int = 0
    column: int = 0

    def __init__(self, type: int, value: str, context: Optional[Tuple[str, Tuple[int, int]]] = None, prefix: Optional[str] = None, fixers_applied: List[Any] = []) -> None:
        assert 0 <= type < 256, type
        if context is not None:
            self._prefix, (self.lineno, self.column) = context
        self.type = type
        self.value = value
        if prefix is not None:
            self._prefix = prefix
        self.fixers_applied = fixers_applied[:]

    def __repr__(self) -> str:
        return '%s(%r, %r)' % (self.__class__.__name__, self.type, self.value)

    def __unicode__(self) -> str:
        return self.prefix + str(self.value)

    if sys.version_info > (3, 0):
        __str__ = __unicode__

    def _eq(self, other: Base) -> bool:
        return (self.type, self.value) == (other.type, other.value)

    def clone(self) -> 'Leaf':
        return Leaf(self.type, self.value, (self.prefix, (self.lineno, self.column)), fixers_applied=self.fixers_applied)

    def leaves(self) -> Generator['Leaf', None, None]:
        yield self

    def post_order(self) -> Generator[Base, None, None]:
        yield self

    def pre_order(self) -> Generator[Base, None, None]:
        yield self

    def _prefix_getter(self) -> str:
        return self._prefix

    def _prefix_setter(self, prefix: str) -> None:
        self.changed()
        self._prefix = prefix

    prefix = property(_prefix_getter, _prefix_setter)

def convert(gr: Any, raw_node: Tuple[int, str, Optional[Any], List[Base]]) -> Base:
    type, value, context, children = raw_node
    if children or type in gr.number2symbol:
        if len(children) == 1:
            return children[0]
        return Node(type, children, context=context)
    else:
        return Leaf(type, value, context=context)

class BasePattern(object):
    type: Optional[int] = None
    content: Optional[Any] = None
    name: Optional[str] = None

    def __new__(cls, *args: Any, **kwds: Any) -> 'BasePattern':
        assert cls is not BasePattern, 'Cannot instantiate BasePattern'
        return object.__new__(cls)

    def __repr__(self) -> str:
        args = [type_repr(self.type), self.content, self.name]
        while args and args[-1] is None:
            del args[-1]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr, args)))

    def optimize(self) -> 'BasePattern':
        return self

    def match(self, node: Base, results: Optional[Dict[str, Base]] = None) -> bool:
        if self.type is not None and node.type != self.type:
            return False
        if self.content is not None:
            r: Optional[Dict[str, Base]] = None
            if results is not None:
                r = {}
            if not self._submatch(node, r):
                return False
            if r:
                results.update(r)
        if results is not None and self.name:
            results[self.name] = node
        return True

    def match_seq(self, nodes: List[Base], results: Optional[Dict[str, Base]] = None) -> bool:
        if len(nodes) != 1:
            return False
        return self.match(nodes[0], results)

    def generate_matches(self, nodes: List[Base]) -> Generator[Tuple[int, Dict[str, Base]], None, None]:
        r: Dict[str, Base] = {}
        if nodes and self.match(nodes[0], r):
            yield (1, r)

class LeafPattern(BasePattern):
    def __init__(self, type: Optional[int] = None, content: Optional[str] = None, name: Optional[str] = None) -> None:
        if type is not None:
            assert 0 <= type < 256, type
        if content is not None:
            assert isinstance(content, str), repr(content)
        self.type = type
        self.content = content
        self.name = name

    def match(self, node: Base, results: Optional[Dict[str, Base]] = None) -> bool:
        if not isinstance(node, Leaf):
            return False
        return BasePattern.match(self, node, results)

    def _submatch(self, node: Base, results: Optional[Dict[str, Base]] = None) -> bool:
        return self.content == node.value

class NodePattern(BasePattern):
    wildcards: bool = False

    def __init__(self, type: Optional[int] = None, content: Optional[List[BasePattern]] = None, name: Optional[str] = None) -> None:
        if type is not None:
            assert type >= 256, type
        if content is not None:
            assert not isinstance(content, str), repr(content)
            content = list(content)
            for i, item in enumerate(content):
                assert isinstance(item, BasePattern), (i, item)
                if isinstance(item, WildcardPattern):
                    self.wildcards = True
        self.type = type
        self.content = content
        self.name = name

    def _submatch(self, node: Base, results: Optional[Dict[str, Base]] = None) -> bool:
        if self.wildcards:
            for c, r in generate_matches(self.content, node.children):
                if c == len(node.children):
                    if results is not None:
                        results.update(r)
                    return True
            return False
        if len(self.content) != len(node.children):
            return False
        for subpattern, child in zip(self.content, node.children):
            if not subpattern.match(child, results):
                return False
        return True

class WildcardPattern(BasePattern):
    def __init__(self, content: Optional[List[List[BasePattern]]] = None, min: int = 0, max: int = HUGE, name: Optional[str] = None) -> None:
        assert 0 <= min <= max <= HUGE, (min, max)
        if content is not None:
            content = tuple(map(tuple, content))
            assert len(content), repr(content)
            for alt in content:
                assert len(alt), repr(alt)
        self.content = content
        self.min = min
        self.max = max
        self.name = name

    def optimize(self) -> BasePattern:
        subpattern: Optional[BasePattern] = None
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

    def match(self, node: Base, results: Optional[Dict[str, Base]] = None) -> bool:
        return self.match_seq([node], results)

    def match_seq(self, nodes: List[Base], results: Optional[Dict[str, Base]] = None) -> bool:
        for c, r in self.generate_matches(nodes):
            if c == len(nodes):
                if results is not None:
                    results.update(r)
                    if self.name:
                        results[self.name] = list(nodes)
                return True
        return False

    def generate_matches(self, nodes: List[Base]) -> Generator[Tuple[int, Dict[str, Base]], None, None]:
        if self.content is None:
            for count in range(self.min, 1 + min(len(nodes), self.max)):
                r: Dict[str, Base] = {}
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

    def _iterative_matches(self, nodes: List[Base]) -> Generator[Tuple[int, Dict[str, Base]], None, None]:
        nodelen: int = len(nodes)
        if 0 >= self.min:
            yield (0, {})
        results: List[Tuple[int, Dict[str, Base]]] = []
        for alt in self.content:
            for c, r in generate_matches(alt, nodes):
                yield (c, r)
                results.append((c, r))
        while results:
            new_results: List[Tuple[int, Dict[str, Base]]] = []
            for c0, r0 in results:
                if c0 < nodelen and c0 <= self.max:
                    for alt in self.content:
                        for c1, r1 in generate_matches(alt, nodes[c0:]):
                            if c1 > 0:
                                r: Dict[str, Base] = {}
                                r.update(r0)
                                r.update(r1)
                                yield (c0 + c1, r)
                                new_results.append((c0 + c1, r))
            results = new_results

    def _bare_name_matches(self, nodes: List[Base]) -> Tuple[int, Dict[str, Base]]:
        count: int = 0
        r: Dict[str, Base] = {}
        done: bool = False
        max: int = len(nodes)
        while not done and count < max:
            done = True
            for leaf in self.content:
                if leaf[0].match(nodes[count], r):
                    count += 1
                    done = False
                    break
        r[self.name] = nodes[:count]
        return (count, r)

    def _recursive_matches(self, nodes: List[Base], count: int) -> Generator[Tuple[int, Dict[str, Base]], None, None]:
        assert self.content is not None
        if count >= self.min:
            yield (0, {})
        if count < self.max:
            for alt in self.content:
                for c0, r0 in generate_matches(alt, nodes):
                    for c1, r1 in self._recursive_matches(nodes[c0:], count + 1):
                        r: Dict[str, Base] = {}
                        r.update(r0)
                        r.update(r1)
                        yield (c0 + c1, r)

class NegatedPattern(BasePattern):
    def __init__(self, content: Optional[BasePattern] = None) -> None:
        if content is not None:
            assert isinstance(content, BasePattern), repr(content)
        self.content = content

    def match(self, node: Base) -> bool:
        return False

    def match_seq(self, nodes: List[Base]) -> bool:
        return len(nodes) == 0

    def generate_matches(self, nodes: List[Base]) -> Generator[Tuple[int, Dict[str, Base]], None, None]:
        if self.content is None:
            if len(nodes) == 0:
                yield (0, {})
        else:
            for c, r in self.content.generate_matches(nodes):
                return
            yield (0, {})

def generate_matches(patterns: List[BasePattern], nodes: List[Base]) -> Generator[Tuple[int, Dict[str, Base]], None, None]:
    if not patterns:
        yield (0, {})
    else:
        p, rest = (patterns[0], patterns[1:])
        for c0, r0 in p.generate_matches(nodes):
            if not rest:
                yield (c0, r0)
            else:
                for c1, r1 in generate_matches(rest, nodes[c0:]):
                    r: Dict[str, Base] = {}
                    r.update(r0)
                    r.update(r1)
                    yield (c0 + c1, r)
