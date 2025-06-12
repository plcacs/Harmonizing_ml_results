"""
Python parse tree definitions.

This is a very concrete parse tree; we need to keep every token and
even the comments and whitespace between tokens.

There's also a pattern matching implementation here.
"""
from collections.abc import Iterable, Iterator
from typing import Any, Optional, TypeVar, Union, Dict, List, Tuple, Sequence, Set, Generic, cast
from blib2to3.pgen2.grammar import Grammar
from typing import overload

__author__ = 'Guido van Rossum <guido@python.org>'
import sys
from io import StringIO
from typing import Type, TypeVar, Generic

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
    """
    Abstract base class for Node and Leaf.

    This provides some default functionality and boilerplate using the
    template pattern.

    A node may be a subnode of at most one parent.
    """
    parent: Optional['Node'] = None
    was_changed: bool = False
    was_checked: bool = False
    prev_sibling_map: Optional[Dict[int, Optional['Base']]] = None
    next_sibling_map: Optional[Dict[int, Optional['Base']]] = None

    def __new__(cls: Type[_P], *args: Any, **kwds: Any) -> _P:
        """Constructor that prevents Base from being instantiated."""
        assert cls is not Base, 'Cannot instantiate Base'
        return object.__new__(cls)

    def __eq__(self, other: Any) -> bool:
        """
        Compare two nodes for equality.

        This calls the method _eq().
        """
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self._eq(other)

    @property
    def prefix(self) -> str:
        raise NotImplementedError

    def _eq(self, other: 'Base') -> bool:
        """
        Compare two nodes for equality.

        This is called by __eq__ and __ne__.  It is only called if the two nodes
        have the same type.  This must be implemented by the concrete subclass.
        Nodes should be considered equal if they have the same structure,
        ignoring the prefix string and other context information.
        """
        raise NotImplementedError

    def __deepcopy__(self, memo: Any) -> 'Base':
        return self.clone()

    def clone(self) -> 'Base':
        """
        Return a cloned (deep) copy of self.

        This must be implemented by the concrete subclass.
        """
        raise NotImplementedError

    def post_order(self) -> Iterator['Base']:
        """
        Return a post-order iterator for the tree.

        This must be implemented by the concrete subclass.
        """
        raise NotImplementedError

    def pre_order(self) -> Iterator['Base']:
        """
        Return a pre-order iterator for the tree.

        This must be implemented by the concrete subclass.
        """
        raise NotImplementedError

    def replace(self, new: Union['Base', List['Base']]) -> None:
        """Replace this node with a new one in the parent."""
        assert self.parent is not None, str(self)
        assert new is not None
        if not isinstance(new, list):
            new = [new]
        l_children: List['Base'] = []
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
        """Return the line number which generated the invocant node."""
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
        """
        Remove the node from the tree. Returns the position of the node in its
        parent's children before it was removed.
        """
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
        """
        The node immediately following the invocant in their parent's children
        list. If the invocant does not have a next sibling, it is None
        """
        if self.parent is None:
            return None
        if self.parent.next_sibling_map is None:
            self.parent.update_sibling_maps()
        assert self.parent.next_sibling_map is not None
        return self.parent.next_sibling_map[id(self)]

    @property
    def prev_sibling(self) -> Optional['Base']:
        """
        The node immediately preceding the invocant in their parent's children
        list. If the invocant does not have a previous sibling, it is None.
        """
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
        """
        Return the string immediately following the invocant node. This is
        effectively equivalent to node.next_sibling.prefix
        """
        next_sib = self.next_sibling
        if next_sib is None:
            return ''
        prefix = next_sib.prefix
        return prefix

class Node(Base):
    """Concrete implementation for interior nodes."""

    def __init__(self, type: int, children: Sequence[Base], context: Optional[Context] = None, 
                 prefix: Optional[str] = None, fixers_applied: Optional[List[Any]] = None) -> None:
        """
        Initializer.

        Takes a type constant (a symbol number >= 256), a sequence of
        child nodes, and an optional context keyword argument.

        As a side effect, the parent pointers of the children are updated.
        """
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
        """Return a canonical string representation."""
        assert self.type is not None
        return '{}({}, {!r})'.format(self.__class__.__name__, type_repr(self.type), self.children)

    def __str__(self) -> str:
        """
        Return a pretty string representation.

        This reproduces the input source exactly.
        """
        return ''.join(map(str, self.children))

    def _eq(self, other: 'Node') -> bool:
        """Compare two nodes for equality."""
        return (self.type, self.children) == (other.type, other.children)

    def clone(self) -> 'Node':
        assert self.type is not None
        'Return a cloned (deep) copy of self.'
        return Node(self.type, [ch.clone() for ch in self.children], fixers_applied=self.fixers_applied)

    def post_order(self) -> Iterator[Base]:
        """Return a post-order iterator for the tree."""
        for child in self.children:
            yield from child.post_order()
        yield self

    def pre_order(self) -> Iterator[Base]:
        """Return a pre-order iterator for the tree."""
        yield self
        for child in self.children:
            yield from child.pre_order()

    @property
    def prefix(self) -> str:
        """
        The whitespace and comments preceding this node in the input.
        """
        if not self.children:
            return ''
        return self.children[0].prefix

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        if self.children:
            self.children[0].prefix = prefix

    def set_child(self, i: int, child: Base) -> None:
        """
        Equivalent to 'node.children[i] = child'. This method also sets the
        child's parent attribute appropriately.
        """
        child.parent = self
        self.children[i].parent = None
        self.children[i] = child
        self.changed()
        self.invalidate_sibling_maps()

    def insert_child(self, i: int, child: Base) -> None:
        """
        Equivalent to 'node.children.insert(i, child)'. This method also sets
        the child's parent attribute appropriately.
        """
        child.parent = self
        self.children.insert(i, child)
        self.changed()
        self.invalidate_sibling_maps()

    def append_child(self, child: Base) -> None:
        """
        Equivalent to 'node.children.append(child)'. This method also sets the
        child's parent attribute appropriately.
        """
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
    """Concrete implementation for leaf nodes."""
    opening_bracket: Optional[Any] = None
    _prefix: str = ''
    lineno: int = 0
    column: int = 0
    fmt_pass_converted_first_leaf: Optional[Any] = None

    def __init__(self, type: int, value: str, context: Optional[Context] = None, 
                 prefix: Optional[str] = None, fixers_applied: List[Any] = [], 
                 opening_bracket: Optional[Any] = None, 
                 fmt_pass_converted_first_leaf: Optional[Any] = None) -> None:
        """
        Initializer.

        Takes a type constant (a token number < 256), a string value, and an
        optional context keyword argument.
        """
        assert 0 <= type < 256, type
        if context is not None:
            self._prefix, (self.lineno, self.column) = context
        self.type = type
        self.value = value
        if prefix is not None:
            self._prefix = prefix
        self.fixers_applied = fixers_applied[:]
        self.children: List[Base] = []
        self.opening_bracket = opening_bracket
        self.fmt_pass_converted_first_leaf = fmt_pass_converted_first_leaf

    def __repr__(self) -> str:
        """Return a canonical string representation."""
        from .pgen2.token import tok_name
        assert self.type is not None
        return '{}({}, {!r})'.format(self.__class__.__name__, tok_name.get(self.type, self.type), self.value)

    def __str__(self) -> str:
        """
        Return a pretty string representation.

        This reproduces the input source exactly.
        """
        return self._prefix + str(self.value)

    def _eq(self, other: 'Leaf') -> bool:
        """Compare two nodes for equality."""
        return (self.type, self.value) == (other.type, other.value)

    def clone(self) -> 'Leaf':
        assert self.type is not None
        'Return a cloned (deep) copy of self.'
        return Leaf(self.type, self.value, (self.prefix, (self.lineno, self.column)), fixers_applied=self.fixers_applied)

    def leaves(self) -> Iterator['Leaf']:
        yield self

    def post_order(self) -> Iterator[Base]:
        """Return a post-order iterator for the tree."""
        yield self

    def pre_order(self) -> Iterator[Base]:
        """Return a pre-order iterator for the tree."""
        yield self

    @property
    def prefix(self) -> str:
        """
        The whitespace and comments preceding this token in the input.
        """
        return self._prefix

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        self.changed()
        self._prefix = prefix

def convert(gr: Grammar, raw_node: RawNode) -> NL:
    """
    Convert raw node information to a Node or Leaf instance.

    This is passed to the parser driver which calls it whenever a reduction of a
    grammar rule produces a new complete node, so that the tree is build
    strictly bottom-up.
    """
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
    """
    A pattern is a tree matching pattern.

    It looks for a specific node type (token or symbol), and
    optionally for a specific content.

    This is an abstract base class.  There are three concrete
    subclasses:

    - LeafPattern matches a single leaf node;
    - NodePattern matches a single node (usually non-leaf);
    - WildcardPattern matches a sequence of nodes of variable length.
    """
    type: Optional[int] = None
    content: Optional[Any] = None
    name: Optional[str] = None

    def __new__(cls: Type['BasePattern'], *args: Any, **kwds: Any) -> 'BasePattern':
        """Constructor that prevents BasePattern from being instantiated."""
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
        """
        A subclass can define this as a hook for optimizations.

        Returns either self or another node with the same effect.
        """
        return self

    def match(self, node: NL, results: Optional[_Results] = None) -> bool:
        """
        Does this pattern exactly match a node?

        Returns True if it matches, False if not.

        If results is not None, it must be a dict which will be
        updated with the nodes matching named subpatterns.

        Default implementation for non-wildcard patterns.
        """
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

    def match_seq(self, nodes: Sequence[NL], results: Optional[_Results] = None) -> bool:
        """
        Does this pattern exactly match a sequence of nodes?

        Default implementation for non-wildcard patterns.
        """
        if len(nodes) != 1:
            return False
        return self.match(nodes[0], results)

    def generate_matches(self, nodes: Sequence[NL]) -> Iterator[Tuple[int, _Results]]:
        """
        Generator yielding all matches for this pattern.

        Default implementation for non-wildcard patterns.
        """
        r: _Results = {}
        if nodes and self.match(nodes[0], r):
            yield (1, r)

class LeafPattern(BasePattern):

    def __init__(self, type: Optional[int] = None, content: Optional[str] = None, name: Optional[str] = None) -> None:
        """
        Initializer.  Takes optional type, content, and name.

        The type, if given must be a token type (< 256).  If not given,
        this matches any *leaf* node; the content may still be required.

        The content, if given, must be a string.

        If a name is given, the matching node is stored in the results
        dict under that key.
        """
        if type is not None:
            assert 0 <= type < 256, type
        if content is not None:
            assert isinstance(content