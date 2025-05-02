"""
Python parse tree definitions.

This is a very concrete parse tree; we need to keep every token and
even the comments and whitespace between tokens.

There's also a pattern matching implementation here.
"""
__author__ = 'Guido van Rossum <guido@python.org>'
import sys
import warnings
from io import StringIO
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Type, TypeVar, Union, overload

HUGE = 2147483647
_type_reprs: Dict[int, str] = {}

def type_repr(type_num: int) -> Union[str, int]:
    global _type_reprs
    if not _type_reprs:
        from .pygram import python_symbols
        for name, val in python_symbols.__dict__.items():
            if type(val) == int:
                _type_reprs[val] = name
    return _type_reprs.setdefault(type_num, type_num)

T = TypeVar('T', bound='Base')

class Base(object):
    """
    Abstract base class for Node and Leaf.

    This provides some default functionality and boilerplate using the
    template pattern.

    A node may be a subnode of at most one parent.
    """
    type: Optional[int] = None
    parent: Optional['Node'] = None
    children: Tuple[Any, ...] = ()
    was_changed: bool = False
    was_checked: bool = False

    def __new__(cls: Type[T], *args: Any, **kwds: Any) -> T:
        """Constructor that prevents Base from being instantiated."""
        assert cls is not Base, 'Cannot instantiate Base'
        return object.__new__(cls)

    def __eq__(self: T, other: object) -> bool:
        """
        Compare two nodes for equality.

        This calls the method _eq().
        """
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self._eq(other)
    __hash__ = None

    def __ne__(self: T, other: object) -> bool:
        """
        Compare two nodes for inequality.

        This calls the method _eq().
        """
        if self.__class__ is not other.__class__:
            return NotImplemented
        return not self._eq(other)

    def _eq(self: T, other: T) -> bool:
        """
        Compare two nodes for equality.

        This is called by __eq__ and __ne__.  It is only called if the two nodes
        have the same type.  This must be implemented by the concrete subclass.
        Nodes should be considered equal if they have the same structure,
        ignoring the prefix string and other context information.
        """
        raise NotImplementedError

    def clone(self: T) -> T:
        """
        Return a cloned (deep) copy of self.

        This must be implemented by the concrete subclass.
        """
        raise NotImplementedError

    def post_order(self) -> Generator['Base', None, None]:
        """
        Return a post-order iterator for the tree.

        This must be implemented by the concrete subclass.
        """
        raise NotImplementedError

    def pre_order(self) -> Generator['Base', None, None]:
        """
        Return a pre-order iterator for the tree.

        This must be implemented by the concrete subclass.
        """
        raise NotImplementedError

    def set_prefix(self, prefix: str) -> None:
        """
        Set the prefix for the node (see Leaf class).

        DEPRECATED; use the prefix property directly.
        """
        warnings.warn('set_prefix() is deprecated; use the prefix property', DeprecationWarning, stacklevel=2)
        self.prefix = prefix

    def get_prefix(self) -> str:
        """
        Return the prefix for the node (see Leaf class).

        DEPRECATED; use the prefix property directly.
        """
        warnings.warn('get_prefix() is deprecated; use the prefix property', DeprecationWarning, stacklevel=2)
        return self.prefix

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
        self.parent.changed()
        self.parent.children = l_children
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
                    self.parent.changed()
                    del self.parent.children[i]
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
        for i, child in enumerate(self.parent.children):
            if child is self:
                try:
                    return self.parent.children[i + 1]
                except IndexError:
                    return None
        return None

    @property
    def prev_sibling(self) -> Optional['Base']:
        """
        The node immediately preceding the invocant in their parent's children
        list. If the invocant does not have a previous sibling, it is None.
        """
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
        """
        Return the string immediately following the invocant node. This is
        effectively equivalent to node.next_sibling.prefix
        """
        next_sib = self.next_sibling
        if next_sib is None:
            return ''
        return next_sib.prefix
    if sys.version_info < (3, 0):

        def __str__(self) -> str:
            return str(self).encode('ascii')

class Node(Base):
    """Concrete implementation for interior nodes."""

    def __init__(self, type: int, children: Sequence[Base], context: Optional[Tuple[str, Tuple[int, int]]] = None, prefix: Optional[str] = None, fixers_applied: Optional[List[Any]] = None):
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
        if prefix is not None:
            self.prefix = prefix
        if fixers_applied:
            self.fixers_applied = fixers_applied[:]
        else:
            self.fixers_applied = None

    def __repr__(self) -> str:
        """Return a canonical string representation."""
        return '%s(%s, %r)' % (self.__class__.__name__, type_repr(self.type), self.children)

    def __unicode__(self) -> str:
        """
        Return a pretty string representation.

        This reproduces the input source exactly.
        """
        return ''.join(map(str, self.children))
    if sys.version_info > (3, 0):
        __str__ = __unicode__

    def _eq(self, other: 'Node') -> bool:
        """Compare two nodes for equality."""
        return (self.type, self.children) == (other.type, other.children)

    def clone(self) -> 'Node':
        """Return a cloned (deep) copy of self."""
        return Node(self.type, [ch.clone() for ch in self.children], fixers_applied=self.fixers_applied)

    def post_order(self) -> Generator['Base', None, None]:
        """Return a post-order iterator for the tree."""
        for child in self.children:
            for node in child.post_order():
                yield node
        yield self

    def pre_order(self) -> Generator['Base', None, None]:
        """Return a pre-order iterator for the tree."""
        yield self
        for child in self.children:
            for node in child.pre_order():
                yield node

    def _prefix_getter(self) -> str:
        """
        The whitespace and comments preceding this node in the input.
        """
        if not self.children:
            return ''
        return self.children[0].prefix

    def _prefix_setter(self, prefix: str) -> None:
        if self.children:
            self.children[0].prefix = prefix
    prefix = property(_prefix_getter, _prefix_setter)

    def set_child(self, i: int, child: 'Base') -> None:
        """
        Equivalent to 'node.children[i] = child'. This method also sets the
        child's parent attribute appropriately.
        """
        child.parent = self
        self.children[i].parent = None
        self.children[i] = child
        self.changed()

    def insert_child(self, i: int, child: 'Base') -> None:
        """
        Equivalent to 'node.children.insert(i, child)'. This method also sets
        the child's parent attribute appropriately.
        """
        child.parent = self
        self.children.insert(i, child)
        self.changed()

    def append_child(self, child: 'Base') -> None:
        """
        Equivalent to 'node.children.append(child)'. This method also sets the
        child's parent attribute appropriately.
        """
        child.parent = self
        self.children.append(child)
        self.changed()

class Leaf(Base):
    """Concrete implementation for leaf nodes."""
    _prefix: str = ''
    lineno: int = 0
    column: int = 0

    def __init__(self, type: int, value: str, context: Optional[Tuple[str, Tuple[int, int]]] = None, prefix: Optional[str] = None, fixers_applied: List[Any] = []):
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

    def __repr__(self) -> str:
        """Return a canonical string representation."""
        return '%s(%r, %r)' % (self.__class__.__name__, self.type, self.value)

    def __unicode__(self) -> str:
        """
        Return a pretty string representation.

        This reproduces the input source exactly.
        """
        return self.prefix + str(self.value)
    if sys.version_info > (3, 0):
        __str__ = __unicode__

    def _eq(self, other: 'Leaf') -> bool:
        """Compare two nodes for equality."""
        return (self.type, self.value) == (other.type, other.value)

    def clone(self) -> 'Leaf':
        """Return a cloned (deep) copy of self."""
        return Leaf(self.type, self.value, (self.prefix, (self.lineno, self.column)), fixers_applied=self.fixers_applied)

    def leaves(self) -> Generator['Leaf', None, None]:
        yield self

    def post_order(self) -> Generator['Leaf', None, None]:
        """Return a post-order iterator for the tree."""
        yield self

    def pre_order(self) -> Generator['Leaf', None, None]:
        """Return a pre-order iterator for the tree."""
        yield self

    def _prefix_getter(self) -> str:
        """
        The whitespace and comments preceding this token in the input.
        """
        return self._prefix

    def _prefix_setter(self, prefix: str) -> None:
        self.changed()
        self._prefix = prefix
    prefix = property(_prefix_getter, _prefix_setter)

def convert(gr: Any, raw_node: Tuple[int, str, Optional[Tuple[str, Tuple[int, int]]], List[Base]]) -> Base:
    """
    Convert raw node information to a Node or Leaf instance.

    This is passed to the parser driver which calls it whenever a reduction of a
    grammar rule produces a new complete node, so that the tree is build
    strictly bottom-up.
    """
    type, value, context, children = raw_node
    if children or type in gr.number2symbol:
        if len(children) == 1:
            return children[0]
        return Node(type, children, context=context)
    else:
        return Leaf(type, value, context=context)

class BasePattern(object):
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
        args = [type_repr(self.type), self.content, self.name]
        while args and args[-1] is None:
            del args[-1]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr, args)))

    def optimize(self) -> 'BasePattern':
        """
        A subclass can define this as a hook for optimizations.

        Returns either self or another node with the same effect.
        """
        return self

    def match(self, node: Base, results: Optional[Dict[str, Base]] = None) -> bool:
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
                results.update(r)
        if results is not None and self.name:
            results[self.name] = node
        return True

    def match_seq(self, nodes: Sequence[Base], results: Optional[Dict[str, Base]] = None) -> bool:
        """
        Does this pattern exactly match a sequence of nodes?

        Default implementation for non-wildcard patterns.
        """
        if len(nodes) != 1:
            return False
        return self.match(nodes[0], results)

    def generate_matches(self, nodes: Sequence[Base]) -> Generator[Tuple[int, Dict[str, Base]], None, None]:
        """
        Generator yielding all matches for this pattern.

        Default implementation for non-wildcard patterns.
        """
        r: Dict[str, Base] = {}
        if nodes and self.match(nodes[0], r):
            yield (1, r)

class LeafPattern(BasePattern):

    def __init__(self, type: Optional[int] = None, content: Optional[str] = None, name: Optional[str] = None):
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
            assert isinstance(content, str), repr(content)
        self.type = type
        self.content = content
        self.name = name

    def match(self, node: Base, results: Optional[Dict[str, Leaf]] = None) -> bool:
        """Override match() to insist on a leaf node."""
        if not isinstance(node, Leaf):
            return False
        return BasePattern.match(self, node, results)

    def _submatch(self, node: Leaf, results: Optional[Dict[str, Leaf]] = None) -> bool:
        """
        Match the pattern's content to the node's children.

        This assumes the node type matches and self.content is not None.

        Returns True if it matches, False if not.

        If results is not None, it must be a dict which will be
        updated with the nodes matching named subpatterns.

        When returning False, the results dict may still be updated.
        """
        return