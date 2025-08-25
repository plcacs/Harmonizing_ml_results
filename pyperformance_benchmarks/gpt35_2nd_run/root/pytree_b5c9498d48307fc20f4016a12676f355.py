from typing import Dict, List, Tuple, Union

_type_reprs: Dict[int, str] = {}

class Base(object):
    type: Union[int, None] = None
    parent: Union[Base, None] = None
    children: Tuple = ()
    was_changed: bool = False
    was_checked: bool = False

    def __new__(cls, *args, **kwds) -> object:
        assert (cls is not Base), 'Cannot instantiate Base'
        return object.__new__(cls)

    def __eq__(self, other) -> Union[bool, NotImplemented]:
        if (self.__class__ is not other.__class__):
            return NotImplemented
        return self._eq(other)

    def _eq(self, other) -> None:
        raise NotImplementedError

    def clone(self) -> None:
        raise NotImplementedError

    def post_order(self) -> None:
        raise NotImplementedError

    def pre_order(self) -> None:
        raise NotImplementedError

    def replace(self, new) -> None:
        assert (self.parent is not None), str(self)
        assert (new is not None)
        if (not isinstance(new, list)):
            new = [new]
        l_children = []
        found = False
        for ch in self.parent.children:
            if (ch is self):
                assert (not found), (self.parent.children, self, new)
                if (new is not None):
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

    def get_lineno(self) -> Union[int, None]:
        node = self
        while (not isinstance(node, Leaf)):
            if (not node.children):
                return
            node = node.children[0]
        return node.lineno

    def changed(self) -> None:
        if self.parent:
            self.parent.changed()
        self.was_changed = True

    def remove(self) -> Union[int, None]:
        if self.parent:
            for (i, node) in enumerate(self.parent.children):
                if (node is self):
                    self.parent.changed()
                    del self.parent.children[i]
                    self.parent = None
                    return i

    @property
    def next_sibling(self) -> Union[Base, None]:
        if (self.parent is None):
            return None
        for (i, child) in enumerate(self.parent.children):
            if (child is self):
                try:
                    return self.parent.children[(i + 1)]
                except IndexError:
                    return None

    @property
    def prev_sibling(self) -> Union[Base, None]:
        if (self.parent is None):
            return None
        for (i, child) in enumerate(self.parent.children):
            if (child is self):
                if (i == 0):
                    return None
                return self.parent.children[(i - 1)]

    def leaves(self) -> None:
        for child in self.children:
            (yield from child.leaves())

    def depth(self) -> int:
        if (self.parent is None):
            return 0
        return (1 + self.parent.depth())

    def get_suffix(self) -> str:
        next_sib = self.next_sibling
        if (next_sib is None):
            return ''
        return next_sib.prefix

    if (sys.version_info < (3, 0)):

        def __str__(self) -> bytes:
            return str(self).encode('ascii')

class Node(Base):
    type: int = None
    children: List = []

    def __init__(self, type: int, children: List[Base], context=None, prefix=None, fixers_applied=None) -> None:
        assert (type >= 256), type
        self.type = type
        self.children = list(children)
        for ch in self.children:
            assert (ch.parent is None), repr(ch)
            ch.parent = self
        if (prefix is not None):
            self.prefix = prefix
        if fixers_applied:
            self.fixers_applied = fixers_applied[:]
        else:
            self.fixers_applied = None

    def __repr__(self) -> str:
        return ('%s(%s, %r)' % (self.__class__.__name__, type_repr(self.type), self.children))

    def __unicode__(self) -> str:
        return ''.join(map(str, self.children))

    if (sys.version_info > (3, 0)):
        __str__ = __unicode__

    def _eq(self, other) -> bool:
        return ((self.type, self.children) == (other.type, other.children))

    def clone(self) -> Node:
        return Node(self.type, [ch.clone() for ch in self.children], fixers_applied=self.fixers_applied)

    def post_order(self) -> None:
        for child in self.children:
            (yield from child.post_order())
        (yield self)

    def pre_order(self) -> None:
        (yield self)
        for child in self.children:
            (yield from child.pre_order())

    @property
    def prefix(self) -> str:
        if (not self.children):
            return ''
        return self.children[0].prefix

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        if self.children:
            self.children[0].prefix = prefix

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
    type: int = None
    value: str = ''
    _prefix: str = ''
    lineno: int = 0
    column: int = 0

    def __init__(self, type: int, value: str, context=None, prefix=None, fixers_applied=[]) -> None:
        assert (0 <= type < 256), type
        if (context is not None):
            (self._prefix, (self.lineno, self.column)) = context
        self.type = type
        self.value = value
        if (prefix is not None):
            self._prefix = prefix
        self.fixers_applied = fixers_applied[:]

    def __repr__(self) -> str:
        return ('%s(%r, %r)' % (self.__class__.__name__, self.type, self.value))

    def __unicode__(self) -> str:
        return (self.prefix + str(self.value))

    if (sys.version_info > (3, 0)):
        __str__ = __unicode__

    def _eq(self, other) -> bool:
        return ((self.type, self.value) == (other.type, other.value))

    def clone(self) -> Leaf:
        return Leaf(self.type, self.value, (self.prefix, (self.lineno, self.column)), fixers_applied=self.fixers_applied)

    def leaves(self) -> None:
        (yield self)

    def post_order(self) -> None:
        (yield self)

    def pre_order(self) -> None:
        (yield self)

    @property
    def prefix(self) -> str:
        return self._prefix

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        self.changed()
        self._prefix = prefix

def convert(gr, raw_node) -> Union[Node, Leaf]:
    (type, value, context, children) = raw_node
    if (children or (type in gr.number2symbol)):
        if (len(children) == 1):
            return children[0]
        return Node(type, children, context=context)
    else:
        return Leaf(type, value, context=context)

class BasePattern(object):
    type: Union[int, None] = None
    content: Union[str, None] = None
    name: Union[str, None] = None

    def __new__(cls, *args, **kwds) -> object:
        assert (cls is not BasePattern), 'Cannot instantiate BasePattern'
        return object.__new__(cls)

    def optimize(self) -> BasePattern:
        return self

    def match(self, node, results=None) -> bool:
        if ((self.type is not None) and (node.type != self.type)):
            return False
        if (self.content is not None):
            r = None
            if (results is not None):
                r = {}
            if (not self._submatch(node, r)):
                return False
            if r:
                results.update(r)
        if ((results is not None) and self.name):
            results[self.name] = node
        return True

    def match_seq(self, nodes, results=None) -> bool:
        if (len(nodes) != 1):
            return False
        return self.match(nodes[0], results)

    def generate_matches(self, nodes) -> None:
        r = {}
        if (nodes and self.match(nodes[0], r)):
            (yield (1, r))

class LeafPattern(BasePattern):

    def __init__(self, type=None, content=None, name=None) -> None:
        if (type is not None):
            assert (0 <= type < 256), type
        if (content is not None):
            assert isinstance(content, str), repr(content)
        self.type = type
        self.content = content
        self.name = name

    def match(self, node, results=None) -> bool:
        if (not isinstance(node, Leaf)):
            return False
        return BasePattern.match(self, node, results)

    def _submatch(self, node, results=None) -> bool:
        return (self.content == node.value)

class NodePattern(BasePattern):
    type: Union[int, None] = None
    content: Union[List[BasePattern], None] = None
    name: Union[str, None] = None
    wildcards: bool = False

    def __init__(self, type=None, content=None, name=None) -> None:
        if (type is not None):
            assert (type >= 256), type
        if (content is not None):
            assert (not isinstance(content, str)), repr(content)
            content = list(content)
            for (i, item) in enumerate(content):
                assert isinstance(item, BasePattern), (i, item)
                if isinstance(item, WildcardPattern):
                    self.wildcards = True
        self.type = type
        self.content = content
        self.name = name

    def _submatch(self, node, results=None) -> bool:
        if self.wildcards:
            for (c, r) in generate_matches(self.content, node.children):
                if (c == len(node.children)):
                    if (results is not None):
                        results.update(r)
                    return True
            return False
        if (len(self.content) != len(node.children)):
            return False
        for (subpattern, child) in zip(self.content, node.children):
            if (not subpattern.match(child, results)):
                return False
        return True

class WildcardPattern(BasePattern):
    content: Union[Tuple[Tuple[BasePattern]], None] = None
    min: int = 0
    max: int = 2147483647
    name: Union[str, None] = None

    def __init__(self, content=None, min=0, max=2147483647, name=None) -> None:
        assert (0 <= min <= max <= 2147483647), (min, max)
        if (content is not None):
            content = tuple(map(tuple, content))
            assert len(content), repr(content)
            for alt in content:
                assert len(alt), repr(alt)
        self.content = content
        self.min = min
        self.max = max
        self.name = name

    def optimize(self) -> WildcardPattern:
        subpattern = None
        if ((self.content is not None) and (len(self.content) == 1) and (len(self.content[0]) == 1)):
            subpattern = self.content[0][0]
        if ((self.min == 1) and (self.max == 1)):
            if (self.content is None):
                return NodePattern(name=self.name)
            if ((subpattern is not None) and (self.name == subpattern.name)):
                return subpattern.optimize()
        if ((self.min <= 1) and isinstance(subpattern, WildcardPattern) and (subpattern.min <= 1) and (self.name == subpattern.name)):
            return WildcardPattern(subpattern.content, (self.min * subpattern.min), (self.max * subpattern.max), subpattern.name)
        return self

    def match(self, node, results=None) -> bool:
        return self.match_seq([node], results)

    def match_seq(self, nodes, results=None) -> bool:
        for (c, r) in self.generate_matches(nodes):
            if (c == len(nodes)):
                if (results is not None):
                    results.update(r)
                    if self.name:
                        results[self.name] = list(nodes)
                return True
        return False

    def generate_matches(self, nodes) -> None:
        if (self.content is None):
            for count in range(self.min, (1 + min(len(nodes), self.max))):
                r = {}
                if self.name:
                    r[self.name] = nodes[:count]
                (yield (count, r))
        elif (self.name == 'bare_name'):
            (yield self._bare_name_matches(nodes))
        else:
            if hasattr(sys, 'getrefcount'):
                save_stderr = sys.stderr
                sys.stderr = StringIO()
            try:
                for (count, r) in self._recursive_matches(nodes, 0):
                    if self.name:
                        r[self.name] = nodes[:count]
                    (yield (count, r))
            except RuntimeError:
                for (count, r) in self._iterative_matches(nodes):
                    if self.name:
                        r[self.name] = nodes[:count]
                    (yield (count, r))
            finally:
                if hasattr(sys, 'getrefcount'):
                    sys.stderr = save_stderr

    def _iterative_matches(self, nodes) -> None:
        nodelen = len(nodes)
        if (0 >= self.min):
            (yield (0, {}))
        results = []
        for alt in self.content:
            for (c, r) in generate_matches(alt, nodes):
                (yield (c, r))
                results.append((c, r))
        while results:
            new_results = []
            for (c0, r0) in results:
                if ((c0 < nodelen) and (c0 <= self.max)):
                    for alt in self.content:
                        for (c1, r1) in generate_matches(alt, nodes[c0:]):
                            if (c1 > 0):
                                r = {}
                                r.update(r0)
                                r.update(r1)
                                (yield ((c0 + c1), r))
                                new_results.append(((c0 + c1), r))
            results = new_results

    def _bare_name_matches(self, nodes) -> Tuple[int, Dict]:
        count = 0
        r = {}
        done = False
        max = len(nodes)
        while ((not done) and (count < max)):
            done = True
            for leaf in self.content:
                if leaf[0].match(nodes[count], r):
                    count += 1
                    done = False
                    break
        r[self.name] = nodes[:count]
        return (count, r)

    def _recursive_matches(self, nodes, count) -> None:
        assert (self.content is not None)
        if (count >= self.min):
            (yield (0, {}))
        if (count < self.max):
            for alt in self.content:
                for (c0, r0) in generate_matches(alt, nodes):
                    for (c1, r1) in self._recursive_matches(nodes[c0:], (count + 1)):
                        r = {}
                        r.update(r0)
                        r.update(r1)
                        (yield ((c0 + c1), r))
