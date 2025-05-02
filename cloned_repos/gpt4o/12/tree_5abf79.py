from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines

def search_ancestor(node: 'NodeOrLeaf', *node_types: str) -> Optional['NodeOrLeaf']:
    n = node.parent
    while n is not None:
        if n.type in node_types:
            return n
        n = n.parent
    return None

class NodeOrLeaf:
    __slots__ = ('parent',)

    def get_root_node(self) -> 'NodeOrLeaf':
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def get_next_sibling(self) -> Optional['NodeOrLeaf']:
        parent = self.parent
        if parent is None:
            return None
        for i, child in enumerate(parent.children):
            if child is self:
                try:
                    return self.parent.children[i + 1]
                except IndexError:
                    return None

    def get_previous_sibling(self) -> Optional['NodeOrLeaf']:
        parent = self.parent
        if parent is None:
            return None
        for i, child in enumerate(parent.children):
            if child is self:
                if i == 0:
                    return None
                return self.parent.children[i - 1]

    def get_previous_leaf(self) -> Optional['Leaf']:
        if self.parent is None:
            return None
        node = self
        while True:
            c = node.parent.children
            i = c.index(node)
            if i == 0:
                node = node.parent
                if node.parent is None:
                    return None
            else:
                node = c[i - 1]
                break
        while True:
            try:
                node = node.children[-1]
            except AttributeError:
                return node

    def get_next_leaf(self) -> Optional['Leaf']:
        if self.parent is None:
            return None
        node = self
        while True:
            c = node.parent.children
            i = c.index(node)
            if i == len(c) - 1:
                node = node.parent
                if node.parent is None:
                    return None
            else:
                node = c[i + 1]
                break
        while True:
            try:
                node = node.children[0]
            except AttributeError:
                return node

    @abstractproperty
    def start_pos(self) -> Tuple[int, int]:
        pass

    @abstractproperty
    def end_pos(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_first_leaf(self) -> 'Leaf':
        pass

    @abstractmethod
    def get_last_leaf(self) -> 'Leaf':
        pass

    @abstractmethod
    def get_code(self, include_prefix: bool = True) -> str:
        pass

    def search_ancestor(self, *node_types: str) -> Optional['NodeOrLeaf']:
        node = self.parent
        while node is not None:
            if node.type in node_types:
                return node
            node = node.parent
        return None

    def dump(self, *, indent: Union[int, str, None] = 4) -> str:
        if indent is None:
            newline = False
            indent_string = ''
        elif isinstance(indent, int):
            newline = True
            indent_string = ' ' * indent
        elif isinstance(indent, str):
            newline = True
            indent_string = indent
        else:
            raise TypeError(f"expect 'indent' to be int, str or None, got {indent!r}")

        def _format_dump(node: 'NodeOrLeaf', indent: str = '', top_level: bool = True) -> str:
            result = ''
            node_type = type(node).__name__
            if isinstance(node, Leaf):
                result += f'{indent}{node_type}('
                if isinstance(node, ErrorLeaf):
                    result += f'{node.token_type!r}, '
                elif isinstance(node, TypedLeaf):
                    result += f'{node.type!r}, '
                result += f'{node.value!r}, {node.start_pos!r}'
                if node.prefix:
                    result += f', prefix={node.prefix!r}'
                result += ')'
            elif isinstance(node, BaseNode):
                result += f'{indent}{node_type}('
                if isinstance(node, Node):
                    result += f'{node.type!r}, '
                result += '['
                if newline:
                    result += '\n'
                for child in node.children:
                    result += _format_dump(child, indent=indent + indent_string, top_level=False)
                result += f'{indent}])'
            else:
                raise TypeError(f'unsupported node encountered: {node!r}')
            if not top_level:
                if newline:
                    result += ',\n'
                else:
                    result += ', '
            return result
        return _format_dump(self)

class Leaf(NodeOrLeaf):
    __slots__ = ('value', 'line', 'column', 'prefix')

    def __init__(self, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None:
        self.value = value
        self.start_pos = start_pos
        self.prefix = prefix
        self.parent = None

    @property
    def start_pos(self) -> Tuple[int, int]:
        return (self.line, self.column)

    @start_pos.setter
    def start_pos(self, value: Tuple[int, int]) -> None:
        self.line = value[0]
        self.column = value[1]

    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        previous_leaf = self.get_previous_leaf()
        if previous_leaf is None:
            lines = split_lines(self.prefix)
            return (self.line - len(lines) + 1, 0)
        return previous_leaf.end_pos

    def get_first_leaf(self) -> 'Leaf':
        return self

    def get_last_leaf(self) -> 'Leaf':
        return self

    def get_code(self, include_prefix: bool = True) -> str:
        if include_prefix:
            return self.prefix + self.value
        else:
            return self.value

    @property
    def end_pos(self) -> Tuple[int, int]:
        lines = split_lines(self.value)
        end_pos_line = self.line + len(lines) - 1
        if self.line == end_pos_line:
            end_pos_column = self.column + len(lines[-1])
        else:
            end_pos_column = len(lines[-1])
        return (end_pos_line, end_pos_column)

    def __repr__(self) -> str:
        value = self.value
        if not value:
            value = self.type
        return '<%s: %s>' % (type(self).__name__, value)

class TypedLeaf(Leaf):
    __slots__ = ('type',)

    def __init__(self, type: str, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None:
        super().__init__(value, start_pos, prefix)
        self.type = type

class BaseNode(NodeOrLeaf):
    __slots__ = ('children',)

    def __init__(self, children: List[NodeOrLeaf]) -> None:
        self.children = children
        self.parent = None
        for child in children:
            child.parent = self

    @property
    def start_pos(self) -> Tuple[int, int]:
        return self.children[0].start_pos

    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        return self.children[0].get_start_pos_of_prefix()

    @property
    def end_pos(self) -> Tuple[int, int]:
        return self.children[-1].end_pos

    def _get_code_for_children(self, children: List[NodeOrLeaf], include_prefix: bool) -> str:
        if include_prefix:
            return ''.join((c.get_code() for c in children))
        else:
            first = children[0].get_code(include_prefix=False)
            return first + ''.join((c.get_code() for c in children[1:]))

    def get_code(self, include_prefix: bool = True) -> str:
        return self._get_code_for_children(self.children, include_prefix)

    def get_leaf_for_position(self, position: Tuple[int, int], include_prefixes: bool = False) -> Optional['Leaf']:
        def binary_search(lower: int, upper: int) -> Optional['Leaf']:
            if lower == upper:
                element = self.children[lower]
                if not include_prefixes and position < element.start_pos:
                    return None
                try:
                    return element.get_leaf_for_position(position, include_prefixes)
                except AttributeError:
                    return element
            index = int((lower + upper) / 2)
            element = self.children[index]
            if position <= element.end_pos:
                return binary_search(lower, index)
            else:
                return binary_search(index + 1, upper)
        if not (1, 0) <= position <= self.children[-1].end_pos:
            raise ValueError('Please provide a position that exists within this node.')
        return binary_search(0, len(self.children) - 1)

    def get_first_leaf(self) -> 'Leaf':
        return self.children[0].get_first_leaf()

    def get_last_leaf(self) -> 'Leaf':
        return self.children[-1].get_last_leaf()

    def __repr__(self) -> str:
        code = self.get_code().replace('\n', ' ').replace('\r', ' ').strip()
        return '<%s: %s@%s,%s>' % (type(self).__name__, code, self.start_pos[0], self.start_pos[1])

class Node(BaseNode):
    __slots__ = ('type',)

    def __init__(self, type: str, children: List[NodeOrLeaf]) -> None:
        super().__init__(children)
        self.type = type

    def __repr__(self) -> str:
        return '%s(%s, %r)' % (self.__class__.__name__, self.type, self.children)

class ErrorNode(BaseNode):
    __slots__ = ()
    type = 'error_node'

class ErrorLeaf(Leaf):
    __slots__ = ('token_type',)
    type = 'error_leaf'

    def __init__(self, token_type: str, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None:
        super().__init__(value, start_pos, prefix)
        self.token_type = token_type

    def __repr__(self) -> str:
        return '<%s: %s:%s, %s>' % (type(self).__name__, self.token_type, repr(self.value), self.start_pos)
