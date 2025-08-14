from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union

from parso.utils import split_lines


def search_ancestor(node: "NodeOrLeaf", *node_types: str) -> Optional["BaseNode"]:
    """
    Recursively looks at the parents of a node and returns the first found node
    that matches ``node_types``. Returns ``None`` if no matching node is found.

    This function is deprecated, use :meth:`NodeOrLeaf.search_ancestor` instead.

    :param node: The ancestors of this node will be checked.
    :param node_types: type names that are searched for.
    """
    n: Optional["BaseNode"] = node.parent
    while n is not None:
        if n.type in node_types:
            return n
        n = n.parent
    return None


class NodeOrLeaf:
    """
    The base class for nodes and leaves.
    """
    __slots__ = ("parent",)
    type: str
    """
    The type is a string that typically matches the types of the grammar file.
    """
    parent: Optional["BaseNode"]
    """
    The parent :class:`BaseNode` of this node or leaf.
    None if this is the root node.
    """

    def get_root_node(self) -> "NodeOrLeaf":
        """
        Returns the root node of a parser tree. The returned node doesn't have
        a parent node like all the other nodes/leaves.
        """
        scope: NodeOrLeaf = self
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def get_next_sibling(self) -> Optional["NodeOrLeaf"]:
        """
        Returns the node immediately following this node in this parent's
        children list. If this node does not have a next sibling, it is None
        """
        parent = self.parent
        if parent is None:
            return None

        for i, child in enumerate(parent.children):
            if child is self:
                try:
                    return parent.children[i + 1]
                except IndexError:
                    return None
        return None

    def get_previous_sibling(self) -> Optional["NodeOrLeaf"]:
        """
        Returns the node immediately preceding this node in this parent's
        children list. If this node does not have a previous sibling, it is
        None.
        """
        parent = self.parent
        if parent is None:
            return None

        for i, child in enumerate(parent.children):
            if child is self:
                if i == 0:
                    return None
                return parent.children[i - 1]
        return None

    def get_previous_leaf(self) -> Optional["NodeOrLeaf"]:
        """
        Returns the previous leaf in the parser tree.
        Returns `None` if this is the first element in the parser tree.
        """
        if self.parent is None:
            return None

        node: NodeOrLeaf = self
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

    def get_next_leaf(self) -> Optional["NodeOrLeaf"]:
        """
        Returns the next leaf in the parser tree.
        Returns None if this is the last element in the parser tree.
        """
        if self.parent is None:
            return None

        node: NodeOrLeaf = self
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
        """
        Returns the starting position of the prefix as a tuple, e.g. `(3, 4)`.

        :return tuple of int: (line, column)
        """

    @abstractproperty
    def end_pos(self) -> Tuple[int, int]:
        """
        Returns the end position of the prefix as a tuple, e.g. `(3, 4)`.

        :return tuple of int: (line, column)
        """

    @abstractmethod
    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        """
        Returns the start_pos of the prefix.
        """
    
    @abstractmethod
    def get_first_leaf(self) -> "NodeOrLeaf":
        """
        Returns the first leaf of a node or itself if this is a leaf.
        """

    @abstractmethod
    def get_last_leaf(self) -> "NodeOrLeaf":
        """
        Returns the last leaf of a node or itself if this is a leaf.
        """

    @abstractmethod
    def get_code(self, include_prefix: bool = True) -> str:
        """
        Returns the code that was the input for the parser for this node.

        :param include_prefix: Removes the prefix (whitespace and comments) of
            e.g. a statement.
        """

    def search_ancestor(self, *node_types: str) -> Optional["BaseNode"]:
        """
        Recursively looks at the parents of this node or leaf and returns the
        first found node that matches ``node_types``. Returns ``None`` if no
        matching node is found.

        :param node_types: type names that are searched for.
        """
        node: Optional["BaseNode"] = self.parent
        while node is not None:
            if node.type in node_types:
                return node
            node = node.parent
        return None

    def dump(self, *, indent: Optional[Union[int, str]] = 4) -> str:
        """
        Returns a formatted dump of the parser tree rooted at this node or leaf.
        This is mainly useful for debugging purposes.
        """
        if indent is None:
            newline = False
            indent_string = ""
        elif isinstance(indent, int):
            newline = True
            indent_string = " " * indent
        elif isinstance(indent, str):
            newline = True
            indent_string = indent
        else:
            raise TypeError(f"expect 'indent' to be int, str or None, got {indent!r}")

        def _format_dump(node: "NodeOrLeaf", indent_str: str = "", top_level: bool = True) -> str:
            result = ""
            node_type = type(node).__name__
            if isinstance(node, Leaf):
                result += f"{indent_str}{node_type}("
                if isinstance(node, ErrorLeaf):
                    result += f"{node.token_type!r}, "
                elif isinstance(node, TypedLeaf):
                    result += f"{node.type!r}, "
                result += f"{node.value!r}, {node.start_pos!r}"
                if node.prefix:
                    result += f", prefix={node.prefix!r}"
                result += ")"
            elif isinstance(node, BaseNode):
                result += f"{indent_str}{node_type}("
                if isinstance(node, Node):
                    result += f"{node.type!r}, "
                result += "["
                if newline:
                    result += "\n"
                for child in node.children:
                    result += _format_dump(child, indent_str + indent_string, top_level=False)
                result += f"{indent_str}])"
            else:
                raise TypeError(f"unsupported node encountered: {node!r}")
            if not top_level:
                if newline:
                    result += ",\n"
                else:
                    result += ", "
            return result

        return _format_dump(self)


class Leaf(NodeOrLeaf):
    """
    Leafs are basically tokens with a better API.
    """
    __slots__ = ("value", "line", "column", "prefix")
    prefix: str

    def __init__(self, value: str, start_pos: Tuple[int, int], prefix: str = "") -> None:
        self.value = value
        self.start_pos = start_pos  # This will trigger the setter.
        self.prefix = prefix
        self.parent: Optional["BaseNode"] = None

    @property
    def start_pos(self) -> Tuple[int, int]:
        return (self.line, self.column)

    @start_pos.setter
    def start_pos(self, value: Tuple[int, int]) -> None:
        self.line, self.column = value

    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        previous_leaf: Optional[NodeOrLeaf] = self.get_previous_leaf()
        if previous_leaf is None:
            lines = split_lines(self.prefix)
            return self.line - len(lines) + 1, 0
        return previous_leaf.end_pos

    def get_first_leaf(self) -> "NodeOrLeaf":
        return self

    def get_last_leaf(self) -> "NodeOrLeaf":
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
        return end_pos_line, end_pos_column

    def __repr__(self) -> str:
        val = self.value
        if not val:
            val = self.type
        return "<%s: %s>" % (type(self).__name__, val)


class TypedLeaf(Leaf):
    __slots__ = ("type",)

    def __init__(self, type: str, value: str, start_pos: Tuple[int, int], prefix: str = "") -> None:
        super().__init__(value, start_pos, prefix)
        self.type = type


class BaseNode(NodeOrLeaf):
    """
    The super class for all nodes.
    A node has children, a type and possibly a parent node.
    """
    __slots__ = ("children",)

    def __init__(self, children: List[NodeOrLeaf]) -> None:
        self.children: List[NodeOrLeaf] = children
        self.parent: Optional["BaseNode"] = None
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
            return "".join(c.get_code() for c in children)
        else:
            first = children[0].get_code(include_prefix=False)
            return first + "".join(c.get_code() for c in children[1:])

    def get_code(self, include_prefix: bool = True) -> str:
        return self._get_code_for_children(self.children, include_prefix)

    def get_leaf_for_position(
        self, position: Tuple[int, int], include_prefixes: bool = False
    ) -> Optional["Leaf"]:
        """
        Get the :py:class:`parso.tree.Leaf` at ``position``
        """
        def binary_search(lower: int, upper: int) -> Optional["Leaf"]:
            if lower == upper:
                element = self.children[lower]
                if not include_prefixes and position < element.start_pos:
                    return None
                try:
                    return element.get_leaf_for_position(position, include_prefixes)  # type: ignore
                except AttributeError:
                    return element  # type: ignore
            index = (lower + upper) // 2
            element = self.children[index]
            if position <= element.end_pos:
                return binary_search(lower, index)
            else:
                return binary_search(index + 1, upper)
        if not ((1, 0) <= position <= self.children[-1].end_pos):
            raise ValueError("Please provide a position that exists within this node.")
        return binary_search(0, len(self.children) - 1)

    def get_first_leaf(self) -> "NodeOrLeaf":
        return self.children[0].get_first_leaf()

    def get_last_leaf(self) -> "NodeOrLeaf":
        return self.children[-1].get_last_leaf()

    def __repr__(self) -> str:
        code = self.get_code().replace("\n", " ").replace("\r", " ").strip()
        return "<%s: %s@%s,%s>" % (type(self).__name__, code, self.start_pos[0], self.start_pos[1])


class Node(BaseNode):
    """Concrete implementation for interior nodes."""
    __slots__ = ("type",)

    def __init__(self, type: str, children: List[NodeOrLeaf]) -> None:
        super().__init__(children)
        self.type = type

    def __repr__(self) -> str:
        return "%s(%s, %r)" % (self.__class__.__name__, self.type, self.children)


class ErrorNode(BaseNode):
    """
    A node that contains valid nodes/leaves that we're follow by a token that
    was invalid.
    """
    __slots__ = ()
    type = "error_node"


class ErrorLeaf(Leaf):
    """
    A leaf that is either completely invalid in a language or is invalid at that position.
    """
    __slots__ = ("token_type",)
    type = "error_leaf"

    def __init__(self, token_type: str, value: str, start_pos: Tuple[int, int], prefix: str = "") -> None:
        super().__init__(value, start_pos, prefix)
        self.token_type = token_type

    def __repr__(self) -> str:
        return "<%s: %s:%s, %s>" % (type(self).__name__, self.token_type, repr(self.value), self.start_pos)
