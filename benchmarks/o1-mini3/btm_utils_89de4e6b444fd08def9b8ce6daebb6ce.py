from typing import List, Optional, Any, Generator, Union, Tuple, Callable
from . import pytree
from .pgen2 import grammar, token
from .pygram import pattern_symbols, python_symbols

syms = pattern_symbols
pysyms = python_symbols
tokens = grammar.opmap
token_labels = token

TYPE_ANY = -1
TYPE_ALTERNATIVES = -2
TYPE_GROUP = -3


class MinNode:
    """This class serves as an intermediate representation of the
    pattern tree during the conversion to sets of leaf-to-root
    subpatterns
    """

    def __init__(self, type: Optional[int] = None, name: Optional[str] = None) -> None:
        self.type: Optional[int] = type
        self.name: Optional[str] = name
        self.children: List['MinNode'] = []
        self.leaf: bool = False
        self.parent: Optional['MinNode'] = None
        self.alternatives: List[List[Any]] = []
        self.group: List[Any] = []

    def __repr__(self) -> str:
        return f"{self.type} {self.name}"

    def leaf_to_root(self) -> List[Union[int, str, Tuple[Any, ...]]]:
        """Internal method. Returns a characteristic path of the
        pattern tree. This method must be run for all leaves until the
        linear subpatterns are merged into a single
        """
        node: Optional['MinNode'] = self
        subp: List[Union[int, str]] = []
        while node:
            if node.type == TYPE_ALTERNATIVES:
                node.alternatives.append(subp)
                if len(node.alternatives) == len(node.children):
                    subp = [tuple(node.alternatives)]
                    node.alternatives = []
                    node = node.parent
                    continue
                else:
                    node = node.parent
                    subp = []
                    break
            if node.type == TYPE_GROUP:
                node.group.append(subp)
                if len(node.group) == len(node.children):
                    subp = get_characteristic_subpattern(node.group)
                    node.group = []
                    node = node.parent
                    continue
                else:
                    node = node.parent
                    subp = []
                    break
            if node.type == token_labels.NAME and node.name:
                subp.append(node.name)
            else:
                subp.append(node.type)
            node = node.parent
        return subp

    def get_linear_subpattern(self) -> Optional[List[Union[int, str, Tuple[Any, ...]]]]:
        """Drives the leaf_to_root method. The reason that
        leaf_to_root must be run multiple times is because we need to
        reject 'group' matches; for example the alternative form
        (a | b c) creates a group [b c] that needs to be matched. Since
        matching multiple linear patterns overcomes the automaton's
        capabilities, leaf_to_root merges each group into a single
        choice based on 'characteristic'ity,

        i.e. (a|b c) -> (a|b) if b more characteristic than c

        Returns: The most 'characteristic'(as defined by
          get_characteristic_subpattern) path for the compiled pattern
          tree.
        """
        for l in self.leaves():
            subp = l.leaf_to_root()
            if subp:
                return subp
        return None

    def leaves(self) -> Generator['MinNode', None, None]:
        """Generator that returns the leaves of the tree"""
        for child in self.children:
            yield from child.leaves()
        if not self.children:
            yield self


def reduce_tree(node: pytree.Node, parent: Optional[MinNode] = None) -> Optional[MinNode]:
    """
    Internal function. Reduces a compiled pattern tree to an
    intermediate representation suitable for feeding the
    automaton. This also trims off any optional pattern elements(like
    [a], a*).
    """
    new_node: Optional[MinNode] = None
    if node.type == syms.Matcher:
        node = node.children[0]
    if node.type == syms.Alternatives:
        if len(node.children) <= 2:
            new_node = reduce_tree(node.children[0], parent)
        else:
            new_node = MinNode(type=TYPE_ALTERNATIVES)
            for child in node.children:
                if node.children.index(child) % 2:
                    continue
                reduced = reduce_tree(child, new_node)
                if reduced is not None:
                    new_node.children.append(reduced)
    elif node.type == syms.Alternative:
        if len(node.children) > 1:
            new_node = MinNode(type=TYPE_GROUP)
            for child in node.children:
                reduced = reduce_tree(child, new_node)
                if reduced:
                    new_node.children.append(reduced)
            if not new_node.children:
                new_node = None
        else:
            new_node = reduce_tree(node.children[0], parent)
    elif node.type == syms.Unit:
        if isinstance(node.children[0], pytree.Leaf) and node.children[0].value == '(':
            return reduce_tree(node.children[1], parent)
        if (
            (isinstance(node.children[0], pytree.Leaf) and node.children[0].value == '[')
            or (
                len(node.children) > 1
                and hasattr(node.children[1], 'value')
                and node.children[1].value == '['
            )
        ):
            return None
        leaf = True
        details_node: Optional[pytree.Node] = None
        alternatives_node: Optional[pytree.Node] = None
        has_repeater: bool = False
        repeater_node: Optional[pytree.Node] = None
        has_variable_name: bool = False
        for child in node.children:
            if child.type == syms.Details:
                leaf = False
                details_node = child
            elif child.type == syms.Repeater:
                has_repeater = True
                repeater_node = child
            elif child.type == syms.Alternatives:
                alternatives_node = child
            if hasattr(child, 'value') and child.value == '=':
                has_variable_name = True
        if has_variable_name:
            name_leaf = node.children[2]
            if hasattr(name_leaf, 'value') and name_leaf.value == '(':
                name_leaf = node.children[3]
        else:
            name_leaf = node.children[0]
        if name_leaf.type == token_labels.NAME:
            if name_leaf.value == 'any':
                new_node = MinNode(type=TYPE_ANY)
            elif hasattr(token_labels, name_leaf.value):
                new_node = MinNode(type=getattr(token_labels, name_leaf.value))
            else:
                new_node = MinNode(type=getattr(pysyms, name_leaf.value))
        elif name_leaf.type == token_labels.STRING:
            name = name_leaf.value.strip("'")
            if name in tokens:
                new_node = MinNode(type=tokens[name])
            else:
                new_node = MinNode(type=token_labels.NAME, name=name)
        elif name_leaf.type == syms.Alternatives:
            new_node = reduce_tree(alternatives_node, parent)
        if has_repeater:
            if repeater_node.children[0].value == '*':
                new_node = None
            elif repeater_node.children[0].value == '+':
                pass
            else:
                raise NotImplementedError
        if details_node and new_node is not None:
            for child in details_node.children[1:-1]:
                reduced = reduce_tree(child, new_node)
                if reduced is not None:
                    new_node.children.append(reduced)
    if new_node:
        new_node.parent = parent
    return new_node


def get_characteristic_subpattern(subpatterns: Any) -> Any:
    """Picks the most characteristic from a list of linear patterns
    Current order used is:
    names > common_names > common_chars
    """
    if not isinstance(subpatterns, list):
        return subpatterns
    if len(subpatterns) == 1:
        return subpatterns[0]
    subpatterns_with_names: List[Any] = []
    subpatterns_with_common_names: List[Any] = []
    common_names: List[str] = ['in', 'for', 'if', 'not', 'None']
    subpatterns_with_common_chars: List[Any] = []
    common_chars: str = '[]().,:'
    for subpattern in subpatterns:
        if any(rec_test(subpattern, lambda x: isinstance(x, str))):
            if any(rec_test(subpattern, lambda x: isinstance(x, str) and x in common_chars)):
                subpatterns_with_common_chars.append(subpattern)
            elif any(rec_test(subpattern, lambda x: isinstance(x, str) and x in common_names)):
                subpatterns_with_common_names.append(subpattern)
            else:
                subpatterns_with_names.append(subpattern)
    if subpatterns_with_names:
        subpatterns = subpatterns_with_names
    elif subpatterns_with_common_names:
        subpatterns = subpatterns_with_common_names
    elif subpatterns_with_common_chars:
        subpatterns = subpatterns_with_common_chars
    return max(subpatterns, key=len)


def rec_test(sequence: Any, test_func: Callable[[Any], bool]) -> Generator[bool, None, None]:
    """Tests test_func on all items of sequence and items of included
    sub-iterables
    """
    for x in sequence:
        if isinstance(x, (list, tuple)):
            yield from rec_test(x, test_func)
        else:
            yield test_func(x)
