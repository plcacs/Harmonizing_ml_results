from typing import List, Tuple, Union, Optional, Iterator, Any, Callable, Generator
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
    def __init__(self, type: Optional[int] = None, name: Optional[str] = None) -> None:
        self.type: Optional[int] = type
        self.name: Optional[str] = name
        self.children: List['MinNode'] = []
        self.leaf: bool = False
        self.parent: Optional['MinNode'] = None
        self.alternatives: List[List[Union[int, str]]] = []
        self.group: List[List[Union[int, str]]] = []

    def __repr__(self) -> str:
        return f"{self.type} {self.name}"

    def leaf_to_root(self) -> Optional[List[Union[int, str]]]:
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
                    subp = None
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
                    subp = None
                    break
            if (node.type == token_labels.NAME) and node.name:
                subp.append(node.name)
            else:
                subp.append(node.type)
            node = node.parent
        return subp

    def get_linear_subpattern(self) -> Optional[List[Union[int, str]]]:
        for l in self.leaves():
            subp = l.leaf_to_root()
            if subp:
                return subp
        return None

    def leaves(self) -> Generator['MinNode', None, None]:
        for child in self.children:
            yield from child.leaves()
        if not self.children:
            yield self

def reduce_tree(node: Any, parent: Optional[MinNode] = None) -> Optional[MinNode]:
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
        if (isinstance(node.children[0], pytree.Leaf) and node.children[0].value == '[') or (
            len(node.children) > 1 and hasattr(node.children[1], 'value') and node.children[1].value == '['):
            return None
        leaf = True
        details_node = None
        alternatives_node = None
        has_repeater = False
        repeater_node = None
        has_variable_name = False
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

def get_characteristic_subpattern(subpatterns: Union[List[List[Union[int, str]]], List[Union[int, str]]]) -> Union[List[Union[int, str]], Tuple]:
    if not isinstance(subpatterns, list):
        return subpatterns
    if len(subpatterns) == 1:
        return subpatterns[0]
    subpatterns_with_names: List[List[Union[int, str]]] = []
    subpatterns_with_common_names: List[List[Union[int, str]]] = []
    common_names = ['in', 'for', 'if', 'not', 'None']
    subpatterns_with_common_chars: List[List[Union[int, str]]] = []
    common_chars = '[]().,:'
    for subpattern in subpatterns:
        if any(rec_test(subpattern, lambda x: type(x) is str)):
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

def rec_test(sequence: Union[List[Any], Tuple[Any, ...]], test_func: Callable[[Any], bool]) -> Generator[bool, None, None]:
    for x in sequence:
        if isinstance(x, (list, tuple)):
            yield from rec_test(x, test_func)
        else:
            yield test_func(x)
