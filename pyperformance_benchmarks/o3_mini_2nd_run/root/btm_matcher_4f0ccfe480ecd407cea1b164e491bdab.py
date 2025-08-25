from collections import defaultdict
import itertools
import logging
from typing import Any, DefaultDict, Dict, Iterator, List, Union, ClassVar

from . import pytree
from .btm_utils import reduce_tree

class BMNode(object):
    count: ClassVar[Iterator[int]] = itertools.count()

    def __init__(self) -> None:
        self.transition_table: Dict[Any, "BMNode"] = {}
        self.fixers: List[Any] = []
        self.id: int = next(BMNode.count)
        self.content: str = ''

class BottomMatcher(object):
    def __init__(self) -> None:
        self.match: set = set()
        self.root: BMNode = BMNode()
        self.nodes: List[BMNode] = [self.root]
        self.fixers: List[Any] = []
        self.logger: logging.Logger = logging.getLogger('RefactoringTool')

    def add_fixer(self, fixer: Any) -> None:
        self.fixers.append(fixer)
        tree = reduce_tree(fixer.pattern_tree)
        linear = tree.get_linear_subpattern()
        match_nodes: List[BMNode] = self.add(linear, start=self.root)
        for match_node in match_nodes:
            match_node.fixers.append(fixer)

    def add(self, pattern: List[Any], start: BMNode) -> List[BMNode]:
        if not pattern:
            return [start]
        if isinstance(pattern[0], tuple):
            match_nodes: List[BMNode] = []
            for alternative in pattern[0]:
                end_nodes = self.add(alternative, start=start)
                for end in end_nodes:
                    match_nodes.extend(self.add(pattern[1:], end))
            return match_nodes
        else:
            if pattern[0] not in start.transition_table:
                next_node = BMNode()
                start.transition_table[pattern[0]] = next_node
            else:
                next_node = start.transition_table[pattern[0]]
            if pattern[1:]:
                end_nodes = self.add(pattern[1:], start=next_node)
            else:
                end_nodes = [next_node]
            return end_nodes

    def run(self, leaves: List[Any]) -> DefaultDict[Any, List[Any]]:
        current_ac_node: BMNode = self.root
        results: DefaultDict[Any, List[Any]] = defaultdict(list)
        for leaf in leaves:
            current_ast_node = leaf
            while current_ast_node:
                current_ast_node.was_checked = True
                for child in current_ast_node.children:
                    if isinstance(child, pytree.Leaf) and (child.value == ';'):
                        current_ast_node.was_checked = False
                        break
                if current_ast_node.type == 1:
                    node_token = current_ast_node.value
                else:
                    node_token = current_ast_node.type
                if node_token in current_ac_node.transition_table:
                    current_ac_node = current_ac_node.transition_table[node_token]
                    for fixer in current_ac_node.fixers:
                        results[fixer].append(current_ast_node)
                else:
                    current_ac_node = self.root
                    if (current_ast_node.parent is not None) and current_ast_node.parent.was_checked:
                        break
                    if node_token in current_ac_node.transition_table:
                        current_ac_node = current_ac_node.transition_table[node_token]
                        for fixer in current_ac_node.fixers:
                            results[fixer].append(current_ast_node)
                current_ast_node = current_ast_node.parent
        return results

    def print_ac(self) -> None:
        print('digraph g{')

        def print_node(node: BMNode) -> None:
            for subnode_key in node.transition_table.keys():
                subnode = node.transition_table[subnode_key]
                print(f'{node.id} -> {subnode.id} [label={type_repr(subnode_key)}] //{str(subnode.fixers)}')
                if subnode_key == 1:
                    print(subnode.content)
                print_node(subnode)
        print_node(self.root)
        print('}')

_type_reprs: Dict[int, Union[str, int]] = {}

def type_repr(type_num: int) -> Union[str, int]:
    global _type_reprs
    if not _type_reprs:
        from .pygram import python_symbols
        for name, val in python_symbols.__dict__.items():
            if isinstance(val, int):
                _type_reprs[val] = name
    return _type_reprs.setdefault(type_num, type_num)