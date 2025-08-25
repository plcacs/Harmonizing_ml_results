from typing import List, Tuple

syms: dict
pysyms: dict
tokens: dict
token_labels: dict
TYPE_ANY: int
TYPE_ALTERNATIVES: int
TYPE_GROUP: int

class MinNode:
    def __init__(self, type: int = None, name: str = None) -> None:
    def __repr__(self) -> str:
    def leaf_to_root(self) -> List:
    def get_linear_subpattern(self) -> List:
    def leaves(self) -> List
    def reduce_tree(node, parent=None) -> MinNode
    def get_characteristic_subpattern(subpatterns: List) -> List
    def rec_test(sequence, test_func) -> List
