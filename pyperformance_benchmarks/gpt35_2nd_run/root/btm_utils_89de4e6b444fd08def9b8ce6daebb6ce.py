from typing import List, Tuple, Union

syms: dict
pysyms: dict
tokens: dict
token_labels: dict
TYPE_ANY: int
TYPE_ALTERNATIVES: int
TYPE_GROUP: int

class MinNode:
    def __init__(self, type: int = None, name: str = None) -> None:
        self.type: int = type
        self.name: str = name
        self.children: List[MinNode] = []
        self.leaf: bool = False
        self.parent: MinNode = None
        self.alternatives: List[List[Tuple[Union[int, str]]] = []
        self.group: List[List[Tuple[Union[int, str]]] = []

    def __repr__(self) -> str:
        return str(self.type) + ' ' + str(self.name)

    def leaf_to_root(self) -> List[Union[int, str]]:
        ...

    def get_linear_subpattern(self) -> List[Union[int, str]]:
        ...

    def leaves(self) -> List[MinNode]:
        ...

def reduce_tree(node: MinNode, parent: MinNode = None) -> MinNode:
    ...

def get_characteristic_subpattern(subpatterns: List[List[Tuple[Union[int, str]]]) -> List[Union[int, str]]:
    ...

def rec_test(sequence: List[Union[int, str]], test_func: callable) -> List[bool]:
    ...
