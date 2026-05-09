from typing import List, Optional, Tuple, Union
from parso.utils import split_lines

def search_ancestor(node: NodeOrLeaf, *node_types: str) -> Optional[NodeOrLeaf]:
    ...

class NodeOrLeaf:
    parent: Optional[BaseNode]
    __slots__: Tuple[str, ...] = ('parent',)

    def get_root_node(self) -> BaseNode:
        ...

    def get_next_sibling(self) -> Optional[NodeOrLeaf]:
        ...

    def get_previous_sibling(self) -> Optional[NodeOrLeaf]:
        ...

    def get_previous_leaf(self) -> Optional[Leaf]:
        ...

    def get_next_leaf(self) -> Optional[Leaf]:
        ...

    @property
    def start_pos(self) -> Tuple[int, int]:
        ...

    @property
    def end_pos(self) -> Tuple[int, int]:
        ...

    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        ...

    def get_first_leaf(self) -> NodeOrLeaf:
        ...

    def get_last_leaf(self) -> NodeOrLeaf:
        ...

    def get_code(self, include_prefix: bool = True) -> str:
        ...

    def search_ancestor(self, *node_types: str) -> Optional[NodeOrLeaf]:
        ...

    def dump(self, indent: Union[int, str, None] = 4) -> str:
        ...

class Leaf(NodeOrLeaf):
    value: str
    line: int
    column: int
    prefix: str
    __slots__: Tuple[str, ...] = ('value', 'line', 'column', 'prefix')

    def __init__(self, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None:
        ...

    @property
    def start_pos(self) -> Tuple[int, int]:
        ...

    @start_pos.setter
    def start_pos(self, value: Tuple[int, int]) -> None:
        ...

    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        ...

    def get_first_leaf(self) -> Leaf:
        ...

    def get_last_leaf(self) -> Leaf:
        ...

    def get_code(self, include_prefix: bool = True) -> str:
        ...

    @property
    def end_pos(self) -> Tuple[int, int]:
        ...

    def __repr__(self) -> str:
        ...

class TypedLeaf(Leaf):
    type: str
    __slots__: Tuple[str, ...] = ('type',)

    def __init__(self, type: str, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None:
        ...

class BaseNode(NodeOrLeaf):
    children: List[NodeOrLeaf]
    __slots__: Tuple[str, ...] = ('children',)

    def __init__(self, children: List[NodeOrLeaf]) -> None:
        ...

    @property
    def start_pos(self) -> Tuple[int, int]:
        ...

    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        ...

    @property
    def end_pos(self) -> Tuple[int, int]:
        ...

    def _get_code_for_children(self, children: List[NodeOrLeaf], include_prefix: bool) -> str:
        ...

    def get_code(self, include_prefix: bool = True) -> str:
        ...

    def get_leaf_for_position(self, position: Tuple[int, int], include_prefixes: bool = False) -> Optional[Leaf]:
        ...

    def get_first_leaf(self) -> NodeOrLeaf:
        ...

    def get_last_leaf(self) -> NodeOrLeaf:
        ...

    def __repr__(self) -> str:
        ...

class Node(BaseNode):
    type: str
    __slots__: Tuple[str, ...] = ('type',)

    def __init__(self, type: str, children: List[NodeOrLeaf]) -> None:
        ...

    def __repr__(self) -> str:
        ...

class ErrorNode(BaseNode):
    type: str
    __slots__: Tuple[str, ...] = ()

class ErrorLeaf(Leaf):
    token_type: str
    __slots__: Tuple[str, ...] = ('token_type',)

    def __init__(self, token_type: str, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None:
        ...

    def __repr__(self) -> str:
        ...