from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines

def search_ancestor(node: NodeOrLeaf, *node_types: str) -> Optional[NodeOrLeaf]:
    ...

class NodeOrLeaf:
    __slots__ = ('parent',)

    def get_root_node(self) -> NodeOrLeaf:
        ...

    def get_next_sibling(self) -> Optional[NodeOrLeaf]:
        ...

    def get_previous_sibling(self) -> Optional[NodeOrLeaf]:
        ...

    def get_previous_leaf(self) -> Optional[NodeOrLeaf]:
        ...

    def get_next_leaf(self) -> Optional[NodeOrLeaf]:
        ...

    @abstractproperty
    def start_pos(self) -> Tuple[int, int]:
        ...

    @abstractproperty
    def end_pos(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def get_first_leaf(self) -> NodeOrLeaf:
        ...

    @abstractmethod
    def get_last_leaf(self) -> NodeOrLeaf:
        ...

    @abstractmethod
    def get_code(self, include_prefix: bool = True) -> str:
        ...

    def search_ancestor(self, *node_types: str) -> Optional[NodeOrLeaf]:
        ...

    def dump(self, *, indent: int = 4) -> str:
        ...

class Leaf(NodeOrLeaf):
    __slots__ = ('value', 'line', 'column', 'prefix')

    def __init__(self, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None:
        ...

    @property
    def start_pos(self) -> Tuple[int, int]:
        return self.start_pos

    @start_pos.setter
    def start_pos(self, value: Tuple[int, int]) -> None:
        ...

    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        ...

    def get_first_leaf(self) -> NodeOrLeaf:
        return self

    def get_last_leaf(self) -> NodeOrLeaf:
        return self

    def get_code(self, include_prefix: bool = True) -> str:
        ...

    @property
    def end_pos(self) -> Tuple[int, int]:
        ...

    def __repr__(self) -> str:
        ...

class TypedLeaf(Leaf):
    __slots__ = ('type',)

    def __init__(self, type: str, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None:
        ...

class BaseNode(NodeOrLeaf):
    __slots__ = ('children',)

    def __init__(self, children: List[NodeOrLeaf]) -> None:
        ...

    @property
    def start_pos(self) -> Tuple[int, int]:
        return self.children[0].start_pos

    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        return self.children[0].get_start_pos_of_prefix()

    @property
    def end_pos(self) -> Tuple[int, int]:
        return self.children[-1].end_pos

    def get_code(self, include_prefix: bool = True) -> str:
        ...

    def get_leaf_for_position(self, position: Tuple[int, int], include_prefixes: bool = False) -> Optional[Leaf]:
        ...

    def get_first_leaf(self) -> NodeOrLeaf:
        return self.children[0].get_first_leaf()

    def get_last_leaf(self) -> NodeOrLeaf:
        return self.children[-1].get_last_leaf()

    def __repr__(self) -> str:
        ...

class Node(BaseNode):
    __slots__ = ('type',)

    def __init__(self, type: str, children: List[NodeOrLeaf]) -> None:
        ...

class ErrorNode(BaseNode):
    __slots__ = ()
    type = 'error_node'

class ErrorLeaf(Leaf):
    __slots__ = ('token_type',)
    type = 'error_leaf'

    def __init__(self, token_type: str, value: str, start_pos: Tuple[int, int], prefix: str = '') -> None:
        ...
