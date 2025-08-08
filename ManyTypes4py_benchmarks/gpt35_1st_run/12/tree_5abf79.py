from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines

def search_ancestor(node: NodeOrLeaf, *node_types: str) -> Optional[NodeOrLeaf]:
    ...

class NodeOrLeaf:
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

    def dump(self, *, indent: Union[int, str, None] = 4) -> str:
        ...

class Leaf(NodeOrLeaf):
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

class TypedLeaf(Leaf):
    ...

class BaseNode(NodeOrLeaf):
    ...

    @property
    def start_pos(self) -> Tuple[int, int]:
        ...

    def get_start_pos_of_prefix(self) -> Tuple[int, int]:
        ...

    @property
    def end_pos(self) -> Tuple[int, int]:
        ...

    def get_code(self, include_prefix: bool = True) -> str:
        ...

    def get_leaf_for_position(self, position: Tuple[int, int], include_prefixes: bool = False) -> Optional[Leaf]:
        ...

    def get_first_leaf(self) -> NodeOrLeaf:
        ...

    def get_last_leaf(self) -> NodeOrLeaf:
        ...

class Node(BaseNode):
    ...

class ErrorNode(BaseNode):
    ...

class ErrorLeaf(Leaf):
    ...
